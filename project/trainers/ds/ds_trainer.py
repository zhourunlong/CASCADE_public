from project.utils.cascade_utils import (
    build_cascade_batch,
    get_logits,
    get_ensemble_logp,
)


class MultiModeDsTrainer(DsTrainer):
    def __init__(self, eval_confs, loss_only_half, **kwargs):
        super().__init__(**kwargs)
        self.eval_confs = eval_confs
        self.eval_all_tags = set([x["tags"] for conf in eval_confs.values() for x in conf])

        self.loss_only_half = loss_only_half

        self.sample_batch_enlarger = kwargs["args"].sample_batch_enlarger
        self.update_per_sample = kwargs["args"].update_per_sample
        self.num_sample_batch = kwargs["args"].num_sample_batch
        self.grad_acc_steps = self.engine.gradient_accumulation_steps()

        assert self.grad_acc_steps * self.num_sample_batch % self.sample_batch_enlarger == 0, "Currently don't support this case"

    def _do_evaluate(self, step):
        if self.eval_confs is None:
            super()._do_evaluate(step)
            return

        bs = self.args.eval_micro_batch_size_per_gpu

        self.engine.eval()
        results = {tag: [] for tag in self.eval_all_tags}
        for start_pos, conf in self.eval_confs.items():
            tot_data = len(conf)
            num_per_device = (tot_data - 1) // self.engine.world_size + 1
            start_idx = num_per_device * self.engine.global_rank
            end_idx = min(start_idx + num_per_device, tot_data)

            input_ids = []
            tags = []
            for i in range(start_idx, end_idx):
                input_ids.append(conf[i]["input_ids"])
                tags.append(conf[i]["tags"])
            
            input_ids = torch.stack(input_ids, dim=0).to(self.engine.device)

            log_probs = []
            for batch_start in range(0, len(input_ids), bs):
                batch_end = min(batch_start + bs, len(input_ids))
                batch_input = input_ids[batch_start:batch_end]
                batch_logp = torch.zeros((batch_input.shape[0], batch_input.shape[1] - start_pos), dtype=torch.float, device=self.engine.device)
                
                for cur_pos in range(start_pos, batch_input.shape[1], self.seq_len // 2):
                    L = max(cur_pos - self.seq_len // 2, 0)
                    R = min(L + self.seq_len, batch_input.shape[1])
                    split_pos = cur_pos - L
                    with torch.no_grad():
                        pos_logp = torch.log_softmax(
                            self.engine(batch_input[:, L:R], return_dict=True).logits[:, split_pos-1:-1, :],
                            dim=-1
                        )

                    labels = batch_input[:, cur_pos:R]
                    gathered_logp = torch.gather(pos_logp, 2, labels.unsqueeze(-1)).squeeze(-1)
                    batch_logp[:, cur_pos-start_pos:R-start_pos] = gathered_logp
                
                log_probs.append(batch_logp.mean(dim=-1))
            
            log_probs = torch.cat(log_probs, dim=0)

            for i in range(log_probs.shape[0]):
                results[tags[i]].append(log_probs[i].item())
        self.engine.train()

        type_logp = {}
        for k, v in results.items():
            type_logp[k] = {
                "count": len(v),
                "sum": sum(v),
            }

        eval_metrics = {"train/step": step}
        for k, v in sorted(type_logp.items()):
            c = torch.tensor(v["count"], device=self.engine.device)
            s = torch.tensor(v["sum"], dtype=torch.float, device=self.engine.device)
            torch.distributed.reduce(c, 0)
            torch.distributed.reduce(s, 0)

            if c.item() > 0:
                eval_metrics[f"eval/mean_log_prob/{k}"] = s.item() / c.item()

        self.callback_handler.on_evaluate(self.engine, self.args, self.client_state)

        eval_metrics.update({"eval/idx": step // self.args.eval_steps})

        if self.engine.global_rank == 0:
            self.client_state["log_history"].append(eval_metrics)
            logger.info(eval_metrics)

class CascadeDsTrainer(MultiModeDsTrainer):
    def _train_batch_without_pipe_parallel(
        self, data_iter: Optional[Iterator] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        gradient_accumulation_steps = self.engine.gradient_accumulation_steps()

        total_loss = torch.tensor(0.0, device=self.engine.device)
        total_aux_loss = torch.tensor(0.0, device=self.engine.device)
        total_dropped_tokens = torch.tensor(0.0, device=self.engine.device)

        type_losses = {tag: [] for tag in self.all_tags}
        
        for _ in range(gradient_accumulation_steps):
            _batch = next(data_iter)
            _batch = {k: v.to(self.engine.device) if isinstance(v, torch.Tensor) else v for k, v in _batch.items()}

            cascade_batch = build_cascade_batch(_batch, self.args.seq_overlap)
            for batch in cascade_batch:
                # `start_pos` is the position of the first token to be predicted
                if hasattr(self, "loss_only_half") and self.loss_only_half:
                    start_pos = batch["input_ids"].shape[1] // 2
                else:
                    start_pos = 1

                outputs = self.engine(**batch)

                losses_flat = F.cross_entropy(
                    outputs.logits[:, start_pos-1:-1].contiguous().view(-1, outputs.logits.shape[-1]),
                    batch["labels"][:, start_pos:].contiguous().view(-1),
                    reduction="none"
                )
                loss = losses_flat.mean() / len(cascade_batch)

                losses = losses_flat.detach().view(outputs.logits.shape[0], -1).mean(dim=1)
                for i, type_tag in enumerate(batch["type_tags"]):
                    type_losses[type_tag].append(losses[i].item())

                if self.use_moe:
                    raise NotImplementedError("CascadeDsTrainer doesn't support MoE")
                    aux_loss = outputs.aux_loss.mean()
                    dropped_tokens = outputs.dropped_tokens.mean()

                    loss += aux_loss

                    total_aux_loss += aux_loss
                    total_dropped_tokens += dropped_tokens

                self.engine.backward(loss)
            self.engine.step()

            total_loss += loss

        total_loss /= gradient_accumulation_steps
        total_aux_loss /= gradient_accumulation_steps
        total_dropped_tokens /= gradient_accumulation_steps
        
        extra_info = {}
        for key in type_losses:
            extra_info[f"{key}"] = {
                "count": len(type_losses[key]),
                "sum": sum(type_losses[key]),
            }

        return total_loss, total_aux_loss, total_dropped_tokens, extra_info
    
    def _do_evaluate(self, step):
        _shard_len = 128

        bs = self.args.eval_micro_batch_size_per_gpu

        self.engine.eval()
        results = {tag: [] for tag in self.eval_all_tags}
        for start_pos, conf in self.eval_confs.items():
            tot_data = len(conf)
            num_per_device = (tot_data - 1) // self.engine.world_size + 1
            start_idx = num_per_device * self.engine.global_rank
            end_idx = min(start_idx + num_per_device, tot_data)

            input_ids = []
            tags = []
            for i in range(start_idx, end_idx):
                input_ids.append(conf[i]["input_ids"])
                tags.append(conf[i]["tags"])
            
            input_ids = torch.stack(input_ids, dim=0).to(self.engine.device)

            log_probs = []
            for batch_start in range(0, len(input_ids), bs):
                batch_end = min(batch_start + bs, len(input_ids))
                batch_input = input_ids[batch_start:batch_end]
                batch_logp = []
                seq_len = batch_input.shape[1]

                for _start_pos in range(start_pos, seq_len, _shard_len):
                    _seq_len = min(_start_pos + _shard_len, seq_len)
                    logits = []
                    context_len = 8
                    while context_len <= seq_len:
                        with torch.no_grad():
                            logits.append(get_logits(
                                model=self.engine,
                                input_ids=batch_input,
                                start_pos=_start_pos,
                                seq_len=_seq_len,
                                context_len=context_len,
                                meta_batch_size=seq_len // context_len,
                            ))

                        context_len *= 2

                    ensemble_logp, weights = get_ensemble_logp(
                        logits,
                        batch_input[:, _start_pos:_seq_len],
                    )
                    batch_logp.append(ensemble_logp)
                
                batch_logp = torch.cat(batch_logp, dim=1)
                log_probs.append(batch_logp.mean(dim=-1))
            
            log_probs = torch.cat(log_probs, dim=0)

            for i in range(log_probs.shape[0]):
                results[tags[i]].append(log_probs[i].item())

        self.engine.train()

        type_logp = {}
        for k, v in results.items():
            type_logp[k] = {
                "count": len(v),
                "sum": sum(v),
            }

        eval_metrics = {"train/step": step}
        for k, v in sorted(type_logp.items()):
            c = torch.tensor(v["count"], device=self.engine.device)
            s = torch.tensor(v["sum"], dtype=torch.float, device=self.engine.device)
            torch.distributed.reduce(c, 0)
            torch.distributed.reduce(s, 0)

            if c.item() > 0:
                eval_metrics[f"eval/mean_log_prob/{k}"] = s.item() / c.item()

        self.callback_handler.on_evaluate(self.engine, self.args, self.client_state)

        eval_metrics.update({"eval/idx": step // self.args.eval_steps})

        if self.engine.global_rank == 0:
            self.client_state["log_history"].append(eval_metrics)
            logger.info(eval_metrics)