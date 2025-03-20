def build_cascade_batch(batch, original_data_is_overlap):
    len = batch["input_ids"].shape[1]
    cascade_batch = []
    l = 8
    end_pos = len // 2 if original_data_is_overlap else len

    while l <= len:
        input_ids = []
        labels = []
        type_tags = []

        for i in range(0, end_pos, l):
            input_ids.append(batch["input_ids"][:, i:i+l])
            labels.append(batch["labels"][:, i:i+l])
            type_tags.extend(batch["type_tags"])
        
        if len % l != 0 and len - l < end_pos:
            input_ids.append(batch["input_ids"][:, len-l:])
            labels.append(batch["labels"][:, len-l:])
            type_tags.extend(batch["type_tags"])
        
        cascade_batch.append({
            "input_ids": torch.cat(input_ids, dim=0),
            "labels": torch.cat(labels, dim=0),
            "type_tags": type_tags,
        })

        l *= 2
    
    return cascade_batch

def get_logits(model, input_ids, start_pos, seq_len, context_len, meta_batch_size):
    batch_size = input_ids.shape[0]
    _input_ids = []
    for i in range(start_pos, seq_len, context_len // 2):
        L = max(i - context_len // 2, 0)
        R = min(i + context_len // 2, seq_len)
        _input_ids.append(input_ids[:, L:R].clone())

    # first batch may have incomplete context
    first_batch_loc = start_pos - max(start_pos - context_len // 2, 0)
    model_k_logits = [model(_input_ids[0], return_dict=True).logits[:, first_batch_loc-1:-1]]

    # middle batches
    for i in range(1, len(_input_ids) - 1, meta_batch_size):
        c_input_ids = torch.cat(
            _input_ids[i:min(i+meta_batch_size,len(_input_ids)-1)],
            dim=0
        )
        _logits = model(c_input_ids, return_dict=True).logits[:, context_len//2-1:-1]
        for j in range(0, c_input_ids.shape[0], batch_size):
            model_k_logits.append(_logits[j:j+batch_size])
    
    # last batch may have incomplete completion
    if len(_input_ids) > 1:
        model_k_logits.append(model(_input_ids[-1], return_dict=True).logits[:, context_len//2-1:-1])
    
    torch.cuda.empty_cache()
    
    return torch.cat(model_k_logits, dim=1)

def get_ensemble_logp(logits, labels):
    """
    Args:
        `logits`: (bs, num_models, len, vocab_size)
        `labels`: (bs, len)
    
    Returns:
        `ensemble_logp`: (bs, len)
        `weights`: (bs, num_models, len)
    """
    device = logits.device
    batch_size, completion_len = labels.shape

    logp_by_model = torch.log_softmax(logits, dim=-1) # (bs, num_models, len, vocab_size)

    max1, argmax1 = torch.max(logp_by_model, dim=-1) # (bs, num_models, len)
    weights = 1 / (1e-9 - max1)
    weights = weights / torch.sum(weights, dim=1, keepdim=True) # (bs, num_models, len)
    ensemble_logp = torch.sum(weights.unsqueeze(-1) * logp_by_model, dim=1) # (bs, len, vocab_size)
    ensemble_logp = torch.log_softmax(ensemble_logp, dim=-1) # (bs, len, vocab_size)
    
    ax1 = torch.arange(0, batch_size, device=device, dtype=torch.long).unsqueeze(1).expand(labels.shape)
    ax2 = torch.arange(0, completion_len, device=device, dtype=torch.long).unsqueeze(0).expand(labels.shape)
    return ensemble_logp[ax1, ax2, labels], weights

def eval_cascade(models, input_ids, start_pos, ablation=False, device="cuda", _shard_len=64):
    """
    Args:
        `models`: list of models, each being a dict of
            {"model": model,
             "context_len": context_len,}
        `input_ids`: shape (batch_size, seq_len)
        `start_pos`: int, the position of the first token to predict
        `ablation`: bool, whether to do ablation on models
        `device`: str, device to use
        `_shard_len`: int, the length of each shard
    
    Returns:
        `log_probs`: (bs, seq_len-start_pos)
        `weights`: (bs, num_models, seq_len-start_pos)
        `log_probs_wo_i`: list of (bs, seq_len-start_pos) for each model
    """
    max_ctx_len = 0
    for dict in models:
        max_ctx_len = max(max_ctx_len, dict["context_len"])
        model = dict["model"]
        model = to_device_map(model, device_map=device)
        model = model.half()
        model.eval()

    bs, seq_len = input_ids.shape # bs is for the longest context length
    log_probs = []
    weights = []
    log_probs_wo_i = [[] for _ in models]
    with torch.no_grad():
        for _start_pos in range(start_pos, seq_len, _shard_len):
            _seq_len = min(_start_pos + _shard_len, seq_len)
            logits = []
            for dict in models:
                model, context_len = dict["model"], dict["context_len"]
                
                logits.append(
                    get_logits(
                        model=model,
                        input_ids=input_ids,
                        start_pos=_start_pos,
                        seq_len=_seq_len,
                        context_len=context_len,
                        meta_batch_size=max_ctx_len // context_len,
                    ) # (bs, _seq_len-_start_pos, vocab_size)
                )
    
            logits = torch.stack(logits, dim=1) # (bs, num_models, _seq_len-_start_pos, vocab_size)

            ensemble_logp, weight = get_ensemble_logp(
                logits,
                input_ids[:, _start_pos:_seq_len],
            )
            
            log_probs.append(
                ensemble_logp # (bs, _seq_len-_start_pos)
            )
            weights.append(
                weight # (bs, num_models, _seq_len-_start_pos)
            )

            if ablation:
                for i in range(len(models)):
                    # delete model i
                    logits_wo_i = torch.cat([logits[:, :i], logits[:, i+1:]], dim=1)

                    ensemble_logp_wo_i, _ = get_ensemble_logp(
                        logits_wo_i,
                        input_ids[:, _start_pos:_seq_len],
                    )

                    log_probs_wo_i[i].append(ensemble_logp_wo_i)

    log_probs = torch.cat(log_probs, dim=-1)
    weights = torch.cat(weights, dim=-1)
    log_probs_wo_i = [torch.cat(log_probs_wo_i[i], dim=-1) for i in range(len(models))]

    return log_probs, weights, log_probs_wo_i


def gen_cascade(models, input_ids, max_new_tokens=5, device="cuda"):
    """
    Args:
        `models`: list of models, each being a dict of
            {"model": model,
             "context_len": context_len,}
        `input_ids`: shape (batch_size, seq_len)
        `max_new_tokens`: int, number of tokens to generate
        `predict_len`: int, number of tokens to predict for each position
        `device`: str, device to use
    """
    for dict in models:
        model = dict["model"]
        model = to_device_map(model, device_map=device)
        model = model.half()
        model.eval()
    
    bs, seq_len = input_ids.shape
    axis = torch.arange(0, bs, device=device, dtype=torch.long)
    log_probs = []
    ks = []
    with torch.no_grad():    
        for i in range(seq_len, seq_len + max_new_tokens):
            logits = []
            for dict in models:
                model, context_len = dict["model"], dict["context_len"]
                L = max(i - context_len // 2, 0)
                c_input_ids = input_ids[:, L:].clone()

                logits.append(model(c_input_ids, return_dict=True).logits[:, -1])
            
            logits = torch.stack(logits, dim=1) # (bs, num_models, vocab_size)

            logp_by_model = torch.log_softmax(logits, dim=-1) # (bs, num_models, vocab_size)

            max1, argmax1 = torch.max(logp_by_model, dim=-1) # (bs, num_models)
            weights = torch.softmax(max1, dim=1) # (bs, num_models)
            ensemble_logp = torch.sum(weights.unsqueeze(-1) * logp_by_model, dim=1) # (bs, vocab_size)
            ensemble_p = torch.softmax(ensemble_logp, dim=-1) # (bs, vocab_size)
            ensemble_logp = torch.log_softmax(ensemble_logp, dim=-1) # (bs, vocab_size)

            generated_ids = torch.multinomial(ensemble_p, num_samples=1).squeeze(-1) # (bs,)
            log_probs.append(ensemble_logp[axis, generated_ids]) # (bs,)
            ks.append(weights) # (bs, num_models)

            input_ids = torch.cat([input_ids, generated_ids.unsqueeze(-1)], dim=1)
    
    log_probs = torch.stack(log_probs, dim=1) # (bs, max_new_tokens)

    return input_ids, log_probs, torch.stack(ks, dim=-1).mean(-1)
