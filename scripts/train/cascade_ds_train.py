from project.trainers.ds.ds_trainer import CascadeDsTrainer

from project.utils.utils import gen_random_strs, gen_eval_confs, find_unique_prefixes, set_seed, build_datasets

if __name__ == "__main__":
    args, extra_args = parse_args()
    config = process_config(load_config(*args.config_file_path, extra_args))

    node_id = os.environ.get("USER_NODE_CODE_PACKAGE_INSTANCE_ID", None)
    if node_id is not None:
        print(f"USER_NODE_CODE_PACKAGE_INSTANCE_ID: {node_id}")

    # Load model from configuration
    assert "model" in config, "`model` must be available in configuration."
    model = get_model(**config["model"])

    # Load previous model, finetune
    is_finetune = config.get("is_finetune", False)
    if is_finetune:
        # Copy dataset config
        # No need to process previous config
        prev_config = load_config(
            os.path.join(config["model"]["pretrained_model_name_or_path"],
                         "config.yaml"),
            extra_args
        )
        for d in config["dataset"]:
            for pd in prev_config["dataset"]:
                if "insert_random_str" in d and (d["train"] == pd["train"]) and (d["eval"] == pd["eval"]):
                    d["insert_random_str"].update({k: v for k, v in pd["insert_random_str"].items() if k not in d["insert_random_str"]})
                    break

    # Load training arguments from configuration and asserts whether
    # `pipe_parallel_loss_fn` should be specified
    assert "output_dir" in config, "`output_dir` must be available in configuration."
    assert "training_args" in config, "`training_args` must be available in configuration."

    training_args = DsTrainingArguments(config["output_dir"], **config["training_args"])

    training_args.pipe_parallel_loss_fn = model.loss if training_args.pipe_parallel_size > 0 else None
    if args.checkpoint_dir is None:
        args.checkpoint_dir = training_args.output_dir

    assert "dataset" in config, "`dataset` must be available in configuration."

    set_seed(config["dataset_seed"])

    # use last 8 tokens from vocab as random strings
    token_range = range(model.config.vocab_size - 8, model.config.vocab_size)

    train_datasets = []
    all_tokens = []
    for i, dataset in enumerate(config["dataset"]):
        if "insert_random_str" not in dataset:
            continue
        if "tokens" in dataset["insert_random_str"]:
            # check if the lengths are consistent
            target_rstr_len_range = range(
                dataset["insert_random_str"]["min_len"],
                dataset["insert_random_str"]["max_len"]
            )
            d_all_tokens = []
            for tokens in dataset["insert_random_str"]["tokens"]:
                if len(tokens) not in target_rstr_len_range:
                    # tokens = np.repeat(tokens, (tgt_rstr_len - 1) // len(tokens) + 1)[:tgt_rstr_len].tolist()
                    # TODO
                    raise ValueError(f"Random string length mismatch: {len(tokens)} not in {target_rstr_len_range}")
                
                d_all_tokens.append(tokens)
            dataset["insert_random_str"]["tokens"] = d_all_tokens
        else:
            gen_random_strs(dataset, token_range)
        all_tokens.extend(dataset["insert_random_str"]["tokens"])
    rstr_len = len(all_tokens[0]) if len(all_tokens) > 0 else 0
    
    fiqj_status = [dataset.get("insert_random_str", {}).get("cross_mode_each_occur", 0) > 0 for dataset in config["dataset"]]
    add_fiqj = any(fiqj_status)
    
    train_dataset = build_datasets(
        config=config,
        training_args=training_args,
        all_occur=is_finetune,
        add_fiqj=add_fiqj,
        split="train",
    )

    # Save the input configuration only on the global main process
    if training_args.is_main_process:
        save_config(config, os.path.join(training_args.output_dir, "config.yaml"))
    deepspeed.comm.barrier()

    eval_confs = gen_eval_confs(
        config=config,
        prefix=find_unique_prefixes(all_tokens),
        seq_len=training_args.eval_seq_len,
        # fiqj_remove_half=add_fiqj,
        fiqj_remove_half=True,
        multiplicity=training_args.eval_multiplicity,
    )

    set_seed(training_args.seed)
    trainer = CascadeDsTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=[],
        eval_confs=eval_confs,
        loss_only_half=training_args.seq_overlap,
    )

    trainer.train()

    wandb.finish()
    