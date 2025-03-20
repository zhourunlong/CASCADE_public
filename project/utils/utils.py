from project.datasets.lm.lm_dataset import LMDataset, CompositeLMDataset, CompositeEvenDistrLMDataset

def gen_random_strs(config, token_range):
    rstr_conf = config["insert_random_str"]
    rstr_conf["tokens"] = [
        random.choices(token_range,
                       k=random.randrange(rstr_conf["min_len"],
                                          rstr_conf["max_len"] + 1))
        for _ in range(rstr_conf["num_strs"])
    ]

def find_unique_prefixes(all_tokens):
    """
    Returns the first position where all tokens have unique prefixes.
    This position is the first token to generate.
    """
    start_pos = 1
    while True:
        prefixes = [tuple(tokens[:start_pos]) for tokens in all_tokens]
        if len(set(prefixes)) == len(all_tokens):
            break
        start_pos += 1
    return start_pos

def build_datasets(config, training_args, add_fiqj, all_occur, split):
    lm_datasets = []

    min_len = min([len(np.fromfile(dataset[split], dtype=np.uint16)) for dataset in config["dataset"]])
    seq_len = training_args.seq_len if split == "train" else training_args.eval_seq_len
    seq_overlap = training_args.seq_overlap if split == "train" else 0

    for i, dataset in enumerate(config["dataset"]):
        # trunc to same length
        input_ids = np.fromfile(dataset[split], dtype=np.uint16)[:min_len]
        
        if "insert_random_str" not in dataset:
            lm_datasets.append(LMDataset(input_ids=input_ids,
                                         seq_len=seq_len,
                                         seq_overlap=seq_overlap,
                                         dataset_tag=dataset["tag"]))
            continue

        original_conf = dataset["insert_random_str"]

        new_conf = {
            "tokens": original_conf["tokens"].copy(),
            "occurs": [original_conf["in_mode_each_occur"] for _ in range(original_conf["num_strs"])],
            "tags": [dataset["tag"] for _ in range(len(original_conf["tokens"]))],
            "fix_rstr_loc": original_conf.get("fix_rstr_loc", None),
        }

        if add_fiqj:
            for j, d in enumerate(config["dataset"]):
                if j != i and "insert_random_str" in d:
                    num_strs = (len(d["insert_random_str"]["tokens"]) + 1) // 2
                    new_conf["tokens"].extend(d["insert_random_str"]["tokens"][:num_strs].copy())
                    new_conf["occurs"].extend([original_conf["cross_mode_each_occur"] for _ in range(num_strs)])
                    new_conf["tags"].extend([d["tag"] for _ in range(num_strs)])
        
        if all_occur:
            tmp_dataset = LMDataset(input_ids, seq_len=seq_len, portion=training_args.num_train_epochs)
            multiplicity = len(tmp_dataset) // sum(new_conf["occurs"])
            new_conf["occurs"] = [o * multiplicity for o in new_conf["occurs"]]
            remain = len(tmp_dataset) - sum(new_conf["occurs"])
            for i in range(remain):
                new_conf["occurs"][i % len(new_conf["occurs"])] += 1

        # first insert random strings with seq_len=1024
        _dataset = LMDataset(input_ids=input_ids,
                             seq_len=1024,
                             dataset_tag=dataset["tag"],
                             **new_conf)

        lm_datasets.append(LMDataset(input_ids=_dataset._input_ids,
                                     seq_len=seq_len,
                                     seq_overlap=seq_overlap,
                                     dataset_tag=dataset["tag"]))

    if (split == "train" and not training_args.dataloader_shuffle):
        return CompositeEvenDistrLMDataset(lm_datasets)
    else:
        return CompositeLMDataset(lm_datasets)

def gen_eval_confs(config, prefix, seq_len, fiqj_remove_half, multiplicity):
    """
    Returns a dict: {start_pos: [eval_conf]}
        `start_pos` is the position of the first token to be generated
        `[eval_conf]` is a list of dict {"input_ids": input_ids, "tags": tag}
    """
    eval_confs = {}

    def add_entries(input_ids, start_pos, tag):
        if start_pos not in eval_confs:
            eval_confs[start_pos] = []
        for k in range(input_ids.shape[0]):
            eval_confs[start_pos].append({
                "input_ids": input_ids[k],
                "tags": tag,
            })

    for i, dataset in enumerate(config["dataset"]):
        if "insert_random_str" not in dataset:
            continue

        di = LMDataset(input_ids=np.fromfile(dataset["eval"], dtype=np.uint16),
                       seq_len=seq_len)
        
        input_ids = torch.stack([di[k]["input_ids"].clone()
                                 for k in range(multiplicity)])
        add_entries(input_ids, 10, dataset['tag'])

        cnt = multiplicity
        for j, d in enumerate(config["dataset"]):
            if "insert_random_str" not in d:
                continue
            
            if j != i and fiqj_remove_half:
                r_tokens = d["insert_random_str"]["tokens"][(d["insert_random_str"]["num_strs"] + 1) // 2:].copy()
            else:
                r_tokens = d["insert_random_str"]["tokens"].copy()

            for tokens in r_tokens:
                tokens = torch.as_tensor(tokens)
                input_ids = torch.stack([di[cnt + k]["input_ids"].clone() for k in range(multiplicity)])
                cnt += multiplicity
                input_ids[:, -tokens.shape[0]:] = tokens

                start_pos = input_ids.shape[1] - tokens.shape[0] + prefix
                tag = f"f{dataset['tag']}q{d['tag']}"
                add_entries(input_ids, start_pos, tag)
    
    return eval_confs

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
