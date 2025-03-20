class LMDataset(Dataset):
    """Language modeling dataset.

    This dataset is used for language modeling tasks. It takes a contiguous array
    of tokens and convert it into a dataset of sequences of length ``seq_len`` by
    sliding a window of size ``seq_len`` over the array of tokens.

    Each sequence is composed of ``input_ids`` and ``labels`` tensors.

    """

    def __init__(
        self,
        input_ids: np.array,
        seq_len: int = 1024,
        seq_overlap: bool = False,
        shift_labels: bool = False,
        ignore_token_id: int = -100,
        mask_prob: Optional[float] = None,
        mask_offset: int = 32_768,
        occurs = [],
        tokens = None,
        dataset_tag = None,
        tags = None,
        fix_rstr_loc = None,
        **kwargs
    ) -> None:
        """Initialize the dataset.

        Args:
            input_ids: Inputs array (encoded data).
            seq_len: Sequence length.
            shift_labels: Whether labels should be shifted by one position.
            ignore_token_id: Index to be ignored in the labels.
            mask_prob: Percentage of sample/seq_len applied to the ``no_loss`` logic.
                If ``None``, ignores masking.
            mask_offset: Value added to the token that will cause it to be masked.
            seed: Seed for the mask random number generator.

        """

        super().__init__()

        assert seq_len % 2 == 0, "Sequence length must be even."

        # `input_ids` should not be sliced since they could be memory mapped
        if seq_overlap:
            self._input_ids = np.concatenate([input_ids, input_ids[seq_len // 2: -seq_len // 2]])
        else:
            self._input_ids = input_ids
        self._seq_len = seq_len

        self._n_input_ids = ((len(self._input_ids) - 1) // self._seq_len) * self._seq_len + 1
        self._n_sequences = math.ceil((self._n_input_ids - 1) / self._seq_len)

        self._label_offset = 1 if shift_labels else 0
        self._ignore_token_id = ignore_token_id
        self._mask_prob = mask_prob
        self._mask_offset = mask_offset

        self.dataset_tag = dataset_tag or ""

        self.occurs = occurs
        self.tokens = tokens
        self.type_tags = ["" for _ in range(self._n_sequences)]
        self.all_tags = [dataset_tag]
        if tags is not None:
            for tag in set(tags):
                self.all_tags.append(f"f{dataset_tag}q{tag}")
        
        if fix_rstr_loc == None:
            fix_rstr_loc = "none"
        
        if fix_rstr_loc not in ["start", "end", "none"]:
            raise ValueError(f"Invalid fix_rstr_loc value: {fix_rstr_loc}")

        if sum(self.occurs) > 0:
            # use a fixed dataset for all configurations
            self._rng = np.random.default_rng(42)

            self.overwrite_idxs = self._rng.choice(self._n_sequences, size=sum(self.occurs), replace=False)

            str_idx, cur_cnt = 0, 0
            for idx in tqdm(self.overwrite_idxs):
                self.type_tags[idx] = tags[str_idx]

                start_idx = idx * self._seq_len
                seq_len = min(self._seq_len, self._n_input_ids - 1 - start_idx)

                tokens = np.array(self.tokens[str_idx])
                if fix_rstr_loc == "none":
                    start_pos = self._rng.choice(range(0, seq_len - tokens.shape[0] + 1))
                else:
                    if fix_rstr_loc == "start":
                        start_pos = 0
                    elif fix_rstr_loc == "end":
                        start_pos = seq_len - tokens.shape[0]
                    else:
                        raise ValueError(f"Invalid fix_rstr_loc value: {fix_rstr_loc}")
                    
                self._input_ids[start_idx + start_pos: start_idx + start_pos + tokens.shape[0]] = tokens

                cur_cnt += 1
                if cur_cnt == self.occurs[str_idx]:
                    str_idx += 1
                    cur_cnt = 0

    def __len__(self) -> int:
        return self._n_sequences

    def _apply_no_loss(
        self, input_ids: torch.LongTensor, labels: torch.LongTensor
    ) -> Tuple[torch.LongTensor, torch.LongTensor]:
        """Apply the ``no_loss`` logic to the ``input_ids`` and ``labels`` tensors.

        The ``no_loss`` logic is a way of ignoring tokens using a custom logic. If a token is to
        be ignored, the value ``(type_info.max + 1) // 2 -- mask_offset`` is added to it.

        For ``input_ids``, we remove that value while for ``labels`` we mask the tokens that
        have that value and remove it.

        Args:
            input_ids: Input tensor.
            labels: Labels tensor.

        Returns:
            Tuple with the updated ``input_ids`` and ``labels`` tensors.

        """

        mask = None
        input_ids = input_ids % self._mask_offset

        if self._rng.random() < self._mask_prob:
            mask = labels >= self._mask_offset

        labels = labels % self._mask_offset
        if mask is not None:
            labels[mask] = self._ignore_token_id

        return input_ids, labels

    def __getitem__(self, idx: int) -> Dict[str, torch.LongTensor]:
        start_idx = idx * self._seq_len
        seq_len = min(self._seq_len, self._n_input_ids - 1 - start_idx)

        input_ids = torch.as_tensor(self._input_ids[start_idx : (start_idx + seq_len)].astype(np.int64))
        
        if self._label_offset == 0:
            labels = input_ids.clone()
        else:
            raise ValueError("Shifted labels are not supported.")
        
        if self._mask_prob is not None:
            input_ids, labels = self._apply_no_loss(input_ids, labels)
        
        if self.type_tags[idx] == "":
            type_tags = self.dataset_tag
        else:
            type_tags = f"f{self.dataset_tag}q{self.type_tags[idx]}"

        return {
            "input_ids": input_ids,
            "labels": labels,
            "type_tags": type_tags,
        }
    
    def get_type_indexes(self):
        idxs = {}

        for idx in tqdm(range(self._n_sequences)):
            if self.type_tags[idx] == "":
                tag = self.dataset_tag
            else:
                tag = f"f{self.dataset_tag}q{self.type_tags[idx]}"

            if tag not in idxs:
                idxs[tag] = []
            idxs[tag].append(idx)    

        return idxs
    
    @property
    def seq_len(self) -> int:
        return self._seq_len

class CompositeLMDataset(Dataset):
    def __init__(self, datasets):
        self.datasets = datasets
        self._lens = [len(dataset) for dataset in self.datasets]
        self._cum_lens = np.cumsum(self._lens)
        self.all_tags = set(sum([d.all_tags for d in datasets], []))

        seq_lens = [dataset.seq_len for dataset in self.datasets]
        assert all([seq_lens[0] == seq_len for seq_len in seq_lens]), "Sequence lengths must be equal."
    
    def __len__(self):
        return self._cum_lens[-1]

    def __getitem__(self, idx):
        dataset_idx = np.searchsorted(self._cum_lens, idx, side="right")
        if dataset_idx == 0:
            return self.datasets[0][idx]
        return self.datasets[dataset_idx][idx - self._cum_lens[dataset_idx - 1]]

    @property
    def seq_len(self) -> int:
        return self.datasets[0].seq_len

class CompositeEvenDistrLMDataset(Dataset):
    def __init__(self, datasets):
        self._datasets = datasets
        self._len = sum([len(dataset) for dataset in datasets])
        self.all_tags = set(sum([d.all_tags for d in datasets], []))

        seq_lens = [dataset.seq_len for dataset in self.datasets]
        assert all([seq_lens[0] == seq_len for seq_len in seq_lens]), "Sequence lengths must be equal."

        self._rng = np.random.default_rng(42)
        
        idxs = {}
        for i, d in enumerate(datasets):
            d_idxs = d.get_type_indexes()

            for k, v in d_idxs.items():
                if k not in idxs:
                    idxs[k] = []
                idxs[k].extend([(i, idx) for idx in v])

        self._idxs = []
        for k, v in sorted(idxs.items()):
            self._idxs.append(self._rng.permutation(v).tolist())
                
        positions = {}
        for v in self._idxs:
            interval = self._len / len(v)
            pos = 0
            for x in v:
                npos = pos
                while npos in positions:
                    npos += 0.01
                positions[npos] = x
                pos += interval
        
        self._schedule = [positions[i] for i in sorted(positions)]
    
    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        dataset_idx, idx = self._schedule[idx]
            
        return self._datasets[dataset_idx][idx]
    
    @property
    def seq_len(self) -> int:
        return self.datasets[0].seq_len
