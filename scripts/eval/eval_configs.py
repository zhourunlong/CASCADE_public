import os

ALL_CTX_LEN = [8, 16, 32, 64, 128, 256, 512, 1024]


CASCADE_OVERLAP = [
    [
        {
            "ckpt": "[REDACTED]/CASCADE_ctx1024_dataseed42_str32_L8_512_occur8192_epoch2_locnone_seed42_shuffle1_overlap1_0304112347",
            "ctx_len": ALL_CTX_LEN,
        }
    ],
    [
        {
            "ckpt": "[REDACTED]/CASCADE_ctx1024_dataseed42_str32_L8_512_occur8192_epoch2_locnone_seed142857_shuffle1_overlap1_0304221720",
            "ctx_len": ALL_CTX_LEN,
        }
    ],
    [
        {
            "ckpt": "[REDACTED]/CASCADE_ctx1024_dataseed42_str32_L8_512_occur8192_epoch2_locnone_seed2225393_shuffle1_overlap1_0305091044",
            "ctx_len": ALL_CTX_LEN,
        }
    ],
    [
        {
            "ckpt": "[REDACTED]/CASCADE_ctx1024_dataseed42_str32_L8_512_occur8192_epoch2_locnone_seed20000308_shuffle1_overlap1_0305200421",
            "ctx_len": ALL_CTX_LEN,
        }
    ],
    [
        {
            "ckpt": "[REDACTED]/CASCADE_ctx1024_dataseed42_str32_L8_512_occur8192_epoch2_locnone_seed2018011309_shuffle1_overlap1_0306065944",
            "ctx_len": ALL_CTX_LEN,
        }
    ],
]

CASCADE_NON_OVERLAP = [
    [
        {
            "ckpt": "[REDACTED]/CASCADE_ctx1024_dataseed42_str32_L8_512_occur8192_epoch2_locnone_seed42_shuffle1_overlap0_0304063234",
            "ctx_len": ALL_CTX_LEN,
        }
    ],
    [
        {
            "ckpt": "[REDACTED]/CASCADE_ctx1024_dataseed42_str32_L8_512_occur8192_epoch2_locnone_seed142857_shuffle1_overlap0_0304172745",
            "ctx_len": ALL_CTX_LEN,
        }
    ],
    [
        {
            "ckpt": "[REDACTED]/CASCADE_ctx1024_dataseed42_str32_L8_512_occur8192_epoch2_locnone_seed2225393_shuffle1_overlap0_0305042130",
            "ctx_len": ALL_CTX_LEN,
        }
    ],
    [
        {
            "ckpt": "[REDACTED]/CASCADE_ctx1024_dataseed42_str32_L8_512_occur8192_epoch2_locnone_seed20000308_shuffle1_overlap0_0305151524",
            "ctx_len": ALL_CTX_LEN,
        }
    ],
    [
        {
            "ckpt": "[REDACTED]/CASCADE_ctx1024_dataseed42_str32_L8_512_occur8192_epoch2_locnone_seed2018011309_shuffle1_overlap0_0306020850",
            "ctx_len": ALL_CTX_LEN,
        }
    ],
]

def parse_args(run_name):
    ctx = int(run_name[run_name.find("ctx")+len("ctx"):run_name.find("dataseed")-1])

    _ = run_name[run_name.find("dataseed")+len("dataseed"):]
    _ = _[_.find("seed")+len("seed"):]
    seed = int(_.split("_")[0])

    _ = run_name[run_name.find("dataseed")+len("dataseed"):]
    dataseed = int(_.split("_")[0])

    _ = run_name[run_name.find("overlap")+len("overlap"):]
    overlap = int(_.split("_")[0])

    return {
        "ctx": ctx,
        "seed": seed,
        "dataseed": dataseed,
        "overlap": overlap,
    }

def find_ckpts(dir, prefix, overlap):
    ckpts = os.listdir(dir)

    ckpts = [ckpt for ckpt in ckpts if ckpt.startswith(prefix)]

    configs = {}

    for ckpt in ckpts:
        args = parse_args(ckpt)

        if args["overlap"] != overlap:
            continue

        if args["seed"] not in configs:
            configs[args["seed"]] = []

        configs[args["seed"]].append({
            "ckpt": os.path.join(dir, ckpt),
            "ctx_len": [args["ctx"]],
        })

    ret = []
    for seed in configs:
        if len(configs[seed]) != 8:
            continue
        configs[seed].sort(key=lambda x: x["ctx_len"][0])
        ret.append(configs[seed])
    
    return ret

ORIGINAL_CASCADE_OVERLAP = find_ckpts("[REDACTED]", "ORIGINAL_CASCADE", 1)
ORIGINAL_CASCADE_NON_OVERLAP = find_ckpts("[REDACTED]", "ORIGINAL_CASCADE", 0)
