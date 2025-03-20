import hashlib
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import pdb
import re
import torch

from tqdm import tqdm

from project.utils.utils import find_unique_prefixes, gen_eval_confs
from project.utils.cascade_utils import eval_cascade

from eval_configs import (
    ALL_CTX_LEN,
    CASCADE_OVERLAP,
    CASCADE_NON_OVERLAP,
    ORIGINAL_CASCADE_OVERLAP,
    ORIGINAL_CASCADE_NON_OVERLAP,
)

KEYS = {
    # "ts": "$f_{\\text{ts}}$",
    # "wiki": "$f_{\\text{wiki}}$",
    "ftsqts": "$f_{\\text{ts}} q_{\\text{ts}}$",
    "fwikiqwiki": "$f_{\\text{wiki}} q_{\\text{wiki}}$",
    "ftsqwiki": "$f_{\\text{ts}} q_{\\text{wiki}}$",
    "fwikiqts": "$f_{\\text{wiki}} q_{\\text{ts}}$",
}

def hash_config(config):
    """
    Generate a consistent hash string for a configuration.
    
    Args:
        config: A list of dictionaries, each containing 'ckpt' and 'ctx_len' keys.
        
    Returns:
        A string hash that uniquely identifies this configuration.
    """
    normalized_config = []
    for item in config:
        normalized_item = {
            "ckpt": item["ckpt"],
            "ctx_len": sorted(item["ctx_len"])
        }
        normalized_config.append(normalized_item)

    config_str = json.dumps(normalized_config, sort_keys=True)
    
    hash_obj = hashlib.md5(config_str.encode())
    hash_str = hash_obj.hexdigest()
    
    return hash_str

def scientific_notation(number, digits=3):
    """
    Convert a float to scientific notation with specified number of effective digits
    in LaTeX format.
    
    Args:
        number (float): The number to convert
        digits (int): Number of effective digits (default 3)
    
    Returns:
        str: LaTeX formatted scientific notation string
    """
    # Format the number to scientific notation with specified precision
    formatted = f"{number:.{digits-1}e}"
    
    # Split into mantissa and exponent parts
    parts = formatted.split('e')
    mantissa = parts[0]
    exponent = int(parts[1])
    
    # Create LaTeX formatted string
    latex_format = f"${mantissa} \\times 10^{{{exponent}}}$"
    
    return latex_format

def plot_weights(weights, ctx_lens, path):
    """
    Args:
        weights: list of numpy arrays, each of shape (num_models, len)
        ctx_lens: list of context lengths
        path: output path for the figure
        window_size: size of the moving average window for smoothing
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    n = len(weights)
    l = len(weights[0])
    positions = np.arange(len(weights[0]))

    if l <= 50:
        window_size = 1
    elif l <= 100:
        window_size = 3
    elif l <= 300:
        window_size = 9
    elif l <= 500:
        window_size = 27
    else:
        window_size = 81

    # Apply moving average smoothing to weights
    smoothed_weights = []
    for i in range(n):
        # Apply moving average smoothing
        window = np.ones(window_size) / window_size
        smoothed = np.convolve(weights[i], window, mode='same')
        
        # Fix edge effects
        if window_size > 1:
            half_window = window_size // 2
            # Adjust the first half_window points
            for j in range(half_window):
                window_size_start = j + half_window + 1
                if window_size_start > 0:
                    smoothed[j] = np.sum(weights[i][:window_size_start]) / window_size_start
            # Adjust the last half_window points
            for j in range(l - half_window, l):
                window_size_end = l - j + half_window
                if window_size_end > 0:
                    smoothed[j] = np.sum(weights[i][j-half_window:]) / window_size_end
        
        # Ensure values stay within [0, 1]
        smoothed = np.clip(smoothed, 0, 1)
        smoothed_weights.append(smoothed)

    # Plot stacked areas
    stack = np.zeros(l)
    for i in range(n):
        ax.fill_between(positions, stack, stack + smoothed_weights[i], 
                        label=f'ctx_len {ctx_lens[i]}', alpha=0.7)
        stack += smoothed_weights[i]

    # Renormalize stack if it exceeds 1 at any point
    if np.any(stack > 1.0):
        # Create a correction factor for each position
        correction = np.ones(l)
        mask = stack > 1.0
        correction[mask] = 1.0 / stack[mask]
        
        # Replot with normalized weights
        stack = np.zeros(l)
        ax.clear()
        for i in range(n):
            normalized = smoothed_weights[i] * correction
            ax.fill_between(positions, stack, stack + normalized, 
                            label=f'ctx_len {ctx_lens[i]}', alpha=0.7)
            stack += normalized

    # Add labels and legend
    ax.set_xlabel('Sequence Position', fontsize=12)
    ax.set_ylabel('Weight Distribution', fontsize=12)
    ax.set_title('Model Weight Distribution', fontsize=14)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set_xlim(0, l-1)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def load_latest_model_and_config(checkpoint_path):
    checkpoints = get_checkpoints_info(
        checkpoint_path,
        checkpoint_regex=re.compile(r"^((\d+)|last)$|checkpoint-\d+"),
        n_checkpoints=None,
        reverse=True,
    )

    config = process_config(load_config(os.path.join(checkpoint_path, "config.yaml")))
    for d in config["dataset"]:
        for k in ["train", "eval"]:
            d[k] = d[k].replace("/mnt/task_runtime/data/", "/home/v-vectorzhou/data/")

    model = get_model(checkpoints[0]["checkpoint_path"])
    model = to_device_map(model, device_map="cuda")

    return model, config

def get_results_ablation(exp_config, entire_eval=False):
    hash_name = hash_config(exp_config)
    file_name = f"eval_results/gen/cascade_eval_{hash_name}.pkl"
    os.makedirs("eval_results/gen", exist_ok=True)

    if os.path.exists(file_name):
        with open(file_name, "rb") as f:
            file = pickle.load(f)
        
        return file["results"], file["ablation_results"]

    models = []

    for ckpt in exp_config:
        model, config = load_latest_model_and_config(ckpt["ckpt"])

        for ctx_len in ckpt["ctx_len"]:
            models.append({
                "model": model,
                "context_len": ctx_len,
            })
    
    models = sorted(models, key=lambda x: x["context_len"])

    all_tokens = []
    for dataset in config["dataset"]:
        if "insert_random_str" in dataset:
            all_tokens.extend(dataset["insert_random_str"]["tokens"])       

    eval_confs = gen_eval_confs(
        config=config,
        prefix=find_unique_prefixes(all_tokens),
        seq_len=1024,
        fiqj_remove_half=False,
        multiplicity=config["training_args"]["eval_multiplicity"],
    )

    batch_size = config["training_args"]["ds_config"]["eval_micro_batch_size_per_gpu"]

    results = {}
    ablation_results = [{} for _ in models]
    for start_pos, conf in tqdm(eval_confs.items()):
        tot_data = len(conf)
        
        input_ids = torch.stack([c["input_ids"] for c in conf], dim=0).to("cuda")
        completion_len = input_ids.shape[1] - (1 if entire_eval else start_pos)
        tags = [c["tags"] for c in conf]

        for i in range(0, tot_data, batch_size):
            input_ids_batch = input_ids[i:i+batch_size]
            tags_batch = tags[i:i+batch_size]

            log_probs, weights, log_probs_wo_i = eval_cascade(
                models=models,
                input_ids=input_ids_batch,
                start_pos=1 if entire_eval else start_pos,
                ablation=True,
                device="cuda",
            )

            for j, tag in enumerate(tags_batch):
                if tag not in results:
                    results[tag] = {}
                    for ar in ablation_results:
                        ar[tag] = {}
                
                if completion_len not in results[tag]:
                    results[tag][completion_len] = {
                        "log_probs": [],
                        "weights": [],
                    }
                    for ar in ablation_results:
                        ar[tag][completion_len] = []
                
                results[tag][completion_len]["log_probs"].append(log_probs[j].cpu().numpy())
                results[tag][completion_len]["weights"].append(weights[j].cpu().numpy())
                for k, logp in enumerate(log_probs_wo_i):
                    ablation_results[k][tag][completion_len].append(logp[j].cpu().numpy())
    
    for tag in results:
        for completion_len in results[tag]:
            results[tag][completion_len]["log_probs"] = np.stack(results[tag][completion_len]["log_probs"], axis=0)
            results[tag][completion_len]["weights"] = np.stack(results[tag][completion_len]["weights"], axis=0)
            for k, ar in enumerate(ablation_results):
                ablation_results[k][tag][completion_len] = np.stack(ablation_results[k][tag][completion_len], axis=0)
    
    with open(file_name, "wb") as f:
        pickle.dump({
            "results": results,
            "ablation_results": ablation_results,
        }, f)
        f.flush()

    return results, ablation_results

def eval(exp_configs, loc=-1, entire_eval=False):
    all_results = {}

    for exp_config in exp_configs:
        results, ablation_results = get_results_ablation(exp_config, entire_eval=entire_eval)

        if all_results == {}:
            all_results = results
            all_ablation_results = ablation_results
            continue

        for tag in results:
            for completion_len in results[tag]:
                for key in results[tag][completion_len]:
                    all_results[tag][completion_len][key] = \
                        np.concatenate((all_results[tag][completion_len][key], results[tag][completion_len][key]), axis=0)
                for k, ar in enumerate(ablation_results):
                    all_ablation_results[k][tag][completion_len] = \
                        np.concatenate((all_ablation_results[k][tag][completion_len], ar[tag][completion_len]), axis=0)

    # os.makedirs("eval_results/fig", exist_ok=True)
    # for tag in KEYS:
    #     for completion_len in all_results[tag]:
    #         plot_weights(all_results[tag][completion_len]["weights"][0], sum([m["ctx_len"] for m in exp_configs[0]], []), f"eval_results/fig/{tag}_{completion_len}.pdf")


    for tag in KEYS:
        all_avg_log_probs = 0
        for completion_len in all_results[tag]:
            avg_log_probs = all_results[tag][completion_len]["log_probs"].mean(0).cumsum() / np.arange(1, completion_len + 1)

            all_avg_log_probs += avg_log_probs[loc].item()

            # print(completion_len, avg_log_probs[-1].item())

            # avg_log_probs_wo_i = [logp.cumsum(dim=1) / torch.arange(1, completion_len + 1, device="cuda", dtype=torch.float32) for logp in log_probs_wo_i]
        
        print(" &", scientific_notation(all_avg_log_probs / len(all_results[tag])), end=" ")
    print("\\\\")
    print("\\hline")

    ctx_lens = sum([m["ctx_len"] for m in exp_configs[0]], [])

    for i, ar in enumerate(all_ablation_results):
        print(f"$ {ctx_lens[i]} $", end=" ")

        for tag in KEYS:
            all_avg_log_probs = 0
            for completion_len in ar[tag]:
                avg_log_probs = ar[tag][completion_len].mean(0).cumsum() / np.arange(1, completion_len + 1)

                all_avg_log_probs += avg_log_probs[loc].item()

                # print(completion_len, avg_log_probs[-1].item())

                # avg_log_probs_wo_i = [logp.cumsum(dim=1) / torch.arange(1, completion_len + 1, device="cuda", dtype=torch.float32) for logp in log_probs_wo_i]
            
            print(" &", scientific_notation(all_avg_log_probs / len(ar[tag])), end=" ")
        print("\\\\")
        print("\\hline")



if __name__ == "__main__":
    eval(CASCADE_OVERLAP)
    # eval(CASCADE_NON_OVERLAP)
    # eval(ORIGINAL_CASCADE_OVERLAP)
    # eval(ORIGINAL_CASCADE_NON_OVERLAP)

    # for ctx_len in ALL_CTX_LEN:
    #     exp_configs = []
    #     for config in EXP_CONFIGS:
    #         exp_configs.append([{
    #             "ckpt": config[0]["ckpt"],
    #             "ctx_len": [ctx_len],
    #         }])
        
    #     print("$", ctx_len, "$", end=" ")
    #     eval(exp_configs, 0, entire_eval=False)
    
