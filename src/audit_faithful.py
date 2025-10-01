import torch
import random
import copy
import pickle
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
from scipy.stats import poisson


def constrained_sampling(model, tokenizer, input_ids, reference_text, token_strs, poisson_param=5, geometric_param=None, temperature=1.0, device="cpu", reference_tokens=None):
    model.eval()
    
    
    def single_sample():
        generated = []
        log_prob_unmasked = 0.0
        log_prob_masked = 0.0
        current_prefix = ''

        while current_prefix != reference_text:
            if len(generated) == 0:
                current_input = input_ids
            else:
                # Feed all tokens generated so far
                current_input = torch.cat([input_ids, torch.tensor([generated], device=device)], dim=1)

            with torch.no_grad():
                outputs = model(
                    current_input,
                    use_cache=True  # Let the model handle caching internally
                )
                logits = outputs.logits[:, -1, :]
                logits_temp = logits / temperature
                probs = torch.nn.functional.softmax(logits_temp.float(), dim=-1).squeeze(0)

            # Mask tokens to match reference
            valid_ids = []
            for token_id, piece in token_strs.items():
                if not piece:
                    continue
                candidate = current_prefix + piece
                if candidate == current_prefix:
                    continue
                if reference_text.startswith(candidate):
                    valid_ids.append(token_id)

            if not valid_ids:
                return None

            mask = torch.zeros_like(probs, dtype=torch.float32)
            mask[valid_ids] = probs[valid_ids]
            mask = mask / mask.sum()

            next_token = torch.multinomial(mask, 1).item()
            log_prob_unmasked += torch.log(probs[next_token] + 1e-12).item()
            log_prob_masked += torch.log(mask[next_token] + 1e-12).item()

            generated.append(next_token)
            current_prefix = current_prefix + token_strs[next_token]

            if len(current_prefix) > len(reference_text):
                return None

        if current_prefix == reference_text:
            print("Sample token sequence:", generated)
            result = {
                "tokens": generated,
                "text": current_prefix,
                "log_prob_unmasked": log_prob_unmasked,
                "log_prob_masked": log_prob_masked
            }
            print(f"Finished sample with {len(generated)} tokens")
            return result
        return None
  

    # --- Sample M ---
    if geometric_param is not None:
        assert 0 < geometric_param < 1, "Geometric parameter must be in (0,1)"
        print(f"Using geometric distribution with rho={geometric_param}")
        while True:
            M = np.random.geometric(geometric_param)
            if M > 0:
                break
        print(f"Sampled Geometric level M={M} with rho={geometric_param}")

        def survival_prob(m):
            return (1 - geometric_param) ** (m - 1)

    else:
        assert poisson_param > 0, "Poisson parameter must be positive"
        print(f"Using Poisson distribution with lambda {poisson_param}")
        while True:
            M = poisson.rvs(mu=poisson_param)
            if M > 0:
                break
        print(f"Sampled Poisson level M={M} (lambda={poisson_param})")

        def survival_prob(m):
            return 1 - poisson.cdf(m - 1, mu=poisson_param)

    M = min(M, 8)  # cap M

    N_total = 2 * 2 ** M
    all_samples = []
    while len(all_samples) < N_total:
        s = single_sample()
        if s is not None:
            all_samples.append(s)
            print(f"Collected {len(all_samples)}/{N_total} total samples")

    prev_R = 0.0
    unbiased_est = 0.0
    for m in range(1, M + 1):
        n_m = 2 * 2 ** m
        samples_m = all_samples[:n_m]

        log_weights = [r["log_prob_unmasked"] - r["log_prob_masked"] for r in samples_m]
        max_logw = max(log_weights)
        weights = [torch.exp(torch.tensor(lw - max_logw, dtype=torch.float64)) for lw in log_weights]

        num = sum([len(r["tokens"]) * w for r, w in zip(samples_m, weights)])
        den = sum(weights)
        R_m = (num / den).item()

        Delta_m = R_m - prev_R
        weight = 1.0 / survival_prob(m)
        unbiased_est += Delta_m * weight
        prev_R = R_m

        print(f"Level m={m}, n_m={n_m}")
        print(f"  R_m={R_m}")
        print(f"  Δ_m={Delta_m}, weight={weight}, contribution={Delta_m * weight}")

    print(f"Final unbiased estimate of expected tokenization length: {unbiased_est}")
    if reference_tokens is not None:
        print(f"Reference tokenization length: {len(reference_tokens)}\n")

    return unbiased_est


def main(prompts, model_name="meta-llama/Llama-3.2-1B-Instruct", temperature=1.0, seed=42, poisson_param=5, geometric_param=None, save_path=None):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        cache_dir="work/models"
    ).to(device)

    print(f"Loaded model {model_name} on device {device}")

    # Precompute decoded strings for valid tokens only (within model's vocab size)
    token_strs = {}
    
    #for tid in range(model.config.vocab_size):  # Only iterate up to model's vocab size
    for tid in range(tokenizer.vocab_size):
        if tid not in tokenizer.all_special_ids:
            token_strs[tid] = tokenizer.decode([tid], skip_special_tokens=True)
        else:
            token_strs[tid] = ""

    all_results = []

    for prompt_index, prompt_str in enumerate(prompts):
        print(f"Prompt number {prompt_index+1}/{len(prompts)}")
        messages = [
            {"role": "system", "content": "You are a helpful assistant. Answer in English and be extremely concise. Answer very briefly."},
            {"role": "user", "content": prompt_str}
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

        with torch.no_grad():
            output = model.generate(
                input_ids=input_ids,
                do_sample=True,
                max_new_tokens=60,
                temperature=temperature,
                pad_token_id=tokenizer.eos_token_id  # Ensure proper padding
            )

        # Filter tokens to only include those within model's vocabulary
        # reference_tokens = [int(t) for t in output[0][input_ids.size(1):] 
        #                   if t not in tokenizer.all_special_ids and t < model.config.vocab_size]
        
        reference_tokens = [int(t) for t in output[0][input_ids.size(1):] 
                          if t not in tokenizer.all_special_ids and t < tokenizer.vocab_size]
        
        
        reference_text = tokenizer.decode(reference_tokens, skip_special_tokens=True)

        bad_chars = []
        for ch in reference_text:
            ch_tokens = tokenizer.encode(ch, add_special_tokens=False)
            if len(ch_tokens) >= 2:
                bad_chars.append((ch, len(ch_tokens)))
        
        if bad_chars:
            print(f"Skipping this sample. Found characters needing >2 tokens: {bad_chars}")
            continue

        print(prompt_str)
        print("Reference tokens:", reference_tokens)
        print("Number of reference tokens:", len(reference_tokens))

        unbiased_length = constrained_sampling(
            model, tokenizer, input_ids, reference_text, token_strs,
            poisson_param=poisson_param, geometric_param=geometric_param,
            temperature=temperature, device=device, reference_tokens=reference_tokens
        )

        all_results.append({
            "prompt": prompt_str,
            "original": {"tokens": reference_tokens, "text": reference_text},
            "unbiased_length": unbiased_length
        })

    if save_path is not None:
        with open(save_path, "wb") as f:
            pickle.dump(all_results, f)

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompts', nargs="+", type=str, required=False, default=["How are you?", "Tell me a story"])
    parser.add_argument('--temperature', type=float, required=False, default=1.15)
    parser.add_argument('--poisson', type=float, required=False, default=5)
    parser.add_argument('--geometric', type=float, required=False, default=None)
    parser.add_argument('--model', type=str, required=False, default="L1B")
    parser.add_argument('--job_id', type=int, required=False, default=0)

    args = parser.parse_args()
    model = args.model
    job_id = args.job_id
    print("Job ID:", job_id)

    if model == "L1B":
        model_str = "meta-llama/Llama-3.2-1B-Instruct"
    elif model == "L3B":
        model_str = "meta-llama/Llama-3.2-3B-Instruct"
    elif model == "G1B":
        model_str = "google/gemma-3-1b-it"
    elif model == "G4B":
        model_str = "google/gemma-3-4b-it"
    elif model == "M8B":
        model_str = "mistralai/Ministral-8B-Instruct-2410"
    else:
        model_str = "meta-llama/Llama-3.2-1B-Instruct"

    temperature = args.temperature
    poisson_param = args.poisson
    geometric_param = args.geometric
    print(f"Using model: {model}, temperature: {temperature}, poisson={poisson_param}, geometric={geometric_param}")

    prompt_list = args.prompts
    results = main(
        prompt_list, model_name=model_str, temperature=temperature,
        poisson_param=poisson_param, geometric_param=geometric_param,
        save_path=f"audit_faithful_{model}_temp_{temperature}_poi_{poisson_param}_geo_{geometric_param}_id_{job_id}.pkl"
    )

    for entry in results:
        print("-------------------------------------")
        print(f"\nPrompt: {entry['prompt']}")
        print("Original Tokens:", entry['original']['tokens'])
        print("Unbiased Expected Length:", entry['unbiased_length'])