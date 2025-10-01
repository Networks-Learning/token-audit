import torch
import random
import copy
import pickle
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
from scipy.stats import poisson
from tokenizations import verify_sampling_conditions

def split_token(sequence, tokenizer, vocab):
    
    """
    Attempts to split a token in the given sequence into two valid subtokens based on the provided vocabulary.
    The function identifies the token in the sequence (with at least two characters when decoded) that has the highest token ID.
    It then tries all possible binary splits of this token, checking if both resulting parts exist in the vocabulary.
    Among all valid splits, it selects the one where the minimum of the two resulting token IDs is maximized.
    If a valid split is found, the original token in the sequence is replaced with the two subtokens; otherwise, the original sequence is returned.
    Args:
        sequence (list of int): The sequence of token IDs to process.
        tokenizer: An object with a `decode` method that converts token IDs to strings.
        vocab (dict): A mapping from token strings to their corresponding token IDs.
    Returns:
        list of int: The updated sequence with the split applied, or the original sequence if no valid split is found.
    """



    # Reverse mapping: ID -> Token
    id_to_token = {v: k for k, v in vocab.items()}


    
    #Get all token IDs in the sequence that have at least two characters
    valid_ids = [token_id for token_id in sequence if len(tokenizer.decode([token_id])) > 1]
    if len(valid_ids) == 0:
        print("No valid token IDs found, returning original sequence", sequence)
        return sequence
    
    token_id_to_split = max(valid_ids)

    # Get the token corresponding to the selected ID
    token_to_split = id_to_token[token_id_to_split]

    
    
    # Initialize variables to store the best split
    best_split = None
    
    
    max_index = -float('inf')  # Start with a very low number for comparison

    # Try all possible splits 
    for mid_index in range(1, len(token_to_split)):  # Split at various points
        Y = token_to_split[:mid_index]
        Z = token_to_split[mid_index:]
        
        # Get the token IDs for Y and Z
        Y_id = vocab.get(Y)  # No default value; will return None if Y isn't valid
        Z_id = vocab.get(Z)  # No default value; will return None if Z isn't valid


        # Skip this split if either Y or Z is invalid
        if Y_id is None or Z_id is None:
            continue

        # Calculate the sum of the indices
        index_min = min(Y_id, Z_id)

        # If the sum of the indices is the largest found so far, update best split
        if index_min > max_index:
            best_split = (Y, Z)
            max_index = index_min




    # If no valid split was found, return the original sequence
    if best_split is None:
        return sequence

    # Replace the token X with its split subtokens Y and Z in the sequence
    new_sequence = []
    updated = False
    for token_id in sequence:
        if token_id == token_id_to_split and not updated:
            # Replace token X with subtokens Y and Z
            new_sequence.extend([vocab[best_split[0]], vocab[best_split[1]]])
            updated = True
        else:
            new_sequence.append(token_id)

    return new_sequence



def main (model_name="L1B", model_str="meta-llama/Llama-3.2-1B-Instruct", temperature=1.0, iter=2, top_p=0.99, seed=42, poisson_param=5, geometric_param=None, save_path=None):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(model_str)
    vocab = tokenizer.get_vocab()
    model = AutoModelForCausalLM.from_pretrained(
        model_str, 
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        cache_dir="/models"
    ).to(device)

    print(f"Loaded model {model_name} on device {device}")


    #Load the generated strings,
    
    data = []
    #Gemma [1, 401, 801, 1201, 1601, 2401, 2801, 3201, 3601]
    #Else [1, 401, 801, 1201, 1601, 2001, 2801, 3201, 3601]
    
    if model_name == "G1B":
        idx_list = [1, 401, 801, 1201, 1601, 2401, 2801, 3201, 3601, 3901]
    else:
        idx_list = [1, 401, 801, 1201, 1601, 2401, 2801, 3201, 3601]
    

    for idx in  idx_list:

        with open(f"outputs/audit_faithfull/audit_faithful_{model_name}_temp_{temperature}_poi_6.0_geo_None_id_{idx}.pkl", "rb") as f:
            data_temp = pickle.load(f)
            data= data + data_temp
            
            
    #Compute all prompts:
    prompt_list = [data[i]["prompt"] for i in range(len(data))]
    output_list = [data[i]["original"]["tokens"] for i in range(len(data))]
    unbiased_output_list = [data[i]["unbiased_length"] for i in range(len(data))]     

    all_results = []

    for output_index in range(len(prompt_list)):
        
        prompt_str = prompt_list[output_index]
        output = output_list[output_index]
        unbiased_len = unbiased_output_list[output_index]
        
        print(f"Prompt number {output_index+1}/{len(prompt_list)}")
        
        print("Prompt: ", prompt_str)
        print("Original Tokens: ", output)
        
        misreported_tokens = output
        input_ids = tokenizer(prompt_str, add_special_tokens=False, return_tensors="pt").input_ids.to(device)

        for _ in range(iter):
            
            misreported_tokens = split_token(misreported_tokens, tokenizer, vocab)
        
        sampling_condition = verify_sampling_conditions(input_ids[0].tolist() + misreported_tokens, len(input_ids[0].tolist()), top_p=top_p, model=model, tokenizer=tokenizer, temp = temperature)
        
        print("Misreported Tokens: ", misreported_tokens)
        
        if sampling_condition["all_top_p_met"]:
            print("Plausible extension found after misreporting.")
            
            all_results.append({
                "prompt": prompt_str,
                "original": {"tokens": output, "text": None},
                "misreported_tokens": misreported_tokens
            })
        else:
            print("IMplausible extension found after misreporting.")
            all_results.append({
                "prompt": prompt_str,
                "original": {"tokens": output, "text": None},
                "misreported_tokens": output,
                "unbias_length": unbiased_len
            })
            

    if save_path is not None:
        with open(save_path, "wb") as f:
            pickle.dump(all_results, f)

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--temperature', type=float, required=False, default=1.15)
    parser.add_argument('--p', type=float, required=False, default=0.99)
    parser.add_argument('--model_name', type=str, required=False, default="L1B")
    parser.add_argument('--iter', type=int, required=False, default=1)
    parser.add_argument('--job_id', type=int, required=False, default=0)

    args = parser.parse_args()
    model_name = args.model_name
    job_id = args.job_id
    print("Job ID:", job_id)

    if model_name == "L1B":
        model_str = "meta-llama/Llama-3.2-1B-Instruct"
    elif model_name == "L3B":
        model_str = "meta-llama/Llama-3.2-3B-Instruct"
    elif model_name == "G1B":
        model_str = "google/gemma-3-1b-it"
    elif model_name == "G4B":
        model_str = "google/gemma-3-4b-it"
    elif model_name == "M8B":
        model_str = "mistralai/Ministral-8B-Instruct-2410"
    else:
        model_str = "meta-llama/Llama-3.2-1B-Instruct"

    temperature = args.temperature
    iter=args.iter
    top_p = args.p
    print(f"Using model: {model_name}, temperature: {temperature}, iter: {iter}, top_p: {top_p}")

    results = main(
        model_name=model_name, model_str=model_str, temperature=temperature, iter=iter, top_p=top_p,
        save_path=f"audit_heuristic_{model_name}_temp_{temperature}_iter_{iter}_p_{top_p}_id_{job_id}.pkl"
    )

