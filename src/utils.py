#This file contains utility functions for tokenization, plotting, and other helper functions. 

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from transformers import AutoTokenizer
import torch






def get_fig_dim(width, fraction=1, aspect_ratio=None):
    """Set figure dimensions to avoid scaling in LaTeX.

    Parameters
    ----------
    width: float
            Document textwidth or columnwidth in pts
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy
    aspect_ratio: float, optional
            Aspect ratio of the figure

    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    # Width of figure (in pts)
    fig_width_pt = width * fraction

    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    if aspect_ratio is None:
        # If not specified, set the aspect ratio equal to the Golden ratio (https://en.wikipedia.org/wiki/Golden_ratio)
        aspect_ratio = (1 + 5**.5) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in / aspect_ratio

    fig_dim = (fig_width_in, fig_height_in)

    return fig_dim


def latexify(font_serif='Computer Modern', mathtext_font='cm', font_size=10, small_font_size=None, usetex=True):
    """Set up matplotlib's RC params for LaTeX plotting.
    Call this before plotting a figure.

    Parameters
    ----------
    font_serif: string, optional
		Set the desired font family
    mathtext_font: float, optional
    	Set the desired math font family
    font_size: int, optional
    	Set the large font size
    small_font_size: int, optional
    	Set the small font size
    usetex: boolean, optional
        Use tex for strings
    """

    if small_font_size is None:
        small_font_size = font_size

    params = {
        'backend': 'ps',
        'text.latex.preamble': '\\usepackage{gensymb} \\usepackage{bm}',
            
        'axes.labelsize': font_size,
        'axes.titlesize': font_size,
        'font.size': font_size,
        
        # Optionally set a smaller font size for legends and tick labels
        'legend.fontsize': small_font_size,
        'legend.title_fontsize': small_font_size,
        'xtick.labelsize': small_font_size,
        'ytick.labelsize': small_font_size,
        
        'text.usetex': usetex,    
        'font.family' : 'serif',
        'font.serif' : font_serif,
        'mathtext.fontset' : mathtext_font
    }

    matplotlib.rcParams.update(params)
    plt.rcParams.update(params)



def is_in_voc(string, model_name="meta-llama/Llama-3.2-1B"):
    """
    Checks if a given string is present as an exact token in the vocabulary of a specified tokenizer model.
    Args:
        string (str): The string to check for presence in the tokenizer's vocabulary.
        model_name (str, optional): The name or path of the pretrained model to load the tokenizer from.
            Defaults to "meta-llama/Llama-3.2-1B".
    Prints:
        If the string is found as an exact token in the vocabulary, prints the token ID(s) and token string(s).
        Otherwise, prints a message indicating that the string is not a token in the vocabulary.
    """



    
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Get the vocabulary: token -> ID

    vocab = tokenizer.get_vocab()

    # Reverse vocab: ID -> token string (for decoding)
    inv_vocab = {v: k for k, v in vocab.items()}

    # Go through all token strings, decode them, and compare
    found_matches = []
    for token, token_id in vocab.items():
        decoded = tokenizer.decode([token_id])
        if decoded == string:
            found_matches.append((token_id, token))

    if found_matches:
        print("Found exact match(es):")
        for token_id, token_str in found_matches:
            print(f"  Token ID: {token_id}, Token string: {repr(token_str)}")
    else:
        print("Not a token in the vocabulary.")
        
def sort_tensors(x,y):
    """
    Sorts and aggregates tensor values based on unique elements in the first tensor.
    Given two tensors `x` and `y`, this function finds the unique values in `x` and, for each unique value,
    sums the corresponding values in `y` where the elements in `x` match that unique value.
    Args:
        x (torch.Tensor): A 1D tensor containing values to group by.
        y (torch.Tensor): A 1D tensor containing values to be summed according to the grouping in `x`.
                          Must be the same shape as `x`.
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - unique_x (torch.Tensor): The sorted unique values from `x`.
            - sorted_y (torch.Tensor): The sum of `y` values corresponding to each unique value in `x`.
    Example:
        >>> x = torch.tensor([1, 2, 1, 3])
        >>> y = torch.tensor([10, 20, 30, 40])
        >>> unique_x, sorted_y = sort_tensors(x, y)
        >>> unique_x
        tensor([1, 2, 3])
        >>> sorted_y
        tensor([40., 20., 40.], dtype=torch.float64)
    """




    unique_x = torch.unique(x)

    # Initialize an empty tensor to accumulate the summed y values
    sorted_y = torch.zeros_like(unique_x, dtype=torch.float64)

    # Sum the corresponding y values for each unique x
    for i, ux in enumerate(unique_x):
        sorted_y[i] = y[x == ux].sum()

    return unique_x, sorted_y



def optimal_tokenization(s, tokenizer, max_token_length=30):
    """
    Computes an optimal (shortest) tokenization of the input string `s` using the provided tokenizer,
    minimizing the number of tokens while respecting a maximum token length constraint.
    The function normalizes the input string, builds a normalized vocabulary from the tokenizer,
    and applies dynamic programming to find the minimal token split. It reconstructs the optimal
    token sequence and returns both the token strings and their corresponding token IDs.
    Args:
        s (str): The input string to tokenize.
        tokenizer: A tokenizer object with `encode`, `convert_ids_to_tokens`, and `get_vocab` methods.
        max_token_length (int, optional): The maximum length of a token substring to consider. Defaults to 30.
    Returns:
        dict: A dictionary with two keys:
            - "strings": List of token strings (normalized and human-readable).
            - "ids": List of corresponding token IDs as produced by the tokenizer.
    Notes:
        - The function handles smart apostrophes and normalizes spaces for tokenizers using special space tokens (e.g., 'Ġ').
        - If a token cannot be encoded, it is skipped and an error message is printed.
    """

    

    s = s.replace("’", "'")  # Replace smart apostrophes with straight ones
    s = s.replace("‘", "'")  # Handle left single quotes as well
    s = tokenizer.convert_ids_to_tokens(tokenizer.encode(s, add_special_tokens=False))
    s = "".join(s).replace("Ġ", " ")  # Normalize spaces if applicable.

    # Build a normalized vocabulary.
    V = {}
    for token in tokenizer.get_vocab().keys():
        if token.startswith("Ġ"):
            normalized = " " + token[1:]
        else:
            normalized = token
        V[normalized] = token

    n = len(s)
    opt_counter = [float("inf")] * (n + 1)
    opt_counter[0] = 0

    # Array to track the optimal split points
    opt_index = [-1] * (n + 1)

    for i in range(1, n + 1):
        for j in range(max(0, i - max_token_length), i):
            substring = s[j:i]
            if substring in V:
                if opt_counter[j] + 1 < opt_counter[i]:
                    opt_counter[i] = opt_counter[j] + 1
                    opt_index[i] = j

    # Reconstruct the tokenization
    tokens = []
    i = n
    while i > 0:
        j = opt_index[i]
        token_str = s[j:i]
        if token_str in V:
            tokens.append(V[token_str])
        i = j

    tokens.reverse()

    # Format back to readable strings
    tokens = [token.replace("Ġ", " ") for token in tokens]

    # Generate token IDs and handle encoding errors gracefully
    ids = []
    for token in tokens:
        try:
            encoded = tokenizer.encode(token, add_special_tokens=False)
            if encoded:
                ids.append(encoded[0])
        except Exception as e:
            print(f"Error encoding token '{token}': {e}")

    return {"strings": tokens, "ids": ids}




def optimal_token_count(text, tokenizer):
    """Returns the number of tokens generated by our optimal tokenization routine."""
    tokens = optimal_tokenization(text, tokenizer)
    return len(tokens["strings"]) if tokens is not None else None

 
