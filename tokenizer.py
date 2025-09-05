from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("nghuyong/ernie-3.0-base-zh")

# --- Tokenization helper ---
def tokenize_function(dataset: dict, max_length: int = 64):
    return tokenizer(
        dataset["text"],
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_token_type_ids=True,
        return_tensors="pt" # optional, if using PyTorch directly
    )