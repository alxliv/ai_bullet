from transformers import AutoTokenizer
import os

def download_tokenizers():
    """Download all required tokenizers to local directories."""

    tokenizers_to_download = [
        ("Qwen/Qwen2.5-Coder-7B-Instruct", "./tokenizers/qwen"),
        # Add more as needed
    ]

    for model_name, save_path in tokenizers_to_download:
        print(f"Downloading {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        os.makedirs(save_path, exist_ok=True)
        tokenizer.save_pretrained(save_path)
        print(f"Saved to {save_path}")

if __name__ == "__main__":
    download_tokenizers()