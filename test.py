# config
CONFIG = {
    "model_name": "llama-2-7b",
    "max_length": 1024,
    "temperature": 0.1,
    "top_p": 0.9,
    "repetition_penalty": 1.0
}


import os
# import torch
import urllib.request

if not os.path.exists("the-verdict.txt"):
    url = "https://raw.githubusercontent.com/rasbt/" \
        "LLMs-from-scratch/main/ch02/01_main-chapter-code/" \
        "the-verdict.txt"
    file_name = "the-verdict.txt"
    urllib.request.urlretrieve(url, file_name)

with open("the-verdict.txt", "r") as file:
    raw_text = file.read()

print(raw_text[:99])

import tiktoken
tokenizer = tiktoken.get_encoding("gpt2")
print(tokenizer.encode(raw_text[:99]))