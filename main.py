from llama_cpp import Llama
from dotenv import load_dotenv
import asyncio
import discord
import torch
import yaml
import gc
import os


load_dotenv()

TOKEN = os.getenv("BOT_TOKEN")

llm = None


stops = ["<|eot_id|>", "<|end_of_text|>"]
generation_kwargs = {
    "max_tokens": 450,
    "temperature": 0.8,
    "top_k": 50,
    "top_p": 0.9,
    "stop": stops,
}


def free_memory():
    global llm
    
    llm = None
    gc.collect()
    torch.cuda.empty_cache()

def load_llm():
    free_memory()
    global llm
    
    if llm is None:
        llm = Llama.from_pretrained(
            "QuantFactory/Meta-Llama-3-8B-Instruct-GGUF",
            filename="*Q4_K_M.gguf",
            n_gpu_layers=-1,
            n_ctx=8192,
            verbose=True,
        )

def llm_response(prompt):
    global llm
    if llm is None:
        load_llm()
    
    response =  llm(prompt, **generation_kwargs)
    return response["choices"][0]["text"]


def main():
    load_llm()
    print(llm_response("What is the meaning of life?"))

if __name__ == "__main__":
    main()