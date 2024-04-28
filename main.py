from llama_cpp import Llama
from dotenv import load_dotenv
import asyncio
import discord
import torch
import time
import yaml
import gc
import os


load_dotenv()


# Discord bot prequisites
TOKEN = os.getenv("BOT_TOKEN")
intents = discord.Intents.default()
intents.message_content = True

client = discord.Client(intents=intents)


# Global variables
llm = None

#templates
prompt_templates = None
system_prompts = None
with open("templates/prompt_templates.yaml", "r") as file:
    prompt_templates = yaml.safe_load(file)
with open("templates/system_prompts.yaml", "r") as file:
    system_prompts = yaml.safe_load(file)

prompt_template = prompt_templates["template"]
message_template = prompt_templates["message"]

# Generation parameters
stops = ["<|eot_id|>", "<|end_of_text|>"]
generation_kwargs = {
    "max_tokens": 450,
    "temperature": 0.9,
    "top_k": 50,
    "top_p": 0.9,
    "stop": stops,
}

n_ctx = 8192
max_message_tokens = n_ctx - generation_kwargs["max_tokens"] - 1

def timing(start_time, checkpoints):
    for key in checkpoints:
        print(f"{key} - {checkpoints[key] - start_time} seconds")
        start_time = checkpoints[key]

# Frees up mamory from the GPU and RAM
def free_memory():
    global llm
    
    llm = None
    gc.collect()
    torch.cuda.empty_cache()

# Loads the model
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

# Returns the response from the model
def llm_response(prompt):
    global llm
    if llm is None:
        load_llm()
    
    response =  llm(prompt, **generation_kwargs)
    return response["choices"][0]["text"]


def count_tokens(text):
    tokens = llm.tokenize(text.encode())
    return len(tokens)

async def handle_messages(message):
    channel = message.channel

    token_count = 0
    history_length = 100
    messages = []

    async for message in channel.history(limit=history_length):
        if message.author == client.user:
            templated_message = message_template.format(user="assistant", user_message=message.content)
        else:
            templated_message = message_template.format(user=message.author.name, user_message=message.content)
        token_count += count_tokens(templated_message)
        if token_count > max_message_tokens:
            print("token count: ",token_count)
            break


        if message.content == "!split":
            break

        messages.append(templated_message)
    
    messages.reverse()
    print("message count: ",len(messages))
    messages = "".join(messages)
    print("messages: \n\n",messages)
    print("max token count: ",max_message_tokens)
    print("token count: ",token_count, "\n\n")
    return messages

def construct_prompt(messages):
    global max_message_tokens
    with open("templates/system_prompts.yaml", "r") as file:
        system_prompts = yaml.safe_load(file)
    system_prompt = system_prompts["default"] + "\n" + system_prompts["users"]

    max_message_tokens = n_ctx - generation_kwargs["max_tokens"] - 1 - count_tokens(system_prompt)

    return prompt_template.format(system_prompt=system_prompt,messages=messages)

async def text_pipeline(message):
    async with message.channel.typing():
        start_time = time.time()
        checkpoints = {}
        if llm is None:
            load_llm()
            checkpoints["load_llm"] = time.time()

        messages = await handle_messages(message)
        checkpoints["messages"] = time.time()

        prompt = construct_prompt(messages)
        checkpoints["prompt"] = time.time()

        response = llm_response(prompt)
        checkpoints["response"] = time.time()

        print("response: \n\n",response)
        await message.channel.send(response)
        checkpoints["send"] = time.time()

        timing(start_time, checkpoints)
        return response



def handle_prefix(message):
    prefix = message.content.split(" ")[0]
    if prefix == "!s":
        return True
    return False

# Discord bot
@client.event
async def on_ready():
    print(f"We have logged in as {client.user}")

@client.event
async def on_message(message):
    if message.author == client.user:
        return

    if handle_prefix(message):
        #await message.channel.send("Hello. Currently under maintenance. Please wait a moment.")
        await text_pipeline(message)



client.run(TOKEN)