from diffusers import StableDiffusionPipeline, UniPCMultistepScheduler
from llama_cpp import Llama
from dotenv import load_dotenv
import random
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
intents.members = True

client = discord.Client(intents=intents)

# models
llm = None
diff = None

# globals
bot_message = False
bot_message_count = 0
bot_message_limit = 5
user_cache = {}

# templates


system_prompts_filename = "templates/system_prompts_showcase.yaml"
with open("templates/prompt_templates.yaml", "r") as file:
    prompt_templates = yaml.safe_load(file)
with open(system_prompts_filename, "r") as file:
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


# Frees up memory from the GPU and RAM
def free_memory():
    global llm, diff

    llm = None
    diff = None
    gc.collect()
    torch.cuda.empty_cache()


# Loads the model
def load_llm():
    free_memory()
    global llm

    if llm is None:
        llm = Llama.from_pretrained(
            "Orenguteng/Llama-3-8B-Lexi-Uncensored-GGUF",
            filename="*Q4_K_M.gguf",
            n_gpu_layers=-1,
            n_ctx=8192,
            verbose=True,
            flash_attn=True,
        )


def load_diff():
    free_memory()
    global diff
    if diff is None:
        diff = StableDiffusionPipeline.from_pretrained(
            "Lykon/DreamShaper",
            torch_dtype=torch.float16,
            safety_checker=None,
            requires_safety_checker=False
        )
        diff.scheduler = UniPCMultistepScheduler.from_config(diff.scheduler.config)
        diff = diff.to("cuda")


# Returns the response from the model
def llm_response(prompt):
    global llm
    if llm is None:
        load_llm()

    response = llm(prompt, **generation_kwargs)
    return response["choices"][0]["text"]


def diff_response(prompt):
    global diff
    if diff is None:
        load_diff()

    generator = torch.manual_seed(random.randint(0, 2 ** 32 - 1))
    image = diff(prompt, generator=generator, num_inference_steps=15).images[0]
    image.save("image.png")

    return image


async def get_user(uid, message):
    # Convert uid to int
    uid = int(uid)

    # If the user is in the cache, return it
    if uid in user_cache:
        return user_cache[uid]

    # Otherwise, fetch the user and add it to the cache
    user = await message.channel.guild.fetch_member(uid)
    user_cache[uid] = user
    return user


async def parse_message_content(message):
    # ping format <@661591351581999187>
    content = message.content
    while "<@" in content:
        start = content.find("<@")
        end = content.find(">")
        ping = content[start:end + 1]
        uid = ping[2:-1]

        user = await get_user(uid, message)
        username = "user"
        nickname = "nick"
        if user is not None:
            username = user.name
            nickname = user.nick if user.nick is not None else username

        fping = f"@{username} ({nickname})"

        content = content[:start] + fping + content[end + 1:]
    return content


def count_tokens(text):
    tokens = llm.tokenize(text.encode())
    return len(tokens)


async def handle_messages(message, history_length=500):
    channel = message.channel

    token_count = 0
    messages = []

    async for message in channel.history(limit=history_length):
        message_content = await parse_message_content(message)
        author = message.author
        if message.author == client.user:
            templated_message = message_template.format(user="assistant (Sparky)", user_message=message_content)
        else:
            name = f'{message.author.name} ({author.nick if author.nick is not None else author.name})'
            templated_message = message_template.format(user=name, user_message=message_content)
        token_count += count_tokens(templated_message)
        if token_count > max_message_tokens:
            print("token count: ", token_count)
            break

        if message.content == "!split":
            break

        messages.append(templated_message)

    messages.reverse()
    print("message count: ", len(messages))
    messages = "".join(messages)
    print("max token count: ", max_message_tokens)
    print("token count: ", token_count, "\n\n")
    return messages


def construct_prompt(messages):
    global max_message_tokens, system_prompts
    with open(system_prompts_filename, "r") as file:
        system_prompts = yaml.safe_load(file)

    config = system_prompts["config"].split(" ")
    system_prompt = ""
    for subprompt in config:
        system_prompt += system_prompts[subprompt] + "\n"

    max_message_tokens = n_ctx - generation_kwargs["max_tokens"] - 1 - count_tokens(system_prompt) - 500

    return prompt_template.format(system_prompt=system_prompt, messages=messages)


# functions are in the middle of the llm response
# they are in a format like this:
# the whole function is surrounded by square brackets
# [function_name arg1 arg2 arg3 ...]


async def handle_functions(response, message):
    # find curly brackets
    if "[" not in response or "]" not in response:
        return None

    start = response.find("[")
    end = response.find("]")
    slice = response[start + 1:end]

    function = slice.split(" ")[0].lower()
    args = slice.split(" ")[1:]
    print("function: ", function)
    print("args: ", args)
    match function:
        case "img":
            diff_response(" ".join(args))
            await message.channel.send(file=discord.File("image.png"))
            if llm is None:
                load_llm()
        case _:
            return None


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
        print("prompt: \n", prompt)
        checkpoints["prompt"] = time.time()

        response = llm_response(prompt)
        checkpoints["response"] = time.time()

        print("response: \n\n", response)
        await message.channel.send(response)
        checkpoints["send"] = time.time()

        await handle_functions(response, message)
        checkpoints["functions"] = time.time()

        timing(start_time, checkpoints)
        return response


async def manual_image_pipeline(message):
    async with message.channel.typing():
        start_time = time.time()
        checkpoints = {}
        if diff is None:
            load_diff()
            checkpoints["load_diff"] = time.time()

        prompt = " ".join(message.content.split(" ")[1:])
        checkpoints["prompt"] = time.time()
        print("prompt: ", prompt)

        diff_response(prompt)
        checkpoints["response"] = time.time()

        await message.channel.send(file=discord.File("image.png"))
        checkpoints["send"] = time.time()

        load_llm()
        checkpoints["free and load_llm"] = time.time()

        timing(start_time, checkpoints)


def handle_prefix(message):
    prefix = message.content.split(" ")[0].lower()
    if prefix == "!s":
        return "!s"
    elif prefix == "!i":
        return "!i"
    elif prefix == "!img":
        return "!img"
    else:
        return None


# Discord bot
@client.event
async def on_ready():
    print(f"We have logged in as {client.user}")


@client.event
async def on_message(message):
    # if message.author == client.user:
    #     bot_message = True

    if handle_prefix(message) == "!s":  # Normal text generation
        await text_pipeline(message)
    elif handle_prefix(message) == "!img":  # Manual image generation
        await manual_image_pipeline(message)


client.run(TOKEN)
