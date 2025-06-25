import torch
import chainlit as cl
from pathlib import Path

from model.gpt import GPTModel
from model.generate import generate
from model.tokenizer_utils import text_to_token_ids, token_ids_to_text
from model.model_config import GPT_CONFIG_355M

# Load tokenizer (use tiktoken or your own)
import tiktoken
tokenizer = tiktoken.get_encoding("gpt2")

# Load model
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model_path = Path(__file__).parent / "gpt2-medium355M-sft.pth"

@cl.on_chat_start
def load_model():
    checkpoint = torch.load(model_path, map_location=device)
    model = GPTModel(GPT_CONFIG_355M)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    cl.user_session.set("model", model)

@cl.on_message
async def respond(message: cl.Message):
    model = cl.user_session.get("model")
    prompt = f"""Below is an instruction that describes a task. Write a response that completes the request.

### Instruction:
{message.content}
"""

    idx = text_to_token_ids(prompt, tokenizer).to(device)
    token_ids = generate(
        model=model,
        idx=idx,
        max_new_tokens=50,
        context_size=GPT_CONFIG_355M["context_length"],
        eos_id=50256
    )

    full_output = token_ids_to_text(token_ids, tokenizer)
    response = full_output[len(prompt):].strip()

    await cl.Message(content=response).send()
