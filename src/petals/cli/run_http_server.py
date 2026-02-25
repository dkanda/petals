import argparse
import asyncio
import json
import logging
import threading
import time
from contextlib import asynccontextmanager
from typing import AsyncIterator, Dict, List, Optional, Union

import configargparse
import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from hivemind import get_logger
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, TextIteratorStreamer

from petals import AutoDistributedModelForCausalLM
from petals.constants import PUBLIC_INITIAL_PEERS

logger = get_logger(__name__)

model = None
tokenizer = None
args = None

class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    stream: Optional[bool] = False
    max_tokens: Optional[int] = None
    stop: Optional[Union[str, List[str]]] = None


class CompletionRequest(BaseModel):
    model: str
    prompt: Union[str, List[str]]
    max_tokens: Optional[int] = 16
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    stream: Optional[bool] = False
    echo: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, tokenizer, args
    if args is None:
        # This can happen during tests if args are not set
        yield
        return

    logger.info(f"Loading model {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False, token=args.token)
    model = AutoDistributedModelForCausalLM.from_pretrained(
        args.model,
        initial_peers=args.initial_peers,
        torch_dtype=args.torch_dtype,
        token=args.token,
    )
    logger.info("Model loaded successfully.")
    yield


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/v1/models")
async def list_models():
    model_id = args.model if args else "unknown"
    return {
        "object": "list",
        "data": [
            {
                "id": model_id,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "petals",
            }
        ],
    }


def _generate_stream(inputs, **generation_kwargs):
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    generation_kwargs["streamer"] = streamer

    # Handle inputs
    if hasattr(inputs, "input_ids"):
        generation_kwargs["input_ids"] = inputs.input_ids.to(model.device)
        if hasattr(inputs, "attention_mask"):
            generation_kwargs["attention_mask"] = inputs.attention_mask.to(model.device)
    else:
        generation_kwargs["input_ids"] = inputs.to(model.device)

    thread = threading.Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    for new_text in streamer:
        yield new_text

    thread.join()


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    if args and request.model != args.model:
        raise HTTPException(status_code=404, detail="Model not found")

    conversation = [{"role": m.role, "content": m.content} for m in request.messages]

    # Apply chat template
    if tokenizer.chat_template is not None:
        prompt = tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
    else:
        # Fallback for models without chat template: simple concatenation
        prompt = ""
        for msg in conversation:
            prompt += f"{msg['role'].upper()}: {msg['content']}\n"
        prompt += "ASSISTANT:"

    inputs = tokenizer(prompt, return_tensors="pt")

    generation_kwargs = {
        "max_new_tokens": request.max_tokens if request.max_tokens else 512,
        "temperature": request.temperature,
        "top_p": request.top_p,
        "do_sample": request.temperature > 0,
    }

    if request.stream:
        return StreamingResponse(
            _chat_event_generator(inputs, request.model, generation_kwargs),
            media_type="text/event-stream",
        )
    else:
        def generate_full():
            text = ""
            for token in _generate_stream(inputs, **generation_kwargs):
                text += token
            return text

        loop = asyncio.get_running_loop()
        generated_text = await loop.run_in_executor(None, generate_full)

        return {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request.model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": generated_text,
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": inputs.input_ids.shape[1],
                "completion_tokens": len(tokenizer.encode(generated_text)),
                "total_tokens": inputs.input_ids.shape[1] + len(tokenizer.encode(generated_text)),
            },
        }


def _chat_event_generator(inputs, model_name, generation_kwargs):
    for token in _generate_stream(inputs, **generation_kwargs):
        chunk = {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model_name,
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": token},
                    "finish_reason": None,
                }
            ],
        }
        yield f"data: {json.dumps(chunk)}\n\n"

    yield "data: [DONE]\n\n"


@app.post("/v1/completions")
async def completions(request: CompletionRequest):
    if args and request.model != args.model:
        raise HTTPException(status_code=404, detail="Model not found")

    prompt = request.prompt
    if isinstance(prompt, list):
        prompt = prompt[0]  # Only support single prompt for now

    inputs = tokenizer(prompt, return_tensors="pt")

    generation_kwargs = {
        "max_new_tokens": request.max_tokens,
        "temperature": request.temperature,
        "top_p": request.top_p,
        "do_sample": request.temperature > 0,
    }

    if request.stream:
        return StreamingResponse(
            _completion_event_generator(inputs, request.model, generation_kwargs),
            media_type="text/event-stream",
        )
    else:
        def generate_full():
            text = ""
            for token in _generate_stream(inputs, **generation_kwargs):
                text += token
            return text

        loop = asyncio.get_running_loop()
        generated_text = await loop.run_in_executor(None, generate_full)

        if request.echo:
            generated_text = prompt + generated_text

        return {
            "id": f"cmpl-{int(time.time())}",
            "object": "text_completion",
            "created": int(time.time()),
            "model": request.model,
            "choices": [
                {
                    "text": generated_text,
                    "index": 0,
                    "logprobs": None,
                    "finish_reason": "length", # Simplified
                }
            ],
            "usage": {
                "prompt_tokens": inputs.input_ids.shape[1],
                "completion_tokens": len(tokenizer.encode(generated_text)),
                "total_tokens": inputs.input_ids.shape[1] + len(tokenizer.encode(generated_text)),
            },
        }

def _completion_event_generator(inputs, model_name, generation_kwargs):
    for token in _generate_stream(inputs, **generation_kwargs):
        chunk = {
            "id": f"cmpl-{int(time.time())}",
            "object": "text_completion",
            "created": int(time.time()),
            "choices": [
                {
                    "text": token,
                    "index": 0,
                    "logprobs": None,
                    "finish_reason": None,
                }
            ],
            "model": model_name,
        }
        yield f"data: {json.dumps(chunk)}\n\n"

    yield "data: [DONE]\n\n"


def main():
    global args
    parser = configargparse.ArgParser(default_config_files=["config.yml"],
                                      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add('-c', '--config', required=False, is_config_file=True, help='config file path')

    parser.add_argument("--model", type=str, required=True, help="The model to serve")
    parser.add_argument("--initial_peers", type=str, nargs='+', required=False, default=PUBLIC_INITIAL_PEERS,
                       help='Multiaddrs of one or more DHT peers from the target swarm. Default: connects to the public swarm')
    parser.add_argument("--torch_dtype", type=str, default="auto", help="The torch dtype to use")
    parser.add_argument("--token", type=str, default=None, help="Hugging Face hub auth token")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")

    args = parser.parse_args()

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
