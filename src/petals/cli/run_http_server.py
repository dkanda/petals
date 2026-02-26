import argparse
import asyncio
import json
import logging
import os
import threading
import time
import uuid
from typing import Any, AsyncIterator, Dict, List, Optional, Union

import configargparse
import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, TextIteratorStreamer

from petals import AutoDistributedModelForCausalLM

logger = logging.getLogger(__name__)

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    max_tokens: Optional[int] = Field(default=None, description="The maximum number of tokens to generate")
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None

class CompletionRequest(BaseModel):
    model: str
    prompt: Union[str, List[str]]
    max_tokens: Optional[int] = 16
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None
    echo: Optional[bool] = False

class ModelCard(BaseModel):
    id: str
    object: str = "model"
    owned_by: str = "petals"
    permission: List[Any] = []

class ModelList(BaseModel):
    object: str = "list"
    data: List[ModelCard]

# Global variables
model = None
tokenizer = None
model_name = None

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/v1/models", response_model=ModelList)
async def list_models():
    return ModelList(data=[ModelCard(id=model_name)])

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    if request.model != model_name:
        raise HTTPException(status_code=404, detail=f"Model {request.model} not found. Available: {model_name}")

    # Convert messages to prompt
    try:
        prompt = tokenizer.apply_chat_template(
            [msg.model_dump() for msg in request.messages],
            tokenize=False,
            add_generation_prompt=True
        )
    except Exception as e:
        logger.warning(f"Failed to apply chat template: {e}")
        prompt = ""
        for msg in request.messages:
            prompt += f"{msg.role}: {msg.content}\n"
        prompt += "assistant:"

    return await generate_response(prompt, request, chat=True)

@app.post("/v1/completions")
async def completions(request: CompletionRequest):
    if request.model != model_name:
        raise HTTPException(status_code=404, detail=f"Model {request.model} not found. Available: {model_name}")

    prompt = request.prompt
    if isinstance(prompt, list):
        prompt = prompt[0]

    return await generate_response(prompt, request, chat=False)

async def generate_response(prompt: str, request: Union[ChatCompletionRequest, CompletionRequest], chat: bool):
    global model, tokenizer

    kwargs = {
        "max_new_tokens": request.max_tokens if request.max_tokens else 512,
        "temperature": request.temperature,
        "top_p": request.top_p,
        "do_sample": request.temperature > 0,
    }

    inputs = tokenizer(prompt, return_tensors="pt")["input_ids"]

    if request.stream:
        return StreamingResponse(stream_generator(inputs, kwargs, chat, request.model), media_type="text/event-stream")
    else:
        outputs = model.generate(inputs, **kwargs)
        text = tokenizer.decode(outputs[0])

        if not chat:
            if not getattr(request, "echo", False):
                 if text.startswith(prompt):
                     text = text[len(prompt):]
        else:
            input_length = inputs.shape[1]
            text = tokenizer.decode(outputs[0][input_length:])

        # Calculate token usage
        prompt_tokens = len(inputs[0])
        completion_tokens = len(outputs[0]) - len(inputs[0])

        return create_response(text, chat, request.model, prompt_tokens, completion_tokens)

def create_response(text: str, chat: bool, model_id: str, prompt_tokens: int = 0, completion_tokens: int = 0):
    req_id = f"gen-{uuid.uuid4()}"
    created = int(time.time())

    usage = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens
    }

    if chat:
        return {
            "id": req_id,
            "object": "chat.completion",
            "created": created,
            "model": model_id,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": text
                },
                "finish_reason": "stop"
            }],
            "usage": usage
        }
    else:
        return {
            "id": req_id,
            "object": "text_completion",
            "created": created,
            "model": model_id,
            "choices": [{
                "text": text,
                "index": 0,
                "logprobs": None,
                "finish_reason": "stop"
            }],
            "usage": usage
        }

async def stream_generator(input_ids, kwargs, chat: bool, model_id: str):
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True)
    generation_kwargs = dict(input_ids=input_ids, streamer=streamer, **kwargs)

    thread = threading.Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    req_id = f"gen-{uuid.uuid4()}"
    created = int(time.time())

    for new_text in streamer:
        if chat:
            chunk = {
                "id": req_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model_id,
                "choices": [{
                    "index": 0,
                    "delta": {
                        "content": new_text
                    },
                    "finish_reason": None
                }]
            }
        else:
            chunk = {
                "id": req_id,
                "object": "text_completion",
                "created": created,
                "model": model_id,
                "choices": [{
                    "text": new_text,
                    "index": 0,
                    "logprobs": None,
                    "finish_reason": None
                }]
            }
        yield f"data: {json.dumps(chunk)}\n\n"

    if chat:
        chunk = {
            "id": req_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model_id,
            "choices": [{
                "index": 0,
                "delta": {},
                "finish_reason": "stop"
            }]
        }
        yield f"data: {json.dumps(chunk)}\n\n"
    else:
        chunk = {
            "id": req_id,
            "object": "text_completion",
            "created": created,
            "model": model_id,
            "choices": [{
                "text": "",
                "index": 0,
                "logprobs": None,
                "finish_reason": "stop"
            }]
        }
        yield f"data: {json.dumps(chunk)}\n\n"
    yield "data: [DONE]\n\n"

def main():
    global model, tokenizer, model_name

    parser = configargparse.ArgParser(default_config_files=["config.yml"],
                                      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add('-c', '--config', required=False, is_config_file=True, help='config file path')

    parser.add_argument("model", type=str, help="Model name or path")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to listen on")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use for inference client")

    args, unknown_args = parser.parse_known_args()
    model_name = args.model

    logging.basicConfig(level=logging.INFO)
    logger.info(f"Loading model {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoDistributedModelForCausalLM.from_pretrained(model_name)
    logger.info(f"Model loaded.")

    uvicorn.run(app, host=args.host, port=args.port)

if __name__ == "__main__":
    main()
