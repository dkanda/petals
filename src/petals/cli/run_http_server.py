import argparse
import logging
import os
import threading
import time
from typing import List, Optional, Union, AsyncIterator
import json
import asyncio

import configargparse
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from transformers import AutoTokenizer, TextIteratorStreamer

from petals import AutoDistributedModelForCausalLM
from petals.constants import PUBLIC_INITIAL_PEERS, DTYPE_MAP
from hivemind.utils.logging import get_logger

logger = get_logger(__name__)

def create_app(model, tokenizer, args):
    app = FastAPI()

    class ChatMessage(BaseModel):
        role: str
        content: str

    class ChatCompletionRequest(BaseModel):
        model: str
        messages: List[ChatMessage]
        max_tokens: Optional[int] = None
        temperature: Optional[float] = 1.0
        top_p: Optional[float] = 1.0
        n: Optional[int] = 1
        stream: Optional[bool] = False

    class CompletionRequest(BaseModel):
        model: str
        prompt: str
        max_tokens: Optional[int] = None
        temperature: Optional[float] = 1.0
        top_p: Optional[float] = 1.0
        n: Optional[int] = 1
        stream: Optional[bool] = False

    @app.get("/v1/models")
    async def list_models():
        return {
            "object": "list",
            "data": [
                {
                    "id": args.model,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "petals",
                }
            ]
        }

    async def generate_streaming_async(prompt, max_new_tokens, temperature, top_p):
        loop = asyncio.get_event_loop()
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True)

        generation_kwargs = dict(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            streamer=streamer,
            do_sample=True if temperature > 0 else False,
        )

        def run_generation():
             model.generate(**generation_kwargs)

        thread = threading.Thread(target=run_generation)
        thread.start()

        def get_next():
            try:
                return next(streamer)
            except StopIteration:
                return None

        while True:
            text = await loop.run_in_executor(None, get_next)
            if text is None:
                break
            yield text

    @app.post("/v1/chat/completions")
    async def chat_completions(request: ChatCompletionRequest):
        if hasattr(tokenizer, "apply_chat_template"):
            prompt = tokenizer.apply_chat_template(
                [m.model_dump() for m in request.messages],
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            prompt = ""
            for m in request.messages:
                prompt += f"{m.role}: {m.content}\n"
            prompt += "assistant: "

        max_new_tokens = request.max_tokens if request.max_tokens else args.max_new_tokens

        if request.stream:
            async def stream_generator():
                created = int(time.time())
                async for text in generate_streaming_async(prompt, max_new_tokens, request.temperature, request.top_p):
                    chunk = {
                        "id": "chatcmpl-stream",
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": args.model,
                        "choices": [{
                            "index": 0,
                            "delta": {"content": text},
                            "finish_reason": None
                        }]
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"

                final_chunk = {
                    "id": "chatcmpl-stream",
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": args.model,
                    "choices": [{
                        "index": 0,
                        "delta": {},
                        "finish_reason": "stop"
                    }]
                }
                yield f"data: {json.dumps(final_chunk)}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(stream_generator(), media_type="text/event-stream")
        else:
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids
            outputs = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                do_sample=True if request.temperature > 0 else False
            )
            new_tokens = outputs[0][input_ids.shape[1]:]
            text_output = tokenizer.decode(new_tokens, skip_special_tokens=True)

            return {
                "id": "chatcmpl-123",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": args.model,
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": text_output
                    },
                    "finish_reason": "stop"
                }],
                "usage": {
                    "prompt_tokens": len(input_ids[0]),
                    "completion_tokens": len(new_tokens),
                    "total_tokens": len(input_ids[0]) + len(new_tokens)
                }
            }

    @app.post("/v1/completions")
    async def completions(request: CompletionRequest):
        max_new_tokens = request.max_tokens if request.max_tokens else args.max_new_tokens

        if request.stream:
            async def stream_generator():
                created = int(time.time())
                async for text in generate_streaming_async(request.prompt, max_new_tokens, request.temperature, request.top_p):
                    chunk = {
                        "id": "cmpl-stream",
                        "object": "text_completion",
                        "created": created,
                        "model": args.model,
                        "choices": [{
                            "text": text,
                            "index": 0,
                            "logprobs": None,
                            "finish_reason": None
                        }]
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"

                final_chunk = {
                    "id": "cmpl-stream",
                    "object": "text_completion",
                    "created": created,
                    "model": args.model,
                    "choices": [{
                        "text": "",
                        "index": 0,
                        "logprobs": None,
                        "finish_reason": "stop"
                    }]
                }
                yield f"data: {json.dumps(final_chunk)}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(stream_generator(), media_type="text/event-stream")
        else:
            input_ids = tokenizer(request.prompt, return_tensors="pt").input_ids
            outputs = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                do_sample=True if request.temperature > 0 else False
            )
            new_tokens = outputs[0][input_ids.shape[1]:]
            text_output = tokenizer.decode(new_tokens, skip_special_tokens=True)

            return {
                "id": "cmpl-123",
                "object": "text_completion",
                "created": int(time.time()),
                "model": args.model,
                "choices": [{
                    "text": text_output,
                    "index": 0,
                    "logprobs": None,
                    "finish_reason": "stop"
                }],
                "usage": {
                    "prompt_tokens": len(input_ids[0]),
                    "completion_tokens": len(new_tokens),
                    "total_tokens": len(input_ids[0]) + len(new_tokens)
                }
            }

    return app

def main():
    parser = configargparse.ArgParser(default_config_files=["config.yml"],
                                      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add('-c', '--config', required=False, is_config_file=True, help='config file path')

    parser.add_argument("--model", type=str, required=True, help="Model name or path")
    parser.add_argument("--port", type=int, default=8000, help="Port to run the server on")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--initial_peers", type=str, nargs='+', default=PUBLIC_INITIAL_PEERS,
                        help="Multiaddrs of one or more DHT peers")
    parser.add_argument("--torch_dtype", type=str, choices=DTYPE_MAP.keys(), default="auto",
                        help="Use this dtype to store block weights and do computations")
    parser.add_argument("--revision", type=str, default=None,
                        help="The specific model version to use.")
    parser.add_argument("--max_new_tokens", type=int, default=1024, help="Default max new tokens")
    parser.add_argument("--connect_timeout", type=float, default=10, help="Timeout for connecting to the swarm")

    args = parser.parse_args()

    # Initialize model and tokenizer
    logger.info(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision, use_fast=False)
    model = AutoDistributedModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=args.torch_dtype,
        initial_peers=args.initial_peers,
        revision=args.revision
    )

    app = create_app(model, tokenizer, args)

    uvicorn.run(app, host=args.host, port=args.port)

if __name__ == "__main__":
    main()
