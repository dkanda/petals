import argparse
import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from threading import Thread
from typing import Any, Dict, List, Optional, Union

import configargparse
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, HTMLResponse
from hivemind.utils.logging import get_logger
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, TextIteratorStreamer

from petals import AutoDistributedModelForCausalLM
from petals.constants import PUBLIC_INITIAL_PEERS

logger = get_logger(__name__)

model = None
tokenizer = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, tokenizer
    args = app.state.args

    logger.info(f"Loading tokenizer for {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, token=args.token, revision=args.revision)

    logger.info(f"Loading model {args.model}...")
    model = AutoDistributedModelForCausalLM.from_pretrained(
        args.model,
        token=args.token,
        revision=args.revision,
        initial_peers=args.initial_peers,
        torch_dtype=args.torch_dtype,
    )
    logger.info("Model loaded successfully.")
    yield
    # Clean up resources if needed


class ModelCard(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "petals"


class ModelList(BaseModel):
    object: str = "list"
    data: List[ModelCard]


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    max_tokens: Optional[int] = Field(default=1024)
    temperature: Optional[float] = Field(default=0.7)
    top_p: Optional[float] = Field(default=0.9)
    stream: Optional[bool] = Field(default=False)
    stop: Optional[Union[str, List[str]]] = None


class ChatChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Optional[str] = None


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatChoice]
    usage: Optional[Dict[str, int]] = None


class Delta(BaseModel):
    role: Optional[str] = None
    content: Optional[str] = None


class ChatChunkChoice(BaseModel):
    index: int
    delta: Delta
    finish_reason: Optional[str] = None


class ChatCompletionChunk(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatChunkChoice]


class CompletionRequest(BaseModel):
    model: str
    prompt: Union[str, List[str]]
    max_tokens: Optional[int] = Field(default=16)
    temperature: Optional[float] = Field(default=1.0)
    top_p: Optional[float] = Field(default=1.0)
    stream: Optional[bool] = Field(default=False)
    echo: Optional[bool] = Field(default=False)


class CompletionChoice(BaseModel):
    text: str
    index: int
    finish_reason: Optional[str] = None


class CompletionResponse(BaseModel):
    id: str
    object: str = "text_completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[CompletionChoice]
    usage: Optional[Dict[str, int]] = None


def create_app(args):
    app = FastAPI(lifespan=lifespan)
    app.state.args = args

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/v1/models", response_model=ModelList)
    async def list_models():
        return ModelList(data=[ModelCard(id=args.model)])

    @app.get("/api/v1/status")
    async def status():
        device_str = str(model.device) if hasattr(model, "device") else "unknown"
        gpu_usage = "N/A"
        if hasattr(model, "device") and hasattr(model.device, "type") and model.device.type == "cuda":
            allocated = torch.cuda.memory_allocated(model.device)
            gpu_usage = f"{allocated / 1024**3:.2f} GiB"

        connection_status = "Healthy" if model is not None else "Initializing"
        block_health = "N/A"
        inference_throughput = "N/A"
        if model is not None and hasattr(model, "get_client_state"):
            client_state = model.get_client_state()
            if client_state:
                block_health = "Healthy" # Mock for now
                inference_throughput = client_state.get("inference_throughput", "N/A")

        return {
            "model": args.model,
            "device": device_str,
            "gpu_usage": gpu_usage,
            "connection_status": connection_status,
            "block_health": block_health,
            "inference_throughput": inference_throughput,
        }

    @app.get("/dashboard", response_class=HTMLResponse)
    async def dashboard():
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Petals Server Dashboard</title>
            <style>
                body { font-family: sans-serif; margin: 40px; background-color: #f4f4f9; color: #333; }
                h1 { color: #5a5a5a; }
                .card { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin-bottom: 20px; }
                .stat { font-size: 1.2em; margin: 10px 0; }
                .label { font-weight: bold; }
                .status-good { color: green; }
                .status-bad { color: red; }
                .status-warn { color: orange; }
            </style>
        </head>
        <body>
            <h1>Petals Server Dashboard</h1>
            <div class="card">
                <h2>Server Status</h2>
                <div class="stat"><span class="label">API Status:</span> <span id="api-status" class="status-bad">Checking...</span></div>
                <div class="stat"><span class="label">Connection Status:</span> <span id="connection-status">Loading...</span></div>
                <div class="stat"><span class="label">Block Health:</span> <span id="block-health">Loading...</span></div>
                <div class="stat"><span class="label">Model:</span> <span id="model-name">Loading...</span></div>
                <div class="stat"><span class="label">Device:</span> <span id="device-name">Loading...</span></div>
                <div class="stat"><span class="label">GPU Usage:</span> <span id="gpu-usage">Loading...</span></div>
                <div class="stat"><span class="label">Inference Throughput:</span> <span id="inference-throughput">Loading...</span></div>
            </div>

            <script>
                async function fetchStatus() {
                    try {
                        const response = await fetch('/api/v1/status');
                        if (response.ok) {
                            const data = await response.json();
                            document.getElementById('api-status').textContent = 'Online';
                            document.getElementById('api-status').className = 'status-good';

                            document.getElementById('model-name').textContent = data.model;
                            document.getElementById('device-name').textContent = data.device;
                            document.getElementById('gpu-usage').textContent = data.gpu_usage;

                            const connStatusEl = document.getElementById('connection-status');
                            connStatusEl.textContent = data.connection_status;
                            connStatusEl.className = data.connection_status === 'Healthy' ? 'status-good' : 'status-warn';

                            document.getElementById('block-health').textContent = data.block_health;
                            document.getElementById('inference-throughput').textContent = data.inference_throughput;

                        } else {
                            document.getElementById('api-status').textContent = 'Error';
                            document.getElementById('api-status').className = 'status-bad';
                        }
                    } catch (err) {
                        document.getElementById('api-status').textContent = 'Offline';
                        document.getElementById('api-status').className = 'status-bad';
                    }
                }

                fetchStatus();
                setInterval(fetchStatus, 5000);
            </script>
        </body>
        </html>
        """
        return HTMLResponse(content=html_content, status_code=200)

    @app.post("/v1/chat/completions")
    async def chat_completions(request: ChatCompletionRequest):
        # Format prompts
        prompt = tokenizer.apply_chat_template(
            [m.model_dump() for m in request.messages], tokenize=False, add_generation_prompt=True
        )

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        generation_kwargs = dict(
            **inputs,
            max_new_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            do_sample=request.temperature > 0,
        )

        if request.stream:
            streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
            generation_kwargs["streamer"] = streamer

            thread = Thread(target=model.generate, kwargs=generation_kwargs)
            thread.start()

            async def event_generator():
                request_id = f"chatcmpl-{uuid.uuid4()}"
                created = int(time.time())

                # Send role first
                chunk = ChatCompletionChunk(
                    id=request_id,
                    created=created,
                    model=request.model,
                    choices=[ChatChunkChoice(index=0, delta=Delta(role="assistant"))],
                )
                yield f"data: {chunk.model_dump_json()}\n\n"

                for new_text in streamer:
                    chunk = ChatCompletionChunk(
                        id=request_id,
                        created=created,
                        model=request.model,
                        choices=[ChatChunkChoice(index=0, delta=Delta(content=new_text))],
                    )
                    yield f"data: {chunk.model_dump_json()}\n\n"

                chunk = ChatCompletionChunk(
                    id=request_id,
                    created=created,
                    model=request.model,
                    choices=[ChatChunkChoice(index=0, delta=Delta(), finish_reason="stop")],
                )
                yield f"data: {chunk.model_dump_json()}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(event_generator(), media_type="text/event-stream")

        else:
            outputs = model.generate(**generation_kwargs)
            # Decode only the new tokens
            input_len = inputs.input_ids.shape[1]
            new_tokens = outputs[0, input_len:]
            response_text = tokenizer.decode(new_tokens, skip_special_tokens=True)

            return ChatCompletionResponse(
                id=f"chatcmpl-{uuid.uuid4()}",
                model=request.model,
                choices=[
                    ChatChoice(
                        index=0, message=ChatMessage(role="assistant", content=response_text), finish_reason="stop"
                    )
                ],
                usage={
                    "prompt_tokens": input_len,
                    "completion_tokens": len(new_tokens),
                    "total_tokens": input_len + len(new_tokens),
                },
            )

    @app.post("/v1/completions")
    async def completions(request: CompletionRequest):
        if isinstance(request.prompt, list):
            # For simplicity, only support single string prompt for now
            if len(request.prompt) != 1:
                raise HTTPException(status_code=400, detail="Batch processing not supported yet")
            prompt = request.prompt[0]
        else:
            prompt = request.prompt

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        generation_kwargs = dict(
            **inputs,
            max_new_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            do_sample=request.temperature > 0,
        )

        if request.stream:
            streamer = TextIteratorStreamer(tokenizer, skip_prompt=not request.echo, skip_special_tokens=True)
            generation_kwargs["streamer"] = streamer

            thread = Thread(target=model.generate, kwargs=generation_kwargs)
            thread.start()

            async def event_generator():
                request_id = f"cmpl-{uuid.uuid4()}"
                created = int(time.time())

                for new_text in streamer:
                    chunk = CompletionResponse(
                        id=request_id,
                        created=created,
                        model=request.model,
                        choices=[CompletionChoice(index=0, text=new_text, finish_reason=None)],
                    )
                    # Note: CompletionResponse usually has object="text_completion", reusing model is fine
                    yield f"data: {chunk.model_dump_json()}\n\n"

                chunk = CompletionResponse(
                    id=request_id,
                    created=created,
                    model=request.model,
                    choices=[CompletionChoice(index=0, text="", finish_reason="stop")],
                )
                yield f"data: {chunk.model_dump_json()}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(event_generator(), media_type="text/event-stream")

        else:
            outputs = model.generate(**generation_kwargs)

            if request.echo:
                text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            else:
                input_len = inputs.input_ids.shape[1]
                new_tokens = outputs[0, input_len:]
                text = tokenizer.decode(new_tokens, skip_special_tokens=True)

            return CompletionResponse(
                id=f"cmpl-{uuid.uuid4()}",
                model=request.model,
                choices=[CompletionChoice(index=0, text=text, finish_reason="stop")],
                usage={
                    "prompt_tokens": inputs.input_ids.shape[1],
                    "completion_tokens": len(outputs[0]) - inputs.input_ids.shape[1],
                    "total_tokens": len(outputs[0]),
                },
            )

    return app


def main():
    parser = configargparse.ArgParser(
        default_config_files=["config.yml"], formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add("-c", "--config", required=False, is_config_file=True, help="config file path")

    group = parser.add_argument_group("HTTP Server")
    group.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind the server to")
    group.add_argument("--port", type=int, default=8000, help="Port to bind the server to")

    model_group = parser.add_argument_group("Model")
    model_group.add_argument("--model", type=str, required=True, help="Model name or path to serve")
    model_group.add_argument("--revision", type=str, default="main", help="Model revision")
    model_group.add_argument("--token", type=str, default=None, help="Hugging Face hub auth token")
    model_group.add_argument("--torch_dtype", type=str, default="auto", help="Data type for the model")

    swarm_group = parser.add_argument_group("Swarm")
    swarm_group.add_argument(
        "--initial_peers",
        type=str,
        nargs="+",
        required=False,
        default=PUBLIC_INITIAL_PEERS,
        help="Multiaddrs of one or more DHT peers from the target swarm",
    )
    swarm_group.add_argument("--connect_timeout", type=float, default=60, help="Timeout for connecting to the swarm")

    args = parser.parse_args()

    app = create_app(args)

    logger.info(f"Listening on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
