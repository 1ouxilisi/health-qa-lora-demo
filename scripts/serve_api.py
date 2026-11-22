# -*- coding: utf-8 -*-
"""
本地推理 API：加载合并后的模型，暴露 OpenAI 兼容接口。
- GET  /health
- POST /v1/chat/completions
启动：python scripts/serve_api.py  （默认 127.0.0.1:8000）
"""
import os
import time
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MERGED_DIR = os.path.join(ROOT, "output", "merged")
ADAPTER_DIR = os.path.join(ROOT, "output", "lora_adapter")

app = FastAPI(title="Health QA LoRA Demo", version="0.1.0")
_state = {"model": None, "tokenizer": None, "device": "cuda" if torch.cuda.is_available() else "cpu"}


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    max_tokens: Optional[int] = 256
    stream: Optional[bool] = False


def load_model():
    if _state["model"] is not None:
        return
    if os.path.isdir(MERGED_DIR) and os.path.exists(os.path.join(MERGED_DIR, "config.json")):
        model_dir = MERGED_DIR
        print(f"loading merged model: {model_dir}")
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16, device_map=_state["device"])
    else:
        raise FileNotFoundError("merged model not found, run merge_and_save.py first")
    model.eval()
    _state["model"] = model
    _state["tokenizer"] = tokenizer


@app.on_event("startup")
def startup():
    load_model()


@app.get("/health")
def health():
    return {"status": "ok", "device": _state["device"], "model_loaded": _state["model"] is not None}


@app.post("/v1/chat/completions")
def chat_completions(req: ChatRequest):
    if not req.messages:
        raise HTTPException(status_code=400, detail="messages is empty")
    model, tokenizer = _state["model"], _state["tokenizer"]

    messages = [{"role": m.role, "content": m.content} for m in req.messages]
    enc = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
    )
    enc = {k: v.to(model.device) for k, v in enc.items()}

    t0 = time.time()
    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=req.max_tokens,
            do_sample=req.temperature > 0,
            temperature=req.temperature,
            top_p=req.top_p,
            pad_token_id=tokenizer.eos_token_id,
        )
    dt = time.time() - t0
    text = tokenizer.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)

    return {
        "id": f"chat-{int(time.time())}",
        "object": "chat.completion",
        "model": "qwen2.5-0.5b-health-lora",
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}
        ],
        "usage": {
            "prompt_tokens": int(enc["input_ids"].shape[1]),
            "completion_tokens": int(out.shape[1] - enc["input_ids"].shape[1]),
        },
        "latency_sec": round(dt, 2),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8123)
