# -*- coding: utf-8 -*-
"""底座模型最小推理验证：加载到 GPU，走 chat template 生成，记录显存与时延。"""
import os
import glob
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_ROOT = os.path.join(ROOT, "models")


def locate_model_dir():
    candidates = glob.glob(os.path.join(MODELS_ROOT, "**", "snapshots", "*"), recursive=True)
    candidates = [c for c in candidates if os.path.isdir(c)]
    if not candidates:
        raise FileNotFoundError(f"no model snapshot under {MODELS_ROOT}")
    # 取最新修改的快照
    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return candidates[0]


def main():
    model_dir = locate_model_dir()
    print(f"model dir: {model_dir}")
    print(f"torch {torch.__version__} cuda {torch.cuda.is_available()}")

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        dtype=torch.bfloat16,
        device_map="cuda",
    )
    model.eval()

    prompts = [
        "高血压患者日常应该怎么监测血压？",
        "糖尿病患者饮食怎么搭配？",
    ]

    for prompt in prompts:
        messages = [
            {"role": "system", "content": "你是健康管理助手，回答通用、保守，提示异常及时就医。"},
            {"role": "user", "content": prompt},
        ]
        inputs = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
            )
        dt = time.time() - t0
        text = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        peak_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
        print("=" * 60)
        print(f"Q: {prompt}")
        print(f"A: {text}")
        print(f"latency {dt:.2f}s | peak GPU mem {peak_mb:.1f} MB")


if __name__ == "__main__":
    main()
