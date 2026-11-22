# -*- coding: utf-8 -*-
"""合并 LoRA adapter 到底座模型，导出为独立可部署模型到 output/merged。"""
import os
import glob
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_ROOT = os.path.join(ROOT, "models")
ADAPTER_DIR = os.path.join(ROOT, "output", "lora_adapter")
MERGED_DIR = os.path.join(ROOT, "output", "merged")


def locate_model_dir():
    candidates = [p for p in glob.glob(os.path.join(MODELS_ROOT, "**", "snapshots", "*"), recursive=True) if os.path.isdir(p)]
    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return candidates[0]


def dir_size_mb(path):
    total = sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fs in os.walk(path) for f in fs)
    return total / 1024 / 1024


def main():
    base_dir = locate_model_dir()
    print(f"base: {base_dir}")
    print(f"adapter: {ADAPTER_DIR}")

    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_DIR)
    base = AutoModelForCausalLM.from_pretrained(base_dir, dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(base, ADAPTER_DIR)
    model = model.merge_and_unload()

    os.makedirs(MERGED_DIR, exist_ok=True)
    model.save_pretrained(MERGED_DIR, safe_serialization=True)
    tokenizer.save_pretrained(MERGED_DIR)

    params = sum(p.numel() for p in model.parameters())
    print(f"merged params: {params/1e6:.1f} M")
    print(f"merged dir size: {dir_size_mb(MERGED_DIR):.1f} MB")
    print(f"merged -> {MERGED_DIR}")


if __name__ == "__main__":
    main()
