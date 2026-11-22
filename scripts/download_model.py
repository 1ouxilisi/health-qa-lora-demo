# -*- coding: utf-8 -*-
"""从 ModelScope 下载底座模型到本地 models/ 目录（HF 直连不通，走 ModelScope）。"""
import os
from modelscope import snapshot_download

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(ROOT, "models")
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    print(f"downloading {MODEL_ID} -> {MODELS_DIR}")
    model_dir = snapshot_download(MODEL_ID, cache_dir=MODELS_DIR)
    print(f"model dir: {model_dir}")
    # 列出关键文件
    for f in sorted(os.listdir(model_dir)):
        p = os.path.join(model_dir, f)
        if os.path.isfile(p):
            print(f"  {f}  {os.path.getsize(p)/1024/1024:.1f} MB")


if __name__ == "__main__":
    main()
