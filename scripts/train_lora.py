# -*- coding: utf-8 -*-
"""
LoRA 微调：Qwen2.5-0.5B-Instruct + 健康咨询垂类指令数据。
- completion-only 掩码：只监督 assistant 回答部分，prompt 部分 label 置 -100
- 记录 train loss、峰值显存、训练时长、样本数，导出 meta 与日志供作品集引用
"""
import os
import json
import glob
import time
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_ROOT = os.path.join(ROOT, "models")
DATA_FILE = os.path.join(ROOT, "data", "health_qa.jsonl")
OUTPUT_DIR = os.path.join(ROOT, "output")
ADAPTER_DIR = os.path.join(OUTPUT_DIR, "lora_adapter")

SYSTEM = "你是健康管理助手，回答通用、保守，不替代医生诊断；提示异常及时就医。"
MAX_LEN = 512
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
LR = 1e-4
EPOCHS = 3
BATCH_SIZE = 1
GRAD_ACCUM = 4


def locate_model_dir():
    candidates = [p for p in glob.glob(os.path.join(MODELS_ROOT, "**", "snapshots", "*"), recursive=True) if os.path.isdir(p)]
    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    if not candidates:
        raise FileNotFoundError(f"no model snapshot under {MODELS_ROOT}")
    return candidates[0]


def main():
    os.makedirs(ADAPTER_DIR, exist_ok=True)
    model_dir = locate_model_dir()
    print(f"model dir: {model_dir}")

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16)
    model.to("cuda")

    # 读数据
    records = []
    with open(DATA_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    print(f"samples: {len(records)}")

    def build_example(rec):
        user = rec["instruction"] + (("\n" + rec["input"]) if rec["input"].strip() else "")
        prompt_enc = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": user},
            ],
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
        prompt_ids = prompt_enc["input_ids"][0].tolist()
        out_ids = tokenizer.encode(rec["output"], add_special_tokens=False) + [tokenizer.eos_token_id]
        input_ids = (prompt_ids + out_ids)[:MAX_LEN]
        labels = ([-100] * len(prompt_ids) + out_ids)[:MAX_LEN]
        return {"input_ids": input_ids, "labels": labels}

    ds = Dataset.from_list(records).map(build_example, remove_columns=Dataset.from_list(records).column_names)
    ds = ds.filter(lambda x: len(x["input_ids"]) > 0)
    print(f"tokenized examples: {len(ds)}")

    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    collator = DataCollatorForSeq2Seq(tokenizer, padding=True, label_pad_token_id=-100)

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        num_train_epochs=EPOCHS,
        logging_steps=1,
        bf16=True,
        save_strategy="no",
        report_to=[],
        max_grad_norm=1.0,
        warmup_steps=2,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds,
        data_collator=collator,
        processing_class=tokenizer,
    )

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    train_result = trainer.train()
    train_secs = time.time() - t0
    peak_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

    # 保存 adapter
    model.save_pretrained(ADAPTER_DIR)
    tokenizer.save_pretrained(ADAPTER_DIR)

    # 导出 loss 日志
    log_history = [h for h in trainer.state.log_history if "loss" in h]
    with open(os.path.join(OUTPUT_DIR, "train_log.json"), "w", encoding="utf-8") as f:
        json.dump(log_history, f, ensure_ascii=False, indent=2)
    with open(os.path.join(OUTPUT_DIR, "train_log.csv"), "w", encoding="utf-8") as f:
        f.write("step,loss,epoch\n")
        for h in log_history:
            f.write(f"{h.get('step','')},{h.get('loss','')},{h.get('epoch','')}\n")

    meta = {
        "base_model": model_dir,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "samples": len(ds),
        "max_len": MAX_LEN,
        "lora": {"r": LORA_R, "alpha": LORA_ALPHA, "dropout": LORA_DROPOUT,
                 "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"]},
        "train": {"lr": LR, "epochs": EPOCHS, "batch_size": BATCH_SIZE,
                  "grad_accum": GRAD_ACCUM, "bf16": True},
        "train_secs": round(train_secs, 2),
        "peak_gpu_mem_mb": round(peak_mb, 1),
        "final_train_loss": round(float(train_result.training_loss), 4),
        "adapter_dir": ADAPTER_DIR,
    }
    with open(os.path.join(OUTPUT_DIR, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("=" * 60)
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    print(f"adapter saved -> {ADAPTER_DIR}")


if __name__ == "__main__":
    main()
