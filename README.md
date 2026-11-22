# 健康咨询小模型 LoRA 微调与本地推理部署 Demo

基于 [Qwen2.5-0.5B-Instruct](https://modelscope.cn/models/Qwen/Qwen2.5-0.5B-Instruct) 的垂类轻量微调流程验证 Demo：在单张 8GB 消费级笔记本 GPU 上跑通「数据准备 → LoRA 训练 → 权重合并导出 → 本地 OpenAI 兼容推理 API」完整模型工程闭环。

> 本仓库为流程验证项目，所有指标来自本机实测，未夸大效果；数据与回答仅为通用健康管理信息，**不构成医疗诊断**。蒸馏、全参微调、预训练等环节未在本项目实践，不计入成果。

## 环境与硬件

| 项 | 值 |
|---|---|
| GPU | NVIDIA RTX 5060 Laptop GPU（8GB，sm_120） |
| CUDA / PyTorch | CUDA 12.8 / torch 2.11.0+cu128 |
| Python | 3.12 |
| 框架 | Transformers 5.24.0、PEFT 0.22.0、Accelerate 1.16.0、Datasets 5.2.0、FastAPI 0.165.0 |
| 底座模型 | Qwen2.5-0.5B-Instruct（494M 参数，ModelScope 拉取） |

## 技术链路（实际完成）

1. **数据准备**：154 条健康/慢病管理咨询指令数据，覆盖高血压、糖尿病、血脂、尿酸痛风、饮食营养、运动、睡眠情绪、体重管理、随访复查、危险信号识别等主题；JSONL（instruction / input / output），按问题去重，格式校验与长度截断。
2. **LoRA 微调**：PEFT 挂载 LoRA，completion-only 标签掩码（只监督回答部分，prompt 部分置 -100），bf16 训练。
3. **权重合并导出**：LoRA adapter 与底座合并，导出独立可部署模型；同时保留轻量 adapter 权重。
4. **推理部署**：FastAPI 封装 OpenAI 兼容 `POST /v1/chat/completions`，本地起服务提供对话接口。

## 实测指标（本机实测，非估算）

| 项目 | 数值 |
|---|---|
| 训练样本数 | 154 条 |
| LoRA 配置 | r=8, alpha=16, dropout=0.05, 目标 q/k/v/o_proj |
| 可训练参数 | 1,081,344（占总参数 0.218%） |
| 训练超参 | lr 1e-4，epochs 3，batch=1，梯度累积 4，bf16，max_len 512 |
| 训练步数 / 时长 | 117 步 / 327.32 秒（约 5 分 27 秒） |
| 训练峰值显存 | 1420.2 MB |
| 最终训练 loss | 2.60（epoch 平均：2.84 → 2.54 → 2.42） |
| LoRA adapter 权重 | 4.15 MB（目录内另附 tokenizer） |
| 合并后模型 | 494.0M 参数，953.2 MB |
| 推理显存 | 模型常驻约 1.0–1.1GB（整卡含后台 Ollama 约 2.0GB） |
| 推理实测时延 | 7.48s（55 tokens）；6.79s（66 tokens），均正常 stop 未截断 |

> 训练 loss 明细见 `output/train_log.csv` / `train_log.json`；推理样例见 `output/api_sample.json`；训练元信息见 `output/meta.json`。

## 目录结构

```
model-demo/
├── scripts/
│   ├── build_dataset.py      # 生成 / 校验指令数据集
│   ├── download_model.py     # ModelScope 拉取底座模型
│   ├── infer_test.py         # 底座 GPU 推理验证
│   ├── train_lora.py         # LoRA 微调，导出 adapter / meta / 日志
│   ├── merge_and_save.py     # 合并 adapter 到底座并导出独立模型
│   └── serve_api.py          # FastAPI OpenAI 兼容推理服务
├── data/
│   └── health_qa.jsonl       # 指令数据集（154 条）
├── output/
│   ├── meta.json             # 训练元信息与实测指标
│   ├── train_log.csv/json    # 逐步 loss 日志
│   └── api_sample.json       # 推理接口实测样例
├── requirements.txt
└── README.md
```

> `models/`（底座权重）、`output/merged/`（合并模型）、`output/lora_adapter/`（adapter 与 tokenizer）体积较大，已在 `.gitignore` 中排除，可按下方步骤复现生成。

## 快速复现

```powershell
# 1. 建环境（Python 3.12）
uv venv .venv

# 2. 安装 torch（CUDA 12.8，走上海交大镜像）
.venv\Scripts\python -m pip install torch==2.11.0+cu128 --index-url https://mirror.sjtu.edu.cn/pytorch-wheels/cu128/

# 3. 安装其余依赖（走清华镜像）
.venv\Scripts\python -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 4. 生成数据集
.venv\Scripts\python scripts\build_dataset.py

# 5. 下载底座模型（ModelScope）
.venv\Scripts\python scripts\download_model.py

# 6. LoRA 微调
.venv\Scripts\python scripts\train_lora.py

# 7. 合并导出独立模型
.venv\Scripts\python scripts\merge_and_save.py

# 8. 起本地推理 API（127.0.0.1:8123）
.venv\Scripts\python scripts\serve_api.py
```

## 接口用法

健康检查：

```powershell
Invoke-RestMethod http://127.0.0.1:8123/health
# {"status":"ok","device":"cuda","model_loaded":true}
```

对话（OpenAI messages 格式）：

```powershell
$body = @{
  messages = @(
    @{ role = "system"; content = "你是健康管理助手，回答通用、保守，不替代医生诊断；提示异常及时就医。" },
    @{ role = "user";   content = "血压偏高日常饮食要注意什么？" }
  )
  max_tokens = 100
  temperature = 0.7
} | ConvertTo-Json -Depth 5

Invoke-RestMethod -Uri http://127.0.0.1:8123/v1/chat/completions `
  -Method Post -Body $body -ContentType "application/json; charset=utf-8"
```

返回 `choices[0].message.content`、`usage`（prompt/completion tokens）与 `latency_sec`。

## 诚实边界

- **数据规模偏小**：154 条为流程验证垂类数据，未设独立评测集，未做人工评分或业务效果验收；loss 下降不代表生产级效果。
- **未做推理优化**：时延为笔记本 GPU warm 后逐条同步生成实测，未使用 vLLM / 量化 / 批处理 / 投机解码等吞吐优化。
- **未实践的环节**：未做知识蒸馏（教师-学生）、全参数微调、预训练；这些方向仅了解原理与适用场景，不列为已交付能力。
- **医疗口径**：数据与回答为通用健康管理信息，不构成诊断；回答保留「咨询专业医生 / 异常及时就医」的保守口径。
- **硬件限制**：单张 8GB 笔记本显存仅适合 0.5B–1.5B 级小模型 LoRA；7B 以上训练、分布式训练不在本机能力范围内。

## 许可与免责

本项目仅用于技术学习与流程演示。健康相关内容不能替代专业医疗建议，如有健康问题请咨询执业医师。
