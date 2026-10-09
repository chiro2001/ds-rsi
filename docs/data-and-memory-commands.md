# 当前可执行的数据与记忆检查

日期：2026-10-10。以下提供数据、模型接入和单 rank A/B 开发命令；正式多芯片训练须先通过健康与协调恢复验证。

## 本地环境

```bash
python -m venv .venv
.venv/bin/python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.10.0
.venv/bin/python -m pip install -e '.[model,test]'
.venv/bin/python -m pytest -q
```

NPU 环境使用对应 torch_npu 与原有 torch 版本，不安装 CPU wheel 替换训练环境。

## 程序生成与原生编译

```bash
.venv/bin/python -m dsrsi.data.generate \
  --output artifacts/generated/pilot_train.jsonl --count 100 \
  --split train --counterfactual-pairs

.venv/bin/python -m dsrsi.data.compiler \
  --input artifacts/generated/pilot_train.jsonl \
  --output artifacts/compiled/pilot_train.jsonl \
  --tokenizer artifacts/tokenizer/Qwen3.5-0.8B
```

输出已存在时默认拒绝覆盖；只有明确需要重建生成缓存时使用 `--overwrite`。不同 split 使用独立 world 标识和问题模板，配对变体必须一起划分。

控制生成器默认独立编码源事件，用于 A/B 的明确证据标注。连续任务编译使用真实的持续源历史，并仅输出缓存之后的新 token；运行执行器仍需落实对应状态语义。

## 真实预训练权重检查

```bash
PYTHONPATH=src .venv/bin/python scripts/verify_pretrained_bridge.py \
  --base /path/to/Qwen3.5-0.8B \
  --output artifacts/pretrained_cpu_integration.json --threads 8
```

需要原始 config、tokenizer 与 safetensors/index 文件。脚本只在 CPU 上检查加载、原生一致性、因果注入和 Writer 梯度，不报告训练后的回忆能力。

## 单 rank 开发训练与恢复

```bash
PYTHONPATH=src .venv/bin/python scripts/train_ab.py \
  --base /path/to/Qwen3.5-0.8B --run-dir runs/ab-development \
  --stage A --student-tokens 30000000 --device cpu --max-micro-steps 6

.venv/bin/python -m dsrsi.training.control --run-dir runs/ab-development --action pause

PYTHONPATH=src .venv/bin/python scripts/train_ab.py \
  --base /path/to/Qwen3.5-0.8B --run-dir runs/ab-development \
  --stage A --student-tokens 30000000 --device cpu --resume --max-micro-steps 10
```

`max-micro-steps` 是累计开发运行限额，触发 stopped 状态而非完成训练。完整恢复需保持 base、配置与数据语义；单 rank CLI 明确拒绝 WORLD_SIZE>1。已在完整权重的 CPU 容器验证，不能据此承诺 NPU 吞吐或多 rank 恢复。

## 仍待实现或验证

正式课程与开发集 gate、长源段证据映射、连续状态 LoRA 和工具训练、Consolidator 语义训练、多 rank/NPU 中断恢复与自由生成评测。基础接口和单 rank 执行已实现；不能将池统计或 CPU smoke 当作完成了能力训练。
