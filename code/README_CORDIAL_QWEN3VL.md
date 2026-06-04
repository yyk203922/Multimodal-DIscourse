# CORDIAL + Qwen3-VL-8B

这里是一个用于 CORDIAL 多模态篇章关系任务的完整项目骨架，覆盖数据检查、训练、推理和测试评估。三个子任务分别是：

- `disrel`: `Similar`, `Complementary`
- `tweet_subtitles`: `Insertion`, `Concretization`, `Projection`, `Restatement`, `Extension`
- `clue`: `Visible`, `Action`, `Meta`, `Subjective`, `Story`

默认认为你已经把 Hugging Face 数据集 `aashananth/CORDIAL` 下载到本地，路径为 `./dataset/CORDIAL`。如果服务器上的路径不同，所有命令都用 `--dataset-root` 指向真实目录即可。

## 代码结构

```text
code/
  cordial_qwen3vl.py      # 推荐入口，负责调用 cordial.cli
  test.py                 # 兼容入口，方便你在 IDE 当前文件里直接运行
  requirements.txt        # 服务器环境依赖
  cordial/
    config.py             # 三个数据集的标签、别名、字段候选名
    data.py               # 本地 JSON/JSONL 数据读取和字段归一化
    prompts.py            # Qwen3-VL 对话 prompt 和标准答案格式
    modeling.py           # Qwen3-VL 加载、4bit/8bit/不量化、LoRA、训练 collator
    train.py              # 单任务训练逻辑
    infer.py              # 推理、保存预测、数据 inspect
    metrics.py            # 输出解析和 Accuracy/F1 指标
    cli.py                # 命令行参数和总调度
```

## 安装依赖

```bash
pip install -r code/requirements.txt
```

服务器上建议使用支持 Qwen3-VL 的新版 `transformers`。脚本默认模型名是 `Qwen/Qwen3-VL-8B-Instruct`。

如果服务器没有安装 `flash-attn`，把命令里的 attention 实现改成 PyTorch SDPA：

```bash
--attn-implementation sdpa
```

## 先检查数据是否能读

```bash
python code/cordial_qwen3vl.py \
  --mode inspect \
  --dataset-root ./dataset/CORDIAL \
  --dataset all
```

如果你的数据集落盘路径不同，把 `--dataset-root` 改成真实路径即可。脚本会在每个任务目录下寻找 `train/dev/validation/test` 这类 JSON 或 JSONL 文件，并把文本、图片路径、标签统一成 `text / image_path / labels`。

CORDIAL 仓库常见目录结构如下，脚本已经兼容 `Clue` 这种首字母大写目录名：

```text
dataset/CORDIAL/
  Clue/
    images/
    train_ml.json
    train_sl.json
    test_ml.json
    test_sl.json
  disrel/
    images/
    train.json
    test.json
  tweet_subtitles/
    images/
    train.json
    test.json
```

检查 CLUE single-label 文件时可以这样跑：

```bash
python code/cordial_qwen3vl.py \
  --mode inspect \
  --dataset clue \
  --label-mode single \
  --dataset-root ./dataset/CORDIAL
```

## 分别训练三个数据集

### 量化选择

训练和测试都支持三种加载方式：

- `--load-in-4bit`：QLoRA，最省显存，单张 3090 24GB 推荐优先用这个。
- `--load-in-8bit`：8bit LoRA，显存高于 4bit，可能略稳，但不保证指标一定更高。
- 不传 `--load-in-4bit` 或 `--load-in-8bit`：不量化 LoRA，显存最高，单张 3090 很可能 OOM；双卡 3090 可以尝试。

`--load-in-4bit` 和 `--load-in-8bit` 是互斥参数，不能同时传。评估 checkpoint 时要使用和训练时相同的量化方式，例如 4bit 训练出来的 LoRA，测试时也加 `--load-in-4bit`。

单卡训练示例：

```bash
python code/cordial_qwen3vl.py \
  --mode train \
  --dataset all \
  --dataset-root ./dataset/CORDIAL \
  --model-name /data/ykyang/models/Qwen3-VL-8B-Instruct \
  --output-dir ./checkpoints/cordial_qwen3vl \
  --load-in-4bit \
  --bf16 \
  --epochs 3 \
  --train-batch-size 1 \
  --gradient-accumulation-steps 16
```

如果想用 SwanLab 监控训练，先确认环境里已经安装 `swanlab`，并在服务器上登录：

```bash
swanlab login
```

然后训练时加 `--report-to swanlab`。`--swanlab-project` 和 `--swanlab-workspace`
会分别写入 `SWANLAB_PROJECT` 和 `SWANLAB_WORKSPACE`，`--run-name` 会传给
Hugging Face Trainer。跑 `--dataset all` 时，每个子任务会自动追加任务名，避免覆盖：

```bash
python code/cordial_qwen3vl.py \
  --mode train \
  --dataset all \
  --dataset-root ./dataset/CORDIAL \
  --model-name /data/ykyang/models/Qwen3-VL-8B-Instruct \
  --output-dir ./checkpoints/cordial_qwen3vl \
  --load-in-4bit \
  --bf16 \
  --epochs 3 \
  --train-batch-size 1 \
  --gradient-accumulation-steps 16 \
  --report-to swanlab \
  --swanlab-project cordial-qwen3vl \
  --run-name qwen3vl-lora
```

多卡训练建议用 `torchrun`。脚本会自动读取 `LOCAL_RANK`，让每个进程只绑定自己的 GPU：

```bash
torchrun --standalone --nproc_per_node 2 code/cordial_qwen3vl.py \
  --mode train \
  --dataset all \
  --dataset-root ./dataset/CORDIAL \
  --model-name Qwen/Qwen3-VL-8B-Instruct \
  --output-dir ./checkpoints/cordial_qwen3vl \
  --load-in-4bit \
  --bf16 \
  --epochs 3 \
  --train-batch-size 1 \
  --gradient-accumulation-steps 16
```

双卡 3090 可尝试 8bit：

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node 2 code/cordial_qwen3vl.py \
  --mode train \
  --dataset all \
  --dataset-root ./dataset/CORDIAL \
  --model-name /data/ykyang/models/Qwen3-VL-8B-Instruct \
  --output-dir ./checkpoints/cordial_qwen3vl_8bit \
  --load-in-8bit \
  --bf16 \
  --epochs 3 \
  --train-batch-size 1 \
  --gradient-accumulation-steps 16
```

如果想尝试不量化训练，去掉 `--load-in-4bit / --load-in-8bit`。不量化显存压力最大，建议先用单任务和较小 batch 做冒烟测试：

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node 2 code/cordial_qwen3vl.py \
  --mode train \
  --dataset disrel \
  --dataset-root ./dataset/CORDIAL \
  --model-name /data/ykyang/models/Qwen3-VL-8B-Instruct \
  --output-dir ./checkpoints/cordial_qwen3vl_full \
  --bf16 \
  --epochs 1 \
  --train-batch-size 1 \
  --gradient-accumulation-steps 16
```

如果 `torchrun` 只显示 `ChildFailedError`，用下面的方式把每个 rank 的真实 traceback 打出来：

```bash
torchrun --standalone --nproc_per_node 2 --tee 3 code/cordial_qwen3vl.py \
  --mode train \
  --dataset disrel \
  --dataset-root ./dataset/CORDIAL \
  --load-in-4bit \
  --bf16
```

这会分别训练：

- `./checkpoints/cordial_qwen3vl/disrel_single/final`
- `./checkpoints/cordial_qwen3vl/tweet_subtitles_single/final`
- `./checkpoints/cordial_qwen3vl/clue_multi/final`

每个目录里保存的是 LoRA adapter，不是合并后的完整 Qwen3-VL 权重。测试时需要用 `--adapter-path` 指向对应任务的 `final` 目录。

CLUE 默认是多标签。如果你想跑论文里的 single-label 版本：

```bash
python code/cordial_qwen3vl.py \
  --mode train \
  --dataset clue \
  --label-mode single \
  --dataset-root ./dataset/CORDIAL
```

## 推理和测试

训练后用对应 adapter 跑测试集。下面是 DisRel 的例子：

```bash
python code/cordial_qwen3vl.py \
  --mode eval \
  --dataset disrel \
  --split test \
  --dataset-root ./dataset/CORDIAL \
  --adapter-path ./checkpoints/cordial_qwen3vl/disrel_single/final \
  --load-in-4bit \
  --prediction-file ./checkpoints/cordial_qwen3vl/disrel_test.jsonl
```

`tweet_subtitles` 和 `clue` 同理，只需要换 `--dataset` 和 `--adapter-path`。如果 checkpoint 是 8bit 训练的，测试时改成 `--load-in-8bit`；如果是不量化训练的，测试时不要传量化参数。单标签任务会输出 Accuracy、Macro-F1 和分类报告；CLUE 多标签会输出 Micro-F1 和 Macro-F1。

Tweet Subtitles 测试示例：

```bash
python code/cordial_qwen3vl.py \
  --mode eval \
  --dataset tweet_subtitles \
  --split test \
  --dataset-root ./dataset/CORDIAL \
  --adapter-path ./checkpoints/cordial_qwen3vl/tweet_subtitles_single/final \
  --load-in-4bit \
  --prediction-file ./checkpoints/cordial_qwen3vl/tweet_subtitles_test.jsonl
```

CLUE 多标签测试示例：

```bash
python code/cordial_qwen3vl.py \
  --mode eval \
  --dataset clue \
  --label-mode multi \
  --split test \
  --dataset-root ./dataset/CORDIAL \
  --adapter-path ./checkpoints/cordial_qwen3vl/clue_multi/final \
  --load-in-4bit \
  --prediction-file ./checkpoints/cordial_qwen3vl/clue_test.jsonl
```

`--mode eval` 和 `--mode predict` 在当前代码里都会执行推理、保存 JSONL，并打印指标；保留两个名字是为了之后如果要区分“只预测”和“带指标评估”，可以直接扩展。

## 重要说明

- `data.py` 已经做了字段名兼容，但如果你本地 CORDIAL 文件字段非常特殊，优先改 `TEXT_KEYS / IMAGE_KEYS / LABEL_KEYS`。
- `modeling.py` 里默认使用 LoRA 训练，不做全量微调，方便 8B VL 模型上服务器跑。
- `prompts.py` 里集中管理任务提示词，后续想做中文 prompt 或加 chain-of-thought 禁止项，可以只改这里。
