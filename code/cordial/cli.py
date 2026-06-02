"""命令行入口。"""

from __future__ import annotations

import argparse
import os

from .config import selected_tasks
from .infer import evaluate_and_save, inspect_dataset
from .train import train_one_task


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["inspect", "train", "eval", "predict"], default="inspect")
    parser.add_argument("--dataset-root", default="./dataset/CORDIAL")
    parser.add_argument("--dataset", default="all", help="all, disrel, tweet_subtitles, or clue")
    parser.add_argument("--label-mode", choices=["single", "multi"], default="multi")
    parser.add_argument("--split", default="test")

    # 模型与输出路径。
    parser.add_argument("--model-name", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--adapter-path", default=None, help="LoRA adapter path for eval/predict.")
    parser.add_argument("--output-dir", default="./checkpoints/cordial_qwen3vl")
    parser.add_argument("--prediction-file", default="./checkpoints/cordial_qwen3vl/predictions.jsonl")

    # 服务器训练常用参数。
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--train-batch-size", type=int, default=1)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=200)
    parser.add_argument("--eval-steps", type=int, default=200)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--resume-from-checkpoint", default=None)
    parser.add_argument("--report-to", default="none")

    # LoRA 参数。
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora-targets",
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    for task in selected_tasks(args.dataset):
        if args.mode == "inspect":
            inspect_dataset(args, task)
        elif args.mode == "train":
            train_one_task(args, task)
        else:
            evaluate_and_save(args, task)
