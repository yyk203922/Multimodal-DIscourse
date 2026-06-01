"""推理、评估和数据检查。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from qwen_vl_utils import process_vision_info

from .data import load_task_dataset
from .metrics import evaluate_predictions, parse_prediction
from .modeling import load_model_and_processor
from .prompts import answer_from_labels, build_messages


@torch.inference_mode()
def generate_predictions(args: argparse.Namespace, task: str, split: str) -> list[dict[str, Any]]:
    """对指定 split 逐样本生成预测。"""
    ds = load_task_dataset(args.dataset_root, task)
    if split not in ds:
        raise ValueError(f"Split `{split}` not found. Available splits: {list(ds.keys())}")

    label_mode = args.label_mode if task == "clue" else "single"
    model, processor = load_model_and_processor(args, for_training=False)
    model.eval()

    rows = []
    for example in ds[split]:
        messages = build_messages(example, task, label_mode, include_answer=False)
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(model.device)

        generated = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
        generated = generated[:, inputs["input_ids"].shape[1] :]
        prediction_text = processor.batch_decode(
            generated, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()
        gold = answer_from_labels(task, example["labels"], label_mode)
        rows.append(
            {
                "text": example["text"],
                "image_path": example["image_path"],
                "gold": gold,
                "prediction": prediction_text,
                "parsed_prediction": parse_prediction(prediction_text, task, label_mode),
            }
        )
    return rows


def evaluate_and_save(args: argparse.Namespace, task: str) -> None:
    """推理、计算指标，并保存 JSONL 预测文件。"""
    rows = generate_predictions(args, task, args.split)
    label_mode = args.label_mode if task == "clue" else "single"
    metrics = evaluate_predictions(rows, task, label_mode)

    output_path = Path(args.prediction_file)
    if args.dataset == "all":
        output_path = output_path.with_name(f"{output_path.stem}_{task}{output_path.suffix}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(
        json.dumps(
            {"task": task, "metrics": metrics, "prediction_file": str(output_path)},
            ensure_ascii=False,
            indent=2,
        )
    )


def inspect_dataset(args: argparse.Namespace, task: str) -> None:
    """打印每个 split 的前两个样本，方便确认字段解析是否正确。"""
    ds = load_task_dataset(args.dataset_root, task)
    print(ds)
    for split in ds:
        print(f"\n[{task}:{split}]")
        for row in ds[split].select(range(min(2, len(ds[split])))):
            sample = {key: row[key] for key in ["text", "image_path", "labels"]}
            print(json.dumps(sample, ensure_ascii=False, indent=2))

