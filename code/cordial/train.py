"""训练入口：每个 CORDIAL 子任务单独训练一个 LoRA adapter。"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from transformers import Trainer, TrainingArguments

from .data import load_task_dataset
from .modeling import QwenVlCollator, load_model_and_processor


def tracker_enabled(report_to: str, tracker: str) -> bool:
    """Return whether a Trainer tracker name is enabled in --report-to."""
    normalized = (report_to or "").lower()
    if normalized in {"none", "no", "false", "0"}:
        return False
    if normalized == "all":
        return True
    return tracker in {item.strip() for item in normalized.split(",")}


def normalize_report_to(report_to: str):
    """Normalize comma-separated tracker names for TrainingArguments."""
    normalized = (report_to or "").strip()
    if "," not in normalized:
        return normalized
    return [item.strip() for item in normalized.split(",") if item.strip()]


def configure_swanlab(args: argparse.Namespace, task: str, label_mode: str) -> str | None:
    """Configure SwanLab through env vars consumed by the Transformers integration."""
    run_name = args.run_name
    if run_name and args.dataset == "all":
        run_name = f"{run_name}_{task}_{label_mode}"
    elif not run_name:
        run_name = f"{task}_{label_mode}"

    if not tracker_enabled(args.report_to, "swanlab"):
        return run_name

    if args.swanlab_project:
        os.environ["SWANLAB_PROJECT"] = args.swanlab_project
    if args.swanlab_workspace:
        os.environ["SWANLAB_WORKSPACE"] = args.swanlab_workspace
    if args.swanlab_mode:
        os.environ["SWANLAB_MODE"] = args.swanlab_mode
    return run_name


def train_one_task(args: argparse.Namespace, task: str) -> None:
    """训练单个数据集。"""
    label_mode = args.label_mode if task == "clue" else "single"
    ds = load_task_dataset(args.dataset_root, task, label_mode=label_mode)
    output_dir = Path(args.output_dir) / f"{task}_{label_mode}"

    train_ds = ds["train"]
    eval_ds = ds.get("validation") or ds.get("test")
    model, processor = load_model_and_processor(args, for_training=True)
    collator = QwenVlCollator(processor, task, label_mode)
    run_name = configure_swanlab(args, task, label_mode)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        run_name=run_name,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        eval_strategy="steps" if eval_ds is not None else "no",
        save_strategy="steps",
        bf16=args.bf16,
        fp16=not args.bf16,
        remove_unused_columns=False,
        gradient_checkpointing=True,
        report_to=normalize_report_to(args.report_to),
        dataloader_num_workers=args.num_workers,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        processing_class=processor,
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(str(output_dir / "final"))
    processor.save_pretrained(str(output_dir / "final"))
