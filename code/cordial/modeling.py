"""模型、Processor、LoRA 和训练 Collator。"""

from __future__ import annotations

import argparse
import os
from typing import Any

import torch
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoProcessor, BitsAndBytesConfig

try:
    from transformers import Qwen3VLForConditionalGeneration
except ImportError:  # pragma: no cover
    from transformers import AutoModelForVision2Seq as Qwen3VLForConditionalGeneration

from qwen_vl_utils import process_vision_info

from .prompts import build_messages


class QwenVlCollator:
    """训练用 Collator：只对 assistant 答案部分计算 loss。"""

    def __init__(self, processor: AutoProcessor, task: str, label_mode: str):
        self.processor = processor
        self.task = task
        self.label_mode = label_mode

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        full_messages = [
            build_messages(feature, self.task, self.label_mode, include_answer=True)
            for feature in features
        ]
        prompt_messages = [
            build_messages(feature, self.task, self.label_mode, include_answer=False)
            for feature in features
        ]

        full_text = [
            self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=False)
            for msg in full_messages
        ]
        prompt_text = [
            self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            for msg in prompt_messages
        ]

        image_inputs, video_inputs = process_vision_info(full_messages)
        model_inputs = self.processor(
            text=full_text,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        prompt_inputs = self.processor(
            text=prompt_text,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        labels = model_inputs["input_ids"].clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        for index in range(labels.size(0)):
            prompt_len = int(prompt_inputs["attention_mask"][index].sum().item())
            labels[index, :prompt_len] = -100
        model_inputs["labels"] = labels
        return model_inputs


def load_model_and_processor(args: argparse.Namespace, for_training: bool):
    """加载 Qwen3-VL-8B；训练时挂 LoRA，评估时可加载 LoRA adapter。"""
    processor = AutoProcessor.from_pretrained(args.model_name, trust_remote_code=True)
    if getattr(processor, "tokenizer", None) is not None:
        processor.tokenizer.padding_side = "right"

    quantization_config = None
    if args.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    device_map = resolve_device_map(args.device_map)

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float16,
        device_map=device_map,
        quantization_config=quantization_config,
        trust_remote_code=True,
        attn_implementation=args.attn_implementation,
    )

    if for_training:
        model.config.use_cache = False
        if args.load_in_4bit:
            model = prepare_model_for_kbit_training(model)
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=args.lora_targets.split(","),
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    elif args.adapter_path:
        model = PeftModel.from_pretrained(model, args.adapter_path)

    return model, processor


def resolve_device_map(device_map_arg: str):
    """根据启动方式决定 device_map。

    单进程时保留 `auto`；torchrun 多进程时，每个 rank 只把完整模型放到
    自己绑定的 GPU，避免 `device_map=auto` 和 DDP/QLoRA 互相打架。
    """
    local_rank = os.environ.get("LOCAL_RANK")
    if local_rank is not None and torch.cuda.is_available():
        rank = int(local_rank)
        torch.cuda.set_device(rank)
        return {"": rank}
    if device_map_arg.lower() in {"none", "null"}:
        return None
    return device_map_arg
