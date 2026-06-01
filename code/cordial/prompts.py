"""Qwen3-VL 对话模板和答案格式。"""

from __future__ import annotations

from typing import Any

from .config import TASKS


def clue_single_label(labels: list[str]) -> str:
    """CLUE 单标签版本：优先选择更抽象、更难的关系。"""
    for label in ["Story", "Subjective", "Meta", "Action", "Visible"]:
        if label in labels:
            return label
    return labels[0]


def task_prompt(task: str, text: str, label_mode: str) -> str:
    """构造用户侧提示词。"""
    spec = TASKS[task]
    instruction = spec["prompt"]
    if task == "clue" and label_mode == "single":
        instruction = (
            "Classify the most specific visual-context relation needed to "
            "interpret the text. Choose exactly one label from: Visible, "
            "Action, Meta, Subjective, Story."
        )
    return (
        f"{instruction}\n\n"
        f"Text:\n{text}\n\n"
        "Return only the label name. For multi-label CLUE, return labels "
        "separated by commas."
    )


def answer_from_labels(task: str, labels: list[str], label_mode: str) -> str:
    """把标准标签列表转换成模型要学习输出的字符串。"""
    if task == "clue" and label_mode == "single":
        return clue_single_label(labels)
    return ", ".join(labels)


def build_messages(
    example: dict[str, Any],
    task: str,
    label_mode: str,
    include_answer: bool,
) -> list[dict[str, Any]]:
    """构造 Qwen3-VL chat message；训练时 include_answer=True。"""
    user_text = task_prompt(task, example["text"], label_mode)
    messages = [
        {
            "role": "system",
            "content": "You are a careful multimodal discourse relation classifier.",
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": example["image_path"]},
                {"type": "text", "text": user_text},
            ],
        },
    ]
    if include_answer:
        messages.append(
            {
                "role": "assistant",
                "content": answer_from_labels(task, example["labels"], label_mode),
            }
        )
    return messages

