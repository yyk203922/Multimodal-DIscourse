"""CORDIAL 本地数据读取与字段归一化。

这个模块尽量兼容 Hugging Face 仓库下载后的不同落盘方式：
只要每个数据集下面能找到 JSON/JSONL，并且行里有文本、图片、标签字段，
就会统一转换成训练脚本需要的 text / image_path / labels。
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable, Optional

from datasets import DatasetDict, load_dataset

from .config import IMAGE_KEYS, LABEL_KEYS, SPLIT_ALIASES, TASKS, TEXT_KEYS


def discover_task_dir(dataset_root: Path, task: str) -> Path:
    """寻找某个子任务所在目录。"""
    candidates = [dataset_root / alias for alias in TASKS[task]["aliases"]]
    candidates += [dataset_root / task, dataset_root]
    for candidate in candidates:
        if candidate.exists() and any(candidate.rglob("*.json*")):
            return candidate
    return dataset_root


def discover_split_files(task_dir: Path) -> dict[str, str]:
    """把 train/dev/test 文件名映射成 datasets 能识别的 split。"""
    files = [p for p in task_dir.rglob("*") if p.suffix.lower() in {".json", ".jsonl"}]
    split_files: dict[str, str] = {}
    for split, aliases in SPLIT_ALIASES.items():
        for path in files:
            stem = path.stem.lower()
            parts = {part for part in re.split(r"[^a-z0-9]+", stem) if part}
            if any(alias in parts or stem == alias for alias in aliases):
                split_files.setdefault(split, str(path))
    if not split_files and files:
        split_files["train"] = str(files[0])
    return split_files


def load_task_dataset(dataset_root: str, task: str) -> DatasetDict:
    """读取单个任务，并统一字段格式。"""
    root = Path(dataset_root).expanduser().resolve()
    task_dir = discover_task_dir(root, task)
    split_files = discover_split_files(task_dir)
    if not split_files:
        raise FileNotFoundError(
            f"No JSON/JSONL files found for task `{task}` under {task_dir}. "
            "Point --dataset-root at the downloaded aashananth/CORDIAL folder."
        )

    loaded = load_dataset("json", data_files=split_files)
    normalized = DatasetDict()
    for split_name, split in loaded.items():
        normalized[split_name] = split.map(
            lambda row: normalize_row(row, task, task_dir),
            remove_columns=split.column_names,
        )
    return normalized


def first_present(row: dict[str, Any], keys: Iterable[str]) -> Any:
    """从一组候选字段里取第一个非空值。"""
    for key in keys:
        if key in row and row[key] not in (None, ""):
            return row[key]
    return None


def normalize_row(row: dict[str, Any], task: str, task_dir: Path) -> dict[str, Any]:
    """把原始样本转换成模型训练/推理共用格式。"""
    image_value = first_present(row, IMAGE_KEYS)
    labels = normalize_labels(row, task)
    return {
        "text": normalize_text(row),
        "image_path": resolve_image_path(image_value, task_dir),
        "labels": labels,
        "raw_json": json.dumps(row, ensure_ascii=False),
    }


def normalize_text(row: dict[str, Any]) -> str:
    """拼接文本字段；如果字段名未知，就保守取前两个非图片标量字段。"""
    values = []
    for key in TEXT_KEYS:
        if key in row and row[key] not in (None, ""):
            values.append(str(row[key]))
    if values:
        return "\n".join(dict.fromkeys(values))

    scalar_values = [
        str(v)
        for v in row.values()
        if isinstance(v, (str, int, float)) and not looks_like_image_path(str(v))
    ]
    return "\n".join(scalar_values[:2])


def looks_like_image_path(value: str) -> bool:
    """判断字符串是否像图片路径。"""
    return Path(value).suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def resolve_image_path(value: Any, task_dir: Path) -> str:
    """把数据里的相对图片路径解析成绝对路径。"""
    if isinstance(value, dict):
        value = first_present(value, ["path", "filename", "file_name", "image"])
    if isinstance(value, list) and value:
        value = value[0]
    if value is None:
        raise ValueError("Could not find an image field in a dataset row.")

    path = Path(str(value))
    if path.exists():
        return str(path.resolve())

    search_roots = [task_dir, task_dir.parent, task_dir / "images", task_dir / "image"]
    for root in search_roots:
        candidate = root / path
        if candidate.exists():
            return str(candidate.resolve())
        matches = list(root.rglob(path.name)) if root.exists() else []
        if matches:
            return str(matches[0].resolve())

    # 最后返回推测路径，便于用户从报错里看到脚本实际找了哪里。
    return str((task_dir / path).resolve())


def normalize_labels(row: dict[str, Any], task: str) -> list[str]:
    """把字符串、列表、布尔列等标签格式统一成标签列表。"""
    raw = first_present(row, LABEL_KEYS)
    task_labels = TASKS[task]["labels"]
    if raw is None:
        joined = " ".join(str(v) for v in row.values())
        found = [label for label in task_labels if re.search(label, joined, re.I)]
        return found[:1] if not TASKS[task]["multi_label"] else found

    if isinstance(raw, str):
        parts = re.split(r"[,;/|]+", raw)
        labels = [match_label(part.strip(), task_labels) for part in parts]
    elif isinstance(raw, list):
        labels = [match_label(str(item), task_labels) for item in raw]
    else:
        labels = [match_label(str(raw), task_labels)]

    labels = [label for label in labels if label]
    if task == "clue" and not labels:
        labels = [
            label
            for label in task_labels
            if bool(row.get(label)) or bool(row.get(label.lower()))
        ]
    if not labels:
        raise ValueError(f"Could not normalize label from row keys: {list(row.keys())}")
    return labels if TASKS[task]["multi_label"] else [labels[0]]


def match_label(value: str, choices: list[str]) -> Optional[str]:
    """把原始标签文本匹配到标准标签名。"""
    lowered = value.strip().lower()
    for choice in choices:
        if lowered == choice.lower() or choice.lower() in lowered:
            return choice
    return None

