"""预测解析与指标计算。"""

from __future__ import annotations

import re
from typing import Any

from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.preprocessing import MultiLabelBinarizer

from .config import TASKS


METRIC_DIGITS = 4


def parse_prediction(text: str, task: str, label_mode: str) -> list[str]:
    """从模型自由文本输出中抽取标准标签。"""
    choices = TASKS[task]["labels"]
    found = [choice for choice in choices if re.search(rf"\b{re.escape(choice)}\b", text, re.I)]
    if found:
        return found if task == "clue" and label_mode == "multi" else [found[0]]
    return [text.split(",")[0].strip()]


def evaluate_predictions(rows: list[dict[str, Any]], task: str, label_mode: str) -> dict[str, Any]:
    """根据任务类型计算单标签或多标签指标。"""
    gold = [parse_prediction(row["gold"], task, label_mode) for row in rows]
    pred = [row["parsed_prediction"] for row in rows]

    if task == "clue" and label_mode == "multi":
        mlb = MultiLabelBinarizer(classes=TASKS[task]["labels"])
        y_true = mlb.fit_transform(gold)
        y_pred = mlb.transform(pred)
        return {
            "micro_f1": round(
                f1_score(y_true, y_pred, average="micro", zero_division=0),
                METRIC_DIGITS,
            ),
            "macro_f1": round(
                f1_score(y_true, y_pred, average="macro", zero_division=0),
                METRIC_DIGITS,
            ),
        }

    gold_single = [items[0] if items else "" for items in gold]
    pred_single = [items[0] if items else "" for items in pred]
    return {
        "accuracy": round(accuracy_score(gold_single, pred_single), METRIC_DIGITS),
        "macro_f1": round(
            f1_score(gold_single, pred_single, average="macro", zero_division=0),
            METRIC_DIGITS,
        ),
        "report": classification_report(
            gold_single,
            pred_single,
            zero_division=0,
            digits=METRIC_DIGITS,
        ),
    }
