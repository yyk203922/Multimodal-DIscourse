"""任务配置：集中放三个数据集的标签、别名和默认提示词。"""

TASKS = {
    "disrel": {
        "aliases": ["disrel", "dis_rel", "discourse_relation"],
        "labels": ["Similar", "Complementary"],
        "multi_label": False,
        "prompt": (
            "Classify the multimodal discourse relation between the text and "
            "the image. Choose exactly one label from: Similar, Complementary."
        ),
    },
    "tweet_subtitles": {
        "aliases": ["tweet_subtitles", "tweet-subtitles", "tweet", "subtitles"],
        "labels": [
            "Insertion",
            "Concretization",
            "Projection",
            "Restatement",
            "Extension",
        ],
        "multi_label": False,
        "prompt": (
            "Classify the multimodal relation between the tweet text and its "
            "subtitle image. Choose exactly one label from: Insertion, "
            "Concretization, Projection, Restatement, Extension."
        ),
    },
    "clue": {
        "aliases": ["clue"],
        "labels": ["Visible", "Action", "Meta", "Subjective", "Story"],
        "multi_label": True,
        "prompt": (
            "Identify all visual-context relations that are needed to interpret "
            "the text. Choose one or more labels from: Visible, Action, Meta, "
            "Subjective, Story."
        ),
    },
}

SPLIT_ALIASES = {
    "train": ["train", "training"],
    "validation": ["validation", "valid", "val", "dev"],
    "test": ["test", "testing"],
}

TEXT_KEYS = [
    "text",
    "tweet",
    "tweet_text",
    "caption",
    "subtitle",
    "sentence",
    "utterance",
    "post",
    "content",
    "arg1",
    "arg2",
]

IMAGE_KEYS = [
    "image",
    "img",
    "image_path",
    "img_path",
    "filename",
    "file_name",
    "filepath",
    "path",
    "media",
]

LABEL_KEYS = [
    "label",
    "labels",
    "relation",
    "relations",
    "gold",
    "answer",
    "category",
    "class",
]


def canonical_task_name(name: str) -> str:
    """把用户输入的数据集名统一成内部使用的任务名。"""
    lowered = name.lower()
    for task, spec in TASKS.items():
        if lowered == task or lowered in spec["aliases"]:
            return task
    raise ValueError(f"Unknown task `{name}`. Choose from {', '.join(TASKS)} or all.")


def selected_tasks(name: str) -> list[str]:
    """解析 --dataset 参数，支持 all 或单个数据集。"""
    if name.lower() == "all":
        return ["disrel", "tweet_subtitles", "clue"]
    return [canonical_task_name(name)]

