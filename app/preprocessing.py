import re
from typing import Iterable


TEXT_COLUMNS = [
    "category",
    "frontend",
    "backend",
    "database",
    "deployment",
    "tags",
    "description",
]


def normalize_text(value: str) -> str:
    """Normalize text while preserving the technology vocabulary."""

    value = str(value).lower().strip()
    value = value.replace("-", " ")
    value = value.replace("/", " ")
    value = re.sub(r"[^a-z0-9+#. ]+", " ", value)

    return re.sub(r"\s+", " ", value)


def build_item_document(row) -> str:
    """Build one content document from all item attributes."""

    parts = [normalize_text(row[column]) for column in TEXT_COLUMNS]

    return " ".join(p for p in parts if p)


def build_user_document(inputs: Iterable[str]) -> str:
    """Combine user choices/interests into one profile document."""

    return " ".join(
        normalize_text(item)
        for item in inputs
        if str(item).strip()
    )
