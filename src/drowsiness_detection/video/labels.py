from __future__ import annotations


RAW_LABEL_TO_NAME = {
    0: "alert",
    5: "low_vigilant",
    10: "drowsy",
}

LABEL_TO_ID = {
    "alert": 0,
    "low_vigilant": 1,
    "drowsy": 2,
}

CANONICAL_ID_TO_LABEL = {value: key for key, value in LABEL_TO_ID.items()}


def raw_label_to_label_name(raw_label: int) -> str:
    if raw_label not in RAW_LABEL_TO_NAME:
        raise ValueError(f"Unsupported UTA-RLDD raw label: {raw_label}")
    return RAW_LABEL_TO_NAME[raw_label]


def raw_label_to_canonical_id(raw_label: int) -> int:
    return LABEL_TO_ID[raw_label_to_label_name(raw_label)]
