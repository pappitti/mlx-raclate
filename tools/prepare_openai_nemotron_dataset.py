import argparse
import ast
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from datasets import ClassLabel, Features, Sequence, Value, load_dataset

from mlx_raclate.utils.tokenizer_utils import load_tokenizer
from mlx_raclate.utils.utils import get_model_path


BASE_LABELS = (
    "O",
    "private_person",
    "private_organization",
    "private_occupation",
    "private_employee_id",
    "private_employment_status",
    "private_email",
    "private_phone",
    "private_url",
    "private_address",
    "private_location",
    "private_date",
    "private_user_id",
    "private_financial_id",
    "private_government_id",
    "private_health_id",
    "private_device_id",
    "private_network_id",
    "private_vehicle_id",
    "secret",
    "sensitive_attribute",
)

RAW_LABEL_TO_BASE_LABEL = {
    "account_number": "private_financial_id",
    "age": "sensitive_attribute",
    "api_key": "secret",
    "bank_routing_number": "private_financial_id",
    "biometric_identifier": "private_government_id",
    "blood_type": "private_health_id",
    "certificate_license_number": "private_government_id",
    "city": "private_address",
    "company_name": "private_organization",
    "coordinate": "private_location",
    "country": "private_address",
    "county": "private_address",
    "credit_debit_card": "private_financial_id",
    "customer_id": "private_user_id",
    "cvv": "private_financial_id",
    "date": "private_date",
    "date_of_birth": "private_date",
    "date_time": "private_date",
    "device_identifier": "private_device_id",
    "education_level": "sensitive_attribute",
    "email": "private_email",
    "employee_id": "private_employee_id",
    "employment_status": "private_employment_status",
    "fax_number": "private_phone",
    "first_name": "private_person",
    "gender": "sensitive_attribute",
    "health_plan_beneficiary_number": "private_health_id",
    "http_cookie": "secret",
    "ipv4": "private_network_id",
    "ipv6": "private_network_id",
    "language": "sensitive_attribute",
    "last_name": "private_person",
    "license_plate": "private_vehicle_id",
    "mac_address": "private_network_id",
    "medical_record_number": "private_health_id",
    "national_id": "private_government_id",
    "occupation": "private_occupation",
    "password": "secret",
    "phone_number": "private_phone",
    "pin": "secret",
    "political_view": "sensitive_attribute",
    "postcode": "private_address",
    "race_ethnicity": "sensitive_attribute",
    "religious_belief": "sensitive_attribute",
    "sexuality": "sensitive_attribute",
    "ssn": "private_government_id",
    "state": "private_address",
    "street_address": "private_address",
    "swift_bic": "private_financial_id",
    "tax_id": "private_government_id",
    "time": "private_date",
    "unique_id": "private_user_id",
    "url": "private_url",
    "user_name": "private_user_id",
    "vehicle_identifier": "private_vehicle_id",
}


def _bioes_labels(base_labels: Iterable[str]) -> List[str]:
    labels = ["O"]
    for label in base_labels:
        if label == "O":
            continue
        labels.extend(f"{prefix}-{label}" for prefix in ("B", "I", "E", "S"))
    return labels


LABELS = _bioes_labels(BASE_LABELS)
LABEL2ID = {label: index for index, label in enumerate(LABELS)}


def _parse_spans(spans: Any) -> List[Dict[str, Any]]:
    if isinstance(spans, str):
        return ast.literal_eval(spans)
    return list(spans or [])


def _span_label(raw_label: str) -> Optional[str]:
    return RAW_LABEL_TO_BASE_LABEL.get(raw_label)


def _apply_bioes(labels: List[str], token_indices: List[int], base_label: str) -> None:
    if not token_indices:
        return

    if len(token_indices) == 1:
        labels[token_indices[0]] = f"S-{base_label}"
        return

    labels[token_indices[0]] = f"B-{base_label}"
    for index in token_indices[1:-1]:
        labels[index] = f"I-{base_label}"
    labels[token_indices[-1]] = f"E-{base_label}"


def _build_converter(tokenizer, max_length: Optional[int]):
    def convert(example):
        text = example["text"]
        encoded = tokenizer(
            text,
            add_special_tokens=False,
            truncation=max_length is not None,
            max_length=max_length,
            return_offsets_mapping=True,
        )
        offsets = encoded["offset_mapping"]
        token_labels = ["O"] * len(encoded["input_ids"])
        conflicts = 0
        skipped = Counter()

        for span in sorted(_parse_spans(example["spans"]), key=lambda item: item["start"]):
            base_label = _span_label(span["label"])
            if base_label is None:
                skipped[span["label"]] += 1
                continue

            token_indices = [
                index
                for index, (start, end) in enumerate(offsets)
                if start < span["end"] and end > span["start"]
            ]
            if not token_indices:
                continue

            if any(token_labels[index] != "O" for index in token_indices):
                conflicts += 1

            _apply_bioes(token_labels, token_indices, base_label)

        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "labels": [LABEL2ID[label] for label in token_labels],
            "num_conflicts": conflicts,
            "num_skipped_spans": sum(skipped.values()),
        }

    return convert


def _prepare_split(
    tokenizer,
    split: str,
    limit: Optional[int],
    max_length: Optional[int],
):
    dataset = load_dataset("nvidia/Nemotron-PII", split=split)
    if limit is not None:
        dataset = dataset.select(range(min(limit, len(dataset))))

    features = Features(
        {
            "input_ids": Sequence(Value("int32")),
            "attention_mask": Sequence(Value("int8")),
            "labels": Sequence(ClassLabel(names=LABELS)),
            "num_conflicts": Value("int32"),
            "num_skipped_spans": Value("int32"),
        }
    )
    return dataset.map(
        _build_converter(tokenizer, max_length),
        remove_columns=dataset.column_names,
        features=features,
        keep_in_memory=False,
        desc=f"Tokenizing {split} with OpenAI tokenizer and aligning spans",
    )


def _write_metadata(output_dir: Path, args, train_dataset, validation_dataset) -> None:
    ontology = {
        "name": "openai_nemotron_company_person_v1",
        "model": args.model,
        "source_dataset": "nvidia/Nemotron-PII",
        "max_length": args.max_length,
        "labels": LABELS,
        "base_labels": list(BASE_LABELS),
        "raw_label_to_base_label": RAW_LABEL_TO_BASE_LABEL,
        "splits": {
            "train": len(train_dataset),
            "validation": len(validation_dataset),
        },
        "alignment_summary": {
            "train_conflicts": int(sum(train_dataset["num_conflicts"])),
            "validation_conflicts": int(sum(validation_dataset["num_conflicts"])),
            "train_skipped_spans": int(sum(train_dataset["num_skipped_spans"])),
            "validation_skipped_spans": int(sum(validation_dataset["num_skipped_spans"])),
        },
    }

    with open(output_dir / "labels.json", "w") as f:
        json.dump(LABELS, f, indent=2)
    with open(output_dir / "ontology.json", "w") as f:
        json.dump(ontology, f, indent=2)


def _optional_positive_int(value: str) -> Optional[int]:
    parsed = int(value)
    if parsed < 0:
        return None
    return parsed


def main():
    parser = argparse.ArgumentParser(
        description="Prepare an OpenAI-tokenized Nemotron PII dataset for token classification."
    )
    parser.add_argument("--model", default="openai/privacy-filter")
    parser.add_argument("--output-dir", default="data/nemotron_openai_tokenized_v1")
    parser.add_argument("--train-limit", type=_optional_positive_int, default=100_000)
    parser.add_argument("--validation-limit", type=_optional_positive_int, default=5_000)
    parser.add_argument("--max-length", type=_optional_positive_int, default=256)
    args = parser.parse_args()

    model_path = get_model_path(args.model)
    tokenizer = load_tokenizer(model_path)._tokenizer

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = _prepare_split(
        tokenizer,
        split="train",
        limit=args.train_limit,
        max_length=args.max_length,
    )
    validation_dataset = _prepare_split(
        tokenizer,
        split="test",
        limit=args.validation_limit,
        max_length=args.max_length,
    )

    train_dataset.to_parquet(str(output_dir / "train.parquet"))
    validation_dataset.to_parquet(str(output_dir / "validation.parquet"))
    _write_metadata(output_dir, args, train_dataset, validation_dataset)

    print(f"Saved OpenAI-tokenized Nemotron dataset to {output_dir}")
    print(f"Labels: {len(LABELS)}")
    print(f"Train rows: {len(train_dataset)}")
    print(f"Validation rows: {len(validation_dataset)}")


if __name__ == "__main__":
    main()
