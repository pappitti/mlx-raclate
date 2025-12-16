import json
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datasets import load_dataset as hf_load_dataset 
from datasets import DatasetDict, ClassLabel, Sequence, Value
from datasets import Dataset as HFDataset
import mlx.core as mx

class DatasetArgs:
    """
    Arguments for dataset loading
    If a remapping of column names is needed, specify the field names here.
    main text : text_field
    label / classification : label_field
    text pair (optional for contrastive learning, sentence similarity or just sequence classification with 2 inputs) : text_pair_field
    negative example (optional for triplet loss) : negative_field
    """
    def __init__(self, data: str, task_type: str, train : bool, 
        text_field: Optional[str] = "text", label_field: Optional[str] = "label",
        text_pair_field: Optional[str] = None, negative_field: Optional[str] = None, test: Optional[bool]=False
    ):
        self.data = data
        self.task_type = task_type
        self.train = train
        self.text_field = text_field
        self.label_field = label_field
        self.text_pair_field = text_pair_field
        self.negative_field = negative_field
        self.test = test # whether to create a test set if not present


def _standardize_column_names(dataset: HFDataset, args: DatasetArgs) -> HFDataset:
    """
    Renames columns to standard 'text', 'label', 'text_pair', 'negative' expected by collators.

    Common mappings for various tasks:
    - similarity : Anchor / Sentence A -> 'text'
    - similarity : The Positive / Reference / Sentence B -> 'text_pair'
    - similarity : The Hard Negative / Sentence C -> 'negative' (optional)
    - similarity : Similarity score for Regression -> 'label' (optional)

    Manual mappings can be specified via args usiing text_field, label_field, text_pair_field, negative_field.
    text_field : column name for the main text input
    label_field : column name for the label / score
    text_pair_field (optional): column name for the paired text input / sentence B (used for cross-encoders or bi-encoders)
    negative_field (optional): column name for the negative example (used for triplet training)
    """

    mapping = {}
    # Manual field mappings
    if args.text_field != "text" and args.text_field in dataset.column_names:
        mapping[args.text_field] = "text"

    if args.text_pair_field and args.text_pair_field != "text_pair" and args.text_pair_field in dataset.column_names:
        mapping[args.text_pair_field] = "text_pair"
    
    if args.label_field != "label" and args.label_field in dataset.column_names:
        mapping[args.label_field] = "label"

    if args.negative_field and args.negative_field != "negative" and args.negative_field in dataset.column_names:
        mapping[args.negative_field] = "negative"

    # Handle common alternative column names for text classification
    if args.task_type == "sentence-similarity" or args.task_type == "sentence-transformers":
         # handle Sequence classification : "sentence1" -> "text", "sentence2" -> "text_pair", "score" = "label"
        if "sentence1" in dataset.column_names and "sentence2" in dataset.column_names and "score" in dataset.column_names:
            mapping["sentence1"] = "text"
            mapping["sentence2"] = "text_pair"
            mapping["score"] = "label"

        # Handle Anchor, Positives and Negatives for Triplet Training
        if "anchor" in dataset.column_names and "positive" in dataset.column_names and "negative" in dataset.column_names:
            mapping["anchor"] = "text"
            mapping["positive"] = "text_pair"
            mapping["negative"] = "negative"

        if "pos" in dataset.column_names:
            mapping["pos"] = "text_pair"
        if "neg" in dataset.column_names:
            mapping["neg"] = "negative"

    # Handle Token Classification: usually "tokens" -> "text", "ner_tags" -> "labels"
    if args.task_type == "token-classification":
        if "tokens" in dataset.column_names and "text" not in mapping.values():
             mapping["tokens"] = "text"
        if "ner_tags" in dataset.column_names and "labels" not in mapping.values():
             mapping["ner_tags"] = "labels"
        
    if mapping:
        dataset = dataset.rename_columns(mapping)

    keep_columns = {"text", "text_pair", "label", "labels", "negative"}
    existing_columns = set(dataset.column_names)
    columns_to_select = list(keep_columns.intersection(existing_columns))
    
    # Check if we have at least 'text'
    if "text" not in columns_to_select:
        print(f"Warning: Standard 'text' column not found in dataset columns: {dataset.column_names}")
    
    dataset = dataset.select_columns(columns_to_select)
        
    return dataset


def get_label_mapping(dataset: HFDataset, args: DatasetArgs) -> Tuple[Optional[Dict[int, str]], Optional[Dict[str, int]]]:
    """
    Derives id2label and label2id from a dataset.
    Prioritizes dataset features (from config), falls back to scanning unique values in data.
    """
    if args.task_type not in ["text-classification", "token-classification"]:
        return None, None

    # Determine the target column name based on task
    target_col = "labels" if args.task_type == "token-classification" else "label"
    if target_col not in dataset.column_names:
        # Fallback: sometimes text-classification uses 'labels' or vice versa
        if "label" in dataset.column_names: target_col = "label"
        elif "labels" in dataset.column_names: target_col = "labels"
        else: return None, None

    labels = []
    
    # Strategy 1: Check Features (Config/Hub Metadata) ---
    feature = dataset.features[target_col]
    
    # Case A: Standard ClassLabel (Text Classification)
    if isinstance(feature, ClassLabel):
        labels = feature.names
    
    # Case B: Sequence of ClassLabels (Token Classification)
    elif isinstance(feature, Sequence) and isinstance(feature.feature, ClassLabel):
        labels = feature.feature.names

    # Strategy 2: Scan Data (Raw JSONL/CSV) ---
    if not labels:
        if len(dataset) > 0:
            if args.task_type == "token-classification":
                # Flatten list of lists to find unique tags
                unique_tags = set()
                for row in dataset[target_col]:
                    unique_tags.update(row)
                labels = sorted(list(unique_tags))
            else:
                # Standard text classification scan
                labels = sorted(list(set(dataset[target_col])))

    if not labels:
        return None, None

    # Construct mappings
    id2label = {k: str(v) for k, v in enumerate(labels)}
    label2id = {str(v): k for k, v in enumerate(labels)}
    
    return id2label, label2id


def load_dataset(args: DatasetArgs) -> Tuple[Optional[HFDataset], Optional[HFDataset], Optional[HFDataset]]:
    if not hasattr(args, "task_type"):
        raise ValueError("Must specify task_type in args")
    
    supported_tasks = ["text-classification", "masked-lm", "token-classification", "sentence-transformers"]
    if args.task_type not in supported_tasks:
        raise ValueError(f"Unsupported task type: {args.task_type}")
    
    # Load from Hub or Local
    data_path = Path(args.data)
    if data_path.exists():
        # Detect format based on extension if it's a file, or assume structure if folder
        if data_path.is_file():
            # Single file loading
            ext = data_path.suffix[1:] # remove dot
            ext = "json" if ext == "jsonl" else ext
            raw_datasets = hf_load_dataset(ext, data_files=str(data_path))
            # If it loaded as 'train' only, we split later
        else:
            # It's a directory. Check for specific files.
            data_files = {}
            for split in ["train", "validation", "test"]:
                for ext in ["jsonl", "json", "parquet", "csv"]:
                    fname = f"{split}.{ext}"
                    if (data_path / fname).exists():
                        data_files[split] = str(data_path / fname)
            
            if not data_files:
                raise ValueError(f"No train/val/test files found in {data_path}")
            
            # Determine loader type from first found file
            first_file = list(data_files.values())[0]
            ext = first_file.split(".")[-1]
            ext = "json" if ext == "jsonl" else ext
            raw_datasets = hf_load_dataset(ext, data_files=data_files)
    
    else:
        # Load from Hub
        try:
            raw_datasets = hf_load_dataset(args.data)
        except Exception as e:
            print(f"Failed to load as standard dataset: {e}. Trying simple load...")
            raw_datasets = hf_load_dataset(args.data, split="train")
            raw_datasets = DatasetDict({"train": raw_datasets})

    if "train" not in raw_datasets:
        raise ValueError("Training split not found in dataset")
    
    # Handle Splits (Standard 70/15/15) or whatever the actual splits are
    if "validation" not in raw_datasets and "test" not in raw_datasets: 
        if args.test:
            t_t_split = raw_datasets["train"].train_test_split(test_size=0.15, seed=42)
            raw_datasets["test"] = t_t_split["test"]
            t_v_split = t_t_split["train"].train_test_split(test_size=0.176, seed=42) 
            raw_datasets["train"] = t_v_split["train"]
            raw_datasets["validation"] = t_v_split["test"]
        else : # create only validation split
            t_v_split = raw_datasets["train"].train_test_split(test_size=0.176, seed=42)
            raw_datasets["train"] = t_v_split["train"]
            raw_datasets["validation"] = t_v_split["test"]
    elif "validation" not in raw_datasets and "test" in raw_datasets:
        if args.test:
            t_v_split = raw_datasets["train"].train_test_split(test_size=0.176, seed=42)
            raw_datasets["train"] = t_v_split["train"]
            raw_datasets["validation"] = t_v_split["test"]
        else : # use test split as validation split
            raw_datasets["validation"] = raw_datasets["test"]
            raw_datasets["test"] = None
    elif "test" not in raw_datasets and args.test:
        t_t_split = raw_datasets["train"].train_test_split(test_size=0.176, seed=42) 
        raw_datasets["train"] = t_t_split["train"]
        raw_datasets["test"] = t_t_split["test"]

    # Standardize Columns
    for split in raw_datasets.keys():
        if raw_datasets[split] is not None:
            print(f"Standardizing columns for split '{split}' ({len(raw_datasets[split])} examples)...")
            raw_datasets[split] = _standardize_column_names(raw_datasets[split], args)

    # Get label mappings if applicable
    id2label, label2id = None, None
    if raw_datasets.get("train") is not None:
        id2label, label2id = get_label_mapping(raw_datasets["train"], args)
        
        if id2label:
            print(f"Found {len(id2label)} labels. First 5: {list(id2label.values())[:5]}")

    return (
        raw_datasets.get("train"), 
        raw_datasets.get("validation"), 
        raw_datasets.get("test"), 
        id2label, 
        label2id
    )