import mlx.core as mx
import numpy as np
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass

@dataclass
class DataCollator:
    tokenizer: Any
    max_length: int = 512
    
    def __call__(self, features: Dict[str, List[Any]]) -> Dict[str, mx.array]:
        raise NotImplementedError

@dataclass
class DataCollatorForSequenceClassification(DataCollator):
    """
    Handles tokenization and padding for classification tasks.
    """
    use_chat_template: bool = False# Whether to use chat templates for decoder models
    force_separator: Optional[str] = None # If set, forces this separator between text pairs
    default_decoder_separator: str = "\n" # Used for decoder models when concatenating text pairs
    label2id: Optional[Dict[str, int]] = None

    def __call__(self, features: Dict[str, List[Any]]) -> Dict[str, mx.array]:
        texts = features.get("text")
        text_pairs = features.get("text_pair", None)

        if text_pairs is not None:
            if getattr(self.tokenizer, "chat_template", None) and self.use_chat_template:
                # This ensures the model sees exactly what it expects for Q&A
                formatted_texts = []
                for prompt, response in zip(texts, text_pairs):
                    messages = [
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": response}
                    ]
                    formatted_texts.append(
                        self.tokenizer.apply_chat_template(messages, tokenize=False)
                    )
                texts = formatted_texts
                text_pairs = None # Handled by template

            elif self.force_separator is not None:
                # Use the forced separator for decoder models
                texts = [
                    f"{t}{self.force_separator}{p}" 
                    for t, p in zip(texts, text_pairs)
                ]
                text_pairs = None

            else :
                # Check if tokenizer has a standard separator (Like [SEP] in BERT)
                # Qwen tokenizer often has sep_token as None or same as EOS
                has_sep_token = getattr(self.tokenizer, "sep_token", None) is not None
                
                if not has_sep_token or self.tokenizer.sep_token == self.tokenizer.eos_token:
                    texts = [
                        f"{t}{self.default_decoder_separator}{p}" 
                        for t, p in zip(texts, text_pairs)
                    ]
                    # Set pairs to None so tokenizer treats it as a single string
                    text_pairs = None 

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        batch = self.tokenizer(
            texts,
            text_pairs,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="mlx"
        )
        
        if "label" in features:
            labels = features["label"]
            # On-the-fly String to ID conversion
            if self.label2id and len(labels) > 0 and isinstance(labels[0], str):
                labels = [self.label2id.get(l, -1) for l in labels] # Default to -1 if missing
            
            # Detect regression (float) vs classification (int)
            if len(labels) > 0 and isinstance(labels[0], float):
                dtype = mx.float32
            else:
                dtype = mx.int32

            batch["labels"] = mx.array(labels, dtype=dtype)
            
        return dict(batch)
    
@dataclass
class DataCollatorForTokenClassification(DataCollator):
    """
    Handles tokenization and aligns labels for token classification.
    """
    label_pad_token_id: int = -100
    # Strategy: 'first' (label only first subword), 'all' (label all subwords with same tag)
    label_all_tokens: bool = False 
    label2id: Optional[Dict[str, int]] = None

    def __call__(self, features: Dict[str, List[Any]]) -> Dict[str, mx.array]:
        texts = features["text"] 
        labels = features["labels"] # Note: usually plural 'labels' list of list

        # SANITY CHECK: The library expects pre-tokenized inputs (List[str])
        if isinstance(texts[0], str):
             raise ValueError(
                 "DataCollatorForTokenClassification expects 'text' to be a list of strings "
                 "(tokens), not a single string. Please pre-tokenize your dataset."
             )
                
        batch = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="mlx",
            is_split_into_words=True
        )

        batch_size, seq_len = batch["input_ids"].shape
        
        # Create a numpy buffer filled with the ignore index
        padded_labels = np.full((batch_size, seq_len), self.label_pad_token_id, dtype=np.int32)

        for i, label_seq in enumerate(labels):
            # On-the-fly conversion for list of strings (to avoid memory issues with dataset.map)
            current_labels = label_seq
            if self.label2id and len(label_seq) > 0 and isinstance(label_seq[0], str):
                current_labels = [self.label2id.get(l, self.label_pad_token_id) for l in label_seq]

            # word_ids returns a list mapping each token to its original word index
            # e.g., [None, 0, 1, 1, 2, None] for "[CLS] My name is John [SEP]"
            word_ids = batch.word_ids(batch_index=i)
            previous_word_idx = None

            for k, word_idx in enumerate(word_ids):
                # Skip Special Tokens (None)
                if word_idx is None:
                    continue
                
                # Safety check: tokenizer truncation might leave word_ids that point to label indices larger than the label list provided.
                if word_idx >= len(current_labels):
                    break 
                
                if word_idx != previous_word_idx:
                    padded_labels[i, k] = current_labels[word_idx]
                else:
                    # This is a subsequent subword of the same word
                    if self.label_all_tokens:
                        padded_labels[i, k] = current_labels[word_idx]
                    else:
                        # Standard BERT NER behavior: ignore subsequent subwords
                        padded_labels[i, k] = self.label_pad_token_id
                
                previous_word_idx = word_idx

        batch["labels"] = mx.array(padded_labels, dtype=mx.int32)

        return dict(batch)

@dataclass
class DataCollatorForMaskedLanguageModeling(DataCollator):
    """
    Handles dynamic masking for MLM.
    """
    mlm_probability: float = 0.15

    def __call__(self, features: Dict[str, List[Any]]) -> Dict[str, mx.array]:
        texts = features["text"]
        
        batch = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="mlx"
        )
        
        input_ids = batch["input_ids"]
        
        # Create Mask
        probability_matrix = mx.random.uniform(input_ids.shape) < self.mlm_probability
        
        # Protect special tokens
        special_tokens_mask = mx.array([
            [1 if token_id in [self.tokenizer.cls_token_id, self.tokenizer.sep_token_id, self.tokenizer.pad_token_id]
             else 0 for token_id in seq]
            for seq in input_ids.tolist() 
        ])
        
        probability_matrix = mx.where(special_tokens_mask, 0, probability_matrix)
        
        # Create labels (-100 for unmasked)
        labels = mx.where(probability_matrix, input_ids, -100)
        
        # Apply masking (80% mask, 10% random, 10% original)
        random_matrix = mx.random.uniform(input_ids.shape)
        mask_indices = (probability_matrix) & (random_matrix < 0.8)
        random_indices = (probability_matrix) & (random_matrix >= 0.8) & (random_matrix < 0.9)
        
        # Create masked input
        masked_inputs = input_ids.copy()
        masked_inputs = mx.where(mask_indices, self.tokenizer.mask_token_id, masked_inputs)
        random_tokens = mx.random.randint(
            0, self.tokenizer.vocab_size, 
            shape=input_ids.shape
        )
        
        # Apply the [MASK] token
        inputs = mx.where(random_indices, random_tokens, masked_inputs)
        
        batch["input_ids"] = inputs
        batch["labels"] = labels
        
        return dict(batch)
    
@dataclass
class DataCollatorForSentenceSimilarity(DataCollator):
    """
    Handles data for Bi-Encoder models (Sentence Similarity / Retrieval).
    Unlike SequenceClassification, this keeps sentences SEPARATE to produce
    independent embeddings.
    
    Expected keys in features (from datasets.py standardization):
    - 'text': The Anchor / Sentence A
    - 'text_pair': The Positive / Reference / Sentence B
    - 'negative' (optional): The Hard Negative / Sentence C
    - 'label' (optional): Similarity score for Regression
    """
    def __call__(self, features: Dict[str, List[Any]]) -> Dict[str, mx.array]:
        batch = {}

        # Tokenize Anchor (Sentence A) -> 'input_ids'
        if "text" in features:
            out_a = self.tokenizer(
                features["text"],
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="mlx"
            )
            batch["input_ids"] = out_a["input_ids"]
            batch["attention_mask"] = out_a["attention_mask"]

        # Tokenize Reference (Sentence B) -> 'reference_input_ids'
        if "text_pair" in features:
            out_b = self.tokenizer(
                features["text_pair"],
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="mlx"
            )
            batch["reference_input_ids"] = out_b["input_ids"]
            batch["reference_attention_mask"] = out_b["attention_mask"]

        # Tokenize Negative (Sentence C) -> 'negative_input_ids'
        neg_key = None
        if "negative" in features: neg_key = "negative"
        elif "text_negative" in features: neg_key = "text_negative"
            
        if neg_key:
            out_n = self.tokenizer(
                features[neg_key],
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="mlx"
            )
            batch["negative_input_ids"] = out_n["input_ids"]
            batch["negative_attention_mask"] = out_n["attention_mask"]

        # Handle Scores (for Regression)
        if "label" in features:
            # Ensure float32 for regression targets
            batch["similarity_scores"] = mx.array(features["label"], dtype=mx.float32)

        return batch
