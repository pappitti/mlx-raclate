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
    label_pad_token_id: int = -100

    def __call__(self, features: Dict[str, List[Any]]) -> Dict[str, mx.array]:
        texts = features.get("text")
        text_pairs = features.get("text_pair", None)
        
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
            # Detect regression (float) vs classification (int)
            dtype = mx.float32 if isinstance(labels[0], float) else mx.int32
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
           # word_ids returns a list mapping each token to its original word index
            # e.g., [None, 0, 1, 1, 2, None] for "[CLS] My name is John [SEP]"
            word_ids = batch.word_ids(batch_index=i)
            
            previous_word_idx = None

            for k, word_idx in enumerate(word_ids):
                # Skip Special Tokens (None)
                if word_idx is None:
                    continue
                
                # Safety check: tokenizer truncation might leave word_ids that point to label indices larger than the label list provided.
                if word_idx >= len(label_seq):
                    break 
                
                if word_idx != previous_word_idx:
                    padded_labels[i, k] = label_seq[word_idx]
                else:
                    # This is a subsequent subword of the same word
                    if self.label_all_tokens:
                        padded_labels[i, k] = label_seq[word_idx]
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
    Handles (Sentence1, Sentence2, Score).
    Assumes datasets.py mapped these to specific keys.
    """
    def __call__(self, features: Dict[str, List[Any]]) -> Dict[str, mx.array]:
        # TODO : is sentence similarity essentially just a regression classification?
        if "text" in features and "text_pair" in features and "label" in features:
             # Just use the classification/regression collator
             delegate = DataCollatorForSequenceClassification(self.tokenizer, self.max_length)
             return delegate(features)
        else :
            raise ValueError("Make sure to pass dataset args to convert sentence1, sentence2, score to text, text_pair, label. DatasetArgs(data=path, task_type=pipeline, text_field='sentence1', text_pair_field-'sentence2', label_field='score') ")
    
@dataclass
class DataCollatorForSentenceTransformers(DataCollator):
    """
    Handles tuples (Anchor, Positive, Negative) or (Sentence1, Sentence2, Score).
    Assumes datasets.py mapped these to specific keys.
    """
    def __call__(self, features: Dict[str, List[Any]]) -> Dict[str, mx.array]:
        if "text" in features and "text_pair" in features and "label" in features:
             # Just use the classification/regression collator
             delegate = DataCollatorForSequenceClassification(self.tokenizer, self.max_length)
             return delegate(features)
        
        # Case: Triplets (Anchor, Positive, Negative)
        elif "anchor" in features and "positive" in features and "negative" in features:
            anchors = features["anchor"]
            positives = features["positive"]
            negatives = features["negative"]
            
            # Concatenate lists: [All Anchors, All Positives, All Negatives]
            # ModelForSentenceTransformers will split the embeddings back into 3 chunks.
            all_texts = anchors + positives + negatives
            
            batch = self.tokenizer(
                all_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="mlx"
            )
            
            return dict(batch)

        raise NotImplementedError("Sentence Transformer collation depends on specific loss function requirements")