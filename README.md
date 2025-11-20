# RACLATE (MLX)

## Installation

```bash
uv add mlx-raclate
```

From source:  
```bash
git clone https://github.com/pappitti/mlx-raclate.git
cd mlx-raclate
uv sync
```

## Quick start
TODO

## Pipelines
In order to use different model classes for different use-cases, a concept of pipeline was introduced to load the model.  
```python
from utils import load
model, tokenizer = load("answerdotai/ModernBERT-base", pipeline='masked-lm')
```  

If no pipeline is provided, information from the model onfig will be used to infer a pipeline:  
 - if config_sentence_transformers.json exists, pipeline will automatically be set to "sentence-transformers"
 - if model architecture in config file is ModernBertForSequenceClassification, pipeline will automatically be set to "text-classification" 
 - if model architecture in config file is ModernBertForMaskedLM, pipeline will automatically be set to "masked-lm"  
If there is no match, the pipeline defaults to masked-lm (original modernBERT model).  

### Pipeline list 
- "embeddings"  
Uses the Model class. Returns the pooled, unnormalized embeddings of the input sequence. Pooling strategy (CLS or mean) is defined by config file. Note that sentence transformers models will automatically switch to the "sentence-transformers" pipeline.  
See examples/raw_embeddings.py  
  
- "sentence-similarity" and "sentence-transformers"  
"sentence-similarity" uses the ModelForSentenceSimilarity class, which extends Model, and returns a dictionary with the embeddings and similarity matrix, using cosine similarity, between input sequences and reference sequences.  
"sentence-transformers" uses the ModelForSentenceTransformers class, which extends ModelForSentenceSimilarity. The only difference is weight sanitization as sentence transformers parameters keys are specific.  
See examples/sentencesimilarity.py  
  
- "masked-lm"  
Uses the ModelForMaskedLM class. Returns logits for all tokens in the input sequence. For now, filtering for the masked token and softmax are handled outside the pipeline.  
See examples/maskedlm.py  
  
- "zero-shot-classification"  
Uses the ModelForMaskedLM class. See above. The reason for using this pipeline is a recent paper by Answer.ai showing that it was a great alternative to other approaches.  
See examples/zeroshot.py  

(a ModelForZeroShotClassification class had also been prepared, which extends Model, and returns probabilities of labels for the sequence. There are other interpretrations of what zero-shot classification means, notably classifications that require fixed labels. Not sure which approach is correct. More work needed so the class is kept for future work even if it is not used. Here, labels must be provided in the config file as a list or as a dictionary {label:description} as a label_candidates parameter in the config file.)  
  
- "text-classification"  
Uses the ModelForSequenceClassification class. Returns probabilities of labels for the sequence. Classification can be a regression (untested for now), binary classification (untested for now) or multilabel classification. For multilabel the config file must contain an id2label dictionary.  
See examples/textclassification.py  
  
- "token-classification" (not tested)  
Uses the ModelForTokenClassification class. Returns probabilities of labels for each token in the sequence.  

Running example file : `uv run python -m examples.raw_embeddings` or if the venv is activated `python -m examples.raw_embeddings`

### Batch processing 
For batch processing, use the _tokenizer method of the tokenizer. 

Unlike the encode method, it does not return a list of tokens but a dictionary with the following keys that are used for the forward pass:  
- input_ids: The token IDs of the input text.
- attention_mask: A mask to avoid performing attention on padding token indices.

```python
input_ids = tokenizer._tokenizer(
    texts, 
    return_tensors="mlx", 
    padding=True, 
    truncation=True, 
    max_length= max_position_embeddings
)

# Generate embeddings
outputs = model(
    input_ids['input_ids'], 
    attention_mask=input_ids['attention_mask'],
)
```

## Supported Models
TODO. 
- ModernBERT 


## Training
Not all pipelines are supported for training yet (WIP).

### Supported pipelines 
TODO

## Acknowledgements
TODO
- transformers
- mlx-examples
- mlx-embeddings
- pylate
