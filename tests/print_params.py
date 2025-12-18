import mlx.core as mx
from mlx.utils import tree_flatten
from mlx_raclate.utils.utils import load
from mlx_raclate.tuner.utils import nparams

def main():
    
    is_regression = False # Set to True for regression models
    
    # Load the model and tokenizer
    model, tokenizer = load(
         "./trained_models/ModernBERT-base_text-classification_20251216_191254/checkpoint-20036",
        pipeline='text-classification' # if the config file includes the architecture "ModernBertForSequenceClassification", the pipeline will be identified automatically so no need to specify it
    ) 

    print(f"Number of model parameters: {nparams(model)}")
    # print(f"Embedding dtype: {model.model.embeddings.tok_embeddings.weight.dtype}")
    # print(f"Linear dtype: {model.model.layers[0].attn.Wqkv.weight.dtype}")

    params_keys = [k for k,_ in dict(tree_flatten(model.parameters())).items()]

    print("Model parameters:")
    for key in params_keys:
        print(key)


if __name__ == "__main__":
    main()