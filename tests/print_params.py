import mlx.core as mx
from mlx.utils import tree_flatten
from mlx_raclate.utils.utils import load

def main():
    
    is_regression = False # Set to True for regression models
    
    # Load the model and tokenizer
    model, tokenizer = load(
         "./trained_models/LFM2-350M_text-classification_20251216_234824/LFM2-350M_text-classification_20251216_234824_E1",
        pipeline='text-classification' # if the config file includes the architecture "ModernBertForSequenceClassification", the pipeline will be identified automatically so no need to specify it
    ) 

    params_keys = [k for k,_ in dict(tree_flatten(model.parameters())).items()]

    print("Model parameters:")
    for key in params_keys:
        print(key)


if __name__ == "__main__":
    main()