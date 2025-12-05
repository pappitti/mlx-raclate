import argparse
import mlx.core as mx
from mlx_raclate.utils.utils import load, PIPELINES
from mlx_raclate.tuner.datasets import load_dataset, DatasetArgs
from mlx_raclate.tuner.trainer import Trainer, TrainingArgs

DEFAULT_MODEL_PATH : str = "answerdotai/ModernBERT-base" #"Qwen/Qwen3-Embedding-0.6B" "answerdotai/ModernBERT-base"
DEFAULT_DATASET : str = "data/20251205_1125" # can be a local path "argilla/synthetic-domain-text-classification" "data/20251205_1125"
DEFAULT_TASK_TYPE : str = "text-classification"

def init_args():
    parser = argparse.ArgumentParser(description="Train or evaluate a classification model using MLX Raclate.")
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH, help="Path to the pre-trained model or model identifier from a model hub.")
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET, help="Local path or HF identifier of the dataset to use for training/evaluation.")
    parser.add_argument("--task_type", type=str, default=DEFAULT_TASK_TYPE, help="Type of task (default: text-classification).")
    parser.add_argument("--is_regression", default=False, action='store_true', help="Set this flag if the task is regression.")
    parser.add_argument("--train", default=True, action='store_true', help="Set this flag to train the model; if not set, only evaluation will be performed.")
    parser.add_argument("--use_chat_template", action='store_true', help="Use chat template for decoder models when there are text pairs.")
    parser.add_argument("--force_separator", type=str, default=None, help="Force a specific separator between text pairs for decoder models, if not using chat template.")
    return parser.parse_args()

def main():
    args = init_args()

    model_path : str = args.model_path 
    dataset : str = args.dataset
    task_type : str = args.task_type
    is_regression : bool = args.is_regression
    train : bool = args.train
    use_chat_template : bool = args.use_chat_template
    force_separator : str = args.force_separator

    if task_type not in PIPELINES:
        raise ValueError(f"Task type {task_type} not supported. Choose from {PIPELINES.items()}")
    
    output_dir = model_path.split("/")[-1] + "_" + task_type

    # Load datasets
    dataset_args = DatasetArgs(
        data=dataset, 
        task_type=task_type, 
        train=train,
        text_field="question",
        text_pair_field="response_anonymized",
        label_field="classification"
    )
    
    train_dataset, valid_dataset, test_dataset, id2label, label2id = load_dataset(dataset_args)

    model_config={}
    if task_type == "text-classification" and is_regression:
        model_config={"is_regression":True}
    if id2label:
        model_config["id2label"] = id2label
    if label2id:
        model_config["label2id"] = label2id
        
    # Load model and tokenizer
    model, tokenizer = load(
        model_path, 
        model_config=model_config, 
        pipeline=task_type,
        train=train,
    )

    # Training arguments
    training_args = TrainingArgs(
        batch_size=8,
        eval_batch_size=8,
        max_length= model.config.max_position_embeddings,
        num_train_epochs=3,
        learning_rate=5e-5, ### 5e-5 
        weight_decay=0.01,
        gradient_accumulation_steps=2, 
        eval_steps=500,
        save_steps=1000,
        logging_steps=96, ### 100
        output_dir=output_dir,
        save_total_limit=None,
        grad_checkpoint=True,
        push_to_hub=False,
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        task_type=task_type,
        training_args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        use_chat_template=use_chat_template if task_type == "text-classification" else False,
        force_separator=force_separator if task_type == "text-classification" else None,
        label2id=label2id
    )
    
    # Train or evaluate
    if train:
        trainer.train()
    if test_dataset:
        trainer.test(test_dataset)

if __name__ == "__main__":
    main()
