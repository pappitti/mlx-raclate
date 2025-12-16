import argparse
import time

from mlx_raclate.utils.utils import load, PIPELINES
from mlx_raclate.tuner.datasets import load_dataset, DatasetArgs
from mlx_raclate.tuner.trainer import Trainer, TrainingArgs

train_tested = {
    "text-classification": [
        {"model": "Qwen/Qwen3-Embedding-0.6B", "special_model_config" : {}, "special_trainer_config" : {"use_chat_template": True}, "special_training_args" : {"max_length":16384}},
        {"model": "answerdotai/ModernBERT-base", "special_model_config" : {}, "special_training_args" : {}},
        {"model": "LiquidAI/LFM2-350M", "special_model_config" : {}, "special_training_args" : {}},
        {"model": "google/t5gemma-b-b-ul2", "special_model_config" : {}, "special_training_args" : {"max_length":16384}} # failed
    ],
}

DEFAULT_MODEL_PATH : str = "LiquidAI/LFM2-350M" #"./trained_models/Qwen3-Embedding-0.6B_text-classification_20251216_174716/checkpoint-14940" #"Qwen/Qwen3-Embedding-0.6B" "answerdotai/ModernBERT-base" "google/t5gemma-b-b-ul2"
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
    parser.add_argument("--resume_from_step", type=int, default=0, help="Step number to resume training from (if applicable).")
    parser.add_argument("--max_length", type=int, default=None, help="Maximum sequence length for the model inputs. If not specified, the model's default max length will be used.")
    parser.add_argument("--freeze_embeddings", default=False, action='store_true', help="Set this flag to freeze embedding layers during training.")
    parser.add_argument("--max_grad_norm", type=float, default=1, help="Maximum gradient norm for gradient clipping (Default: 1).")
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
    resume_from_step : int = args.resume_from_step
    max_length : int = args.max_length
    freeze_embeddings : bool = args.freeze_embeddings
    max_grad_norm : float = args.max_grad_norm

    if task_type not in PIPELINES:
        raise ValueError(f"Task type {task_type} not supported. Choose from {PIPELINES.items()}")
    
    output_dir = model_path.split("/")[-1] + "_" + task_type + "_" + time.strftime("%Y%m%d_%H%M%S")

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

    # testing chat template
    if use_chat_template:
        if not getattr(tokenizer, "chat_template", None):
            raise ValueError("The tokenizer does not support chat templates.")
        messages = [
            {"role": "user", "content": "test_prompt"},
            {"role": "assistant", "content": "test_response"}
        ]
        templated = tokenizer.apply_chat_template(messages, tokenize=False)
        print("Chat template working:", templated)

    # Training arguments
    training_args = TrainingArgs(
        batch_size=2,
        gradient_accumulation_steps=4, 
        max_length= max_length if max_length else model.config.max_position_embeddings,
        resume_from_step=resume_from_step, # warmup will be ingnored if before this step and schedulers will only start after
        num_train_epochs=2,
        learning_rate=2e-5, # 3e-5 for ModernBERT, 5e-5 for T5Gemma, 1e-5 for Qwen
        weight_decay=0.01,
        freeze_embeddings=freeze_embeddings,
        warmup_ratio=0.03, # can use warmup_steps=300 instead (both warmup_ratio and warmup_steps default to 0, steps override ratio)
        lr_scheduler_type="cosine_decay", # would default to "constant", can also use "cosine_decay" or "linear_schedule"
        max_grad_norm=max_grad_norm,
        save_steps=1000,
        logging_steps=12, # will be adjusted to be multiple of gradient_accumulation_steps inside Trainer
        eval_batch_size=2,
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
