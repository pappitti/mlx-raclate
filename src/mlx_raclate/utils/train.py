import argparse
import time
from collections.abc import Sequence

from mlx_raclate.registry import PIPELINES


DEFAULT_TASK_TYPE : str = "text-classification"
DEFAULT_BATCH_SIZE : int = 8
DEFAULT_GRADIENT_ACCUMULATION_STEPS : int = 8
DEFAULT_TRAIN_EPOCHS : int = 2
DEFAULT_WEIGHT_DECAY : float = 0.01
DEFAULT_LR : float = 2e-5 # 3e-5 for ModernBERT, 5e-5 for T5Gemma, 1e-5 for Qwen
DEFAULT_LR_SCHEDULER_TYPE : str = "linear_schedule"
DEFAULT_MIN_LR : float = 2e-6
DEFAULT_WARMUP_RATIO : float = 0.03
DEFAULT_WARMUP_STEPS : int = 0
DEFAULT_SAVE_STEPS : int = 5000
DEFAULT_LOGGING_STEPS : int = 64
DEFAULT_EVAL_BATCH_SIZE : int = 8
DEFAULT_CACHE_CLEAR_STEPS : int = 1

def init_args(argv: Sequence[str] | None = None):
    parser = argparse.ArgumentParser(
        prog="raclate train",
        description="Train or evaluate a classification model using MLX Raclate.",
    )
    # Dataset Init Params
    parser.add_argument("--dataset", type=str, required=True, help="Local path or HF identifier of the dataset to use for training/evaluation.")
    parser.add_argument("--text_field", type=str, default=None, help="Name of the text field in the dataset (if different from default).")
    parser.add_argument("--text_pair_field", type=str, default=None, help="Name of the text pair field in the dataset (if applicable and different from default).")
    parser.add_argument("--label_field", type=str, default=None, help="Name of the label field in the dataset (if different from default).")
    parser.add_argument("--negative_field", type=str, default=None, help="Name of the negative samples field in the dataset (if applicable and different from default).")
    parser.add_argument("--create_test", action='store_true', help="Set this flag to create a test split, if not already present in the dataset, out of the training set (validation set not affected).")
    parser.add_argument("--train_limit", type=int, default=None, help="Limit the number of training examples after loading and split creation.")
    parser.add_argument("--validation_limit", type=int, default=None, help="Limit the number of validation examples after loading and split creation.")
    parser.add_argument("--test_limit", type=int, default=None, help="Limit the number of test examples after loading and split creation.")

    # Trainer / End Model Init Params
    parser.add_argument("--model_path", type=str, required=True, help="Path to the pre-trained model or model identifier from a model hub.")
    parser.add_argument("--dtype", choices=("auto", "float32", "float16", "bfloat16"), help="Override model_config dtype. If omitted, use the checkpoint config.")
    parser.add_argument("--task_type", choices=PIPELINES, default=DEFAULT_TASK_TYPE, help="Type of task (default: text-classification).")
    parser.add_argument("--is_regression", default=False, action='store_true', help="Set this flag if the task is regression.")
    parser.add_argument("--use_late_interaction", default=False, action='store_true', help="Set this flag to use late interaction for retrieval tasks (if applicable).")
    parser.add_argument("--eval_only", dest="train", action='store_false', help="Set this flag to skip training and only evaluate.")
    parser.add_argument("--use_chat_template", default=False, action='store_true', help="Use chat template for decoder models when there are text pairs.")
    parser.add_argument("--force_separator", type=str, default=None, help="Force a specific separator between text pairs for decoder models, if not using chat template.")
    parser.add_argument("--base_model", type=str, default=None, help="Upstream Hugging Face model ID recorded in checkpoint model cards. Required with --push_to_hub.")
    parser.add_argument("--hub_repo_id", type=str, default=None, help="Destination Hugging Face model repository, for example org/model-name.")
    parser.add_argument("--push_to_hub", default=False, action="store_true", help="Upload each saved checkpoint to --hub_repo_id.")

    # Training Params
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE, help="Batch size for training.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=DEFAULT_GRADIENT_ACCUMULATION_STEPS, help="Number of gradient accumulation steps.")
    parser.add_argument("--gradient_accumulation_dtype", choices=("float32", "model"), default="float32", help="Dtype used to accumulate gradients.")
    parser.add_argument("--no_multi_precision", dest="multi_precision", action="store_false", help="Disable FP32 master weights for low-precision full training.")
    parser.add_argument("--num_train_epochs", type=int, default=DEFAULT_TRAIN_EPOCHS, help="Number of training epochs.")
    parser.add_argument("--max_length", type=int, default=None, help="Maximum sequence length for the model inputs. If not specified, the model's default max length will be used.")
    parser.add_argument("--freeze_embeddings", default=False, action='store_true', help="Set this flag to freeze embedding layers during training.")
    parser.add_argument("--weight_decay", type=float, default=DEFAULT_WEIGHT_DECAY, help="Weight decay for the optimizer.")
    # Optimizer and Scheduler Params (AdamW + schedulers)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR, help="Initial learning rate for the optimizer.")
    parser.add_argument("--lr_scheduler_type", type=str, default=DEFAULT_LR_SCHEDULER_TYPE, help="Type of learning rate scheduler to use.")
    parser.add_argument("--min_lr", type=float, default=DEFAULT_MIN_LR, help="Minimum learning rate for the scheduler.")
    parser.add_argument("--warmup_ratio", type=float, default=DEFAULT_WARMUP_RATIO, help="Warmup ratio for learning rate scheduler.")
    parser.add_argument("--warmup_steps", type=int, default=DEFAULT_WARMUP_STEPS, help="Number of warmup steps for learning rate scheduler (if set, steps override ratio).")
    parser.add_argument("--max_grad_norm", type=float, default=1, help="Maximum gradient norm for gradient clipping (Default: 1).")
    parser.add_argument("--resume_from_step", type=int, default=0, help="Step number to resume training from (if applicable). Will override warmup if steps are after warmup period.")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Resume model, optimizer, scheduler, and epoch position from a complete Raclate checkpoint.")
    # Other Training Params
    parser.add_argument("--logging_steps", type=int, default=DEFAULT_LOGGING_STEPS, help="Number of steps between logging training metrics.")
    parser.add_argument("--save_steps", type=int, default=DEFAULT_SAVE_STEPS, help="Number of steps between model checkpoints.")
    parser.add_argument("--eval_batch_size", type=int, default=DEFAULT_EVAL_BATCH_SIZE, help="Batch size for evaluation.")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save model checkpoints and logs.")
    parser.add_argument("--no_grad_checkpoint", dest="grad_checkpoint", action="store_false", help="Disable MLX gradient checkpointing.")
    parser.add_argument("--cache_clear_steps", type=int, default=DEFAULT_CACHE_CLEAR_STEPS, help="Clear the MLX cache every N optimizer updates. Use 0 to disable explicit cache clearing.")
    parser.set_defaults(train=True)
    parser.set_defaults(grad_checkpoint=True)
    parser.set_defaults(multi_precision=True)
    arguments = parser.parse_args(argv)
    if arguments.push_to_hub and not arguments.hub_repo_id:
        parser.error("--hub_repo_id is required with --push_to_hub")
    if arguments.push_to_hub and not arguments.base_model:
        parser.error("--base_model is required with --push_to_hub")
    return arguments

def main(argv: Sequence[str] | None = None):
    args = init_args(argv)

    from mlx_raclate.tuner.datasets import DatasetArgs, load_dataset
    from mlx_raclate.tuner.trainer import Trainer, TrainingArgs
    from mlx_raclate import load

    # Dataset Params
    dataset : str = args.dataset
    text_field : str = args.text_field
    text_pair_field : str = args.text_pair_field
    label_field : str = args.label_field
    negative_field : str = args.negative_field
    create_test : bool = args.create_test
    train_limit : int = args.train_limit
    validation_limit : int = args.validation_limit
    test_limit : int = args.test_limit

    # Trainer / End Model Params
    model_path : str = args.model_path 
    dtype : str = args.dtype
    task_type : str = args.task_type
    is_regression : bool = args.is_regression
    use_late_interaction : bool = args.use_late_interaction
    train : bool = args.train
    use_chat_template : bool = args.use_chat_template
    force_separator : str = args.force_separator
    base_model : str = args.base_model
    hub_repo_id : str = args.hub_repo_id
    push_to_hub : bool = args.push_to_hub

    # Training Params
    batch_size : int = args.batch_size
    gradient_accumulation_steps : int = args.gradient_accumulation_steps
    gradient_accumulation_dtype : str = args.gradient_accumulation_dtype
    multi_precision : bool = args.multi_precision
    num_train_epochs : int = args.num_train_epochs
    weight_decay : float = args.weight_decay
    learning_rate : float = args.lr
    lr_scheduler_type : str = args.lr_scheduler_type
    min_lr : float = args.min_lr
    warmup_ratio : float = args.warmup_ratio
    warmup_steps : int = args.warmup_steps
    logging_steps : int = args.logging_steps
    save_steps : int = args.save_steps
    eval_batch_size : int = args.eval_batch_size
    resume_from_step : int = args.resume_from_step
    resume_from_checkpoint : str = args.resume_from_checkpoint
    max_length : int = args.max_length
    freeze_embeddings : bool = args.freeze_embeddings
    max_grad_norm : float = args.max_grad_norm
    grad_checkpoint : bool = args.grad_checkpoint
    cache_clear_steps : int = args.cache_clear_steps

    if resume_from_step and resume_from_checkpoint:
        raise ValueError("Use either --resume_from_step or --resume_from_checkpoint, not both.")
    if resume_from_checkpoint:
        model_path = resume_from_checkpoint

    print(f"Training Mode : {train}")

    output_dir : str =  args.output_dir if args.output_dir else model_path.split("/")[-1] + "_" + task_type + "_" + time.strftime("%Y%m%d_%H%M%S")

    # Load datasets
    dataset_args = DatasetArgs(
        data=dataset, 
        task_type=task_type, 
        text_field=text_field,
        text_pair_field=text_pair_field,
        label_field=label_field,
        negative_field=negative_field,
        test=create_test,
        train_limit=train_limit,
        validation_limit=validation_limit,
        test_limit=test_limit,
    )
    
    train_dataset, valid_dataset, test_dataset, id2label, label2id = load_dataset(dataset_args)

    model_config={}
    if dtype is not None:
        model_config["dtype"] = dtype
    if task_type == "text-classification" and is_regression:
        model_config["is_regression"] = True
    if use_late_interaction and task_type in ["sentence-transformers","sentence-similarity"]:
        model_config["use_late_interaction"] = True
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
        messages = [
            {"role": "user", "content": "test_prompt"},
            {"role": "assistant", "content": "test_response"}
        ]
        if not getattr(tokenizer, "chat_template", None):
            raise ValueError(
                "--use_chat_template requires the selected tokenizer to define a "
                "chat template. Use --force_separator for plain text pairs."
            )
        
        templated = tokenizer.apply_chat_template(messages, tokenize=False)
        print("Chat template working:", templated)

    # Training arguments
    training_args = TrainingArgs(
        batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps, 
        gradient_accumulation_dtype=gradient_accumulation_dtype,
        multi_precision=multi_precision,
        max_length= max_length if max_length else model.config.max_position_embeddings,
        resume_from_step=resume_from_step, # warmup will be ingnored if before this step and schedulers will only start after
        resume_from_checkpoint=resume_from_checkpoint,
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate, 
        weight_decay=weight_decay,
        freeze_embeddings=freeze_embeddings,
        warmup_ratio=warmup_ratio, # can use warmup_steps or warmup_ratio
        warmup_steps=warmup_steps, # if both set, warmup_steps will be used
        lr_scheduler_type=lr_scheduler_type, # would default to "constant", can also use "cosine_decay" or "linear_schedule"
        min_lr=min_lr,
        max_grad_norm=max_grad_norm,
        save_steps=save_steps,
        logging_steps=logging_steps, # will be adjusted to be multiple of gradient_accumulation_steps inside Trainer
        eval_batch_size=eval_batch_size,
        output_dir=output_dir,
        save_total_limit=None,
        grad_checkpoint=grad_checkpoint,
        cache_clear_steps=cache_clear_steps,
        push_to_hub=push_to_hub,
        hub_repo_id=hub_repo_id,
        base_model=base_model,
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
