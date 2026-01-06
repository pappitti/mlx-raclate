import argparse
import time

from mlx_raclate.utils.utils import load, PIPELINES
from mlx_raclate.tuner.datasets import load_dataset, DatasetArgs
from mlx_raclate.tuner.trainer import Trainer, TrainingArgs

train_tested = {
    "text-classification": [
        {"model": "Qwen/Qwen3-Embedding-0.6B", "special_model_config" : {}, "special_trainer_config" : {"use_chat_template": True}, "special_training_args" : {"max_length":8192}},
        {"model": "answerdotai/ModernBERT-base", "special_model_config" : {}, "special_training_args" : {}},
        {"model": "LiquidAI/LFM2-350M", "special_model_config" : {"use_chat_template": True}, "special_training_args" : {}},
        {"model": "google/t5gemma-b-b-ul2", "special_model_config" : {}, "special_training_args" : {"max_length":8192}}, # failed
        {"model": "google/embeddinggemma-300m", "special_model_config" : {}, "special_training_args" : {}} # failed
    ],
}

# LFM2 CHAT TEMPLATE
FORCED_CHAT_TEMPLATE = """
{- bos_token -}}
{%- set system_prompt = "" -%}
{%- set ns = namespace(system_prompt="") -%}
{%- if messages[0]["role"] == "system" -%}
	{%- set ns.system_prompt = messages[0]["content"] -%}
	{%- set messages = messages[1:] -%}
{%- endif -%}
{%- if tools -%}
	{%- set ns.system_prompt = ns.system_prompt + ("\n" if ns.system_prompt else "") + "List of tools: <|tool_list_start|>[" -%}
	{%- for tool in tools -%}
		{%- if tool is not string -%}
            {%- set tool = tool | tojson -%}
		{%- endif -%}
		{%- set ns.system_prompt = ns.system_prompt + tool -%}
        {%- if not loop.last -%}
            {%- set ns.system_prompt = ns.system_prompt + ", " -%}
        {%- endif -%}
	{%- endfor -%}
	{%- set ns.system_prompt = ns.system_prompt + "]<|tool_list_end|>" -%}
{%- endif -%}
{%- if ns.system_prompt -%}
	{{- "<|im_start|>system\n" + ns.system_prompt + "<|im_end|>\n" -}}
{%- endif -%}
{%- for message in messages -%}
	{{- "<|im_start|>" + message["role"] + "\n" -}}
	{%- set content = message["content"] -%}
	{%- if content is not string -%}
		{%- set content = content | tojson -%}
	{%- endif -%}
	{%- if message["role"] == "tool" -%}
		{%- set content = "<|tool_response_start|>" + content + "<|tool_response_end|>" -%}
	{%- endif -%}
	{{- content + "<|im_end|>\n" -}}
{%- endfor -%}
{%- if add_generation_prompt -%}
	{{- "<|im_start|>assistant\n" -}}
{%- endif -%}
"""

DEFAULT_MODEL_PATH : str = "./trained_models/Qwen3-Embedding-0.6B_text-classification_20251219_001137/checkpoint-39939" #"Qwen/Qwen3-Embedding-0.6B" "answerdotai/ModernBERT-base" "google/t5gemma-b-b-ul2"
DEFAULT_DATASET : str = "data/wines" # can be a local path or HF "argilla/synthetic-domain-text-classification" "data/20251205_1125"
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

def init_args():
    parser = argparse.ArgumentParser(description="Train or evaluate a classification model using MLX Raclate.")
    # Dataset Init Params
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET, help="Local path or HF identifier of the dataset to use for training/evaluation.")
    parser.add_argument("--text_field", type=str, default=None, help="Name of the text field in the dataset (if different from default).")
    parser.add_argument("--text_pair_field", type=str, default=None, help="Name of the text pair field in the dataset (if applicable and different from default).")
    parser.add_argument("--label_field", type=str, default=None, help="Name of the label field in the dataset (if different from default).")
    parser.add_argument("--negative_field", type=str, default=None, help="Name of the negative samples field in the dataset (if applicable and different from default).")
    parser.add_argument("--create_test", action='store_true', help="Set this flag to create a test split, if not already present in the dataset, out of the training set (validation set not affected).")

    # Trainer / End Model Init Params
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH, help="Path to the pre-trained model or model identifier from a model hub.")
    parser.add_argument("--task_type", type=str, default=DEFAULT_TASK_TYPE, help="Type of task (default: text-classification).")
    parser.add_argument("--is_regression", default=False, action='store_true', help="Set this flag if the task is regression.")
    parser.add_argument("--use_late_interaction", default=False, action='store_true', help="Set this flag to use late interaction for retrieval tasks (if applicable).")
    parser.add_argument("--eval_only", dest="train", action='store_false', help="Set this flag to skip training and only evaluate.")
    parser.add_argument("--use_chat_template", default=False, action='store_true', help="Use chat template for decoder models when there are text pairs.")
    parser.add_argument("--force_separator", type=str, default=None, help="Force a specific separator between text pairs for decoder models, if not using chat template.")

    # Training Params
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE, help="Batch size for training.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=DEFAULT_GRADIENT_ACCUMULATION_STEPS, help="Number of gradient accumulation steps.")
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
    # Other Training Params
    parser.add_argument("--logging_steps", type=int, default=DEFAULT_LOGGING_STEPS, help="Number of steps between logging training metrics.")
    parser.add_argument("--save_steps", type=int, default=DEFAULT_SAVE_STEPS, help="Number of steps between model checkpoints.")
    parser.add_argument("--eval_batch_size", type=int, default=DEFAULT_EVAL_BATCH_SIZE, help="Batch size for evaluation.")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save model checkpoints and logs.")
    parser.set_defaults(train=True)
    return parser.parse_args()

def main():
    args = init_args()

    # Dataset Params
    dataset : str = args.dataset
    text_field : str = args.text_field
    text_pair_field : str = args.text_pair_field
    label_field : str = args.label_field
    negative_field : str = args.negative_field
    create_test : bool = args.create_test

    # Trainer / End Model Params
    model_path : str = args.model_path 
    task_type : str = args.task_type
    is_regression : bool = args.is_regression
    use_late_interaction : bool = args.use_late_interaction
    train : bool = args.train
    use_chat_template : bool = args.use_chat_template
    force_separator : str = args.force_separator

    # Training Params
    batch_size : int = args.batch_size
    gradient_accumulation_steps : int = args.gradient_accumulation_steps
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
    max_length : int = args.max_length
    freeze_embeddings : bool = args.freeze_embeddings
    max_grad_norm : float = args.max_grad_norm

    print(f"Training Mode : {train}")

    if task_type not in PIPELINES:
        raise ValueError(f"Task type {task_type} not supported. Choose from {PIPELINES.items()}")
    
    output_dir : str =  args.output_dir if args.output_dir else model_path.split("/")[-1] + "_" + task_type + "_" + time.strftime("%Y%m%d_%H%M%S")

    # Load datasets
    dataset_args = DatasetArgs(
        data=dataset, 
        task_type=task_type, 
        text_field=text_field,
        text_pair_field=text_pair_field,
        label_field=label_field,
        negative_field=negative_field,
        test=create_test
    )
    
    train_dataset, valid_dataset, test_dataset, id2label, label2id = load_dataset(dataset_args)

    model_config={}
    if task_type == "text-classification" and is_regression:
        model_config={"is_regression":True}
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
        if not getattr(tokenizer, "chat_template", None) and FORCED_CHAT_TEMPLATE:
            tokenizer.chat_template = FORCED_CHAT_TEMPLATE
        
        templated = tokenizer.apply_chat_template(messages, tokenize=False)
        print("Chat template working:", templated)

    # Training arguments
    training_args = TrainingArgs(
        batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps, 
        max_length= max_length if max_length else model.config.max_position_embeddings,
        resume_from_step=resume_from_step, # warmup will be ingnored if before this step and schedulers will only start after
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
