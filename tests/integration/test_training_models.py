"""Downloaded-checkpoint training matrix across supported model families."""

import pytest
from datasets import Dataset as HFDataset

from tests.fixtures import (
    generate_classification_dataset,
    generate_id2label,
    generate_label2id,
    generate_mlm_dataset,
    generate_ner_dataset,
    generate_similarity_dataset,
    generate_triplet_dataset,
)
from tests.model_registry import BASE_MODELS, MODEL_FAMILIES, get_trainable_pipelines


pytestmark = [pytest.mark.integration, pytest.mark.slow]


def _get_training_params():
    return [
        pytest.param(family, pipeline, id=f"{family}-{pipeline}")
        for family in MODEL_FAMILIES
        for pipeline in get_trainable_pipelines(family)
    ]


def _get_dataset_for_pipeline(pipeline: str, triplet: bool = False):
    if pipeline == "text-classification":
        data = generate_classification_dataset(n_samples=16, n_classes=3)
        return HFDataset.from_dict(data), generate_id2label(3), generate_label2id(3)
    if pipeline == "sentence-similarity":
        data = (
            generate_triplet_dataset(n_samples=16)
            if triplet
            else generate_similarity_dataset(n_samples=16)
        )
        return HFDataset.from_dict(data), None, None
    if pipeline == "masked-lm":
        return HFDataset.from_dict(generate_mlm_dataset(n_samples=16)), None, None
    if pipeline == "token-classification":
        data = generate_ner_dataset(n_samples=16, n_tags=5)
        id2label = {str(index): f"TAG_{index}" for index in range(5)}
        label2id = {label: int(index) for index, label in id2label.items()}
        return HFDataset.from_dict(data), id2label, label2id
    raise ValueError(f"Unknown trainable pipeline: {pipeline}")


def _ensure_mask_token(model, tokenizer):
    if getattr(tokenizer, "mask_token_id", None) is not None:
        return
    for token in ("<mask >", "[MASK]", "<mask>"):
        if token in tokenizer.vocab:
            tokenizer.mask_token = token
            tokenizer.mask_token_id = tokenizer.vocab[token]
            return
    tokenizer.add_tokens(["[MASK]"], special_tokens=True)
    tokenizer.mask_token = "[MASK]"
    tokenizer.mask_token_id = tokenizer.convert_tokens_to_ids("[MASK]")
    if hasattr(model, "resize_token_embeddings"):
        model.resize_token_embeddings(len(tokenizer))


@pytest.mark.parametrize("model_family,pipeline", _get_training_params())
def test_training_step(model_family: str, pipeline: str):
    from mlx_raclate import load
    from mlx_raclate.tuner.trainer import Trainer, TrainingArgs

    def run_step(triplet: bool = False):
        dataset, id2label, label2id = _get_dataset_for_pipeline(pipeline, triplet)
        model_config = {}
        if id2label:
            model_config.update(id2label=id2label, label2id=label2id)
        model, tokenizer = load(
            BASE_MODELS[model_family],
            model_config=model_config,
            pipeline=pipeline,
            train=True,
        )
        if pipeline == "masked-lm":
            _ensure_mask_token(model, tokenizer)

        args = TrainingArgs(
            batch_size=2,
            num_train_epochs=1,
            max_length=64,
            learning_rate=1e-5,
            logging_steps=1,
            save_steps=10000,
            output_dir=f"/tmp/test_training_{model_family}_{pipeline}",
        )
        trainer = Trainer(
            model=model,
            tokenizer=tokenizer,
            task_type=pipeline,
            training_args=args,
            train_dataset=dataset,
            label2id=label2id,
        )
        batches = list(trainer._create_batches(dataset, 2, shuffle=True))[:2]
        for batch in batches:
            assert model(**trainer.data_collator(batch)).get("loss") is not None

    run_step()
    if pipeline == "sentence-similarity":
        run_step(triplet=True)
