# Repository tools

The installed `raclate` command is the supported interface for training,
serving, and reusable NER workflows:

```shell
raclate train --help
raclate serve --help
raclate ner evaluate --help
raclate ner calibrate --help
```

The scripts in this directory prepare specific source datasets and intentionally stay
repository-local because their schemas and label mappings are not generic
library APIs:

```shell
uv run python tools/prepare_finance_crm_ner_dataset.py --help
uv run python tools/prepare_openai_nemotron_dataset.py --help
```

Reusable behavior belongs under `src/mlx_raclate`; repository scripts should
remain dataset-specific, accept normal CLI arguments, and have focused tests
when they contain transformation logic.
