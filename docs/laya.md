# Optional Laya adapter

This repository contains a **small optional adapter** for
[Laya](https://github.com/NandhaKishorM/laya), a non-autoregressive typed
decision engine. The adapter is an example of how a caller can turn Laya's
`score` outputs into the nine-field `obs_schema` used by this project.

## Upstream attribution

Laya is a project by **Convai Innovations**. The Python package metadata
identifies `Author: Convai Innovations` and `License: Apache-2.0`.

Official upstream links:

- GitHub: https://github.com/NandhaKishorM/laya
- PyPI: https://pypi.org/project/laya/
- Model hub: https://huggingface.co/convaiinnovations/laya
- Multilingual checkpoint: https://huggingface.co/convaiinnovations/laya-multilingual
- Demo: https://huggingface.co/spaces/convaiinnovations/laya-demo
- Apache-2.0: https://www.apache.org/licenses/LICENSE-2.0

Please check the upstream repository and model card for the current release,
license, citation, and usage terms. This project does not claim authorship of
Laya and does not redistribute Laya source code, weights, or checkpoints.

## What this repository provides

Only an optional adapter:

```text
examples/laya_obs_adapter.py
examples/obs_schema.py
```

It imports the upstream `laya` package, sends the nine observation questions in
one typed decision request, and prints the returned scores.

The adapter is **not** a fine-tuned observation model. Laya's base/checkpoint
output is a starting point; its behavior on your event schema should be measured
on your own data. If you need a production extractor, fine-tune a checkpoint
using your own properly licensed labels and keep the evaluation set separate.

## Install and run

Install Laya from its official source:

```bash
python -m pip install laya
```

Then run the adapter:

```bash
python examples/laya_obs_adapter.py \
  --model convaiinnovations/laya \
  --subfolder multilingual \
  --device cpu \
  --text "今天在咖啡馆和朋友聊到了周末的计划"
```

For a local checkpoint that you have fine-tuned yourself:

```bash
python examples/laya_obs_adapter.py \
  --local-path /path/to/your/checkpoint \
  --device cuda \
  --text "..."
```

The optional dependency is declared as:

```bash
python -m pip install -e ".[laya]"
```

## Boundaries

- No Laya model weights, training data, or API keys are included here.
- Do not redistribute third-party weights unless the upstream license allows it.
- Do not describe this adapter as an official Laya release or imply endorsement.
- The nine-field schema is this repository's example, not a claim about the
  downstream project.

