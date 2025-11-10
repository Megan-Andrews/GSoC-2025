# PaliGemma (Flax / JAX / NNX)
An educational and test-covered Flax/NNX implementation of PaliGemma, Google’s vision-language model combining a ViT encoder and a Gemma-style decoder.

## Overview

PaliGemma integrates a Vision Transformer (ViT) with a Gemma language model to process images and text jointly. 
The codebase is organized into three top-level components under the paligemma/ package:
| Directory               | Purpose                                                                                                                                                                                                                 |
| :---------------------- | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **`paligemma/gemma/`**  | The **Gemma language model (decoder)** implementation, including transformer layers, attention, MLP, sowing utilities, and a text sampler. This subpackage mirrors Google’s Gemma architecture but written in Flax NNX. |
| **`paligemma/vit/`**    | The **Vision Transformer (ViT)** encoder, used to embed image patches before concatenation with text tokens. Includes its own transformer and module tests.                                                             |
| **`paligemma/` (root)** | The **multimodal integration layer**, combining ViT and Gemma into `PaliGemmaTransformer`, along with a unified `Sampler`, `sow_lib.py`, and parameter conversion utilities.                                            |

### The architecture supports:

- Loading pretrained .npz weights (e.g., from Kaggle’s paligemma-3b-pt-224)
- Multimodal fusion (image tokens + text tokens)
- Efficient decoding with attention cache prefill
- Optional sowing of embeddings and intermediate activations for interpretability

## What I've Done:
- Implemented modular PaliGemma architecture under src/models/paligemma/:
    - paligema/transformer.py — multimodal PaliGemmaTransformer combining ViT + Gemma.
    - paligema/sampler.py — 
- To make the base Gemma model compatible with the official Kaggle weight files, I modified the implementation so that the parameter shapes and layouts matched the checkpoint format. Found in ```paligemma/gemma/```
    - Stacked parameter tensors: I restructured the linear and MLP projection weights (e.g., q_proj, k_proj, v_proj, o_proj, and gate_proj/up_proj) to be stacked into a single combined tensor. This matches how the official checkpoints store them — as a single concatenated or packed tensor per layer rather than separate per-projection parameters.
    - Adjusted parameter loading and mapping logic: I updated the parameter initialization and checkpoint-loading utilities to unstack (split) or stack (merge) weights during load/save, ensuring consistent alignment with the Kaggle format. I also added logic in params.py and the model constructor to detect the expected parameter structure and remap accordingly.
    - Updated module definitions: I refactored the affected layers in layers.py and modules.py so that the forward passes now reference slices of the stacked weight tensor instead of independent fields. This kept the model’s functionality unchanged while making the weight layout compatible with the pretrained checkpoint.
    - Minor interface and shape consistency fixes: I adjusted positional embeddings, normalization, and attention configurations to ensure shape broadcasting worked correctly with the new stacked parameter layout. I also added helper functions to restore these weights seamlessly.

```bash
notebooks/                    # Jupyter notebooks for demos, experiments, or visualization
│
src/
└── models/
    ├── gemma/                # Original Gemma implementation for reference
    │
    └── paligemma/
        ├── gemma/
        │   ├── helpers.py
        │   ├── helpers_test.py
        │   ├── layers.py
        │   ├── layers_test.py
        │   ├── modules.py
        │   ├── modules_test.py
        │   ├── params.py
        │   ├── positional_embeddings.py
        │   ├── positional_embeddings_test.py
        │   ├── sampler.py
        │   ├── sampler_test.py
        │   ├── sow_lib.py
        │   ├── transformer.py
        │   ├── transformer_test.py
        │   └── README.md
        │
        ├── vit/
        │   ├── helpers.py
        │   ├── helpers_test.py
        │   ├── modules.py
        │   ├── modules_test.py
        │   ├── params.py
        │   ├── transformer.py
        │   └── transformer_test.py
        │
        ├── helpers.py
        ├── helpers_test.py
        ├── params.py
        ├── sampler.py
        ├── sampler_test.py
        ├── sow_lib.py
        ├── temp.md
        ├── test_all.py
        ├── transformer.py
        └── transformer_test.py
```
