PaliGemma (Flax / JAX / NNX)
An educational and test-covered Flax/NNX implementation of PaliGemma, Google’s vision-language model combining a ViT encoder and a Gemma-style decoder.

Overview

PaliGemma integrates a Vision Transformer (ViT) with a Gemma language model to process images and text jointly. 
The codebase is organized into three top-level components under the paligemma/ package:
| Directory               | Purpose                                                                                                                                                                                                                 |
| :---------------------- | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **`paligemma/gemma/`**  | The **Gemma language model (decoder)** implementation, including transformer layers, attention, MLP, sowing utilities, and a text sampler. This subpackage mirrors Google’s Gemma architecture but written in Flax NNX. |
| **`paligemma/vit/`**    | The **Vision Transformer (ViT)** encoder, used to embed image patches before concatenation with text tokens. Includes its own transformer and module tests.                                                             |
| **`paligemma/` (root)** | The **multimodal integration layer**, combining ViT and Gemma into `PaliGemmaTransformer`, along with a unified `Sampler`, `sow_lib.py`, and parameter conversion utilities.                                            |

notebooks/           # Jupyter notebooks for demos, experiments, or visualization
│
src/
└── models/
    ├── gemma/       # Orignial Gemma implementation for reference
    └── paligemma/        
        ├── gemma/
        │   ├── debug.py
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
        ├── params.md
        ├── params.py
        ├── sampler.py
        ├── sampler_test.py
        ├── sow_lib.py
        ├── temp.md
        ├── test_all.py
        ├── transformer.py
        └── transformer_test.py

