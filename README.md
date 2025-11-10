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
- To make the base Gemma model compatible with the official Kaggle weight files, I modified the implementation so that the parameter shapes and layouts matched the checkpoint format. Changes found in ```paligemma/gemma/```. 
    - Stacked parameter tensors: I restructured the linear and MLP projection weights (e.g., q_proj, k_proj, v_proj, o_proj, and gate_proj/up_proj) to be stacked into a single combined tensor. This matches how the official checkpoints store them — as a single concatenated or packed tensor per layer rather than separate per-projection parameters. Example found in ```params.md```
    - Adjusted parameter loading and mapping logic: I updated the parameter initialization and checkpoint-loading utilities to unstack (split) or stack (merge) weights during load/save, ensuring consistent alignment with the Kaggle format. I also added logic in params.py and the model constructor to detect the expected parameter structure and remap accordingly.
    - Updated module definitions: I refactored the affected layers in layers.py and modules.py so that the forward passes now reference slices of the stacked weight tensor instead of independent fields. This kept the model’s functionality unchanged while making the weight layout compatible with the pretrained checkpoint.
        
        For example:
        ```
        @nnx.split_rngs(splits=self.num_layers)
        @nnx.scan(in_axes=(nnx.Carry, 0, 0), out_axes=(nnx.Carry, 0))
        def forward_block(x, block: modules.Block, cache):
          new_cache, x = block(x, positions, cache, attention_mask)
          return x, new_cache
        ```
        vs.
        ```
        for i, layer in enumerate(self.layers):
        layer_name = f'layer_{i}'
        layer_cache = cache[layer_name] if cache else None
        layer_cache, x = layer(
          x,
          positions,
          layer_cache,
          attention_mask,
        )
        if cache is not None:
        new_cache[layer_name] = layer_cache 
        ```
        The former vectorizes the per-layer pass with nnx.scan, carries x cleanly, aligns a per-layer cache as a scanned input/output, and ensures deterministic RNG splitting—all of which stays compatible with stacked weights of the chosen npz weight formatting.
        
        The old code did the same work but in Python, with manual dict indexing and RNG management.

    - Minor interface and shape consistency fixes: I adjusted positional embeddings, normalization, and attention configurations to ensure shape broadcasting worked correctly with the new stacked parameter layout. I also added helper functions to restore these weights seamlessly.
- Implemented modular VisionTransformer architecture under src/models/vit:
    - VisionTransformer in Flax NNX, complete with parameter loading, attention, MLP
- Implemented modular PaliGemma architecture under src/models/paligemma/:
    - paligema/transformer.py — multimodal PaliGemmaTransformer combining ViT + Gemma.
        - A transformer that combines image embeddings and text tokens into one causal sequence, modifies attention masking to allow image-to-text flow, and reuses the Gemma core architecture (RMSNorm, FeedForward, Attention, cache) for efficient image-conditioned text generation.
    - paligema/sampler.py — 
        - Prefill Cache
        Original (Gemma): No separate “prefill pass.” It runs a single while-loop from step 0 and, for the prompt portion, it simply copies the next prompt token instead of sampling. Concretely, it uses jnp.where(decoding_step < num_input_tokens-1, prompt[:, t+1], sampled) so the loop doubles as both “prefill” and “decode.”
        (PaliGemma): Added a prefill phase via _prefill_cache(...). It encodes the image prefix, builds prompt matrices/lengths, and warms the cache, then you start decoding at start_step = max_prompt_len - 1 with the warmed cache and image embeddings zimg.
        - Attention mask
        Original: Plain causal mask for text; it slices the visible window and inverts to a keep-mask per step.
        (PaliGemma): Introduced _compute_paligemma_attention_mask that prepends always-visible image tokens and then applies causal text visibility, producing a [B,1, Zi+S] keep-mask (image visible + causal text).
        
## Current State of the Code:
- Functional, refactored implementation of Gemma, Vision Transformer (ViT)
- PaliGemma integration in progress — architecture implemented but not yet fully verified
- Supports loading pretrained weights into all model variants
- Includes comprehensive test coverage for the models and the sampler
  
## What remains to be done:
- Implement sowing of intermediate activations in the Vision Transformer (ViT)
- Extend configuration options for the PaliGemma model
- Create example notebooks demonstrating Gemma and PaliGemma usage
- Add an example notebook for capturing and visualizing model intermediates

## Acknowledgments:
Huge thanks to Mayuresh for providing continuous support throughout!
    
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
