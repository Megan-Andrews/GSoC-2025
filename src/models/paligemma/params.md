npz file:
Successfully loaded /root/.cache/kagglehub/models/google/paligemma/JAX/paligemma-3b-pt-224/1/paligemma-3b-pt-224.npz
Contents of the .npz file:
  Array Name: 'params/img/Transformer/encoder_norm/bias', Shape: (1152,), Dtype: float32
  Array Name: 'params/img/Transformer/encoder_norm/scale', Shape: (1152,), Dtype: float32
  Array Name: 'params/img/Transformer/encoderblock/LayerNorm_0/bias', Shape: (27, 1152), Dtype: float32
  Array Name: 'params/img/Transformer/encoderblock/LayerNorm_0/scale', Shape: (27, 1152), Dtype: float32
  Array Name: 'params/img/Transformer/encoderblock/LayerNorm_1/bias', Shape: (27, 1152), Dtype: float32
  Array Name: 'params/img/Transformer/encoderblock/LayerNorm_1/scale', Shape: (27, 1152), Dtype: float32
  Array Name: 'params/img/Transformer/encoderblock/MlpBlock_0/Dense_0/bias', Shape: (27, 4304), Dtype: float32
  Array Name: 'params/img/Transformer/encoderblock/MlpBlock_0/Dense_0/kernel', Shape: (27, 1152, 4304), Dtype: float32
  Array Name: 'params/img/Transformer/encoderblock/MlpBlock_0/Dense_1/bias', Shape: (27, 1152), Dtype: float32
  Array Name: 'params/img/Transformer/encoderblock/MlpBlock_0/Dense_1/kernel', Shape: (27, 4304, 1152), Dtype: float32
  Array Name: 'params/img/Transformer/encoderblock/MultiHeadDotProductAttention_0/key/bias', Shape: (27, 16, 72), Dtype: float32
  Array Name: 'params/img/Transformer/encoderblock/MultiHeadDotProductAttention_0/key/kernel', Shape: (27, 1152, 16, 72), Dtype: float32
  Array Name: 'params/img/Transformer/encoderblock/MultiHeadDotProductAttention_0/out/bias', Shape: (27, 1152), Dtype: float32
  Array Name: 'params/img/Transformer/encoderblock/MultiHeadDotProductAttention_0/out/kernel', Shape: (27, 16, 72, 1152), Dtype: float32
  Array Name: 'params/img/Transformer/encoderblock/MultiHeadDotProductAttention_0/query/bias', Shape: (27, 16, 72), Dtype: float32
  Array Name: 'params/img/Transformer/encoderblock/MultiHeadDotProductAttention_0/query/kernel', Shape: (27, 1152, 16, 72), Dtype: float32
  Array Name: 'params/img/Transformer/encoderblock/MultiHeadDotProductAttention_0/value/bias', Shape: (27, 16, 72), Dtype: float32
  Array Name: 'params/img/Transformer/encoderblock/MultiHeadDotProductAttention_0/value/kernel', Shape: (27, 1152, 16, 72), Dtype: float32
  
  Array Name: 'params/img/embedding/bias', Shape: (1152,), Dtype: float32
  Array Name: 'params/img/embedding/kernel', Shape: (14, 14, 3, 1152), Dtype: float32
  Array Name: 'params/img/head/bias', Shape: (2048,), Dtype: float32
  Array Name: 'params/img/head/kernel', Shape: (1152, 2048), Dtype: float32
  Array Name: 'params/img/pos_embedding', Shape: (1, 256, 1152), Dtype: float32




  Array Name: 'params/llm/embedder/input_embedding', Shape: (257152, 2048), Dtype: float32
  Array Name: 'params/llm/final_norm/scale', Shape: (2048,), Dtype: float32
  Array Name: 'params/llm/layers/attn/attn_vec_einsum/w', Shape: (18, 8, 256, 2048), Dtype: float32
  Array Name: 'params/llm/layers/attn/kv_einsum/w', Shape: (18, 2, 1, 2048, 256), Dtype: float32
  Array Name: 'params/llm/layers/attn/q_einsum/w', Shape: (18, 8, 2048, 256), Dtype: float32
  Array Name: 'params/llm/layers/mlp/gating_einsum', Shape: (18, 2, 2048, 16384), Dtype: float32
  Array Name: 'params/llm/layers/mlp/linear', Shape: (18, 16384, 2048), Dtype: float32
  Array Name: 'params/llm/layers/pre_attention_norm/scale', Shape: (18, 2048), Dtype: float32
  Array Name: 'params/llm/layers/pre_ffw_norm/scale', Shape: (18, 2048), Dtype: float32



classifier/bias
'classifier', 'kernel
'cls_token',
dropout', 'rngs', 'default', 'count'
dropout', 'rngs', 'default', 'key'
encoderblock', 'attn', 'key', 'bias'
encoderblock', 'attn', 'key', 'kernel
encoderblock', 'attn', 'out', 'bias'
encoderblock', 'attn', 'out', 'kernel
encoderblock', 'attn', 'query', 'bias'
encoderblock', 'attn', 'query', 'kernel'
encoderblock', 'attn', 'rngs', 'default', 'count'
'encoderblock', 'attn', 'rngs', 'default', 'key'
'encoderblock', 'attn', 'value', 'bias'
'encoderblock', 'attn', 'value', 'kernel'
'encoderblock', 'mlp', 'layers', 0, 'bias'
encoderblock', 'mlp', 'layers', 0, 'kernel
encoderblock', 'mlp', 'layers', 3, 'bias'
encoderblock', 'mlp', 'layers', 3, 'kernel'
encoderblock', 'norm1', 'bias'
encoderblock', 'norm1', 'scale'
encoderblock', 'norm2', 'bias'
encoderblock', 'norm2', 'scale'
final_norm', 'bias'
'final_norm', 'scale
'patch_embeddings', 'bias'
'patch_embeddings', 'kernel'
('position_embeddings'





transformer/embedder/input_embedding

transformer/final_norm/scale

transformer/layer_0/attn/attn_vec_einsum/w

transformer/layer_0/attn/kv_einsum/w

transformer/layer_0/attn/q_einsum/w

transformer/layer_0/mlp/gating_einsum

transformer/layer_0/mlp/linear

transformer/layer_0/pre_attention_norm/scale

transformer/layer_0/pre_ffw_norm/scale