# Copyright 2024 Big Vision Authors.
# Copyright 2024 The Flax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""Utils for loading PaliGemma params."""

from collections.abc import Mapping
import collections
import dataclasses
import os
from typing import Any
import re

import jax
# import jax.numpy as jnp
import flax
import numpy as np
import ml_collections as mlc

Params = Mapping[str, Any]


def load_and_format_params(path: str) -> Params:
  """Loads parameters from an NPZ file and formats them for compatibility.
  """
  print(f"Loading parameters from {path}...")
  # params = load_params_from_npz(path) # <--- Modified: use the NPZ loader
  
  # Nest the parameters into the Flax/NNX PyTree structure
  # nested_params = nest_params(params)
  # return nested_params
  
  return load_params(path)


# TODO: reincorporate param_remapper into load_params function
def param_remapper(orig_params: Params) -> Params:
  """Remaps params to new module layout.

  This is needed here because the model definition  does not have a separate
  `mlp` module.

  Args:
    orig_params: original dict of parameters in Gemma format.

  Returns:
    dict of params with different names.
  """
  new_params = {}
  for k, v in orig_params.items():
    if 'mlp/' in k:
      layer_name, param = k.rsplit('/', maxsplit=1)
      if layer_name not in new_params:
        new_params[layer_name] = {}
      if 'w' in v:
        new_params[layer_name][param] = v['w']
    else:
      new_params[k] = v
  return new_params


# --- MODIFIED FUNCTION ---
# The original `load_params` used orbax.checkpoint.
# We need a new function or modify `load_params` to use np.load.
# @functools.cache
# def load_params_from_npz(path: str) -> Params:
#   """Loads parameters from a .npz file."""
#   print(f"Loading parameters from {path} using numpy.load...")
#   with jnp.load(path) as data:
#     loaded_params = {k: jnp.array(v) for k, v in data.items()}
#   print(f"Loaded {len(loaded_params)} arrays.")
#   return loaded_params


# def nest_params(params: Params) -> Params:
#   """Nests params as a dict of dicts rather than a flat dict."""
#   nested_params = {}
#   for path, param in params.items():
#     parts = path.split('/')
#     subdict = nested_params
#     for i, key in enumerate(parts):
#       if i == len(parts) - 1:  # Last part is the leaf (the actual parameter)
#         subdict[key] = param
#       else:
#         subdict = subdict.setdefault(key, {})
#   return nested_params



def npload(fname):
  """Loads `fname` and returns an np.ndarray or dict thereof."""
  if os.path.exists(fname):
    loaded = np.load(fname, allow_pickle=False)
  else:
    # If you want to support remote paths via gfile:
    # from tensorflow.io import gfile
    # with gfile.GFile(fname, "rb") as f:
    #     data = f.read()
    # loaded = np.load(io.BytesIO(data), allow_pickle=False)

    raise FileNotFoundError(
      f"File '{fname}' does not exist and remote loading is disabled.")

  # Handle both np.save (ndarray) and np.savez (zip dict)
  if isinstance(loaded, np.ndarray):
    return loaded
  elif isinstance(loaded, np.lib.npyio.NpzFile):
    return dict(loaded)
  else:
    raise TypeError(f"Unsupported type returned by np.load: {type(loaded)}")

def recover_tree(keys, values):
  """Recovers a tree as a nested dict from flat names and values.

  This function is useful to analyze checkpoints that are saved by our programs
  without need to access the exact source code of the experiment. In particular,
  it can be used to extract an reuse various subtrees of the scheckpoint, e.g.
  subtree of parameters.

  Args:
    keys: a list of keys, where '/' is used as separator between nodes.
    values: a list of leaf values.

  Returns:
    A nested tree-like dict.
  """
  tree = {}
  sub_trees = collections.defaultdict(list)
  for k, v in zip(keys, values):
    if "/" not in k:
      tree[k] = v
    else:
      k_left, k_right = k.split("/", 1)
      sub_trees[k_left].append((k_right, v))
  for k, kv_pairs in sub_trees.items():
    k_subtree, v_subtree = zip(*kv_pairs)
    tree[k] = recover_tree(k_subtree, v_subtree)
  return tree

def load_checkpoint_np(npz, tree=None):
  """Loads a jax pytree from a npz file.

  Args:
    npz: Either path to the checkpoint file (.npz), or a dict-like.
    tree: deprecated, use None.
      Bwd-compat for old format that only stored values: the pytree structure.

  Returns:
    A pytree that is the checkpoint.
  """
  if isinstance(npz, str):  # If not already loaded, then load.
    npz = npload(npz)
  #TODO: Temporarily only load img params
  filtered_items = {k: v for k, v in npz.items() if k.startswith("params/llm")}
  keys, values = zip(*filtered_items.items())
  if tree:
    checkpoint = tree.unflatten(values)
  else:
    checkpoint = recover_tree(keys, values)
  return checkpoint

def recover_dtype(a):
  """Numpy's `save` stores bfloat16 type as "void" type, so we recover it."""
  if hasattr(a, "dtype") and a.dtype.type is np.void:
    assert a.itemsize == 2, "Unknown dtype!"
    return a.view(jax.numpy.bfloat16)
  else:
    return a

def _traverse_with_names(tree, with_inner_nodes=False):
  """Traverses nested dicts/dataclasses and emits (leaf_name, leaf_val)."""
  if dataclasses.is_dataclass(tree):
    tree = flax.serialization.to_state_dict(tree)
  # Don't output the non-leaf nodes. If the optimizer doesn't have a state
  # the tree leaves can be Nones which was interpreted as a leaf by this
  # function but not by the other functions (like jax.tree.map).
  if tree is None:
    return
  elif isinstance(tree, Mapping):
    keys = sorted(tree.keys())
    for key in keys:
      for path, v in _traverse_with_names(tree[key], with_inner_nodes):
        yield (key + "/" + path).rstrip("/"), v
    if with_inner_nodes:
      yield "", tree
  elif isinstance(tree, (list, tuple)):
    for idx in range(len(tree)):
      for path, v in _traverse_with_names(tree[idx], with_inner_nodes):
        yield (str(idx) + "/" + path).rstrip("/"), v
    if with_inner_nodes:
      yield "", tree
  else:
    yield "", tree

def tree_get(tree, name):
  """Get an entry of pytree by flattened key name, eg a/b/c, with nice error.

  Args:
    tree: the pytree to be queried.
    name: the path to extract from the tree, see below for examples.

  Returns:
    A few examples:
      tree = {'a': 1, 'b': {'c': 2, 'd': 3}}
      tree_get(tree, 'a') == 1
      tree_get(tree, 'b/c') == 2
      tree_get(tree, 'b') == {'c': 2, 'd': 3}
  """
  flattened = dict(_traverse_with_names(tree, with_inner_nodes=True))
  try:
    return flattened[name]
  except KeyError as e:
    class Msg(str):  # Reason: https://stackoverflow.com/a/70114007/2366315
      def __repr__(self):
        return str(self)
    msg = "\n".join([name, "Available keys:", *flattened, ""])
    # Turn into configdict to use its "did you mean?" error message!
    msg = mlc.ConfigDict(flattened)._generate_did_you_mean_message(name, msg)  # pylint: disable=protected-access
    raise KeyError(Msg(msg)) from e


def load_params(ckpt, **kw):
  """Loads the parameters of a big_vision checkpoint, both old or new format.

  Args:
    ckpt: Path to the checkpoint (.npz, .ts) or dict-like.
    **kw: forwarded to the underlying load function (_np or _ts).

  Returns:
    A pytree that is the checkpoint, potentially sharded.

  Notes:
    The `ckpt` string can contain an colon-separated "submodel" indicator, like
    `img` in the example `/path/to/file.npz:img`.
    This is used to load sub-parts of a model, for example the image load the
    image encoder out of a two_tower (SigLIP) checkpoint, or distillation.
    This way, ANY model that uses this function can load itself from a
    checkpoint that contains multiple sub-models.
  """
  key = None  # Whether we want to extract only a sub-key of the model.

  if isinstance(ckpt, str):  # Most common case of passing a checkpoint path.
    # Potentially read out the sub-part to load from after the colon
    # '/path/to/file:img/head' => '/path/to/file', 'img/head'
    # 'gs://path/to/file' => 'gs://path/to/file', None
    if match := re.match(r"^(.*?/.*?)(?::([\w/]+))?$", ckpt):
      ckpt, key = match.groups()
    else:
      raise ValueError(f"Weird ckpt path: {ckpt} ; Maybe prepend ./ ?")

    # Use the checkpoint filename to detect when we're loading old-style .npz
    # checkpoints, as opposed to new-style tensorstore checkpoint folders.
    if ".npz" in ckpt:  # Not a perfect heuristic, but good enough.
      checkpoint = load_checkpoint_np(ckpt, **kw)
      checkpoint = jax.tree.map(recover_dtype, checkpoint)
      if "params" in checkpoint:
        # Checkpoint with optax state (after (internal link)).
        params = checkpoint["params"]
      elif "opt" in checkpoint:
        # Checkpoint with Flax optimizer.
        params = checkpoint["opt"]["target"]
      else:
        # When open-sourcing, we often shared only the params directly.
        params = checkpoint
    # else:
      # Here we're now loading new-style tensorstore checkpoints.
      # We can be a more efficient and load params and `key` only right away.
      # regex = f"params/{key}($|/.*)" if key else "params/.*"
      # assert "regex" not in kw, "For a custom regex, use tsload directly."
      # kw["regex"] = regex
      # checkpoint = load_checkpoint_ts(ckpt, **kw)
      # params = checkpoint["params"]
  if key is not None:
    params = tree_get(params, key)

  return params
