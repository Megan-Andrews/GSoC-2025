import unittest

from gemma import transformer_test
from gemma import sampler_test
from gemma import helpers_test
from gemma import layers_test
from gemma import modules_test
from gemma import positional_embeddings_test

from vit import transformer_test as vision_transformer_test

import transformer_test as paligemma_transformer_test
import sampler_test as paligemma_sampler_test

if __name__ == '__main__':
  loader = unittest.TestLoader()
  runner = unittest.TextTestRunner()

  suite = unittest.TestSuite()

  # Load all tests from each module
  for mod in [
      # transformer_test,
      # sampler_test,
      # helpers_test,
      # layers_test,
      # modules_test,
      # positional_embeddings_test,
      # vision_transformer_test,
      # paligemma_transformer_test
      paligemma_sampler_test
  ]:
    suite.addTests(loader.loadTestsFromModule(mod))

  runner.run(suite)
