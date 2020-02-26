# Lint as: python2, python3
# Copyright 2020 Google LLC. All Rights Reserved.
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
"""Module for TFMA EvalSavedModel flavor model path.

TensorFlow Model Analysis (TFMA) export a model's evaluation graph to a special
[EvalSavedModel](https://www.tensorflow.org/tfx/model_analysis/eval_saved_model)
format under the directory {export_dir_base}/{timestamp}. We call this a
*TFMA-EvalSavedModel-flavored model path*.

Example:

```
gs://your_bucket_name/eval/   # An `export_dir_base`
  1582072718/                 # UTC `timestamp` in seconds
    (Your exported EvalSavedModel)
```
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
from typing import List, Text, Tuple

import tensorflow as tf

_TFMA_EVAL_SAVED_MODEL_PATH_PATTERN = re.compile(
    r'^(?P<export_dir_base>.*)/(?P<timestamp>\d+)$')


def make_model_path(export_dir_base: Text, timestamp: int) -> Text:
  """Make a TFMA-EvalSavedModel-flavored model path.

  Args:
    export_dir_base: An `export_dir_base` parameter for
        `tfma.export.export_eval_savedmodel()` call.
    timestamp: A unix timestamp in seconds.

  Returns:
    `{export_dir_base}/{timestamp}`.
  """
  return os.path.join(export_dir_base, str(timestamp))


def lookup_model_paths(export_dir_base: Text) -> List[Text]:
  """Lookup all model paths in an export_dir_base.

  Args:
    export_dir_base: An export_dir_base as defined from the module docstring.

  Raises:
    tf.errors.NotFoundError: If no models found in the export_dir_base.

  Returns:
    A list of model_path.
  """
  result = []
  for timestamp in tf.io.gfile.listdir(export_dir_base):
    if not timestamp.isdigit():
      continue
    model_path = os.path.join(export_dir_base, timestamp)
    if tf.io.gfile.isdir(model_path):
      result.append(model_path)
  return result


def lookup_only_model_path(export_dir_base: Text) -> Text:
  """Lookup the only model path in an export_dir_base.

  Args:
    export_dir_base: An export_dir_base as defined from the module docstring.

  Raises:
    tf.errors.NotFoundError: If no models found in the export_dir_base.

  Returns:
    The only model_path.
  """
  models_found = lookup_model_paths(export_dir_base)
  if not models_found:
    raise tf.errors.NotFoundError(
        node_def=None, op=None,
        message='No model found in {}'.format(export_dir_base))

  assert len(models_found) == 1, (
      'Multiple models found: {}'.format(models_found))
  return models_found[0]


def parse_model_path(model_path: Text) -> Tuple[Text, int]:
  """Parse the model_path as a TFMA-EvalSavedModel-flavored model path.

  Args:
    model_path: A path to the model.

  Raises:
    ValueError: If the model_path is not TFMA-EvalSavedModel-flavored.

  Returns:
    (export_dir_base, timestamp) tuple.
  """
  match = _TFMA_EVAL_SAVED_MODEL_PATH_PATTERN.match(model_path)
  if not match:
    raise ValueError('{} does not match TFMA EvalSavedModel flavor.'.format(
        model_path))
  match_dict = match.groupdict()
  return match_dict['export_dir_base'], int(match_dict['timestamp'])
