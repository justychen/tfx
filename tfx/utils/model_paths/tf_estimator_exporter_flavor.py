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
"""Modules for TensorFlow Estimator flavor model path.

TensorFlow Estimator
[Exporter](https://www.tensorflow.org/api_docs/python/tf/estimator/Exporter)
export the model under `{export_path}/export/{exporter_name}/{timestamp}`
directory. We call this a *TF-Estimator-flavored model path*.

Example:

```
gs://your_bucket_name/{export_path}/  # An `export_path`
  export/                             # Constant name "export"
    my_exporter/                      # An `exporter_name`
      1582072718/                     # UTC `timestamp` in seconds
        (Your exported SavedModel)
```
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
from typing import List, Text, Tuple

import tensorflow as tf

_EXPORT_SUB_DIR_NAME = 'export'
_TF_ESTIMATOR_EXPORT_MODEL_PATH_PATTERN = re.compile(
    r'^(?P<export_path>.*)/export/(?P<exporter_name>[^/]+)/(?P<timestamp>\d+)$')


def make_model_path(export_path: Text, exporter_name: Text,
                    timestamp: int) -> Text:
  """Make a TF-estimator-flavored model path.

  Args:
    export_path: An `export_path` specified for the Exporter.
    exporter_name: Name of the exporter.
    timestamp: A unix timestamp in seconds.

  Returns:
    `{export_path}/export/{exporter_name}/{timestamp}`.
  """
  return os.path.join(export_path, _EXPORT_SUB_DIR_NAME, exporter_name,
                      str(timestamp))


def lookup_model_paths(export_path: Text) -> List[Text]:
  """Lookup all model paths in an export path.

  Args:
    export_path: An export_path as defined from the module docstring.

  Raises:
    tf.errors.NotFoundError: If no models found in the export_path.

  Returns:
    A list of model_path.
  """
  export_sub_dir = os.path.join(export_path, _EXPORT_SUB_DIR_NAME)
  result = []
  for exporter_name in tf.io.gfile.listdir(export_sub_dir):
    model_sub_dir = os.path.join(export_sub_dir, exporter_name)
    if not tf.io.gfile.isdir(model_sub_dir):
      continue
    for timestamp in tf.io.gfile.listdir(model_sub_dir):
      if not timestamp.isdigit():
        continue
      model_path = os.path.join(model_sub_dir, timestamp)
      if tf.io.gfile.isdir(model_path):
        result.append(model_path)

  return result


def lookup_only_model_path(export_path: Text) -> Text:
  """Lookup the only model path in an export_path.

  Args:
    export_path: An export_path as defined from the module docstring.

  Raises:
    tf.errors.NotFoundError: If no models found in the export_path.

  Returns:
    The only model_path.
  """
  models_found = lookup_model_paths(export_path)
  if not models_found:
    raise tf.errors.NotFoundError(
        node_def=None, op=None,
        message='No model found in {}'.format(export_path))

  assert len(models_found) == 1, (
      'Multiple models found: {}'.format(models_found))
  return models_found[0]


def parse_model_path(model_path: Text) -> Tuple[Text, Text, int]:
  """Parse the model_path as a TF-estimator-flavored model path.

  Args:
    model_path: A path to the model.

  Raises:
    ValueError: If the model_path is not TF-estimator-flavored.

  Returns:
    (export_path, exporter_name, timestamp) tuple.
  """
  match = _TF_ESTIMATOR_EXPORT_MODEL_PATH_PATTERN.match(model_path)
  if not match:
    raise ValueError('{} does not match tensorflow estimator flavor.'.format(
        model_path))
  match_dict = match.groupdict()
  return (
      match_dict['export_path'],
      match_dict['exporter_name'],
      int(match_dict['timestamp'])
  )
