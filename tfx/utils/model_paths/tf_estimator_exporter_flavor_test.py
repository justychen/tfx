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
"""Tests for tfx.utils.model_paths.tf_estimator_exporter_flavor."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf

from tfx.utils.model_paths import tf_estimator_exporter_flavor


class TFEstimatorExporterFlavorTest(tf.test.TestCase):

  def _SetupSingleModel(self):
    export_path = self.get_temp_dir()
    tf.io.gfile.makedirs(
        tf_estimator_exporter_flavor.make_model_path(
            export_path=export_path,
            exporter_name='my_exporter',
            timestamp=1582798459))
    return export_path

  def _SetupMultipleModels(self):
    export_path = self.get_temp_dir()
    tf.io.gfile.makedirs(
        tf_estimator_exporter_flavor.make_model_path(
            export_path=export_path,
            exporter_name='first_exporter',
            timestamp=1582798459))
    tf.io.gfile.makedirs(
        tf_estimator_exporter_flavor.make_model_path(
            export_path=export_path,
            exporter_name='second_exporter',
            timestamp=1582858365))
    return export_path

  def testLookupModelPaths_ForSingleModel(self):
    export_path = self._SetupSingleModel()

    model_paths = tf_estimator_exporter_flavor.lookup_model_paths(
        export_path=export_path)

    self.assertEqual(len(model_paths), 1)
    model_path, exporter_name, timestamp = model_paths[0]
    self.assertEqual(os.path.relpath(model_path, export_path),
                     'export/my_exporter/1582798459')
    self.assertEqual(exporter_name, 'my_exporter')
    self.assertEqual(timestamp, 1582798459)

  def testLookupModelPaths_ForMultipleModels(self):
    export_path = self._SetupMultipleModels()

    model_paths = tf_estimator_exporter_flavor.lookup_model_paths(
        export_path=export_path)

    self.assertEqual(len(model_paths), 2)
    mp1, mp2 = sorted(model_paths)

    model_path, exporter_name, timestamp = mp1
    self.assertEqual(os.path.relpath(model_path, export_path),
                     'export/first_exporter/1582798459')
    self.assertEqual(exporter_name, 'first_exporter')
    self.assertEqual(timestamp, 1582798459)

    model_path, exporter_name, timestamp = mp2
    self.assertEqual(os.path.relpath(model_path, export_path),
                     'export/second_exporter/1582858365')
    self.assertEqual(exporter_name, 'second_exporter')
    self.assertEqual(timestamp, 1582858365)

  def testLookupModelPaths_InvalidPattern(self):
    # Setup invalid directory
    export_path = self.get_temp_dir()
    tf.io.gfile.makedirs(os.path.join(export_path, 'foo', 'bar', 'baz'))

    with self.assertRaises(tf.errors.NotFoundError):
      tf_estimator_exporter_flavor.lookup_model_paths(export_path=export_path)

  def testLookupOnlyModelPath(self):
    export_path = self._SetupSingleModel()

    model_path, exporter_name, timestamp = (
        tf_estimator_exporter_flavor.lookup_only_model_path(
            export_path=export_path))

    self.assertEqual(os.path.relpath(model_path, export_path),
                     'export/my_exporter/1582798459')
    self.assertEqual(exporter_name, 'my_exporter')
    self.assertEqual(timestamp, 1582798459)

  def testLookupOnlyModelPath_FailIfNoModel(self):
    # Setup no model
    with self.assertRaises(tf.errors.NotFoundError):
      tf_estimator_exporter_flavor.lookup_only_model_path(
          export_path=self.get_temp_dir())

  def testLookupOnlyModelPath_FailIfMultipleModels(self):
    export_path = self._SetupMultipleModels()

    with self.assertRaises(AssertionError):
      tf_estimator_exporter_flavor.lookup_only_model_path(
          export_path=export_path)

  def testParseModelPath(self):
    self.assertEqual(
        tf_estimator_exporter_flavor.parse_model_path(
            '/foo/bar/export/my_exporter/1582798459'),
        ('/foo/bar', 'my_exporter', 1582798459))

    # Invalid (non-digit) timestamp segment
    with self.assertRaises(ValueError):
      tf_estimator_exporter_flavor.parse_model_path(
          '/foo/bar/export/my_exporter/not-a-timestamp')

    # No timestamp segment
    with self.assertRaises(ValueError):
      tf_estimator_exporter_flavor.parse_model_path(
          '/foo/bar/export/my_exporter')

    # No exporter_name segment
    with self.assertRaises(ValueError):
      tf_estimator_exporter_flavor.parse_model_path(
          '/foo/bar/export/1582798459')

    # No "export" segment
    with self.assertRaises(ValueError):
      tf_estimator_exporter_flavor.parse_model_path(
          '/foo/bar/my_exporter/1582798459')


if __name__ == '__main__':
  tf.test.main()
