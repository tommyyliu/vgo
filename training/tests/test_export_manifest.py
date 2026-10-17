from __future__ import annotations

import unittest

import onnx
from onnx import TensorProto, helper

from vgo_training.export_onnx import graph_inputs


class ExportManifestTests(unittest.TestCase):
    def test_describes_actual_packed_inputs_and_symbolic_shapes(self) -> None:
        inputs = [
            helper.make_tensor_value_info("bits", TensorProto.UINT8, ["batch", 3, 2048]),
            helper.make_tensor_value_info("dense", TensorProto.FLOAT16, ["batch", 1, 16384]),
            helper.make_tensor_value_info("scalars", TensorProto.FLOAT16, ["batch", 3]),
        ]
        model = helper.make_model(helper.make_graph([], "packed", inputs, []))
        descriptors = graph_inputs(model, 64)
        self.assertEqual([value["name"] for value in descriptors], ["bits", "dense", "scalars"])
        self.assertEqual([value["dtype"] for value in descriptors], ["uint8", "float16", "float16"])
        self.assertEqual(descriptors[0]["shape"], ["batch", 3, 2048])
        self.assertEqual(descriptors[0]["maximum_batch"], 64)

    def test_dense_descriptor_reads_graph_dtype_instead_of_assuming_float32(self) -> None:
        value = helper.make_tensor_value_info("states", TensorProto.FLOAT16, ["batch", 7, 128, 128])
        model = helper.make_model(helper.make_graph([], "dense", [value], []))
        self.assertEqual(graph_inputs(model, 32), [{
            "name": "states", "dtype": "float16", "shape": ["batch", 7, 128, 128],
            "minimum_batch": 1, "maximum_batch": 32,
        }])


if __name__ == "__main__":
    unittest.main()
