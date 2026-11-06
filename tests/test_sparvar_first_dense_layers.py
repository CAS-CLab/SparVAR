"""Check the image CS4A dense-layer guard without importing CUDA kernels."""

import ast
from pathlib import Path
import unittest


class SparsePathReached(Exception):
    pass


class FirstDenseLayersTests(unittest.TestCase):
    def setUp(self):
        path = Path(__file__).resolve().parents[1] / 'models/sparvar/sparse_attn.py'
        module = ast.parse(path.read_text())
        cls = next(n for n in module.body if isinstance(n, ast.ClassDef) and n.name == 'SparseDiffAttn')
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '_fast_attention_image')
        method.decorator_list = []  # Exercise the body without the CUDA compilation wrapper.
        self.config = {'attn': {'first_n_dense_layers': 2}}
        self.calls = []
        self.output = object()

        def dense(q, k, v, *, scale):
            self.calls.append((q, k, v, scale))
            return self.output, None

        def sparse_setup():
            raise SparsePathReached

        scope = dict(Tensor=object, GLOBAL_CONFIG=self.config, dense_attn=dense,
                     get_kernel_config_attn=sparse_setup)
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), 'exec'), scope)
        self.forward = scope['_fast_attention_image']
        self.q, self.k, self.v = object(), object(), object()

    def run_layer(self, layer, scale):
        return self.forward(None, self.q, self.k, self.v, 0.125, 0, scale == 10, scale, layer)

    def test_first_two_layers_skip_decision_and_sparse_work(self):
        for scale in (10, 11, 12):
            for layer in (0, 1):
                self.assertIs(self.run_layer(layer, scale), self.output)
        self.assertEqual(self.calls, [(self.q, self.k, self.v, 0.125)] * 6)

    def test_other_layers_retain_cs4a_path(self):
        for layer in range(2, 40):
            for scale in (10, 11, 12):
                with self.assertRaises(SparsePathReached):
                    self.run_layer(layer, scale)
        self.assertEqual(self.calls, [])

    def test_zero_disables_dense_protection(self):
        self.config['attn']['first_n_dense_layers'] = 0
        with self.assertRaises(SparsePathReached):
            self.run_layer(0, 10)
        self.assertEqual(self.calls, [])


if __name__ == '__main__':
    unittest.main()
