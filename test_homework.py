"""Simple self-checks for Lesson 1, without external test packages."""

import math
import unittest

import torch

from homework_tensors import make_tensors, matrix_operations, extract_slices, reshape_24
from homework_autograd import three_variable_gradients, mse_gradients, chain_rule_gradient
from homework_performance import benchmark_operations, cpu_time_ms


class TensorHomeworkTests(unittest.TestCase):
    def test_creation(self):
        result = make_tensors()
        self.assertEqual(tuple(result["random_3x4"].shape), (3, 4))
        self.assertTrue(torch.all((result["random_3x4"] >= 0) & (result["random_3x4"] <= 1)))
        self.assertTrue(torch.equal(result["zeros_2x3x4"], torch.zeros(2, 3, 4)))
        self.assertTrue(torch.equal(result["ones_5x5"], torch.ones(5, 5)))
        self.assertTrue(torch.equal(result["numbers_4x4"].flatten(), torch.arange(16)))

    def test_operations(self):
        a = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        b = torch.arange(12, dtype=torch.float32).reshape(4, 3)
        r = matrix_operations(a, b)
        self.assertTrue(torch.equal(r["a_transposed"], a.T))
        self.assertTrue(torch.equal(r["matrix_product"], a @ b))
        self.assertTrue(torch.equal(r["elementwise_product"], a * b.T))
        self.assertEqual(r["sum_of_a"].item(), 66)
        with self.assertRaises(ValueError):
            matrix_operations(torch.ones(4, 4), b)

    def test_slices(self):
        t = torch.arange(125).reshape(5, 5, 5)
        r = extract_slices(t)
        self.assertTrue(torch.equal(r["first_row_each_layer"], t[:, 0, :]))
        self.assertTrue(torch.equal(r["last_column_each_layer"], t[:, :, -1]))
        self.assertEqual(tuple(r["central_2x2_each_layer"].shape), (5, 2, 2))
        self.assertTrue(torch.equal(r["even_indices_all_axes"], t[::2, ::2, ::2]))
        with self.assertRaises(ValueError):
            extract_slices(torch.ones(4, 5, 5))

    def test_reshapes(self):
        for item in reshape_24().values():
            self.assertEqual(item.numel(), 24)
            self.assertTrue(torch.equal(item.flatten(), torch.arange(24)))


class AutogradHomeworkTests(unittest.TestCase):
    def test_three_gradients(self):
        r = three_variable_gradients()
        for key in "xyz":
            self.assertAlmostEqual(r["autograd"][key], r["analytical"][key], places=5)

    def test_mse_gradients(self):
        r = mse_gradients()
        self.assertGreater(r["mse"], 0)
        for key in ("w", "b"):
            self.assertAlmostEqual(r["autograd"][key], r["analytical"][key], places=5)

    def test_chain_rule(self):
        for x in (0.7, -1.0, 0.0):
            r = chain_rule_gradient(x)
            self.assertTrue(math.isclose(r["autograd_grad"], r["analytical"], abs_tol=1e-12))


class PerformanceHomeworkTests(unittest.TestCase):
    def test_cpu_timer(self):
        time_ms = cpu_time_ms(lambda: torch.sum(torch.ones(10)), repeats=2)
        self.assertGreaterEqual(time_ms, 0)

    def test_benchmark_cpu_or_gpu(self):
        a = torch.ones(8, 8)
        results = benchmark_operations(a, a, repeats=1)
        self.assertEqual(len(results), 5)
        for result in results:
            self.assertGreaterEqual(result["cpu_ms"], 0)
            if not torch.cuda.is_available():
                self.assertIsNone(result["gpu_ms"])
                self.assertIsNone(result["speedup"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
