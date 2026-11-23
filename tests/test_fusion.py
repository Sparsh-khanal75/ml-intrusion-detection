import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from features import ATTACK_CATEGORY, CLASSES  # noqa: E402
from fusion import entropy_weighted_fusion, normalised_entropy  # noqa: E402


class FusionTests(unittest.TestCase):
    def test_entropy_bounds(self):
        h = normalised_entropy(np.array([[1.0, 0.0], [0.5, 0.5]]))
        self.assertAlmostEqual(h[0], 0.0, places=6)
        self.assertAlmostEqual(h[1], 1.0, places=6)

    def test_confident_model_dominates(self):
        confident = np.array([[0.99, 0.01]])
        unsure = np.array([[0.45, 0.55]])
        fused, w = entropy_weighted_fusion([confident, unsure])
        self.assertGreater(w[0, 0], 0.9)
        self.assertEqual(fused.argmax(1)[0], 0)  # simple averaging would also say 0, but by less
        self.assertGreater(fused[0, 0], (0.99 + 0.45) / 2)

    def test_output_is_a_distribution(self):
        rng = np.random.default_rng(0)
        ps = [rng.dirichlet(np.ones(5), size=50) for _ in range(3)]
        fused, w = entropy_weighted_fusion(ps)
        np.testing.assert_allclose(fused.sum(1), 1.0, atol=1e-9)
        np.testing.assert_allclose(w.sum(1), 1.0, atol=1e-9)

    def test_equal_certainty_equals_mean(self):
        a = np.array([[0.8, 0.2]])
        b = np.array([[0.2, 0.8]])
        fused, _ = entropy_weighted_fusion([a, b])
        np.testing.assert_allclose(fused, [[0.5, 0.5]])


class TaxonomyTests(unittest.TestCase):
    def test_every_category_is_known(self):
        self.assertTrue(set(ATTACK_CATEGORY.values()) <= set(CLASSES))


if __name__ == "__main__":
    unittest.main()
