import unittest

import torch

from config import Config
from experiments.run_ablation import ABLATION_CONFIGS, build_ablation_model


class AblationVariantTests(unittest.TestCase):
    def setUp(self):
        self.cfg = Config()
        self.cfg.apply_frequency('weekly')
        self.cfg.horizon = 1
        self.in_features = 30
        self.x = torch.randn(2, self.cfg.window_size, self.in_features)

    def _variant(self, marker):
        for name, overrides in ABLATION_CONFIGS.items():
            if marker in name:
                return name, overrides
        self.fail(f'Missing ablation variant marker: {marker}')

    def _edge_count(self, model):
        with torch.no_grad():
            node_emb = model.node_encoder(self.x)
            H = model.hg_constructor(node_emb, self.x)
        return H.shape[2]

    def test_all_ablation_variants_forward(self):
        for name, overrides in ABLATION_CONFIGS.items():
            with self.subTest(name=name):
                model = build_ablation_model(name, overrides, self.in_features, self.cfg)
                y = model(self.x)
                self.assertEqual(tuple(y.shape), (2, self.cfg.horizon))

    def test_ablation_variants_remove_hyperedge_channels(self):
        full = build_ablation_model(
            'Full MDHGNN', ABLATION_CONFIGS['Full MDHGNN'], self.in_features, self.cfg
        )
        full_edges = self._edge_count(full)

        for marker in ['A1', 'A2', 'A3']:
            name, overrides = self._variant(marker)
            with self.subTest(name=name):
                model = build_ablation_model(name, overrides, self.in_features, self.cfg)
                self.assertLess(self._edge_count(model), full_edges)


if __name__ == '__main__':
    unittest.main()
