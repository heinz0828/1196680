import json
import os
import tempfile
import unittest

import numpy as np

from experiments.run_all import compute_dm_results, get_default_model_names
from utils.visualization import plot_comparison_bar


class PublicationOutputTests(unittest.TestCase):
    def test_figures_are_saved_as_high_res_and_vector_outputs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, 'comparison.png')
            plot_comparison_bar(
                {'MDHGNN': {'RMSE': 0.08}, 'GRU': {'RMSE': 0.10}},
                'RMSE',
                save_path=save_path,
            )

            base = os.path.join(tmpdir, 'comparison')
            for ext in ['.png', '.pdf', '.svg']:
                path = base + ext
                self.assertTrue(os.path.exists(path), path)
                self.assertGreater(os.path.getsize(path), 0, path)

    def test_dm_results_are_structured_and_json_serializable(self):
        all_errors = {
            'MDHGNN': [np.array([0.1, -0.1, 0.05, -0.05])],
            'GRU': [np.array([0.3, -0.2, 0.25, -0.15])],
        }

        dm_results = compute_dm_results(all_errors, ['MDHGNN', 'GRU'], horizon=1)

        self.assertIn('GRU', dm_results)
        self.assertIn('DM_stat', dm_results['GRU'])
        self.assertIn('p_value', dm_results['GRU'])
        json.dumps(dm_results)

    def test_dm_results_accept_preconcatenated_error_arrays(self):
        all_errors = {
            'MDHGNN': np.array([0.1, -0.1, 0.05, -0.05]),
            'GRU': np.array([0.3, -0.2, 0.25, -0.15]),
        }

        dm_results = compute_dm_results(all_errors, ['MDHGNN', 'GRU'], horizon=1)

        self.assertIn('GRU', dm_results)
        self.assertIn('n_errors', dm_results['GRU'])

    def test_default_baseline_list_includes_naive(self):
        self.assertIn('Naive', get_default_model_names())

    def test_reported_ablation_uses_masked_protocol_and_full_is_best(self):
        table_dir = os.path.join(os.getcwd(), 'results', 'tables')
        summary_path = os.path.join(table_dir, 'ablation_h1_multi_seed.json')
        manifest_path = os.path.join(table_dir, 'ablation_h1_manifest.json')

        with open(summary_path, encoding='utf-8') as f:
            summary = json.load(f)
        with open(manifest_path, encoding='utf-8') as f:
            manifest = json.load(f)

        full_rmse = summary['Full MDHGNN']['RMSE']['mean']
        other_rmses = [
            values['RMSE']['mean']
            for name, values in summary.items()
            if name != 'Full MDHGNN'
        ]

        self.assertLess(full_rmse, min(other_rmses))
        self.assertIn('masked at inference', manifest['protocol'])


if __name__ == '__main__':
    unittest.main()
