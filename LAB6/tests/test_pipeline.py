"""Run with python -m unittest discover -s tests."""
from pathlib import Path
import sys
import unittest

import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from main import CNN, train_epoch


class PipelineTests(unittest.TestCase):
    def test_lab3_checkpoint_matches_architecture(self):
        path = Path(__file__).resolve().parents[1] / "src" / "lab3_cnn_state.pt"
        model = CNN()
        weights = torch.load(path, map_location="cpu", weights_only=True)
        # Lab 3 saved THOP profiling counters alongside the trained weights.
        model.load_state_dict({
            key: value for key, value in weights.items()
            if key.rsplit(".", 1)[-1] not in {"total_ops", "total_params"}
        })
        self.assertEqual(sum(parameter.numel() for parameter in model.parameters()), 20_490)
        self.assertEqual(tuple(model(torch.zeros(2, 1, 28, 28)).shape), (2, 10))

    def test_accumulation_scales_final_partial_update(self):
        torch.set_num_threads(1)
        torch.manual_seed(42)
        direct = CNN()
        accumulated = CNN()
        accumulated.load_state_dict(direct.state_dict())
        data = TensorDataset(torch.randn(13, 1, 28, 28), torch.arange(13) % 10)
        for model, batch, steps in [(direct, 8, 1), (accumulated, 2, 4)]:
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
            loader = DataLoader(data, batch_size=batch)
            train_epoch(model, loader, optimizer, torch.device("cpu"), steps, False)
        for reference, candidate in zip(direct.parameters(), accumulated.parameters()):
            torch.testing.assert_close(reference, candidate, rtol=1e-5, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
