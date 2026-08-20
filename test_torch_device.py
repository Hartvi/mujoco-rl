from __future__ import annotations

import unittest
from unittest.mock import patch

from torch_device import resolve_device, resolve_device_name


class TorchDeviceTest(unittest.TestCase):
    def test_auto_prefers_cuda(self) -> None:
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.backends.mps.is_available", return_value=True),
        ):
            self.assertEqual(resolve_device_name("auto"), "cuda")

    def test_auto_uses_mps_when_cuda_is_unavailable(self) -> None:
        with (
            patch("torch.cuda.is_available", return_value=False),
            patch("torch.backends.mps.is_available", return_value=True),
        ):
            self.assertEqual(resolve_device_name("auto"), "mps")

    def test_auto_falls_back_to_cpu(self) -> None:
        with (
            patch("torch.cuda.is_available", return_value=False),
            patch("torch.backends.mps.is_available", return_value=False),
        ):
            self.assertEqual(resolve_device_name(None), "cpu")

    def test_unavailable_explicit_accelerator_raises(self) -> None:
        with patch("torch.backends.mps.is_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "MPS.*not available"):
                resolve_device("mps")

    def test_explicit_cpu_is_supported(self) -> None:
        self.assertEqual(resolve_device_name("cpu"), "cpu")


if __name__ == "__main__":
    unittest.main()
