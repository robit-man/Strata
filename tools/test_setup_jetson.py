"""Jetson ARM64 detection and unified-memory accounting.  No GPU or network is needed."""

import unittest
from unittest import mock

import setup


class JetsonDetection(unittest.TestCase):
    def test_nvidia_smi_na_falls_back_to_the_cuda_driver(self):
        driver_card = {"index": 0, "name": "Orin", "vram_gb": 30.0, "arch": "87", "driver": "540.3.0",
                       "jetson": True, "uma": True, "dedicated_gb": 0.0, "shared_gb": 30.0}
        smi = "0, Orin, [N/A], 8.7, [N/A]\n"
        with mock.patch.object(setup, "out", return_value=smi), \
                mock.patch.object(setup, "_cuda_driver_gpus", return_value=[driver_card]) as fallback:
            self.assertEqual(setup.gpus(), [driver_card])
        fallback.assert_called_once_with()

    def test_unified_memory_is_not_counted_twice(self):
        gpu = {"vram_gb": 30.0, "uma": True, "dedicated_gb": 0.0, "shared_gb": 30.0}
        self.assertEqual(setup.low_ram_vram(gpu), 0.0)
        self.assertEqual(setup.mem_have("Gyro-S", 30.0, setup.low_ram_vram(gpu)), 30.0)

    def test_arm64_has_no_x86_instruction_floor(self):
        with mock.patch.object(setup.platform, "machine", return_value="aarch64"):
            self.assertEqual(setup.cpu_floor(False), "")

    def test_cpu_clusters_are_not_reported_as_x86_sockets(self):
        with mock.patch.object(setup.platform, "machine", return_value="aarch64"):
            self.assertEqual(setup.two_socket_note((3, 4)), [])


if __name__ == "__main__":
    unittest.main()
