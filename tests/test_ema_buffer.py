"""Tests for the EMA dual-state buffer."""

import numpy as np
import pytest

from luna_streams.streams.ema_buffer import EMABuffer


class TestEMABuffer:
    def test_initial_state(self):
        buf = EMABuffer(dim=64, decay=0.999)
        assert buf.drift_signal == 0.0
        assert buf.state_norm == 0.0
        assert buf.step_count == 0

    def test_single_update(self):
        buf = EMABuffer(dim=64, decay=0.999)
        state = np.ones(64, dtype=np.float32)
        buf.update(state)

        assert buf.step_count == 1
        assert buf.state_norm > 0
        np.testing.assert_array_equal(buf.fast, state)
        # Slow state should be (1-0.999) * state = 0.001 * state
        expected_slow = 0.001 * state
        np.testing.assert_array_almost_equal(buf.slow, expected_slow, decimal=5)

    def test_drift_decreases_over_time(self):
        """After many updates with the same state, drift should decrease."""
        buf = EMABuffer(dim=64, decay=0.99)
        state = np.ones(64, dtype=np.float32)

        drifts = []
        for _ in range(500):
            buf.update(state)
            drifts.append(buf.drift_signal)

        # Drift should decrease as slow state converges to fast state
        assert drifts[-1] < drifts[0]
        assert drifts[-1] < 0.05  # Should be very small after 500 steps

    def test_drift_spikes_on_change(self):
        """Drift should spike when fast state changes abruptly."""
        buf = EMABuffer(dim=64, decay=0.999)

        # Stabilize with one pattern
        for _ in range(100):
            buf.update(np.ones(64, dtype=np.float32))
        drift_before = buf.drift_signal

        # Sudden change
        buf.update(np.ones(64, dtype=np.float32) * -5.0)
        drift_after = buf.drift_signal

        assert drift_after > drift_before

    def test_serialization_roundtrip(self):
        """State should survive a save/restore cycle."""
        buf = EMABuffer(dim=64, decay=0.999)
        for i in range(10):
            buf.update(np.random.randn(64).astype(np.float32))

        state_dict = buf.to_dict()
        restored = EMABuffer.from_dict(state_dict, decay=0.999)

        np.testing.assert_array_equal(buf.fast, restored.fast)
        np.testing.assert_array_equal(buf.slow, restored.slow)
        assert buf.step_count == restored.step_count

    def test_dimension_mismatch_handling(self):
        """Buffer should handle mismatched input dimensions gracefully."""
        buf = EMABuffer(dim=64, decay=0.999)
        # Larger input - should be truncated
        buf.update(np.ones(128, dtype=np.float32))
        assert buf.fast.shape == (64,)

    def test_smaller_input_padding(self):
        """Buffer should pad smaller inputs with zeros."""
        buf = EMABuffer(dim=64, decay=0.999)
        buf.update(np.ones(32, dtype=np.float32))
        assert buf.fast.shape == (64,)
        assert buf.fast[31] == 1.0
        assert buf.fast[32] == 0.0
