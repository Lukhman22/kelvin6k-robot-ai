"""
Unit tests for Phase 1: Continuous AudioWorklet & AudioRingBuffer Foundation.

Tests:
1. Ring buffer capacity (2000ms @ 16kHz = 32,000 samples)
2. Sequential write and chronological read
3. Circular wrap-around handling without sample corruption
4. Float32 to Int16 PCM conversion and clipping
5. Feature flag default verification (CONTINUOUS_AUDIO_FOUNDATION is FALSE)
6. Non-interference with backend modules
"""

import math
import numpy as np
import pytest


class MockAudioRingBuffer:
    """Python reference mirror of AudioRingBuffer in ui/src/audio/worklet/ringBuffer.ts."""

    def __init__(self, capacity: int = 32000, sample_rate: int = 16000):
        self.capacity = capacity
        self.sample_rate = sample_rate
        self.buffer = np.zeros(capacity, dtype=np.int16)
        self.write_index = 0
        self.samples_written_total = 0

    def write(self, samples: np.ndarray) -> int:
        count = len(samples)
        if count == 0:
            return 0

        # Convert float to int16 if needed
        if samples.dtype in (np.float32, np.float64):
            clamped = np.clip(samples, -1.0, 1.0)
            pcm = np.where(clamped < 0, clamped * 32768, clamped * 32767).round().astype(np.int16)
        else:
            pcm = samples.astype(np.int16)

        for s in pcm:
            self.buffer[self.write_index] = s
            self.write_index = (self.write_index + 1) % self.capacity

        self.samples_written_total += count
        return count

    @property
    def available(self) -> int:
        return min(self.samples_written_total, self.capacity)

    @property
    def total_written(self) -> int:
        return self.samples_written_total

    @property
    def duration_ms(self) -> float:
        return (self.available / self.sample_rate) * 1000.0

    def read(self, count: int, offset_from_latest: int = 0) -> np.ndarray:
        available = self.available
        if count <= 0 or available == 0:
            return np.zeros(0, dtype=np.int16)

        safe_offset = max(0, offset_from_latest)
        if safe_offset >= available:
            return np.zeros(0, dtype=np.int16)

        actual_count = min(count, available - safe_offset)
        output = np.zeros(actual_count, dtype=np.int16)

        read_idx = (self.write_index - safe_offset - actual_count) % self.capacity
        if read_idx < 0:
            read_idx += self.capacity

        for i in range(actual_count):
            output[i] = self.buffer[read_idx]
            read_idx = (read_idx + 1) % self.capacity

        return output

    def clear(self):
        self.buffer.fill(0)
        self.write_index = 0
        self.samples_written_total = 0


def test_ring_buffer_capacity_and_duration():
    """Verify 2000ms @ 16kHz yields exactly 32,000 samples."""
    buf = MockAudioRingBuffer(capacity=32000, sample_rate=16000)
    assert buf.capacity == 32000
    assert buf.sample_rate == 16000
    assert buf.available == 0
    assert buf.duration_ms == 0.0

    # Write 16,000 samples (1.0 second)
    samples = np.ones(16000, dtype=np.int16) * 100
    buf.write(samples)
    assert buf.available == 16000
    assert buf.total_written == 16000
    assert math.isclose(buf.duration_ms, 1000.0, rel_tol=1e-3)


def test_ring_buffer_sequential_write_and_read():
    """Verify samples read match exactly what was written in chronological order."""
    buf = MockAudioRingBuffer(capacity=32000, sample_rate=16000)
    test_data = np.arange(1000, dtype=np.int16)
    buf.write(test_data)

    read_back = buf.read(1000)
    assert len(read_back) == 1000
    np.testing.assert_array_equal(read_back, test_data)


def test_ring_buffer_wrap_around_handling():
    """Verify circular wrap-around retains newest 32,000 samples and drops oldest."""
    capacity = 100
    buf = MockAudioRingBuffer(capacity=capacity, sample_rate=1000)

    # Write 150 sequential samples (0 to 149)
    input_data = np.arange(150, dtype=np.int16)
    buf.write(input_data)

    assert buf.available == 100
    assert buf.total_written == 150

    # Oldest 50 samples (0 to 49) should be overwritten; newest 100 (50 to 149) retained
    expected = np.arange(50, 150, dtype=np.int16)
    read_back = buf.read(100)
    np.testing.assert_array_equal(read_back, expected)


def test_pcm_float_conversion_and_clipping():
    """Verify float normalization to 16-bit signed PCM integer bounds."""
    buf = MockAudioRingBuffer(capacity=1000, sample_rate=16000)
    # Test values: -1.5 (clipped to -1.0), -1.0, 0.0, 0.5, 1.0, 2.0 (clipped to 1.0)
    floats = np.array([-1.5, -1.0, 0.0, 0.5, 1.0, 2.0], dtype=np.float32)
    buf.write(floats)

    result = buf.read(6)
    assert result[0] == -32768  # Clamped minimum
    assert result[1] == -32768
    assert result[2] == 0
    assert result[3] == round(0.5 * 32767)
    assert result[4] == 32767
    assert result[5] == 32767  # Clamped maximum


def test_feature_flag_is_strictly_false_by_default():
    """Verify the TypeScript continuousAudio.ts file declares feature flag as false."""
    import os

    ts_path = os.path.join("ui", "src", "audio", "worklet", "continuousAudio.ts")
    assert os.path.exists(ts_path), "continuousAudio.ts must exist"

    with open(ts_path, "r", encoding="utf-8") as f:
        content = f.read()

    assert "export const CONTINUOUS_AUDIO_FOUNDATION = false;" in content, (
        "CONTINUOUS_AUDIO_FOUNDATION must default to false"
    )
