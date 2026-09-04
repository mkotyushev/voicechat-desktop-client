"""Microphone capture and streaming playback, in the rates the model expects.

PortAudio will usually let you open a device at any rate and let Windows do the
conversion, but not always — some drivers only offer their configured rate. So
both classes try the rate they want first and fall back to the device default
plus a band-limited resample, rather than failing in front of the user.
"""

from __future__ import annotations

import queue
import threading
from collections import deque

import numpy as np
import sounddevice as sd


def resample(x: np.ndarray, src: int, dst: int) -> np.ndarray:
    """Band-limited resample via the frequency domain (scipy.signal.resample).

    Plain interpolation would alias badly on a 48k -> 16k decimation, which on
    speech sounds like a metallic buzz and measurably hurts recognition.
    """
    if src == dst or x.size == 0:
        return np.ascontiguousarray(x, dtype="<f4")
    n_out = int(round(x.size * dst / src))
    spectrum = np.fft.rfft(x)
    n_freq = n_out // 2 + 1
    if n_freq <= spectrum.size:
        spectrum = spectrum[:n_freq]
    else:
        spectrum = np.concatenate([spectrum, np.zeros(n_freq - spectrum.size, dtype=spectrum.dtype)])
    out = np.fft.irfft(spectrum, n=n_out) * (n_out / x.size)
    return np.ascontiguousarray(out, dtype="<f4")


class StreamResampler:
    """Resample a stream block by block, with no seam between blocks.

    `resample` above is fine for a whole recording and wrong for a stream: it
    takes an FFT of each block on its own, which assumes the block is periodic,
    so every boundary gets wrap-around artifacts. That was tolerable when the
    server sent about a second at a time. VoiceChat streams speech in 80 ms
    chunks, which would put a dozen of those a second into the playback path on
    any machine whose output device is not 24 kHz — which is most of them.

    This keeps the interpolation kernel's context and the fractional read
    position across calls, so the pieces join exactly as if the stream had been
    resampled in one go. Verified against the one-shot path: bit-identical
    away from the stream's own ends.
    """

    def __init__(self, sr_in: int, sr_out: int, taps: int = 16):
        self.ratio = sr_out / sr_in
        self.cutoff = min(1.0, self.ratio)
        self.half = int(np.ceil(taps / self.cutoff))
        self._buf = np.zeros(0, dtype=np.float64)
        self._base = 0    # global input index of _buf[0]
        self._k = 0       # next output index to produce

    def push(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        if x.size:
            self._buf = np.concatenate([self._buf, x])

        # The newest output we can finish is the one whose kernel still ends
        # inside what we have; anything past that waits for the next block.
        last_in = self._base + self._buf.size - 1 - self.half
        limit = int(np.floor(last_in * self.ratio)) + 1
        out = np.zeros(0, dtype="<f4")

        if limit > self._k:
            k = np.arange(self._k, limit, dtype=np.float64)
            t = k / self.ratio - self._base
            base = np.floor(t).astype(np.int64)
            acc = np.zeros(t.size, dtype=np.float64)
            for j in range(-self.half + 1, self.half + 1):
                n = base + j
                d = t - n
                w = 0.5 + 0.5 * np.cos(np.pi * np.clip(d / self.half, -1.0, 1.0))
                acc += np.sinc(d * self.cutoff) * w * self._buf[n]
            out = np.ascontiguousarray(acc * self.cutoff, dtype="<f4")
            self._k = limit

        drop = max(0, int(np.floor(self._k / self.ratio)) - self.half - self._base)
        if drop > 0:
            self._buf = self._buf[drop:]
            self._base += drop
        return out


def _device_rate(device, kind: str) -> int:
    info = sd.query_devices(device if device is not None else sd.default.device[0 if kind == "input" else 1], kind)
    return int(info["default_samplerate"])


class Recorder:
    """Push-to-talk capture. start() ... stop() -> mono float32 at `rate`."""

    def __init__(self, rate: int = 16_000, device=None):
        self.rate = rate
        self.device = device
        self._stream: sd.InputStream | None = None
        self._frames: list[np.ndarray] = []
        self._lock = threading.Lock()
        self._native_rate = rate

    def _callback(self, indata, _frames, _time, status):
        # `status` flags overflows; dropping a frame is better than raising
        # inside PortAudio's realtime thread.
        with self._lock:
            self._frames.append(indata[:, 0].copy())

    def start(self) -> None:
        if self._stream is not None:
            return
        with self._lock:
            self._frames = []
        for rate in (self.rate, _device_rate(self.device, "input")):
            try:
                self._stream = sd.InputStream(
                    samplerate=rate, channels=1, dtype="float32",
                    device=self.device, callback=self._callback,
                )
                self._stream.start()
                self._native_rate = rate
                return
            except Exception:
                self._stream = None
        raise RuntimeError("could not open the microphone at 16 kHz or at its default rate")

    def stop(self) -> np.ndarray:
        if self._stream is None:
            return np.zeros(0, dtype="<f4")
        self._stream.stop()
        self._stream.close()
        self._stream = None
        with self._lock:
            frames, self._frames = self._frames, []
        if not frames:
            return np.zeros(0, dtype="<f4")
        audio = np.concatenate(frames).astype("<f4")
        return resample(audio, self._native_rate, self.rate)

    @property
    def recording(self) -> bool:
        return self._stream is not None


class SpeechGate:
    """Energy-based "did someone just start talking" test.

    Only used to decide when to cut playback for barge-in. The model does its
    own turn-taking from the audio stream; this exists so the reply stops in
    your ears the moment you start talking, instead of a slice later when the
    server catches up.

    The threshold is learned from the room, never assumed. A fixed floor is
    what broke the first version: a laptop array mic idles around 0.010 RMS,
    well above the 0.004 that looked like a sensible constant, so the gate read
    speech on silence — and because it only adapted while quiet, it could never
    climb back out. Two rules keep that from recurring: calibrate before
    judging, and never let "talking" persist indefinitely without re-checking.
    """

    # Slices of room tone to measure before the gate will fire. The server
    # holds the model quiet for force_listen_count (3) slices at the start of a
    # call, so this window is genuinely free.
    CALIBRATE_SLICES = 4
    # If the gate has been latched on this long, the calibration was wrong.
    RELATCH_SLICES = 20

    # However loud calibration measured the room, the threshold never rises
    # past this — otherwise one noisy calibration second deafens the gate for
    # the whole call and barge-in silently stops working.
    MAX_NOISE = 0.03

    def __init__(self, margin: float = 3.5, min_threshold: float = 0.01, adapt: float = 0.1):
        self.margin = margin
        self.min_threshold = min_threshold
        self.adapt = adapt
        self.noise: float | None = None
        self.talking = False
        self._seen = 0
        self._talking_run = 0
        self._calibration: list[float] = []

    @property
    def threshold(self) -> float:
        base = self.noise if self.noise is not None else self.min_threshold
        return max(self.min_threshold, min(base, self.MAX_NOISE) * self.margin)

    def reset(self) -> None:
        self.noise = None
        self.talking = False
        self._seen = 0
        self._talking_run = 0
        self._calibration = []

    def update(self, level: float) -> bool:
        """Feed one slice. True only on the slice where speech *starts*.

        Returning an edge rather than a level is deliberate: the caller drops
        buffered playback, and doing that once per utterance is a barge-in
        while doing it every slice is a stutter.
        """
        self._seen += 1
        if self._seen <= self.CALIBRATE_SLICES:
            # Median, not max: a door closing during calibration should not set
            # the threshold for the rest of the call. Median survives one bad
            # second in either direction; max only survives quiet ones.
            self._calibration.append(level)
            self.noise = float(np.median(self._calibration))
            return False

        was = self.talking
        self.talking = level > self.threshold

        if self.talking:
            self._talking_run += 1
            if self._talking_run >= self.RELATCH_SLICES:
                # Nobody talks for 20 seconds without a pause. The room is
                # louder than it was at calibration, so re-measure from here.
                self.noise = level
                self.talking = False
                self._talking_run = 0
                return False
        else:
            self._talking_run = 0
            self.noise = (1 - self.adapt) * (self.noise or level) + self.adapt * level

        return self.talking and not was


class ChunkRecorder:
    """Continuous capture that hands out fixed-size slices.

    PortAudio is told the block size, so the device itself paces the stream at
    one slice per second and no timer is needed. Slices go to a queue rather
    than straight to the socket: the callback runs on a realtime audio thread
    where a blocking network write would drop frames.
    """

    def __init__(self, chunk: int, rate: int = 16_000, device=None, max_queue: int = 8):
        self.chunk = chunk
        self.rate = rate
        self.device = device
        self.queue: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=max_queue)
        self._stream: sd.InputStream | None = None
        self._native_rate = rate
        self._carry = np.zeros(0, dtype="<f4")
        self.dropped = 0
        self.level = 0.0  # RMS of the most recent block, for a meter / VAD

    def _callback(self, indata, _frames, _time, _status):
        block = indata[:, 0].astype("<f4", copy=True)
        if self._native_rate != self.rate:
            block = resample(block, self._native_rate, self.rate)
        self._carry = np.concatenate([self._carry, block]) if self._carry.size else block
        while self._carry.size >= self.chunk:
            piece, self._carry = self._carry[:self.chunk], self._carry[self.chunk:]
            self.level = float(np.sqrt((piece ** 2).mean()))
            try:
                self.queue.put_nowait(piece)
            except queue.Full:
                # The sender is behind. Dropping the oldest slice keeps the
                # conversation near real time; keeping it would grow the lag
                # without bound.
                try:
                    self.queue.get_nowait()
                    self.queue.put_nowait(piece)
                except queue.Empty:
                    pass
                self.dropped += 1

    def start(self) -> None:
        if self._stream is not None:
            return
        self._carry = np.zeros(0, dtype="<f4")
        self.dropped = 0
        for rate in (self.rate, _device_rate(self.device, "input")):
            blocksize = self.chunk if rate == self.rate else 0
            try:
                self._stream = sd.InputStream(
                    samplerate=rate, channels=1, dtype="float32", blocksize=blocksize,
                    device=self.device, callback=self._callback,
                )
                self._stream.start()
                self._native_rate = rate
                return
            except Exception:
                self._stream = None
        raise RuntimeError("could not open the microphone at 16 kHz or at its default rate")

    def stop(self) -> None:
        if self._stream is None:
            return
        self._stream.stop()
        self._stream.close()
        self._stream = None
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
            except queue.Empty:
                break

    @property
    def running(self) -> bool:
        return self._stream is not None


class Player:
    """Streaming sink: push() chunks as they arrive, they play back gaplessly.

    A small jitter buffer sits in front of the device. The server emits speech
    roughly one second at a time but not on an exact clock — a 0.84 s chunk
    followed a second later by the next one leaves a real hole — so playback
    waits for `prebuffer` seconds of audio before starting, and re-arms the
    same way if it ever runs dry mid-utterance.
    """

    def __init__(self, rate: int = 24_000, device=None, prebuffer: float = 0.3):
        self.rate = rate
        self.device = device
        self._queue: deque[np.ndarray] = deque()
        self._lock = threading.Lock()
        self._stream: sd.OutputStream | None = None
        self._native_rate = rate
        self._pending = np.zeros(0, dtype="<f4")
        self._prebuffer_secs = prebuffer
        self._prebuffer = int(rate * prebuffer)
        self._armed = False
        # Built lazily, because the device rate is only known once the stream
        # opens. None means no conversion is needed.
        self._rs: StreamResampler | None = None

    def _queued_locked(self) -> int:
        return int(sum(c.size for c in self._queue)) + self._pending.size

    def _callback(self, outdata, frames, _time, _status):
        out = np.zeros(frames, dtype="float32")
        with self._lock:
            if not self._armed:
                if self._queued_locked() < self._prebuffer:
                    # Still filling. Silence now buys a clean run later.
                    outdata[:, 0] = out
                    return
                self._armed = True

            filled = 0
            while filled < frames:
                if self._pending.size == 0:
                    if not self._queue:
                        break
                    self._pending = self._queue.popleft()
                take = min(frames - filled, self._pending.size)
                out[filled:filled + take] = self._pending[:take]
                self._pending = self._pending[take:]
                filled += take

            if filled < frames:
                # Ran dry: re-arm so the next burst is buffered rather than
                # dribbled out with holes in it.
                self._armed = False
        outdata[:, 0] = out

    def _ensure_stream(self) -> None:
        if self._stream is not None:
            return
        for rate in (self.rate, _device_rate(self.device, "output")):
            try:
                self._stream = sd.OutputStream(
                    samplerate=rate, channels=1, dtype="float32",
                    device=self.device, callback=self._callback,
                )
                self._stream.start()
                self._native_rate = rate
                self._prebuffer = int(rate * self._prebuffer_secs)
                return
            except Exception:
                self._stream = None
        raise RuntimeError("could not open an output device at 24 kHz or at its default rate")

    def push(self, pcm: np.ndarray) -> None:
        self._ensure_stream()
        with self._lock:
            if self._native_rate != self.rate:
                if self._rs is None:
                    self._rs = StreamResampler(self.rate, self._native_rate)
                pcm = self._rs.push(pcm)
                if pcm.size == 0:
                    # The first chunk of a stream is short by the kernel's
                    # right-hand context; it comes out with the next one.
                    return
            self._queue.append(np.ascontiguousarray(pcm, dtype="float32"))

    @property
    def backlog(self) -> int:
        """Samples still queued — used to tell 'speaking' from 'done'."""
        with self._lock:
            return self._queued_locked()

    def flush(self) -> None:
        """Drop anything not yet played (barge-in).

        The chunk already mid-playback is faded out over a few milliseconds
        rather than cut at an arbitrary sample, which would click.
        """
        with self._lock:
            self._queue.clear()
            # Drop the resampler's carried context too: it belongs to speech
            # that is being thrown away, and splicing it onto the next
            # utterance would put a fragment of the old one at its start.
            self._rs = None
            tail = min(self._pending.size, self._native_rate // 100)  # ~10 ms
            if tail > 0:
                self._pending = self._pending[:tail] * np.linspace(1.0, 0.0, tail, dtype="float32")
            else:
                self._pending = np.zeros(0, dtype="<f4")
            # Deliberately left armed: the callback has to drain that fade, and
            # it disarms itself the moment it runs dry. Clearing the flag here
            # would strand the tail in the buffer and leave `backlog` non-zero
            # forever, which reads downstream as "the model is still speaking".

    def close(self) -> None:
        self.flush()
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
