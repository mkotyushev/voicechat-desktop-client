"""The OpenAI Realtime WebSocket protocol, as NemotronLabs VoiceChat speaks it.

This is the second server this client can talk to. The first, `omni_client`,
speaks MiniCPM-o's bespoke `/backend` protocol; this one speaks the protocol
NVIDIA documents for their VoiceChat container, which our own deployment on the
3090 reproduces in front of `llama-voicechat`. Writing against the documented
protocol rather than the local one means pointing this at NVIDIA's real
container later is a URL change.

The shape is deliberately the same as `omni_client.DuplexSession` — open,
start with callbacks, push audio, close — so `talk.py` can drive either.

Three differences from MiniCPM-o's duplex mode, all consequences of the model:

**Audio is 24 kHz in both directions, in 80 ms chunks.** MiniCPM-o wants
exactly one second of 16 kHz per slice; here the rate is the client's and the
server resamples to whatever the encoder and codec want.

**Input and output are simultaneous.** Every microphone frame reaches the
model, including while its speech is playing. `speech_started` /
`speech_stopped` are lifecycle hints; they do not gate inference or decide when
a response may start.

**Text and speech both stream.** Text still leads the codec slightly, but audio
arrives while the answer is being generated rather than after a completed wav.
"""

from __future__ import annotations

import base64
import json
import threading
import uuid
from typing import Callable

import numpy as np
from websocket import WebSocket, create_connection

# Both directions. The server resamples to the encoder's 16 kHz and back from
# the codec's 22.05 kHz, so neither of those rates is the client's problem.
IN_RATE = 24_000
OUT_RATE = 24_000

# 80 ms, the chunk size NVIDIA's client uses and the frame the model runs at.
CHUNK = int(IN_RATE * 0.08)


class RealtimeError(RuntimeError):
    pass


def _event_id() -> str:
    return f"event_{uuid.uuid4().hex[:16]}"


def pcm_to_b64(samples: np.ndarray) -> str:
    """float32 [-1, 1) -> base64 of little-endian signed 16-bit PCM.

    Note this is int16, not the float32 MiniCPM-o's protocol carries.
    """
    clipped = np.clip(np.asarray(samples, dtype=np.float32), -1.0, 1.0 - 1.0 / 32768.0)
    return base64.b64encode((clipped * 32768.0).astype("<i2").tobytes()).decode()


def b64_to_pcm(data: str) -> np.ndarray:
    if not data:
        return np.zeros(0, dtype=np.float32)
    raw = base64.b64decode(data)
    return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0


class RealtimeSession:
    """One WebSocket connection, which is one conversation.

    As with MiniCPM-o, memory lives in the server and dies with the socket:
    VoiceChat has a single 12.5 Hz timeline and no chat history to replay, so
    closing this is what forgets the conversation. The server holds one session
    at a time and refuses a second with a `session_in_use` error.
    """

    def __init__(
        self,
        url: str,
        *,
        instructions: str | None = None,
        tools: list[dict] | None = None,
        connect_timeout: float = 15.0,
        load_timeout: float = 600.0,
    ):
        self.url = url
        self.instructions = instructions
        self.tools = tools or []
        self.connect_timeout = connect_timeout
        self.load_timeout = load_timeout

        self.session_id: str | None = None
        self.chunks_sent = 0
        self._ws: WebSocket | None = None
        self._send_lock = threading.Lock()
        self._reader: threading.Thread | None = None
        self._closing = threading.Event()
        self._configured = threading.Event()
        self._input_speaking = False

    # ------------------------------------------------------------- lifecycle

    def open(self, on_status: Callable[[str], None] | None = None) -> None:
        if self._ws is not None:
            return
        self._input_speaking = False
        if on_status:
            on_status("connecting")
        ws = create_connection(self.url, timeout=self.connect_timeout)
        try:
            ws.settimeout(self.load_timeout)
            created = json.loads(ws.recv())
            if created.get("type") == "error":
                raise RealtimeError(
                    (created.get("error") or {}).get("message", "server refused the session")
                )
            if created.get("type") != "session.created":
                raise RealtimeError(
                    f"expected session.created, got {created.get('type')}: {created}"
                )
            self.session_id = (created.get("session") or {}).get("id")

            # Configure once, before any audio. The system prompt still occupies
            # one model-timeline position per token, but the server conditions
            # the complete prefix as a logical llama.cpp prefill batch.
            # session.updated does not come back until that batch is resident.
            if on_status:
                on_status("configuring")
            ws.send(json.dumps({
                "type": "session.update",
                "event_id": _event_id(),
                "session": {
                    "audio": {
                        "input": {"format": {"type": "audio/pcm", "rate": IN_RATE}},
                        "output": {"format": {"type": "audio/pcm", "rate": OUT_RATE}},
                    },
                    "instructions": self.instructions,
                    "tools": self.tools,
                },
            }))
            while True:
                ev = json.loads(ws.recv())
                if ev.get("type") == "session.updated":
                    break
                if ev.get("type") == "error":
                    raise RealtimeError(
                        (ev.get("error") or {}).get("message", "session.update failed")
                    )

            self._configured.set()
            # Long enough to ride out a slow turn, short enough that a dead
            # server surfaces rather than hanging forever.
            ws.settimeout(300.0)
            self._ws = ws
        except Exception:
            try:
                ws.close()
            except Exception:
                pass
            raise

    def start(
        self,
        *,
        on_text: Callable[[str], None] | None = None,
        on_audio: Callable[[np.ndarray], None] | None = None,
        on_speech_started: Callable[[], None] | None = None,
        on_speech_stopped: Callable[[], None] | None = None,
        on_response_start: Callable[[], None] | None = None,
        on_done: Callable[[dict], None] | None = None,
        on_tool_call: Callable[[str, str, str], None] | None = None,
        on_error: Callable[[Exception], None] | None = None,
        on_status: Callable[[str], None] | None = None,
    ) -> None:
        self.open(on_status)
        self._closing.clear()
        self._reader = threading.Thread(
            target=self._read_loop,
            args=(on_text, on_audio, on_speech_started, on_speech_stopped,
                  on_response_start, on_done, on_tool_call, on_error),
            daemon=True,
        )
        self._reader.start()

    def close(self) -> None:
        self._closing.set()
        ws, self._ws = self._ws, None
        self.session_id = None
        self._configured.clear()
        self._input_speaking = False
        if ws is not None:
            with self._send_lock:
                try:
                    # Ask for an orderly end so the server resets the timeline
                    # rather than waiting for the TCP connection to drop.
                    ws.send(json.dumps({"type": "session.close", "event_id": _event_id()}))
                except Exception:
                    pass
                try:
                    # close() performs a WebSocket handshake and may leave a
                    # reader blocked if the peer is stuck. Shutting down the
                    # socket after session.close makes local teardown bounded.
                    ws.shutdown()
                except Exception:
                    pass
                try:
                    ws.close()
                except Exception:
                    pass
        reader, self._reader = self._reader, None
        if reader is not None and reader is not threading.current_thread():
            reader.join(timeout=2.0)

    def __enter__(self) -> RealtimeSession:
        self.open()
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

    @property
    def is_open(self) -> bool:
        return self._ws is not None

    # ------------------------------------------------------------------- i/o

    def send_audio(self, pcm: np.ndarray) -> None:
        """Push one chunk of microphone audio. Any length; 80 ms is the norm.

        Keep sending while the model talks and while nobody is talking at all:
        those frames are the model's continuous duplex timeline.
        """
        ws = self._ws
        if ws is None:
            raise RealtimeError("session is not open")
        if not len(pcm):
            return
        message = json.dumps({
            "type": "input_audio_buffer.append",
            "event_id": _event_id(),
            "audio": pcm_to_b64(pcm),
        })
        with self._send_lock:
            if self._ws is None:
                return
            ws.send(message)
        self.chunks_sent += 1

    def set_input_speaking(self, speaking: bool) -> None:
        """Mute incoming speech locally while the user is actively talking."""
        self._input_speaking = speaking

    def send_tool_result(self, call_id: str, output: str) -> None:
        """Answer a tool call. The model's clock is frozen until this arrives."""
        ws = self._ws
        if ws is None:
            raise RealtimeError("session is not open")
        message = json.dumps({
            "type": "conversation.item.create",
            "event_id": _event_id(),
            "item": {
                "type": "function_call_output",
                "call_id": call_id,
                "output": output,
            },
        })
        with self._send_lock:
            if self._ws is None:
                return
            ws.send(message)

    # ---------------------------------------------------------------- events

    def _read_loop(self, on_text, on_audio, on_speech_started, on_speech_stopped,
                   on_response_start, on_done, on_tool_call, on_error) -> None:
        ws = self._ws
        while not self._closing.is_set() and ws is not None:
            try:
                raw = ws.recv()
            except Exception as exc:
                if not self._closing.is_set() and on_error:
                    on_error(exc)
                return
            if not raw:
                continue
            try:
                ev = json.loads(raw)
            except ValueError:
                continue

            kind = ev.get("type")

            if kind == "response.output_audio_transcript.delta":
                piece = ev.get("delta") or ""
                if piece and on_text:
                    on_text(piece)
            elif kind == "response.output_audio.delta":
                pcm = b64_to_pcm(ev.get("delta") or "")
                if len(pcm) and on_audio and not self._input_speaking:
                    on_audio(pcm)
            elif kind == "input_audio_buffer.speech_started":
                if on_speech_started:
                    on_speech_started()
            elif kind == "input_audio_buffer.speech_stopped":
                if on_speech_stopped:
                    on_speech_stopped()
            elif kind == "response.created":
                if on_response_start:
                    on_response_start()
            elif kind == "response.function_call_arguments.done":
                if on_tool_call:
                    on_tool_call(
                        ev.get("call_id") or "",
                        ev.get("name") or "",
                        ev.get("arguments") or "{}",
                    )
            elif kind == "response.done":
                response = ev.get("response") or {}
                if response.get("status") == "failed":
                    details = response.get("status_details") or {}
                    message = (details.get("error") or {}).get("message", "turn failed")
                    if on_error:
                        on_error(RealtimeError(message))
                elif response.get("status") == "completed" and on_done:
                    on_done(response.get("metrics") or {})
            elif kind == "error":
                error = ev.get("error") or {}
                # A tool result sent with nothing waiting for it is the client
                # racing itself, not a dead session; the rest are worth raising.
                if error.get("code") == "no_pending_call":
                    continue
                if on_error:
                    on_error(RealtimeError(error.get("message", "server error")))
            elif kind == "session.end":
                if not self._closing.is_set() and on_error:
                    on_error(RealtimeError("server ended the session"))
                return
