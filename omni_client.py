"""Client for the llama.cpp-omni WebSocket backend that serves MiniCPM-o 4.5.

Two conversation shapes, one socket each:

* `OmniSession` — turn-based. You speak, then it answers. Push-to-talk.
* `DuplexSession` — full-duplex. Audio flows continuously in 1-second slices
  and the model decides, slice by slice, whether to listen or to speak.

They are different modes on the server and cannot share a loaded model: opening
a session whose mode differs from the resident context makes the server free
and reload the weights (~20 s).

One WebSocket connection is one conversation, not one turn.

That distinction is the whole design. Memory lives in the server's KV cache,
which grows as turns accumulate on an open session; it does *not* come from the
`messages` array. The engine dispatches each turn on exactly one modality — an
`else if` chain over vision, then audio, then text — so a turn that carries
audio never evaluates its own text, and any conversation history packed into
that text is silently dropped. Reconnecting per turn therefore produces a model
with no memory of anything you said out loud. Keeping the socket open produces
one that does, because `stream_decode` leaves the KV ready for the next user
turn.

Wire format, both directions: raw little-endian float32 PCM, mono, base64'd.
Not WAV, not int16.
"""

from __future__ import annotations

import base64
import json
import threading
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from websocket import WebSocket, create_connection

# The audio encoder is Whisper-medium, fixed at 16 kHz. Token2Wav synthesises at
# 24 kHz. Playing one at the other's rate is the classic chipmunk bug.
IN_RATE = 16_000
OUT_RATE = 24_000

# Full-duplex is time-sliced: one input.append is one slice, and the server
# answers each with either a single `listen` delta or a burst of text/audio
# followed by response.done. The slice is 1000 ms by the engine's design — the
# streaming audio encoder and the 1 Hz speak/listen decision are both built
# around it, so this is not a tunable.
DUPLEX_CHUNK = IN_RATE

# Defaults lifted from the reference demo's duplex config schema, with one
# deliberate change. force_listen_count keeps the model quiet while the stream
# warms up, and max_new_speak_tokens_per_chunk is what keeps a long answer
# interruptible instead of monologuing past the point where you talked over it.
#
# listen_prob_scale biases the per-slice speak/listen decision — it is added to
# the <|listen|> logit as (scale - 1) * 2, so below 1.0 makes the model readier
# to speak. The reference default of 1.0 measured as marginal here: sessions
# would sometimes ride out an entire question choosing listen every slice and
# never answer at all. 0.5 answers reliably. Raise it toward 1.0 if the model
# talks over you; lower it toward 0.3 if it sits silent.
DUPLEX_CONFIG = {
    "force_listen_count": 3,
    "max_new_speak_tokens_per_chunk": 20,
    "temperature": 0.7,
    "top_k": 20,
    "top_p": 0.8,
    "length_penalty": 1.1,
    "tts_temperature": 0.8,
    "listen_prob_scale": 0.5,
}


class OmniError(RuntimeError):
    """The server refused a turn, or closed the session under us."""


def pcm_to_b64(samples: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(samples, dtype="<f4").tobytes()).decode("ascii")


def b64_to_pcm(data: str) -> np.ndarray:
    return np.frombuffer(base64.b64decode(data), dtype="<f4")


def user_turn(audio: np.ndarray | None = None, text: str = "") -> dict:
    """A `messages` entry.

    Audio and text can both be present, but the server will act on the audio
    and ignore the text — see the module docstring. Send one or the other.
    """
    content: list[dict] = []
    if audio is not None and len(audio):
        content.append({"type": "audio", "data": pcm_to_b64(audio)})
    if text:
        content.append({"type": "text", "text": text})
    return {"role": "user", "content": content}


@dataclass
class TurnResult:
    text: str = ""
    audio: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype="<f4"))
    metrics: dict = field(default_factory=dict)
    listened: bool = False  # the model chose to keep listening instead of replying

    @property
    def kv_used(self) -> int:
        """Tokens the conversation now occupies on the server."""
        return int(self.metrics.get("kv_cache_length") or 0)


class OmniSession:
    """An open conversation with the server.

    The server allows exactly one session at a time, process-wide, and refuses
    `session.init` while another is active. Always `close()` — a leaked session
    locks out every other client until the TCP connection drops.
    """

    def __init__(
        self,
        url: str,
        *,
        ref_audio: np.ndarray | None = None,
        load_tts: bool = True,
        connect_timeout: float = 15.0,
        load_timeout: float = 600.0,
        reply_timeout: float = 300.0,
    ):
        self.url = url
        self.ref_audio = ref_audio
        # Stays on even for a text-only conversation, deliberately: the server
        # reuses its loaded context only while the session-level TTS flag is
        # unchanged, so flipping it costs a full model reload. Whether a given
        # turn is spoken is decided per turn by `use_tts_template` instead.
        self.load_tts = load_tts
        self.connect_timeout = connect_timeout
        self.load_timeout = load_timeout
        self.reply_timeout = reply_timeout
        self.session_id: str | None = None
        self._ws: WebSocket | None = None

    # ------------------------------------------------------------- lifecycle

    def open(self, on_status: Callable[[str], None] | None = None) -> None:
        if self._ws is not None:
            return
        if on_status:
            on_status("connecting")
        ws = create_connection(self.url, timeout=self.connect_timeout)
        try:
            payload: dict = {"mode": "turn_based", "use_tts": self.load_tts}
            if self.ref_audio is not None and len(self.ref_audio):
                payload["voice"] = {"ref_audio": pcm_to_b64(self.ref_audio)}
            ws.send(json.dumps({"type": "session.init", "payload": payload}))

            # The server does not touch the weights until a client asks for a
            # session, so on a cold server this read blocks for the whole model
            # load — see the server README, "Cold first turn".
            ws.settimeout(self.load_timeout)
            if on_status:
                on_status("opening session")
            created = json.loads(ws.recv())
            if created.get("type") != "session.created":
                raise OmniError(f"expected session.created, got {created.get('type')}: {created}")
            self.session_id = created.get("session_id")
            ws.settimeout(self.reply_timeout)
            self._ws = ws
        except Exception:
            try:
                ws.close()
            except Exception:
                pass
            raise

    def close(self) -> None:
        ws, self._ws = self._ws, None
        self.session_id = None
        if ws is not None:
            try:
                ws.close()
            except Exception:
                pass

    @property
    def is_open(self) -> bool:
        return self._ws is not None

    def __enter__(self) -> OmniSession:
        self.open()
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

    # ------------------------------------------------------------------ turn

    def turn(
        self,
        *,
        audio: np.ndarray | None = None,
        text: str = "",
        speak: bool = True,
        max_new_tokens: int = 512,
        length_penalty: float = 1.1,
        on_text: Callable[[str], None] | None = None,
        on_audio: Callable[[np.ndarray], None] | None = None,
        on_status: Callable[[str], None] | None = None,
    ) -> TurnResult:
        """Send one turn and collect the reply.

        Only this turn goes on the wire. Earlier turns are already in the
        server's KV cache; re-sending them would prefill the conversation twice.
        """
        self.open(on_status)
        ws = self._ws
        assert ws is not None

        ws.send(json.dumps({
            "type": "input.append",
            "input": {
                # turn_based rejects an absent `messages`, so this is required
                # even though the history it would carry is not used.
                "messages": [user_turn(audio=audio, text=text)],
                "streaming": True,
                "use_tts_template": speak,
                "generation": {
                    "max_new_tokens": max_new_tokens,
                    "length_penalty": length_penalty,
                },
            },
        }))
        if on_status:
            on_status("thinking")

        result = TurnResult()
        chunks: list[np.ndarray] = []
        try:
            while True:
                ev = json.loads(ws.recv())
                kind = ev.get("type")

                if kind == "response.output.delta":
                    what = ev.get("kind")
                    if what == "text":
                        piece = ev.get("text") or ""
                        result.text += piece
                        if on_text and piece:
                            on_text(piece)
                    elif what == "audio":
                        pcm = b64_to_pcm(ev.get("audio") or "")
                        if len(pcm):
                            chunks.append(pcm)
                            if on_audio:
                                on_audio(pcm)
                    elif what == "listen":
                        result.listened = True
                elif kind == "response.done":
                    result.metrics = ev.get("metrics") or {}
                    # Non-streaming turns return the whole clip here instead of
                    # as deltas. We ask for streaming, so this is normally absent.
                    if ev.get("audio") and not chunks:
                        pcm = b64_to_pcm(ev["audio"])
                        if len(pcm):
                            chunks.append(pcm)
                            if on_audio:
                                on_audio(pcm)
                    break
                elif kind == "session.closed":
                    self.close()
                    reason = ev.get("reason", "unknown")
                    detail = (ev.get("diagnostic") or {}).get("message", "")
                    raise OmniError(f"server closed the session: {reason} {detail}".strip())
        except OmniError:
            raise
        except Exception:
            # A dead socket is not reusable, and leaving it half-open would hold
            # the server's only session slot.
            self.close()
            raise

        if chunks:
            result.audio = np.concatenate(chunks)
        return result


class DuplexSession:
    """A live, always-on conversation.

    Unlike the turn-based session there is no request/response pairing to wait
    on. You push a 1-second slice of microphone audio whenever you have one —
    silence included, the server rejects an empty slice — and events arrive on
    their own schedule from a reader thread. Every slice draws exactly one of:

      * `on_listen()`                       the model stayed quiet this slice
      * `on_text(str)` / `on_audio(pcm)` …  followed by `on_done(metrics)`

    Interruption is not something the client asks for. The model hears you in
    the stream and stops on its own, which is why the microphone must keep
    streaming while it talks. Cutting playback when that happens is the
    client's job — see `Player.flush` in audio_io.
    """

    def __init__(
        self,
        url: str,
        *,
        ref_audio: np.ndarray | None = None,
        config: dict | None = None,
        connect_timeout: float = 15.0,
        load_timeout: float = 600.0,
    ):
        self.url = url
        self.ref_audio = ref_audio
        self.config = dict(DUPLEX_CONFIG, **(config or {}))
        self.connect_timeout = connect_timeout
        self.load_timeout = load_timeout
        self.session_id: str | None = None
        self._ws: WebSocket | None = None
        self._send_lock = threading.Lock()
        self._reader: threading.Thread | None = None
        self._closing = threading.Event()
        self.slices_sent = 0

    # ------------------------------------------------------------- lifecycle

    def open(self, on_status: Callable[[str], None] | None = None) -> None:
        if self._ws is not None:
            return
        if on_status:
            on_status("connecting")
        ws = create_connection(self.url, timeout=self.connect_timeout)
        try:
            payload: dict = {
                "mode": "full_duplex",
                "use_tts": True,
                "config": self.config,
            }
            if self.ref_audio is not None and len(self.ref_audio):
                payload["voice"] = {"ref_audio": pcm_to_b64(self.ref_audio)}
            ws.send(json.dumps({"type": "session.init", "payload": payload}))

            # On a cold server, or one loaded in the other mode, this blocks for
            # the whole model load.
            ws.settimeout(self.load_timeout)
            if on_status:
                on_status("opening session")
            created = json.loads(ws.recv())
            if created.get("type") != "session.created":
                raise OmniError(f"expected session.created, got {created.get('type')}: {created}")
            if created.get("mode") != "full_duplex":
                raise OmniError(f"server opened mode={created.get('mode')}, wanted full_duplex")
            self.session_id = created.get("session_id")
            # Long enough to ride out a slow slice, short enough that a dead
            # server surfaces rather than hanging the call forever.
            ws.settimeout(120.0)
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
        on_listen: Callable[[], None] | None = None,
        on_done: Callable[[dict], None] | None = None,
        on_error: Callable[[Exception], None] | None = None,
        on_status: Callable[[str], None] | None = None,
    ) -> None:
        self.open(on_status)
        self._closing.clear()
        self._reader = threading.Thread(
            target=self._read_loop,
            args=(on_text, on_audio, on_listen, on_done, on_error),
            daemon=True,
        )
        self._reader.start()

    def close(self) -> None:
        self._closing.set()
        ws, self._ws = self._ws, None
        self.session_id = None
        if ws is not None:
            with self._send_lock:
                try:
                    ws.close()
                except Exception:
                    pass
        reader, self._reader = self._reader, None
        if reader is not None and reader is not threading.current_thread():
            reader.join(timeout=2.0)

    @property
    def is_open(self) -> bool:
        return self._ws is not None

    # ------------------------------------------------------------------ i/o

    def send_slice(self, pcm: np.ndarray, *, force_listen: bool = False) -> None:
        """Push one time slice. Must be DUPLEX_CHUNK samples of 16 kHz mono.

        `force_listen` skips the model's decision for this slice and keeps it
        quiet. The engine already does that for the first `force_listen_count`
        slices; passing it here is for holding the floor deliberately.
        """
        ws = self._ws
        if ws is None:
            raise OmniError("duplex session is not open")
        message = json.dumps({
            "type": "input.append",
            "input": {"audio": pcm_to_b64(pcm), "force_listen": bool(force_listen)},
        })
        with self._send_lock:
            if self._ws is None:
                return
            ws.send(message)
        self.slices_sent += 1

    def _read_loop(self, on_text, on_audio, on_listen, on_done, on_error) -> None:
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
            if kind == "response.output.delta":
                what = ev.get("kind")
                if what == "text":
                    piece = ev.get("text") or ""
                    if piece and on_text:
                        on_text(piece)
                elif what == "audio":
                    pcm = b64_to_pcm(ev.get("audio") or "")
                    if len(pcm) and on_audio:
                        on_audio(pcm)
                elif what == "listen":
                    if on_listen:
                        on_listen()
            elif kind == "response.done":
                if on_done:
                    on_done(ev.get("metrics") or {})
            elif kind == "session.closed":
                if not self._closing.is_set() and on_error:
                    reason = ev.get("reason", "unknown")
                    on_error(OmniError(f"server closed the session: {reason}"))
                return


def run_turn(url: str, *, audio=None, text="", **kwargs) -> TurnResult:
    """One-shot convenience: open a session, take one turn, close it.

    Fine for probes. Not what an interactive client should do — a fresh session
    starts with an empty KV cache and therefore no memory.
    """
    session_kwargs = {
        k: kwargs.pop(k)
        for k in ("ref_audio", "load_tts", "connect_timeout", "load_timeout", "reply_timeout")
        if k in kwargs
    }
    with OmniSession(url, **session_kwargs) as session:
        return session.turn(audio=audio, text=text, **kwargs)
