"""Talk to MiniCPM-o 4.5 on the 3090, in either of two conversation shapes.

**Live** is full-duplex: the microphone streams continuously and the model
decides whether to listen or speak. VoiceChat runs 80 ms frames; MiniCPM-o uses
one-second slices. You interrupt either by talking.

**Push to talk** is turn-based: hold a key, speak, release, hear the answer.
Better in a noisy room, and the only mode that also takes typed turns.

Everything model-side happens on the server; this process only moves audio.
The worker threads own the socket, the Tk thread owns the widgets, and they
meet at a single queue — Tk is not thread-safe, so nothing else crosses.
"""

from __future__ import annotations

import json
import queue
import threading
import tkinter as tk
import wave
from pathlib import Path
from tkinter import scrolledtext, ttk

import numpy as np

import cua_tools
import omni_client
import realtime_client
from audio_io import ChunkRecorder, Player, Recorder, SpeechGate, resample

HERE = Path(__file__).resolve().parent
CONFIG_PATH = HERE / "config.json"

DEFAULTS = {
    # Which server, and therefore which protocol. See README, "Two servers".
    #   "voicechat" — NemotronLabs VoiceChat 11B on the 3090, OpenAI Realtime
    #                 protocol, ws://127.0.0.1:9070/v1/realtime
    #   "minicpm"   — MiniCPM-o 4.5, the bespoke /backend protocol,
    #                 ws://127.0.0.1:9060/backend
    "backend": "voicechat",
    "url": "ws://127.0.0.1:9070/v1/realtime",
    "mode": "live",  # "live" (full-duplex) or "ptt" (push to talk)
    "speak": True,
    "max_new_tokens": 512,
    "length_penalty": 1.1,
    # Voice the TTS clones. Without one the system prompt still opens an audio
    # block but leaves it empty, and the synthesised voice is a coin toss.
    "ref_audio": "ref_voice.wav",
    # Must match N_CTX in the server's .env — nothing reports it over the wire,
    # and it is only used to warn before the conversation outgrows the cache.
    "context_size": 8192,
    # Live mode only. Scales the microphone while the model is talking, so an
    # open speaker does not feed its own voice back into the stream. 1.0 is
    # right with headphones and keeps barge-in most sensitive; lower it toward
    # 0.2 if you use speakers. There is no acoustic echo cancellation here.
    "mic_duck_while_speaking": 1.0,
    # Audio buffered before playback starts. The server emits speech about a
    # second at a time but not on an exact clock, so a little slack here is
    # what keeps the gaps between chunks from being audible. Raise it if speech
    # still sounds broken up; lower it to shave latency off the model's replies.
    "playback_buffer_ms": 1000,
    # Live mode only. Overrides for omni_client.DUPLEX_CONFIG; the one worth
    # touching is listen_prob_scale — lower makes the model readier to speak.
    "duplex": {"listen_prob_scale": 0.5},
    # VoiceChat only. Written into the model's perception channel one token per
    # 80 ms timeline position before live audio. Batched prefill makes this fast
    # in wall time, but a long prompt still reduces the finite session length.
    "instructions": None,
    # VoiceChat only, and off by default — it lets the model act on this
    # machine. Tool definitions are installed with the batched system prefill,
    # but still consume timeline/context positions. `allow` is a list of
    # cua-driver tool names, or a mapping of name to the parameters worth
    # showing the model. See cua_tools.py.
    "tools": {
        "enabled": False,
        "driver": None,
        "allow": {
            "list_apps": None,
            "launch_app": ["name", "urls"],
            "clipboard_read": None,
        },
        "timeout_s": 20,
        "max_result_chars": 300,
    },
    "input_device": None,
    "output_device": None,
}


def backend_module(name: str):
    """The protocol module for a backend name.

    Both expose IN_RATE and OUT_RATE, so the audio devices are opened from
    whichever is in use — 16 kHz in for MiniCPM-o, 24 kHz for VoiceChat.
    """
    if name == "voicechat":
        return realtime_client
    if name == "minicpm":
        return omni_client
    raise SystemExit(f"unknown backend {name!r}; expected 'voicechat' or 'minicpm'")


def load_config() -> dict:
    cfg = dict(DEFAULTS)
    if not CONFIG_PATH.exists():
        return cfg
    for key, value in json.loads(CONFIG_PATH.read_text(encoding="utf-8")).items():
        # One level of merging, so `"tools": {"enabled": true}` in config.json
        # turns tools on without also having to restate the whole default
        # whitelist. Deeper than that, the file wins outright.
        if isinstance(value, dict) and isinstance(cfg.get(key), dict):
            cfg[key] = {**cfg[key], **value}
        else:
            cfg[key] = value
    return cfg


def load_wav_16k(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as handle:
        frames = handle.readframes(handle.getnframes())
        rate, width, channels = handle.getframerate(), handle.getsampwidth(), handle.getnchannels()
    if width == 2:
        audio = np.frombuffer(frames, dtype="<i2").astype("<f4") / 32768.0
    elif width == 4:
        audio = np.frombuffer(frames, dtype="<f4")
    else:
        raise ValueError(f"unsupported WAV sample width: {width} bytes")
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    return resample(audio, rate, omni_client.IN_RATE)


class App:
    def __init__(self, root: tk.Tk, cfg: dict):
        self.root = root
        self.cfg = cfg
        self.events: queue.Queue[tuple[str, object]] = queue.Queue()

        # Turn-based state
        self.busy = False
        self.holding = False
        self.session: omni_client.OmniSession | None = None

        # Live state
        self.live: omni_client.DuplexSession | None = None
        self.live_thread: threading.Thread | None = None
        self.live_stop = threading.Event()
        self.model_speaking = False
        self.spoke_this_line = False

        self.backend_name = cfg.get("backend", "voicechat")
        self.backend = backend_module(self.backend_name)
        self.voicechat = self.backend is realtime_client

        # Rates and slice size come from the protocol in use. VoiceChat streams
        # 80 ms of 24 kHz; MiniCPM-o wants exactly one second of 16 kHz.
        chunk = (
            realtime_client.CHUNK if self.voicechat else omni_client.DUPLEX_CHUNK
        )
        self.recorder = Recorder(self.backend.IN_RATE, cfg.get("input_device"))
        self.chunker = ChunkRecorder(
            chunk, self.backend.IN_RATE, cfg.get("input_device")
        )
        self.player = Player(
            self.backend.OUT_RATE,
            cfg.get("output_device"),
            prebuffer=float(cfg.get("playback_buffer_ms", 1000)) / 1000.0,
        )
        self.gate = SpeechGate()
        self.ref_audio = None if self.voicechat else self._load_ref_audio()
        self.tools = self._load_tools() if self.voicechat else None

        root.title(
            "NemotronLabs VoiceChat 11B — voice chat"
            if self.voicechat
            else "MiniCPM-o 4.5 — voice chat"
        )
        root.geometry("820x620")
        root.minsize(600, 460)

        self._build_ui()
        self._bind_keys()
        self._apply_mode()
        self.root.after(50, self._drain_events)

    # ---------------------------------------------------------------- setup

    def _load_ref_audio(self) -> np.ndarray | None:
        raw = (self.cfg.get("ref_audio") or "").strip()
        if not raw:
            return None
        path = Path(raw)
        if not path.is_absolute():
            path = HERE / path
        return load_wav_16k(path)

    def _load_tools(self) -> cua_tools.ToolBox | None:
        """The desktop tools to offer VoiceChat, or None for a conversation
        without any.

        A missing or broken driver costs you tools, not the app: it lands as a
        note in the transcript once the window is up, and the conversation runs
        as it always did.
        """
        cfg = self.cfg.get("tools") or {}
        if not cfg.get("enabled"):
            return None
        try:
            return cua_tools.ToolBox(
                cfg.get("allow") or [],
                driver=cfg.get("driver"),
                timeout_s=float(cfg.get("timeout_s", 20)),
                max_result_chars=int(cfg.get("max_result_chars", 300)),
            )
        except Exception as exc:
            self.events.put(("note", f"[tools] unavailable — {exc}"))
            return None

    def _build_ui(self) -> None:
        top = ttk.Frame(self.root, padding=(10, 8))
        top.pack(fill="x")
        ttk.Label(top, text="Server").pack(side="left")
        self.url_var = tk.StringVar(value=self.cfg["url"])
        ttk.Entry(top, textvariable=self.url_var).pack(
            side="left", fill="x", expand=True, padx=(6, 10)
        )
        ttk.Button(top, text="New conversation", command=self.reset).pack(side="left")

        modes = ttk.Frame(self.root, padding=(10, 0, 10, 4))
        modes.pack(fill="x")
        self.mode_var = tk.StringVar(value=self.cfg.get("mode", "live"))
        ttk.Radiobutton(
            modes, text="Live conversation", value="live",
            variable=self.mode_var, command=self._apply_mode,
        ).pack(side="left")
        ttk.Radiobutton(
            modes, text="Push to talk", value="ptt",
            variable=self.mode_var, command=self._apply_mode,
        ).pack(side="left", padx=(12, 0))
        self.speak_var = tk.BooleanVar(value=bool(self.cfg["speak"]))
        self.speak_chk = ttk.Checkbutton(modes, text="Speak replies", variable=self.speak_var)
        self.speak_chk.pack(side="left", padx=(20, 0))
        self.state_var = tk.StringVar(value="")
        ttk.Label(modes, textvariable=self.state_var, font=("Segoe UI", 10, "bold")).pack(
            side="right"
        )

        self.transcript = scrolledtext.ScrolledText(
            self.root, wrap="word", state="disabled", font=("Segoe UI", 11), padx=10, pady=8
        )
        self.transcript.pack(fill="both", expand=True, padx=10)
        self.transcript.tag_configure("you", foreground="#1a6fb5", font=("Segoe UI", 11, "bold"))
        self.transcript.tag_configure("model", foreground="#0b7a3b", font=("Segoe UI", 11, "bold"))
        self.transcript.tag_configure("note", foreground="#888888", font=("Segoe UI", 9, "italic"))

        self.typed_row = ttk.Frame(self.root, padding=(10, 6))
        self.typed_row.pack(fill="x")
        self.typed_var = tk.StringVar()
        entry = ttk.Entry(self.typed_row, textvariable=self.typed_var)
        entry.pack(side="left", fill="x", expand=True)
        entry.bind("<Return>", lambda _e: self.send_typed())
        ttk.Button(self.typed_row, text="Send text", command=self.send_typed).pack(
            side="left", padx=(8, 0)
        )

        bottom = ttk.Frame(self.root, padding=(10, 4, 10, 6))
        bottom.pack(fill="x")
        self.talk_btn = tk.Button(
            bottom, text="Hold to talk  (or hold Space)", height=2,
            font=("Segoe UI", 12, "bold"), bg="#e8eef5", activebackground="#cfe0f0",
        )
        self.talk_btn.pack(fill="x")
        self.talk_btn.bind("<ButtonPress-1>", lambda _e: self.start_talking())
        self.talk_btn.bind("<ButtonRelease-1>", lambda _e: self.stop_talking())

        self.live_btn = tk.Button(
            bottom, text="Start conversation", height=2,
            font=("Segoe UI", 12, "bold"), bg="#dff0e4", activebackground="#c6e4d0",
            command=self.toggle_live,
        )

        self.status_var = tk.StringVar(value="Ready.")
        ttk.Label(
            self.root, textvariable=self.status_var, padding=(10, 0, 10, 8), foreground="#555555"
        ).pack(fill="x")

    def _bind_keys(self) -> None:
        # Windows does not synthesise release events for auto-repeat, so a
        # single held-flag is enough to debounce a held key.
        self.root.bind("<KeyPress-space>", self._space_down)
        self.root.bind("<KeyRelease-space>", self._space_up)

    def _typing(self) -> bool:
        return isinstance(self.root.focus_get(), (ttk.Entry, tk.Entry))

    def _space_down(self, _event: tk.Event) -> None:
        if self.mode_var.get() == "ptt" and not self._typing():
            self.start_talking()

    def _space_up(self, _event: tk.Event) -> None:
        if self.mode_var.get() == "ptt" and not self._typing():
            self.stop_talking()

    def _apply_mode(self) -> None:
        live = self.mode_var.get() == "live"
        if self.voicechat:
            # One duplex protocol, two microphone policies. "Live" leaves the
            # microphone open; push-to-talk substitutes silence unless the
            # button is held. Both keep the same 12.5 Hz timeline running.
            self.live_btn.pack(fill="x")
            # VoiceChat has no text input channel — there is one 12.5 Hz
            # timeline and nowhere to put a typed turn — and every reply is
            # spoken, so neither control has anything to do.
            self.typed_row.pack_forget()
            self.speak_chk.state(["disabled"])
            if live:
                self.talk_btn.pack_forget()
                self.status_var.set(
                    "Live. Start the conversation, then just talk — interrupt "
                    "at any time. Headphones recommended."
                )
            else:
                self.talk_btn.pack(fill="x")
                self.status_var.set(
                    "Push to talk. Start the conversation, then hold the button "
                    "or Space while you speak."
                )
            self.state_var.set("")
            return

        if live:
            self._drop_session()
            self.talk_btn.pack_forget()
            self.typed_row.pack_forget()
            self.live_btn.pack(fill="x")
            self.speak_chk.state(["disabled"])
            self.status_var.set(
                "Live mode. Switching modes reloads the model on the server (~20 s on the "
                "first turn). Headphones strongly recommended."
            )
        else:
            self.stop_live()
            self.live_btn.pack_forget()
            self.talk_btn.pack(fill="x")
            self.typed_row.pack(fill="x", before=self.talk_btn.master)
            self.speak_chk.state(["!disabled"])
            self.status_var.set("Push-to-talk mode.")
        self.state_var.set("")

    # ------------------------------------------------------------- ui utils

    def say(self, text: str, tag: str = "") -> None:
        self.transcript.configure(state="normal")
        self.transcript.insert("end", text, tag)
        self.transcript.configure(state="disabled")
        self.transcript.see("end")

    def _drain_events(self) -> None:
        try:
            while True:
                kind, payload = self.events.get_nowait()
                if kind == "status":
                    self.status_var.set(str(payload))
                elif kind == "state":
                    self.state_var.set(str(payload))
                elif kind == "text":
                    self.say(str(payload))
                elif kind == "note":
                    self.say(f"{payload}\n", "note")
                elif kind == "model_start":
                    if not self.spoke_this_line:
                        self.say("Model: ", "model")
                        self.spoke_this_line = True
                elif kind == "model_end":
                    if self.spoke_this_line:
                        self.say("\n\n")
                        self.spoke_this_line = False
                elif kind == "you_spoke":
                    self.say("You: ", "you")
                    self.say(f"{payload}\n", "note")
                elif kind == "done":
                    self._finish_turn(payload)
                elif kind == "live_error":
                    self.say(f"\n[error] {payload}\n", "note")
                    self.stop_live()
        except queue.Empty:
            pass
        self.root.after(50, self._drain_events)

    def _finish_turn(self, payload: object) -> None:
        self.busy = False
        self.talk_btn.configure(state="normal")
        assert isinstance(payload, dict)
        if payload.get("error"):
            self.say(f"\n[error] {payload['error']}\n", "note")
            self.status_var.set("Error — the conversation was dropped; the next turn starts fresh.")
            return
        self.say("\n\n")
        self.status_var.set(self._metrics_line(payload.get("metrics") or {}))

    def _metrics_line(self, metrics: dict) -> str:
        parts = []
        if metrics.get("prefill_ms"):
            parts.append(f"prefill {metrics['prefill_ms']:.0f} ms")
        if metrics.get("generate_ms"):
            parts.append(f"generate {metrics['generate_ms']:.0f} ms")
        if metrics.get("wall_clock_ms"):
            parts.append(f"total {metrics['wall_clock_ms'] / 1000:.1f} s")
        kv = int(metrics.get("kv_cache_length") or 0)
        ctx = int(self.cfg.get("context_size") or 0)
        if kv and ctx:
            parts.append(f"context {kv}/{ctx}")
            if kv > 0.85 * ctx:
                self.say(
                    "(this conversation is nearly out of context — "
                    "start a new one before replies degrade)\n",
                    "note",
                )
        return " · ".join(parts) if parts else "Ready."

    # ------------------------------------------------------------ live mode

    def toggle_live(self) -> None:
        if self.live_thread is not None:
            self.stop_live()
        else:
            self.start_live()

    def start_live(self) -> None:
        if self.live_thread is not None:
            return
        self._drop_session()  # the two modes cannot share a loaded model
        self.live_stop.clear()
        self.spoke_this_line = False
        self.live_btn.configure(text="End conversation", bg="#f5d5d5", activebackground="#eec0c0")
        self.state_var.set("connecting…")
        self.live_thread = threading.Thread(
            target=self._live_worker, args=(self.url_var.get().strip(),), daemon=True
        )
        self.live_thread.start()

    def stop_live(self) -> None:
        self.live_stop.set()
        # On VoiceChat the push-to-talk button gates this worker's microphone
        # rather than a recorder, so a conversation ended mid-hold would leave
        # the gate stuck open for the next one.
        self.holding = False
        self.chunker.stop()
        if self.live is not None:
            self.live.close()
            self.live = None
        thread, self.live_thread = self.live_thread, None
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=3.0)
        self.player.flush()
        self.live_btn.configure(text="Start conversation", bg="#dff0e4", activebackground="#c6e4d0")
        self.state_var.set("")
        if self.spoke_this_line:
            self.events.put(("model_end", None))

    def _live_worker(self, url: str) -> None:
        if self.voicechat:
            self._voicechat_worker(url)
        else:
            self._minicpm_worker(url)

    def _voicechat_worker(self, url: str) -> None:
        """Stream to the Realtime endpoint and play what comes back.

        The server consumes every 80 ms frame while it is speaking as well as
        listening. The local speech gate only cuts queued playback immediately;
        the same microphone frame still goes to the model so it can yield.
        """
        duck = float(self.cfg.get("mic_duck_while_speaking") or 1.0)
        ptt = self.mode_var.get() == "ptt"
        session = realtime_client.RealtimeSession(
            url,
            instructions=self.cfg.get("instructions"),
            tools=self.tools.definitions if self.tools else None,
        )
        self.live = session
        silence = np.zeros(realtime_client.CHUNK, dtype=np.float32)

        if self.tools:
            self.events.put((
                "note",
                f"{len(self.tools.definitions)} desktop tools installed in the "
                "batched system prefill",
            ))

        def on_text(piece: str) -> None:
            self.events.put(("model_start", None))
            self.events.put(("text", piece))

        def on_audio(pcm: np.ndarray) -> None:
            self.model_speaking = True
            self.player.push(pcm)

        def on_speech_started() -> None:
            session.set_input_speaking(True)
            self.player.flush()
            self.events.put(("state", "hearing you"))

        def on_speech_stopped() -> None:
            session.set_input_speaking(False)
            self.events.put(("state", "thinking"))

        def on_response_start() -> None:
            self.events.put(("state", "answering"))

        def on_done(metrics: dict) -> None:
            if self.spoke_this_line:
                self.events.put(("model_end", None))
            ms = metrics.get("generation_ms")
            frames = metrics.get("frames")
            timeline = metrics.get("timeline_frame")
            if ms is not None and frames is not None:
                self.events.put((
                    "status",
                    f"{int(ms) / 1000:.1f} s for {frames} frames "
                    f"({frames / 12.5:.1f} s of timeline)",
                ))
            elif frames is not None:
                suffix = f", timeline at {timeline}" if timeline is not None else ""
                self.events.put(("status", f"{frames} response frames{suffix}"))
            self.events.put(("state", "listening"))

        def on_tool_call(call_id: str, name: str, arguments: str) -> None:
            # A result must always come back: the model stops listening until
            # one does. With no toolbox a call means the model invented the
            # tool, and an empty result is what keeps the timeline moving.
            if self.tools is None:
                self.events.put(("note", f"model called {name}({arguments}) — no tools "
                                         f"are configured, answering empty"))
                try:
                    session.send_tool_result(call_id, "{}")
                except Exception:
                    pass
                return

            # This runs on the session's reader thread, which is also the
            # thread that delivers speech, so the driver is not called here.
            self.events.put(("note", f"[tool] {name}({arguments})"))
            self.events.put(("state", f"running tool: {name}"))
            threading.Thread(
                target=self._run_tool, args=(session, call_id, name, arguments),
                daemon=True,
            ).start()

        def on_error(exc: Exception) -> None:
            self.events.put(("live_error", f"{type(exc).__name__}: {exc}"))

        try:
            session.start(
                on_text=on_text, on_audio=on_audio,
                on_speech_started=on_speech_started,
                on_speech_stopped=on_speech_stopped,
                on_response_start=on_response_start,
                on_done=on_done, on_tool_call=on_tool_call, on_error=on_error,
                on_status=lambda s: self.events.put(("state", s)),
            )
            self.gate.reset()
            self.chunker.start()
            self.events.put(("state", "listening"))
            self.events.put((
                "note",
                "Hold the button and speak." if ptt else
                "Live. Just start talking — interrupt any time.",
            ))

            was_speaking = False
            while not self.live_stop.is_set():
                try:
                    chunk = self.chunker.queue.get(timeout=0.5)
                except queue.Empty:
                    continue

                speaking_now = self.player.backlog > 0
                if speaking_now != was_speaking:
                    self.events.put(("state", "model speaking" if speaking_now else "listening"))
                    was_speaking = speaking_now

                # In push-to-talk the stream never stops; releasing the button
                # substitutes silence on the same continuous timeline.
                if ptt and not self.holding:
                    chunk = silence
                    started_talking = self.gate.update(0.0)
                else:
                    started_talking = self.gate.update(self.chunker.level)

                if started_talking and speaking_now:
                    self.player.flush()
                    session.set_input_speaking(True)
                    self.events.put(("state", "you interrupted"))
                    was_speaking = False

                if duck < 1.0 and speaking_now:
                    chunk = chunk * duck

                try:
                    session.send_audio(chunk)
                except Exception as exc:
                    if not self.live_stop.is_set():
                        on_error(exc)
                    return
        except Exception as exc:
            if not self.live_stop.is_set():
                on_error(exc)
        finally:
            self.chunker.stop()
            session.close()
            if self.live is session:
                self.live = None

    def _run_tool(
        self, session: realtime_client.RealtimeSession,
        call_id: str, name: str, arguments: str,
    ) -> None:
        """Run one driver tool and answer the model, on a thread of its own.

        The model's clock is frozen from the call until the result.
        `ToolBox.call` turns every driver failure into a bounded result, and
        the server has a second timeout that releases a disconnected client.
        """
        assert self.tools is not None
        result = self.tools.call(name, arguments)
        self.events.put(("state", "processing tool result"))
        # The model gets the whole result; the transcript gets one line of it.
        shown = " ".join(result.split())
        self.events.put((
            "note",
            f"[tool] {name} → {shown[:120] + '…' if len(shown) > 120 else shown}",
        ))
        try:
            session.send_tool_result(call_id, result)
            self.events.put(("state", "thinking"))
        except Exception as exc:
            self.events.put(("note", f"[tool] {name} result went nowhere — {exc}"))

    def _minicpm_worker(self, url: str) -> None:
        duck = float(self.cfg.get("mic_duck_while_speaking") or 1.0)
        session = omni_client.DuplexSession(
            url, ref_audio=self.ref_audio, config=self.cfg.get("duplex") or {}
        )
        self.live = session

        def on_text(piece: str) -> None:
            self.events.put(("model_start", None))
            self.events.put(("text", piece))

        def on_audio(pcm: np.ndarray) -> None:
            self.model_speaking = True
            self.player.push(pcm)

        def on_listen() -> None:
            # A listen slice after speech is the end of the model's turn.
            if self.spoke_this_line:
                self.events.put(("model_end", None))
            self.events.put(("state", "listening"))

        def on_done(metrics: dict) -> None:
            self.events.put(("status", self._metrics_line(metrics)))

        def on_error(exc: Exception) -> None:
            self.events.put(("live_error", f"{type(exc).__name__}: {exc}"))

        try:
            session.start(
                on_text=on_text, on_audio=on_audio, on_listen=on_listen,
                on_done=on_done, on_error=on_error,
                on_status=lambda s: self.events.put(("state", s)),
            )
            # Each call re-measures the room; a threshold learned in a quiet
            # room is wrong in a noisy one and vice versa.
            self.gate.reset()
            self.chunker.start()
            self.events.put(("state", "listening"))
            self.events.put(("note", "Live. Just start talking — interrupt any time."))

            was_speaking = False
            while not self.live_stop.is_set():
                try:
                    slice_pcm = self.chunker.queue.get(timeout=0.5)
                except queue.Empty:
                    continue

                speaking_now = self.player.backlog > 0
                if speaking_now != was_speaking:
                    self.events.put(("state", "model speaking" if speaking_now else "listening"))
                    was_speaking = speaking_now

                # Rising edge only. Flushing on every slice the gate reports
                # would chop the reply into one-second fragments instead of
                # interrupting it once.
                started_talking = self.gate.update(self.chunker.level)
                if started_talking and speaking_now:
                    # Barge-in. The model will hear this slice and stop on its
                    # own, but dropping the queued speech now is what makes the
                    # interruption feel immediate rather than a slice late.
                    self.player.flush()
                    self.events.put(("state", "you interrupted"))
                    was_speaking = False

                if duck < 1.0 and speaking_now:
                    slice_pcm = slice_pcm * duck

                try:
                    session.send_slice(slice_pcm)
                except Exception as exc:
                    if not self.live_stop.is_set():
                        on_error(exc)
                    return
        except Exception as exc:
            if not self.live_stop.is_set():
                on_error(exc)
        finally:
            self.chunker.stop()

    # ------------------------------------------------------- turn-based mode

    def reset(self) -> None:
        if self.busy:
            return
        # The KV cache *is* the history, so ending the session is what forgets.
        if self.mode_var.get() == "live":
            was_live = self.live_thread is not None
            self.stop_live()
        else:
            was_live = False
            self._drop_session()
        self.player.flush()
        self.transcript.configure(state="normal")
        self.transcript.delete("1.0", "end")
        self.transcript.configure(state="disabled")
        self.spoke_this_line = False
        self.status_var.set("New conversation.")
        if was_live:
            self.start_live()

    def _drop_session(self) -> None:
        if self.session is not None:
            self.session.close()
            self.session = None

    def start_talking(self) -> None:
        if self.busy or self.holding:
            return
        # On VoiceChat the stream is already open and running; holding the
        # button only opens the microphone gate in the live worker, which is
        # otherwise sending silence. There is no separate recording to start.
        if self.voicechat:
            if self.live_thread is None:
                self.status_var.set("Start the conversation first.")
                return
            self.holding = True
            self.player.flush()
            self.talk_btn.configure(bg="#f5d5d5", text="Listening — release when done")
            return
        self.holding = True
        self.player.flush()
        try:
            self.recorder.start()
        except Exception as exc:
            self.holding = False
            self.say(f"[microphone] {exc}\n", "note")
            return
        self.talk_btn.configure(bg="#f5d5d5", text="Recording — release to send")
        self.status_var.set("Recording…")

    def stop_talking(self) -> None:
        if not self.holding:
            return
        self.holding = False
        self.talk_btn.configure(bg="#e8eef5", text="Hold to talk  (or hold Space)")
        if self.voicechat:
            # Releasing the button makes the worker send silence; there is no
            # buffered turn to submit or close.
            return
        audio = self.recorder.stop()
        seconds = audio.size / omni_client.IN_RATE
        if seconds < 0.3:
            self.status_var.set("Too short — hold the button while you speak.")
            return
        self.say("You: ", "you")
        self.say(f"(spoke {seconds:.1f} s)\n", "note")
        self._send(audio=audio)

    def send_typed(self) -> None:
        text = self.typed_var.get().strip()
        if not text or self.busy:
            return
        self.typed_var.set("")
        self.say("You: ", "you")
        self.say(f"{text}\n")
        self._send(text=text)

    def _send(self, audio: np.ndarray | None = None, text: str = "") -> None:
        self.busy = True
        self.talk_btn.configure(state="disabled")
        self.say("Model: ", "model")

        url = self.url_var.get().strip()
        if self.session is not None and self.session.url != url:
            self._drop_session()

        threading.Thread(
            target=self._worker, args=(url, audio, text, self.speak_var.get()), daemon=True
        ).start()

    def _worker(self, url: str, audio: np.ndarray | None, text: str, speak: bool) -> None:
        try:
            if self.session is None:
                self.session = omni_client.OmniSession(url, ref_audio=self.ref_audio)
            result = self.session.turn(
                audio=audio,
                text=text,
                speak=speak,
                max_new_tokens=int(self.cfg["max_new_tokens"]),
                length_penalty=float(self.cfg["length_penalty"]),
                on_text=lambda piece: self.events.put(("text", piece)),
                on_audio=self.player.push if speak else None,
                on_status=lambda s: self.events.put(("status", s.capitalize() + "…")),
            )
        except Exception as exc:
            self._drop_session()
            self.events.put(("done", {"error": f"{type(exc).__name__}: {exc}"}))
            return

        if result.listened and not result.text:
            self.events.put(("note", "(the model chose to keep listening rather than reply)"))
        self.events.put(("done", {"metrics": result.metrics}))


def main() -> None:
    cfg = load_config()
    root = tk.Tk()
    try:
        ttk.Style().theme_use("vista")
    except tk.TclError:
        pass
    app = App(root, cfg)

    def on_close() -> None:
        # Leaving a session open would hold the server's only slot.
        app.stop_live()
        app._drop_session()
        app.player.close()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    root.mainloop()


if __name__ == "__main__":
    main()
