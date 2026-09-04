"""Command-line probe — one turn, no GUI.

Useful when something is wrong and you need to know whether the fault is the
server, the network, or the desktop app.

  python check.py --wav question.wav --out reply.wav
  python check.py --record 5 --out reply.wav
  python check.py --text "Say hello in one short sentence."   (MiniCPM-o only)

Works against either server; `backend` in config.json picks which, and --backend
overrides it for one run. VoiceChat has no text input channel, so --text is
rejected there rather than silently ignored.

Each run opens a fresh session, so there is no memory between invocations —
that is the probe being a probe, not a limitation of the server. Use talk.py
for a conversation.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import wave
from pathlib import Path

import numpy as np

import omni_client
import realtime_client


def read_wav(path: Path, rate_out: int) -> np.ndarray:
    with wave.open(str(path), "rb") as handle:
        frames = handle.readframes(handle.getnframes())
        rate, width, channels = handle.getframerate(), handle.getsampwidth(), handle.getnchannels()
    if width == 2:
        audio = np.frombuffer(frames, dtype="<i2").astype("<f4") / 32768.0
    elif width == 4:
        audio = np.frombuffer(frames, dtype="<f4")
    else:
        raise SystemExit(f"unsupported WAV sample width: {width} bytes")
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    from audio_io import resample

    return resample(audio, rate, rate_out)


def write_wav(path: Path, audio: np.ndarray, rate: int) -> None:
    pcm = np.clip(audio, -1.0, 1.0)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes((pcm * 32767).astype("<i2").tobytes())


def check_voicechat(args, cfg) -> int:
    """One paced question against the Realtime endpoint.

    Audio is sent on the model's real 80 ms clock, followed by silence while
    its continuously running text and speech channels finish the response.
    """
    import threading

    audio = args.audio
    if audio is None:
        raise SystemExit("VoiceChat needs speech: use --wav or --record")

    started = time.monotonic()
    first_text: list[float | None] = [None]
    first_audio: list[float | None] = [None]
    text_parts: list[str] = []
    reply: list[np.ndarray] = []
    done = threading.Event()
    failure: list[Exception] = []

    def on_text(piece: str) -> None:
        if first_text[0] is None:
            first_text[0] = time.monotonic() - started
        text_parts.append(piece)
        sys.stdout.write(piece)
        sys.stdout.flush()

    def on_audio(pcm: np.ndarray) -> None:
        if first_audio[0] is None:
            first_audio[0] = time.monotonic() - started
        reply.append(pcm)

    metrics: list[dict] = []

    def on_done(m: dict) -> None:
        metrics.append(m)
        done.set()

    def on_error(exc: Exception) -> None:
        failure.append(exc)
        done.set()

    session = realtime_client.RealtimeSession(
        args.url, instructions=cfg.get("instructions")
    )
    chunk = realtime_client.CHUNK
    silence = np.zeros(chunk, dtype=np.float32)

    try:
        session.start(
            on_text=on_text, on_audio=on_audio, on_done=on_done, on_error=on_error,
            on_speech_started=lambda: print("[speech started]", flush=True),
            on_speech_stopped=lambda: print("[speech stopped]", flush=True),
            on_status=lambda s: print(f"[{s}]", flush=True),
        )
        for i in range(0, audio.size, chunk):
            piece = audio[i : i + chunk]
            if piece.size < chunk:
                piece = np.pad(piece, (0, chunk - piece.size))
            session.send_audio(piece)
            # The duplex server consumes one semantic frame every 80 ms. Feed
            # a file on that same clock instead of dumping it into a queue.
            time.sleep(chunk / realtime_client.IN_RATE)

        # Keep the continuous timeline moving until the response settles.
        deadline = time.monotonic() + args.timeout
        while not done.is_set() and time.monotonic() < deadline:
            session.send_audio(silence)
            time.sleep(chunk / realtime_client.IN_RATE)

        if not done.is_set():
            print(f"\nno reply within {args.timeout:g} s", file=sys.stderr)
            return 1
    finally:
        session.close()

    print()
    if failure:
        print(f"error: {failure[0]}", file=sys.stderr)
        return 1

    speech = np.concatenate(reply) if reply else np.zeros(0, dtype=np.float32)
    if first_text[0] is not None:
        print(f"first text after  {first_text[0]:.2f} s")
    if first_audio[0] is not None:
        print(f"first audio after {first_audio[0]:.2f} s")
    rate = realtime_client.OUT_RATE
    print(f"reply audio: {speech.size / rate:.2f} s @ {rate} Hz")
    if metrics:
        print(f"metrics: {json.dumps(metrics[0])}")

    if args.out and speech.size:
        write_wav(args.out, speech, rate)
        print(f"wrote {args.out}")
    return 0


def main() -> int:
    cfg_path = Path(__file__).resolve().parent / "config.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8")) if cfg_path.exists() else {}

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", choices=("voicechat", "minicpm"),
                    default=cfg.get("backend", "voicechat"))
    ap.add_argument("--url", default=cfg.get("url", "ws://127.0.0.1:9070/v1/realtime"))
    ap.add_argument("--timeout", type=float, default=300.0,
                    help="VoiceChat only: how long to wait for the reply")
    ap.add_argument("--text", default="", help="send a typed turn (MiniCPM-o only)")
    ap.add_argument("--wav", type=Path, help="send this WAV as the spoken turn")
    ap.add_argument("--record", type=float, metavar="SECONDS", help="record from the mic instead")
    ap.add_argument("--out", type=Path, help="write the spoken reply to this WAV")
    ap.add_argument("--no-speak", action="store_true", help="text reply only, skip TTS")
    ap.add_argument("--ref-audio", default=cfg.get("ref_audio", ""), help="voice for the TTS to clone")
    ap.add_argument("--max-new-tokens", type=int, default=int(cfg.get("max_new_tokens", 512)))
    args = ap.parse_args()

    backend = realtime_client if args.backend == "voicechat" else omni_client
    in_rate = backend.IN_RATE

    audio = None
    if args.wav:
        audio = read_wav(args.wav, in_rate)
    elif args.record:
        import sounddevice as sd

        print(f"Recording {args.record:g} s — speak now…", flush=True)
        audio = sd.rec(
            int(args.record * in_rate),
            samplerate=in_rate, channels=1, dtype="float32",
        )
        sd.wait()
        audio = audio[:, 0]

    if audio is None and not args.text:
        ap.error("give one of --text, --wav or --record")

    if args.backend == "voicechat":
        if args.text:
            # Not a client limitation: the engine dispatches each turn on one
            # modality and VoiceChat has no text input channel at all.
            ap.error("VoiceChat has no text input channel; --text works only "
                     "with --backend minicpm. For text, use the OpenAI "
                     "endpoint on port 9071 instead.")
        args.audio = audio
        return check_voicechat(args, cfg)

    ref_audio = None
    if args.ref_audio:
        ref_path = Path(args.ref_audio)
        if not ref_path.is_absolute():
            ref_path = cfg_path.parent / ref_path
        ref_audio = read_wav(ref_path, in_rate)

    started = time.monotonic()
    first_text = [None]

    def on_text(piece: str) -> None:
        if first_text[0] is None:
            first_text[0] = time.monotonic() - started
        sys.stdout.write(piece)
        sys.stdout.flush()

    first_audio = [None]

    def on_audio(_pcm) -> None:
        if first_audio[0] is None:
            first_audio[0] = time.monotonic() - started

    result = omni_client.run_turn(
        args.url,
        audio=audio,
        text=args.text,
        speak=not args.no_speak,
        ref_audio=ref_audio,
        max_new_tokens=args.max_new_tokens,
        on_text=on_text,
        on_audio=on_audio,
        on_status=lambda s: print(f"[{s}]", flush=True),
    )

    print()
    if result.listened and not result.text:
        print("(the model chose to keep listening rather than reply)")
    if first_text[0] is not None:
        print(f"first text after  {first_text[0]:.2f} s")
    if first_audio[0] is not None:
        print(f"first audio after {first_audio[0]:.2f} s")
    print(f"reply audio: {result.audio.size / omni_client.OUT_RATE:.2f} s @ {omni_client.OUT_RATE} Hz")
    print(f"context used: {result.kv_used} tokens")
    print(f"metrics: {json.dumps(result.metrics)}")

    if args.out and result.audio.size:
        write_wav(args.out, result.audio, omni_client.OUT_RATE)
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
