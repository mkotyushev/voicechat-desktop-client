# VoiceChat desktop client

One Windows desktop app for multiple speech-to-speech models. `backend` in
`config.json` selects the server protocol; everything below that is shared.

For the NVIDIA VoiceChat server, see
[`nemotron-voicechat-realtime-gguf`](https://github.com/mkotyushev/nemotron-voicechat-realtime-gguf).

| `backend` | Model | Protocol | Deployment |
|---|---|---|---|
| `"voicechat"` *(default)* | NemotronLabs VoiceChat 11B | OpenAI Realtime | [server repo](https://github.com/mkotyushev/nemotron-voicechat-realtime-gguf) |
| `"minicpm"` | MiniCPM-o 4.5 | bespoke `/backend` | your MiniCPM-o deployment |

Each deployment's own README covers the server side. **They cannot both run** —
either one alone fills the GPU.

## Which to use

**VoiceChat** answers in better English and its replies are more coherent. It
runs its native 12.5 Hz full-duplex timeline: the microphone keeps reaching the
model while it speaks, it may answer before you stop, and talking flushes local
playback immediately. It cannot take typed turns.

**MiniCPM-o** is also full-duplex, but makes a coarse listen/speak decision once
per second and accepts typed turns. It is easier to tune but more likely to
mishear you and to answer in Chinese.

Neither is strictly better. VoiceChat has the finer realtime clock and stronger
English answers; MiniCPM-o has typed turns and more explicit pacing controls.

```
talk.cmd            double-click to start the app
setup.cmd           one-time: create .venv and install dependencies

talk.py             the desktop app (Tkinter)
check.py            one-turn command-line probe, for when something is wrong
realtime_client.py  the OpenAI Realtime protocol  (VoiceChat)
cua_tools.py        desktop tools VoiceChat can call, from the Cua driver
omni_client.py      the /backend protocol, both modes  (MiniCPM-o)
audio_io.py         microphone capture, streaming playback, speech gate
config.json         backend, server address, per-mode settings
ref_voice.wav       the voice MiniCPM-o's TTS clones (VoiceChat ignores it)
test_question.wav   a spoken question, for check.py
```

## Setup

Needs `uv` (already on PATH), a microphone, and — for live conversation —
**headphones**. See [Echo](#echo) for why.

```
setup.cmd
```

Set `url` in `config.json` to your server address, then start the configured
server. For VoiceChat:

```
ssh <user>@<server-host> "cd ~/nemotron-voicechat-realtime-gguf && docker compose up -d"
```

VoiceChat needs no warmup — it loads at startup and reports unhealthy until the
weights are resident, so `docker compose ps` showing healthy means ready. For
MiniCPM-o:

```
ssh <user>@<server-host> "cd <minicpm-deployment-dir> && docker compose up -d && ./warmup.sh live"
```

## Talking to VoiceChat

Press **Start conversation** and talk. Every 80 ms microphone frame advances the
model even while its speech is coming back. The model may greet first, answer
before you have completely stopped, or yield when you talk over it. Text leads
the codec slightly and the voice follows in streamed chunks.

In **push to talk** mode the conversation still has to be started first; holding
the button or **Space** only opens the microphone. Releasing it substitutes
silence on the same continuous timeline; it does not submit a buffered turn.

Things VoiceChat will not do, none of which are client limitations:

- **No typed turns.** The model has one 12.5 Hz timeline and no text input
  channel. The typed row is hidden. For text, use the OpenAI-compatible endpoint
  on port 9071 — the deployment README covers it.
- **English only**, whatever you speak to it.
- **Every reply is spoken**, so "Speak replies" is disabled.

### Desktop tools

VoiceChat can call functions, and there is already a supply of them on this
machine: `cua-driver`, the computer-use daemon Open Interpreter drives, which
`~/.openinterpreter/config.toml` registers as an MCP server. Turn it on in
`config.json` — `"tools": {"enabled": true}` — and ask for something the three
default tools can do: *open Notepad*, *open youtube.com*, *what is running*,
*what is on my clipboard*. The call and its result appear in the transcript as
grey notes, and the status changes from `running tool` to `processing tool
result`, so you can see where it is.

The client does not speak MCP to it. It shells out to the same daemon's command
line — `cua-driver describe <tool>` for a schema, `cua-driver call <tool> <json>`
to run one — which is the same tool set Open Interpreter gets, without an MCP
client or a JSON-RPC loop, at about 120 ms of process startup per call. The
driver's own permission mode decides what is allowed; this client adds no
approval step, so only offer tools you would let the model run unattended.

Take that literally with `launch_app`, because a name it cannot match is not an
error: it falls back to a `shell:AppsFolder` search and launches the closest
thing it finds. `{"name": "x"}` launched the Xbox app. This model mishears
ordinary words often enough that a spoken app name lands somewhere unintended
sooner or later — the launches are hidden and unfocused, so the cost is a stray
process rather than a stolen window, but it is a reason to keep the list short
and the feature off when you are not using it.

The bridge renders `instructions` and `tools` into one text system prompt in
Nemotron-Nano-9B-v2's `<AVAILABLE_TOOLS>` format. The server conditions that
prefix as one logical llama.cpp prefill batch: the current three-tool prompt is
514 tokens and measured **0.36 s** to install. Each token still occupies one
position on VoiceChat's finite 12.5 Hz timeline, so definitions are compacted to
leave room for the conversation even though they no longer cause a long startup
wait. Inspect the size of a list with:

```
.venv\Scripts\python cua_tools.py list_apps launch_app:name,urls clipboard_read
```

That is what the parameter filter in `allow` is for. `launch_app` takes ten
parameters, four of them macOS parity no-ops; the default keeps only `name` and
`urls`. Anything the tool marks required is kept whether you list it or not.

Adding a tool is a line in `allow`; `cua-driver list-tools` is the menu.

Results are clipped to `max_result_chars` for the same reason — they are read
at the same 12.5 tokens a second, and `list_apps` unclipped is minutes of
silence before the answer. A clipped result ends in `[truncated]`, and the
model reads it that way.

The model card requires prompts and tool responses to be **ASCII**, and the
bridge enforces that by deleting everything else — an em dash between two words
would arrive as a double space, and `cua-driver`'s text is full of them. So
schemas and results are folded to ASCII here first (`—` to `-`, `…` to `...`,
`café` to `cafe`). Scripts with no ASCII form, Cyrillic included, still do not
survive; that is the card's rule, not the client's.

If the driver is missing or a tool name is wrong, the transcript says so once
and the conversation runs without tools. Nothing here applies to MiniCPM-o: its
protocol has no tool channel at all.

## Talking to MiniCPM-o

Run `talk.cmd`.

### Live conversation

Press **Start conversation** and talk. The indicator on the right shows whether
the model is listening or speaking. Talk over it and it stops — the model hears
you in the stream and yields, and the app cuts the already-buffered speech so
the interruption lands immediately rather than a second later.

Press **End conversation** to hang up. That closes the session, which is also
what forgets the conversation.

VoiceChat can begin returning text about 1.2 s after speech begins and audio
around 1.9 s, before a longer question has finished. MiniCPM-o's balance is
coarser; tune its `listen_prob_scale` below.

### Push to talk

Hold the button or the **Space** bar, speak, release. VoiceChat only gates its
microphone on the same continuous duplex timeline and has no typed turns.
MiniCPM-o also accepts typed turns here; **Speak replies** toggles its audio.

**New conversation** ends the session and starts fresh in either mode.

## Checking it works without the GUI

The quickest check records directly from your microphone and needs no sample
audio file:

```
.venv\Scripts\python check.py --record 5 --out reply.wav
.venv\Scripts\python check.py --wav test_question.wav --out reply.wav
.venv\Scripts\python check.py --text "Say hello."      MiniCPM-o only
```

It uses `backend` from `config.json`; `--backend voicechat` / `--backend
minicpm` overrides that for one run, and `--url` overrides the address.

One turn per run, and it prints time-to-first-text and time-to-first-audio
separately, so it tells "the server is slow" apart from "the app is slow", and
on VoiceChat it tells both apart from "the speech is not being streamed".

Audio recordings are deliberately excluded from this repository. If you use
MiniCPM-o voice cloning, record about six seconds of clean mono speech as
`ref_voice.wav`. For the file-based probe, record any spoken question as
`test_question.wav`. Use your own voice or audio you have permission to use.

A worked example against VoiceChat:

```
[speech started]
Hi there! How can I help you today?The capital of France is Paris.
[speech stopped]
The capital of France is Paris.
first text after  1.19 s
first audio after 1.91 s
reply audio: 8.72 s @ 24000 Hz
metrics: {"frames": 110, "timeline_frame": 161, "reason": "completed"}
```

With `STREAM_FRAMES=8`, first audio normally trails first text by roughly one
640 ms speech chunk. If it trails by several seconds, speech streaming is off —
check `STREAM_FRAMES` in the deployment's `.env`.

## Tuning MiniCPM-o's live mode

VoiceChat has no equivalent: its turn taking is learned on the continuous
80 ms timeline. Its energy detector never withholds audio or delays a response
start; it supplies the hearing-you UI and prevents a completed answer from
reopening until a new speech epoch.

`config.json` -> `duplex`:

| Key | Default | What it does |
|---|---|---|
| `listen_prob_scale` | 0.5 | **The one to touch.** Bias on the per-second speak/listen decision. Lower = readier to speak. |
| `force_listen_count` | 3 | Slices at the start of a call where the model is held quiet. |
| `max_new_speak_tokens_per_chunk` | 20 | How far a reply runs before the model re-checks whether you started talking. Raising it makes interruption sluggish. |

**If it never answers**, lower `listen_prob_scale` toward 0.3. The engine's own
default of 1.0 measured as marginal here — whole questions would go by with the
model choosing to listen every single second and never replying. 0.5 is
reliable and still waits for you to finish.

**If it talks over you**, raise it toward 1.0.

### Echo

There is no acoustic echo cancellation. On speakers the model hears its own
voice, treats it as more conversation, and can end up talking to itself.

Use headphones. If you can't, set `mic_duck_while_speaking` to about `0.2`,
which attenuates the microphone while the model is talking. It is a partial fix
and it makes barge-in less sensitive, which is why the default is `1.0` (off).

## What to expect

### VoiceChat

The cached encoder and LLM advance from the first audio frame. In the paced test
question, first text arrived 1.2 s after the stream began and audio at 1.9 s,
while the question was still being sent. On this 3090, TTS chunk decoding can
briefly put the clock about half a second behind; it catches up between chunks.
NVIDIA quotes ~450 ms turn-taking for its H100 container.

No warmup: the server loads at startup, so the first turn costs the same as the
tenth.

### MiniCPM-o

The first session after the server starts loads ~13 GB of weights and takes
about 20 seconds; `warmup.sh` gets that out of the way. Switching between Live
and Push-to-talk reloads the model too (~7 s warm) — the two modes cannot share
a loaded context, so pick one and stay in it.

In live mode the server spends roughly 90 ms of each second listening and
150–400 ms speaking, so it holds real time with plenty of headroom and the
conversation does not drift behind.

### The conversation is the connection

Memory lives in the server's KV cache, so closing the socket forgets
everything. "New conversation" and "End conversation" really do end it, and if
a turn errors the app drops the session and the next one starts from nothing.

The server allows **one session at a time across all clients**, so leaving the
app open holds that slot; close it when someone else needs the GPU.

### You cannot attach an instruction to a spoken turn

VoiceChat has no mid-call text input channel, so instructions are installed once
in `session.update`; a per-utterance typed instruction has nowhere to go. For
MiniCPM-o push-to-talk, send it as its own typed turn first.

That works for style, but not for language: the model answers in the language
you speak to it, and a typed "answer only in Russian from now on" did not change
the language of the spoken turn that followed.

### What has actually been tested

Spoken English in, correct English answer out, as text and as speech; typed
turns; memory carried across mixed voice and text turns; full-duplex with
proactive turn-taking and repeated sessions; interrupting playback.

Non-English speech has **not** been tested with real recordings. Image and video
input are supported by the protocol but this client does not send them.

## config.json

| Key | Applies to | Meaning |
|---|---|---|
| `backend` | both | `"voicechat"` or `"minicpm"`. Decides the protocol, the audio rates and half the UI. **Change `url` with it.** |
| `url` | both | VoiceChat: `ws://<server-host>:9070/v1/realtime`. MiniCPM-o: `ws://<server-host>:9060/backend`. The checked-in `127.0.0.1` default is safe for a local or SSH-forwarded server; replace it for a remote host. |
| `mode` | both | `"live"` or `"ptt"` — which mode the app opens in. |
| `instructions` | VoiceChat | System prompt, or `null`. Batched prefill is fast, but every token still consumes one logical 80 ms timeline/KV position and shortens the session. |
| `tools` | VoiceChat | Desktop tools from the Cua driver, off by default. `allow` is a list of tool names, or a mapping of name to the parameters worth showing the model (`null` for all of them). `driver` overrides where `cua-driver` is found; `timeout_s` and `max_result_chars` bound one call. See [Desktop tools](#desktop-tools). |
| `speak` | MiniCPM-o | Push-to-talk only: initial state of "Speak replies". VoiceChat always speaks. |
| `max_new_tokens` | MiniCPM-o | Push-to-talk reply length cap. 512 is a few sentences. |
| `length_penalty` | MiniCPM-o | >1 shortens replies. |
| `ref_audio` | MiniCPM-o | WAV whose voice the TTS clones. Any ~6 s of clean mono speech. VoiceChat's voice is fixed and it ignores this. |
| `context_size` | MiniCPM-o | **Must match `N_CTX` in the server's `.env`** (8192). Only used to warn before the conversation outgrows the cache. |
| `mic_duck_while_speaking` | both | See [Echo](#echo). |
| `playback_buffer_ms` | both | 1000. Audio buffered before playback starts. VoiceChat receives 640 ms chunks, so this waits for two and covers the 3090's small generation drift without hiding the whole reply. |
| `duplex` | MiniCPM-o | See [Tuning MiniCPM-o's live mode](#tuning-minicpm-os-live-mode). |
| `input_device` / `output_device` | both | `null` = system default. An index or name substring from `python -m sounddevice`. |

The URL box in the app overrides `url` for that run — but **not** `backend`, so
it cannot point the app at the other server. Edit `config.json` for that.

## Troubleshooting

### VoiceChat

**It never answers, and the indicator never says "hearing you".** First check
the selected microphone. The indicator is an energy hint, so `VAD_MARGIN` can
fix the display, but the audio is sent to the model regardless; a silent input
device or wrong URL is the inference problem.

**It answers before you have finished a sentence.** That is the model's learned
duplex turn taking. `VAD_SILENCE_MS` cannot change it. Keep talking to interrupt,
and use headphones so speaker echo is not mistaken for another voice.

**The words arrive but the voice is seconds late.** Speech streaming is off.
`STREAM_FRAMES` in the deployment's `.env` should be 8; zero disables the
duplex speech chunks because there is no whole-turn wav fallback on this path.

**The speech clicks or ticks about once a second.** The playback resampler has
lost its state. `Player` uses `audio_io.StreamResampler` precisely so that 80 ms
chunks join without a seam on a machine whose output device is not 24 kHz; a
plain per-chunk resample puts an artifact at every boundary.

**The voice repeatedly stops and resumes, but a saved reply is continuous.**
That is playback underrun, not a wav seam. The server build needs
`GGML_CUDA_GRAPHS=ON`, `STREAM_FRAMES=8`, and the client should keep
`playback_buffer_ms` at 1000. On the current 3090 build, 640 ms speech chunks
arrive 683-740 ms apart; the two-chunk cushion covers that residual drift.

**`another session is active`.** The server holds one conversation at a time
across all clients. Close the other copy of the app. Ending this client now
force-closes its socket, and the server skips a pending tool call and releases
the slot in a bounded cleanup path; a persistent error therefore means another
real client is connected and should be checked in the server log.

**It answered in English something you asked in another language.** The model is
English-only. There is no setting.

**Typed turns are missing.** VoiceChat has no text input channel. Use the
OpenAI-compatible endpoint on port 9071.

**Tools are on, but the model never calls one.** It has the list — the note at
the start of the conversation says how many — so this is prompting, not
plumbing. A one-line `instructions` naming what it can do ("you can open apps
on this computer") is usually enough.

**"[tools] unavailable".** `cua-driver` was not found, or a name in `allow` is
not one of its tools. `cua-driver list-tools` prints the real names, and
`cua-driver status` says whether the daemon is running.

### MiniCPM-o

**Live mode never answers.** Lower `listen_prob_scale`. See above — this is the
expected first thing to tune, not a fault.

**It answers itself in a loop.** Speaker echo. Use headphones, or set
`mic_duck_while_speaking` to `0.2`.

**"you interrupted" flashes constantly while you sit in silence**, and the
reply is chopped into roughly one second of speech, one of silence. These are
one fault, not two: the barge-in gate is triggering on room tone, and each
trigger drops the queued audio. The gate measures your room over the first four
slices of a call, so if it happens, something was making noise during those
four seconds — start the call in silence. If it persists, your microphone's
idle level is unusually high; raise `margin` in `SpeechGate` (audio_io.py) from
3.5, or check `python -m sounddevice` for a quieter input device.

**Speech is broken up but the text arrives clean.** The gaps are in playback,
not generation. Raise `playback_buffer_ms` above 1000 — the server emits speech
about a second at a time but not on an exact clock, and a short chunk followed
late by the next one leaves a hole the buffer is there to bridge.

**Nothing happens for a long time on the first turn.** The server loads lazily.
Run `./warmup.sh live` (or `turn`) on the 3090 first.

**`ConnectionRefusedError`.** Check that the selected container is up:
`ssh <user>@<server-host> "cd <deployment-dir> && docker compose ps"`.

**`server closed the session`, or opening hangs.** Something else holds the
server's one session slot — another copy of this app, or a socket that never
closed. Close other instances; if it persists,
`ssh <user>@<server-host> "cd <deployment-dir> && docker compose restart"`, then
warm up again.

**Replies come back in Chinese.** The server's hard-coded Chinese system prompt
is winning over your audio. There is no client-side fix — text in a spoken turn
is dropped. A longer, clearer utterance is what helps.

**The model answers a question you did not ask.** It misheard. There is no
transcript and no confidence signal, so a poorly recognised utterance comes back
as a confident answer to something else. Say it again rather than rephrasing
around the wrong answer.

**Replies get worse late in a long conversation.** Check the context figure in
the status line; past ~85% the app warns. Start a new conversation.

**The voice is too fast or too deep.** A sample-rate mismatch, not a model
problem. Input is 16 kHz and output 24 kHz; both are fixed in `omni_client.py`.

**No microphone / no sound.** `python -m sounddevice` lists devices; set
`input_device` / `output_device`. The app opens the mic at 16 kHz and the
speakers at 24 kHz, falling back to the device's own rate with a band-limited
resample if the driver refuses.
