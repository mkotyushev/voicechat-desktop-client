"""Desktop tools for VoiceChat, taken from the Cua driver Open Interpreter uses.

VoiceChat can already call functions — the Realtime protocol carries a tool
list in `session.update`, the model answers with a `function_call`, and
`realtime_client` surfaces it and sends the result back. What the client never
had was tools to offer. This is a source of them: `cua-driver`, the
computer-use daemon already installed on this machine and registered as an MCP
server in `~/.openinterpreter/config.toml`, which can see and drive the desktop.

Open Interpreter speaks MCP to it. This does not, and does not need to: the
daemon exposes the same tools on its command line — `cua-driver describe <tool>`
for a schema, `cua-driver call <tool> <json>` to run one — and both reach the
same running daemon the MCP server talks to. So the tool set is the one Open
Interpreter gets, without an MCP client, a JSON-RPC loop, or a dependency. A
call costs about 120 ms of process startup, which is nothing next to a turn.

**Tools are not free here, and neither are their results.** Definitions are
conditioned in the server's batched system prefill, so they install quickly,
but each token still occupies a model timeline/KV position. Tool results arrive
inside a live turn and are still injected one token per frame. So the list is a
whitelist, not the driver's full 60 tools; schemas are compacted before they are
sent; and results are clipped. Ask for three tools, not thirty, and prefer the
ones with short schemas.

None of this is MiniCPM-o's: its protocol has no tool channel at all.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tomllib
import unicodedata
from pathlib import Path

# Where to look for the driver, in order, when config.json does not say.
OI_CONFIG = Path.home() / ".openinterpreter" / "config.toml"
DEFAULT_INSTALL = (
    Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Cua" / "cua-driver"
    / "bin" / "cua-driver.exe"
)

# Definitions still consume the finite VoiceChat timeline even though the
# server prefills them as a batch, so schemas are trimmed to what a caller needs
# to fill them in: the first paragraph of the tool's description, one clipped
# line per parameter.
TOOL_DESCRIPTION_CHARS = 220
PARAM_DESCRIPTION_CHARS = 90

# Every tool takes an optional `session` label for multi-call work. The
# implicit session used when it is omitted is the right one for a single
# conversation, and the parameter's description is longer than most tools.
DROPPED_PARAMS = {"session"}

# Schema keys worth the tokens. `enum` is what stops the model inventing
# values; the rest of JSON Schema it will not honour anyway.
KEPT_SCHEMA_KEYS = ("type", "enum")

# Windows only: keep `pythonw` from flashing a console for every call.
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)

# NVIDIA's model card requires system prompts and tool responses to be ASCII,
# and the bridge enforces that by deleting everything else — so an em dash
# between two words arrives as a double space, and a Cyrillic app name arrives
# as nothing at all. The driver's own text is full of typographic punctuation,
# so it is worth spending the character on an ASCII equivalent here instead of
# letting the server drop it.
ASCII_SUBSTITUTES = {
    "—": "-", "–": "-", "−": "-", " ": " ",
    "‘": "'", "’": "'", "“": '"', "”": '"',
    "…": "...", "•": "*", "→": "->", "×": "x",
}
_ASCII_TABLE = str.maketrans(ASCII_SUBSTITUTES)

class ToolError(RuntimeError):
    pass


def find_driver(explicit: str | None = None) -> Path:
    """Locate `cua-driver`, preferring the copy Open Interpreter drives.

    Reading its path out of `~/.openinterpreter/config.toml` is the point: this
    is meant to be the same daemon, the same tools and the same install to
    update — not a second one that drifts.
    """
    if explicit:
        path = Path(explicit).expanduser()
        if not path.exists():
            raise ToolError(f"no cua-driver at {path}")
        return path

    if OI_CONFIG.exists():
        try:
            with OI_CONFIG.open("rb") as handle:
                servers = tomllib.load(handle).get("mcp_servers") or {}
            command = (servers.get("cua-driver") or {}).get("command")
        except (OSError, tomllib.TOMLDecodeError):
            command = None
        if command and Path(command).exists():
            return Path(command)

    if DEFAULT_INSTALL.exists():
        return DEFAULT_INSTALL

    found = shutil.which("cua-driver")
    if found:
        return Path(found)
    raise ToolError(
        "cua-driver not found — install Cua, or set tools.driver in config.json"
    )


# --------------------------------------------------------------- compaction


def to_ascii(text: str) -> str:
    """Everything the model reads, in the ASCII the model card asks for.

    Substitutions first, then NFKD — which turns `é` into `e` and a
    non-breaking hyphen into one — and only then drop what is left. Scripts
    with no ASCII form, Cyrillic among them, still disappear; that is the
    card's rule, and it applies whether the deleting happens here or on the
    server. Doing it here at least means the clip limits count characters that
    will actually survive.
    """
    folded = unicodedata.normalize("NFKD", text.translate(_ASCII_TABLE))
    return folded.encode("ascii", "ignore").decode("ascii")


def _clip(text: str, limit: int) -> str:
    text = " ".join(to_ascii(text).split())
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0] + "..."


def _describe(text: str, limit: int) -> str:
    """The first paragraph of a tool's description, usually.

    A paragraph ending in a colon is introducing something — `list_apps` ends
    "with per-app state flags:" and then lists them — and on its own it
    promises the model information it never gets. Keep reading in that case,
    and let the clip decide where to stop.
    """
    first = text.split("\n\n")[0].strip()
    return _clip(text if first.endswith(":") else first, limit)


def _compact_schema(schema: dict, keep: list[str] | None = None) -> dict:
    """The parts of an input schema a caller needs, and nothing else.

    `keep` narrows it further to named parameters (plus whatever the schema
    marks required, which cannot be dropped without breaking the call). It is
    worth using: `launch_app` takes ten parameters, four of which are macOS
    parity no-ops, and `name` and `urls` are the two anyone asks for by voice.
    """
    required = set(schema.get("required") or [])
    wanted = None if keep is None else set(keep) | required

    properties = {}
    for name, spec in (schema.get("properties") or {}).items():
        if name in DROPPED_PARAMS or not isinstance(spec, dict):
            continue
        if wanted is not None and name not in wanted:
            continue
        kept = {key: spec[key] for key in KEPT_SCHEMA_KEYS if key in spec}
        if isinstance(spec.get("items"), dict):
            kept["items"] = {"type": spec["items"].get("type", "string")}
        if spec.get("description"):
            kept["description"] = _clip(spec["description"], PARAM_DESCRIPTION_CHARS)
        properties[name] = kept

    compact: dict = {"type": "object", "properties": properties}
    still_required = [name for name in required if name in properties]
    if still_required:
        compact["required"] = still_required
    return compact


def parse_describe(text: str) -> tuple[str, dict]:
    """Pull the description and the input schema out of `describe` output.

    `describe` prints for a human — `name:`, `description:`, then the schema as
    pretty JSON under `input_schema:` — and has no `--json` that changes it.
    The schema is a whole document at a known marker, so splitting on the
    marker is the whole parser.
    """
    marker = "\ninput_schema:"
    head, sep, tail = text.partition(marker)
    if not sep:
        raise ToolError(f"unexpected describe output: {text.strip()[:200]}")
    try:
        schema = json.loads(tail)
    except ValueError as exc:
        raise ToolError(f"could not read the input schema: {exc}") from exc

    _, sep, described = head.partition("description:")
    return (described if sep else head).strip(), schema


# ------------------------------------------------------------------ toolbox


class ToolBox:
    """A whitelist of driver tools, in the shape the Realtime protocol wants.

    Built once, when the app starts: `describe` is a subprocess per tool, and
    the definitions do not change while the driver stays installed.

    `allow` is either a list of tool names, or a mapping of name to the
    parameters worth showing the model — `{"launch_app": ["name", "urls"]}` —
    with null or an empty list meaning all of them.
    """

    def __init__(
        self,
        allow: list[str] | dict[str, list[str] | None],
        *,
        driver: str | None = None,
        timeout_s: float = 20.0,
        max_result_chars: int = 300,
    ):
        wanted = allow if isinstance(allow, dict) else {name: None for name in allow}
        self.driver = find_driver(driver)
        self.timeout_s = float(timeout_s)
        self.max_result_chars = int(max_result_chars)
        self.definitions = [
            self._define(name, keep or None) for name, keep in wanted.items()
        ]
        self.names = {definition["name"] for definition in self.definitions}

    def _run(self, args: list[str], timeout: float) -> tuple[int, str]:
        result = subprocess.run(
            [str(self.driver), *args],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            creationflags=NO_WINDOW,
        )
        return result.returncode, (result.stdout or result.stderr or "").strip()

    def _define(self, name: str, keep: list[str] | None = None) -> dict:
        try:
            code, out = self._run(["describe", name], timeout=30.0)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise ToolError(f"cua-driver describe {name}: {exc}") from exc
        if code != 0:
            raise ToolError(f"cua-driver has no tool {name!r}: {out}")
        description, schema = parse_describe(out)
        return {
            "type": "function",
            "name": name,
            "description": _describe(description, TOOL_DESCRIPTION_CHARS),
            "parameters": _compact_schema(schema, keep),
        }

    def call(self, name: str, arguments: str) -> str:
        """Run one tool and return what to hand back to the model.

        Never raises. The model's clock is frozen until a result arrives, so
        every failure has to come back as a result it can read.
        """
        if name not in self.names:
            return json.dumps({"error": f"no tool named {name}"})
        try:
            parsed = json.loads(arguments or "{}")
            # The model is told to write `"arguments": "tool_args"` — a json
            # string — and the bridge passes a string through untouched, so a
            # model that quotes an already-encoded object arrives doubly
            # encoded. One more parse is cheaper than a tool that never works.
            if isinstance(parsed, str):
                parsed = json.loads(parsed)
        except ValueError:
            return json.dumps({"error": "arguments were not valid JSON"})
        if not isinstance(parsed, dict):
            return json.dumps({"error": "arguments must be a JSON object"})

        try:
            code, out = self._run(["call", name, json.dumps(parsed)], self.timeout_s)
        except subprocess.TimeoutExpired:
            return json.dumps({"error": f"{name} timed out after {self.timeout_s:.0f}s"})
        except OSError as exc:
            return json.dumps({"error": f"could not run cua-driver: {exc}"})

        if code != 0:
            return json.dumps({"error": _clip(out, self.max_result_chars)})
        if not out:
            return json.dumps({"ok": True})
        # Tool responses are held to the same ASCII rule as the prompt.
        out = to_ascii(out)
        # Clipping mid-JSON hands the model a truncated document, and that is
        # the intended trade: a whole accessibility tree is minutes of decode,
        # and what the model asked for is usually in the first few fields.
        if len(out) > self.max_result_chars:
            return _clip(out, self.max_result_chars) + " [truncated]"
        return out

def main() -> None:
    """Size a tool list without starting the app.

        python cua_tools.py list_apps launch_app:name,urls

    Prints what each tool costs, so a list can be sized before a conversation
    is spent waiting for it. `tool:a,b` is the command-line spelling of the
    parameter filter that config.json writes as `{"tool": ["a", "b"]}`.
    """
    import sys

    argv = sys.argv[1:] or ["list_apps", "launch_app:name,urls", "clipboard_read"]
    allow: dict[str, list[str] | None] = {}
    for arg in argv:
        name, _, params = arg.partition(":")
        allow[name] = [p.strip() for p in params.split(",") if p.strip()] or None
    box = ToolBox(allow)
    print(f"driver: {box.driver}\n")
    for definition in box.definitions:
        # Sized as it goes over the wire, not as it is printed here.
        print(f"--- {definition['name']} ({len(json.dumps(definition))} chars) ---")
        print(json.dumps(definition, indent=2))
    print(
        f"\n{len(box.definitions)} tools, "
        f"{len(json.dumps(box.definitions))} chars in the batched system prefill"
    )


if __name__ == "__main__":
    main()
