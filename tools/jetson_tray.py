#!/usr/bin/env python3
"""Opt-in Jetson system tray for a source-installed Strata server.

The tray never owns the model process. A user systemd unit owns it, so closing
the indicator does not unexpectedly stop an active API or leave an orphan.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import threading
import urllib.error
import urllib.request
import webbrowser


ROOT = Path(__file__).resolve().parent.parent
UNIT = "strata-jetson.service"
CONFIG_HOME = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
STATE = CONFIG_HOME / "strata" / "jetson-tray.json"
UNIT_FILE = CONFIG_HOME / "systemd" / "user" / UNIT
AUTOSTART = CONFIG_HOME / "autostart" / "strata-jetson-tray.desktop"
ICON = ROOT / "apps" / "desktop" / "src-tauri" / "icons" / "32x32.png"
URL = "http://127.0.0.1:8080"


def models() -> list[dict]:
    """List usable source-install run configs; never scan huge model files."""
    found = []
    for path in sorted(ROOT.glob("strata-*.json")):
        try:
            cfg = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(cfg, dict) or not all(cfg.get(k) for k in ("exe", "args", "model_name")):
                continue
            args = cfg["args"]
            if not isinstance(args, list):
                continue
            native = next((args[i + 1] for i, arg in enumerate(args[:-1]) if arg == "--native"), None)
            pack = next((args[i + 1] for i, arg in enumerate(args[:-1]) if arg == "--pack"), None)
            ready = Path(cfg["exe"]).is_file() and bool(pack and Path(pack).is_dir())
            if native:
                ready = ready and Path(native).is_file()
            found.append({"file": path.name, "name": str(cfg["model_name"]),
                          "ready": ready, "native": native, "pack": pack})
        except (OSError, ValueError, TypeError, IndexError):
            continue
    return found


def selected_model() -> dict | None:
    available = models()
    if not available:
        return None
    try:
        selected = json.loads(STATE.read_text(encoding="utf-8")).get("selected")
    except (OSError, ValueError, AttributeError):
        selected = None
    return next((m for m in available if m["file"] == selected), available[0])


def select_model(filename: str) -> None:
    if filename not in {m["file"] for m in models()}:
        raise ValueError("model config not found")
    if service_active():
        raise RuntimeError("pause the service before changing models")
    STATE.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = STATE.with_suffix(".tmp")
    temporary.write_text(json.dumps({"selected": filename}) + "\n", encoding="utf-8")
    temporary.chmod(0o600)
    temporary.replace(STATE)


def systemctl(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["systemctl", "--user", *args], text=True, capture_output=True, timeout=35)


def unit_state() -> str:
    result = systemctl("show", UNIT, "--property=ActiveState", "--value")
    return result.stdout.strip() if result.returncode == 0 else "not installed"


def service_active() -> bool:
    return unit_state() in ("active", "activating", "reloading", "deactivating")


def port_open() -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.2)
        return sock.connect_ex(("127.0.0.1", 8080)) == 0


def request(path: str, *, post: bool = False, timeout: float = 2.0) -> dict:
    model = selected_model()
    headers = {"Accept": "application/json"}
    if post:
        headers["Content-Type"] = "application/json"
    if model:
        cfg = json.loads((ROOT / model["file"]).read_text(encoding="utf-8"))
        keys = cfg.get("api_key")
        key = keys[0] if isinstance(keys, list) and keys else keys
        if isinstance(key, str) and key:
            headers["Authorization"] = "Bearer " + key.split(",")[0].strip()
    req = urllib.request.Request(URL + path, data=b"{}" if post else None, headers=headers,
                                 method="POST" if post else "GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        try:
            detail = json.load(exc).get("error", {}).get("message", exc.reason)
        except (ValueError, AttributeError):
            detail = exc.reason
        raise RuntimeError(f"API {exc.code}: {detail}") from exc


def status() -> dict:
    unit = unit_state()
    model = selected_model()
    if unit not in ("active", "activating"):
        occupied = port_open()
        if occupied:
            return {"state": "error", "unit": unit, "model": model, "port": True,
                    "error": "port 8080 is used by another process"}
        return {"state": "failed" if unit == "failed" else "paused", "unit": unit,
                "model": model, "port": False}
    try:
        api = request("/v1/status", timeout=1.0)
        if api.get("service") != "strata" or not model or api.get("model") != model["name"]:
            return {"state": "error", "unit": unit, "model": model, "port": True,
                    "error": "port 8080 does not report the selected Strata model"}
        return {"state": "ready" if api.get("loaded") else "unloaded", "unit": unit,
                "model": model, "port": True, "api": api}
    except (OSError, RuntimeError, ValueError):
        return {"state": "starting", "unit": unit, "model": model, "port": False}


def service() -> None:
    model = selected_model()
    if not model or not model["ready"]:
        raise RuntimeError("no prepared model found; run ./setup.sh --no-start first")
    python = ROOT / ".venv" / "bin" / "python"
    if not python.is_file():
        raise RuntimeError("Strata Python environment missing; run ./setup.sh --no-start first")
    os.chdir(ROOT)
    os.execv(str(python), [str(python), str(ROOT / "serve" / "server.py"), "--engine", "strata",
                            "--config", str(ROOT / model["file"]), "--host", "127.0.0.1", "--port", "8080"])


def systemd_quote(path: Path) -> str:
    return '"' + str(path).replace("\\", "\\\\").replace('"', '\\"') + '"'


def install() -> None:
    if not selected_model():
        raise RuntimeError("no Strata model config found; run ./setup.sh --no-start first")
    if UNIT_FILE.exists() and "Managed by tools/jetson_tray.py" not in UNIT_FILE.read_text(encoding="utf-8"):
        raise RuntimeError(f"existing {UNIT_FILE} is not managed by this tool")
    if AUTOSTART.exists() and "Managed by tools/jetson_tray.py" not in AUTOSTART.read_text(encoding="utf-8"):
        raise RuntimeError(f"existing {AUTOSTART} is not managed by this tool")
    UNIT_FILE.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    AUTOSTART.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    UNIT_FILE.write_text(
        "# Managed by tools/jetson_tray.py\n"
        "[Unit]\nDescription=Strata Jetson model server\nAfter=network.target\n\n"
        "[Service]\nType=simple\n"
        f"WorkingDirectory={ROOT}\n"
        f"ExecStart={systemd_quote(Path(sys.executable))} {systemd_quote(Path(__file__).resolve())} service\n"
        "KillSignal=SIGTERM\nTimeoutStopSec=40\n\n"
        "[Install]\nWantedBy=default.target\n", encoding="utf-8")
    AUTOSTART.write_text(
        "# Managed by tools/jetson_tray.py\n"
        "[Desktop Entry]\nType=Application\nName=Strata Jetson tray\n"
        f"Exec={sys.executable} {Path(__file__).resolve()} tray\n"
        f"Icon={ICON}\nTerminal=false\nX-GNOME-Autostart-enabled=true\n", encoding="utf-8")
    result = systemctl("daemon-reload")
    if result.returncode:
        raise RuntimeError(result.stderr.strip())
    print(f"Installed tray autostart and {UNIT}; service remains paused.")


def action(command: str, filename: str | None = None) -> None:
    if command == "resume":
        model = selected_model()
        if not model or not model["ready"]:
            raise RuntimeError("selected model is not prepared")
        if not UNIT_FILE.exists():
            raise RuntimeError("run install first")
        if not service_active() and port_open():
            raise RuntimeError("port 8080 is already used by another process")
        result = systemctl("start", UNIT)
        if result.returncode:
            raise RuntimeError(result.stderr.strip())
    elif command == "pause":
        result = systemctl("stop", UNIT)
        if result.returncode:
            raise RuntimeError(result.stderr.strip())
    elif command in ("load", "unload"):
        if unit_state() != "active":
            raise RuntimeError("resume the service first")
        view = status()
        if view["state"] not in ("ready", "unloaded"):
            raise RuntimeError(view.get("error", "the selected Strata server is not ready"))
        request("/v1/" + command + ("?wait=1" if command == "unload" else ""), post=True,
                timeout=300.0 if command == "load" else 15.0)
    elif command == "select" and filename:
        select_model(filename)
    else:
        raise ValueError("unknown action")


def tray() -> None:
    import gi
    gi.require_version("Gtk", "3.0")
    gi.require_version("AyatanaAppIndicator3", "0.1")
    from gi.repository import AyatanaAppIndicator3 as AppIndicator, GLib, Gtk

    indicator = AppIndicator.Indicator.new("strata-jetson", str(ICON),
                                          AppIndicator.IndicatorCategory.APPLICATION_STATUS)
    indicator.set_status(AppIndicator.IndicatorStatus.ACTIVE)
    busy = False
    refresh_id = 0
    refresh_in_flight = False

    def item(label: str, callback=None, enabled: bool = True):
        widget = Gtk.MenuItem(label=label)
        widget.set_sensitive(enabled)
        if callback:
            widget.connect("activate", callback)
        return widget

    def launch(*argv: str) -> None:
        subprocess.Popen(argv, start_new_session=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def error_dialog(message: str) -> None:
        dialog = Gtk.MessageDialog(message_type=Gtk.MessageType.ERROR, buttons=Gtk.ButtonsType.CLOSE,
                                   text="Strata could not complete the action")
        dialog.format_secondary_text(message)
        dialog.run()
        dialog.destroy()

    def model_details(model: dict) -> None:
        native = Path(model["native"]) if model.get("native") else None
        try:
            size = f"{native.stat().st_size / 1_000_000_000:.2f} GB" if native else "not found"
        except OSError:
            size = "not found"
        dialog = Gtk.MessageDialog(message_type=Gtk.MessageType.INFO, buttons=Gtk.ButtonsType.CLOSE,
                                   text=model["name"])
        dialog.format_secondary_text(
            f"Config: {model['file']}\n"
            f"Weights: {native if native else 'none'}\n"
            f"File size: {size}\n"
            f"Prepared pack: {model['pack'] or 'none'}\n"
            f"State: {'ready' if model['ready'] else 'files missing'}")
        dialog.run()
        dialog.destroy()

    def run_action(command: str, filename: str | None = None) -> None:
        nonlocal busy
        if busy:
            return
        busy = True
        refresh()

        def work():
            try:
                action(command, filename)
                message = None
            except (OSError, RuntimeError, ValueError, subprocess.TimeoutExpired) as exc:
                message = str(exc)

            def done():
                nonlocal busy
                busy = False
                if message:
                    error_dialog(message)
                refresh()
                return False

            GLib.idle_add(done)

        threading.Thread(target=work, daemon=True).start()

    def refresh():
        nonlocal refresh_id, refresh_in_flight
        if refresh_in_flight:
            return
        refresh_in_flight = True
        refresh_id += 1
        this_id = refresh_id
        # systemctl and the local API have bounded timeouts; keep them off the GTK thread.
        def read():
            try:
                result = status()
            except (OSError, subprocess.TimeoutExpired) as exc:
                result = {"state": "error", "model": selected_model(), "error": str(exc)}
            GLib.idle_add(draw, result, this_id)

        threading.Thread(target=read, daemon=True).start()

    def draw(result, this_id):
        nonlocal refresh_in_flight
        refresh_in_flight = False
        if this_id != refresh_id:
            return False
        state = result["state"]
        model = result.get("model")
        menu = Gtk.Menu()
        menu.append(item("Strata · " + state.capitalize(), enabled=False))
        menu.append(item("Model: " + (model["name"] if model else "none"), enabled=False))
        menu.append(Gtk.SeparatorMenuItem())
        menu.append(item("Resume service (load weights)", lambda *_: run_action("resume"),
                         enabled=not busy and state in ("paused", "failed") and bool(model and model["ready"])))
        menu.append(item("Pause service (close port 8080)", lambda *_: run_action("pause"),
                         enabled=not busy and result.get("unit") in ("active", "activating")))
        menu.append(item("Load model", lambda *_: run_action("load"), enabled=not busy and state == "unloaded"))
        menu.append(item("Unload model", lambda *_: run_action("unload"), enabled=not busy and state == "ready"))
        menu.append(Gtk.SeparatorMenuItem())
        submenu = Gtk.Menu()
        available = models()
        if not available:
            submenu.append(item("No model configs found", enabled=False))
        for entry in available:
            label = ("✓ " if model and entry["file"] == model["file"] else "    ") + entry["name"]
            if not entry["ready"]:
                label += " (files missing)"
            submenu.append(item(label, lambda _w, name=entry["file"]: run_action("select", name),
                                enabled=not busy and state in ("paused", "failed") and entry["ready"]))
        choose = item("Models")
        choose.set_submenu(submenu)
        menu.append(choose)
        menu.append(item("View selected model…", lambda *_: model_details(model), enabled=bool(model)))
        native = Path(model["native"]).parent if model and model.get("native") else None
        menu.append(item("Open model folder", lambda *_: launch("xdg-open", str(native)),
                         enabled=bool(native and native.is_dir())))
        menu.append(item("Open Strata", lambda *_: webbrowser.open(URL), enabled=state in ("ready", "unloaded")))
        menu.append(item("View live service log", lambda *_: launch("xterm", "-T", "Strata service log", "-e",
                                                                    "journalctl", "-b", "_SYSTEMD_USER_UNIT=" + UNIT,
                                                                    "-f")))
        menu.append(item("Refresh state", lambda *_: refresh()))
        if result.get("error"):
            menu.append(item("Error: " + result["error"], enabled=False))
        menu.append(Gtk.SeparatorMenuItem())
        menu.append(item("Quit indicator (service keeps its state)", lambda *_: Gtk.main_quit()))
        menu.show_all()
        indicator.set_menu(menu)
        return False

    refresh()
    GLib.timeout_add_seconds(5, lambda: (refresh(), True)[1])
    try:
        Gtk.main()
    except KeyboardInterrupt:
        pass


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("install", "tray", "service", "status", "models", "resume",
                                             "pause", "load", "unload", "select"))
    parser.add_argument("filename", nargs="?")
    args = parser.parse_args()
    try:
        if args.command == "install":
            install()
        elif args.command == "tray":
            tray()
        elif args.command == "service":
            service()
        elif args.command == "status":
            print(json.dumps(status(), indent=2))
        elif args.command == "models":
            print(json.dumps(models(), indent=2))
        else:
            action(args.command, args.filename)
    except (OSError, RuntimeError, ValueError, subprocess.TimeoutExpired) as exc:
        print(f"strata tray: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
