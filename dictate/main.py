import os, sys, sysconfig

base = os.path.join(sysconfig.get_paths()["purelib"], "nvidia")
libs = [p for p in (os.path.join(base, "cublas", "lib"), os.path.join(base, "cudnn", "lib")) if os.path.isdir(p)]
cur = os.environ.get("LD_LIBRARY_PATH", "")
if libs and not set(libs) <= set(cur.split(":")):
    os.environ["LD_LIBRARY_PATH"] = ":".join(libs + cur.split(":") if cur else libs)
    os.execv(sys.executable, [sys.executable, "-m", "dictate.main"] + sys.argv[1:])

import argparse, configparser, gc, queue, select, subprocess, threading, time
import numpy as np
import tkinter as tk
import ctranslate2
from faster_whisper import WhisperModel
from faster_whisper.tokenizer import _LANGUAGE_CODES
from Xlib import X, XK, display
from Xlib.ext import shape

SR = 16000
PIDF = os.path.join(os.environ.get("XDG_RUNTIME_DIR", "/tmp"), "dictate.pid")
CFG = os.path.expanduser("~/.dictate.conf")
DEFAULT_HOTKEY = "Alt+Super+Space"
DEFAULT_FLUSH_HOTKEY = "Enter"
DEFAULT_CANCEL_HOTKEY = "Esc"
DEFAULT_STOP_HOTKEY = "Super"
DEFAULT_LANGUAGE = "en"
DEFAULT_WINDOW = 30
MIN_WINDOW, MAX_WINDOW = 10, 300
WINDOW_COMMENTS = (
    "# window: seconds of recent audio re-transcribed on every preview update, before older text is committed",
    f"# valid range is {MIN_WINDOW} to {MAX_WINDOW}, default {DEFAULT_WINDOW}",
    "# larger values give more context and better accuracy, but the preview updates slower",
    "# smaller values make the preview faster, but sentences are cut with less context",
)
DEFAULT_LANGUAGES = {"F1": "en", "F2": "pt", "F3": "de", "F4": "es", "F5": "pl", "F6": "it", "F7": "fr", "F8": "ja", "F9": "zh", "F10": "ru", "F11": "no", "F12": "id"}
LANG_ALIASES = {"pt-br": "pt"}
UI_TICK_MS = 50
CUDA = ctranslate2.get_cuda_device_count() > 0
LANG = MODEL = WINDOW = None
ui = queue.Queue()

MODIFIERS = {"super": X.Mod4Mask, "ctrl": X.ControlMask, "control": X.ControlMask, "alt": X.Mod1Mask, "shift": X.ShiftMask}
MOD_BITS = X.Mod4Mask | X.ControlMask | X.Mod1Mask | X.ShiftMask
LOCK_MASKS = (0, X.LockMask, X.Mod2Mask, X.LockMask | X.Mod2Mask)
KEY_ALIASES = {"enter": "Return", "esc": "Escape", "space": "space", "backspace": "BackSpace", "super": "Super_L", "alt": "Alt_L", "ctrl": "Control_L", "control": "Control_L", "shift": "Shift_L"}


def running():
    try:
        with open(PIDF) as f:
            pid = int(f.read())
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            return b"dictate" in f.read()
    except (OSError, ValueError):
        return False


def parse_hotkey(disp, hotkey):
    parts = [p.strip() for p in hotkey.split("+") if p.strip()]
    name = KEY_ALIASES.get(parts[-1].lower(), parts[-1]) if parts else ""
    keysym = next((k for k in (XK.string_to_keysym(c) for c in (name, name.capitalize(), name.upper())) if k), 0) if name else 0
    keycode = disp.keysym_to_keycode(keysym) if keysym else 0
    mask = 0
    for mod in parts[:-1]:
        if mod.lower() not in MODIFIERS:
            keycode = 0
            break
        mask |= MODIFIERS[mod.lower()]
    if not keycode:
        raise ValueError(f"Invalid hotkey '{hotkey}' in {CFG}")
    return keycode, mask


def norm_lang(value):
    v = value.strip().lower()
    if v in ("", "auto"):
        return None
    v = LANG_ALIASES.get(v, v)
    if v not in _LANGUAGE_CODES:
        raise ValueError(f"Unknown language '{value}' in {CFG}")
    return v


class HotkeyGrabber(threading.Thread):
    def __init__(self, bindings):
        super().__init__(daemon=True)
        self.events = queue.Queue()
        self.disp = display.Display()
        self.root = self.disp.screen().root
        self.actions = {}
        self.names = {}
        for action, hotkey in bindings:
            key = parse_hotkey(self.disp, hotkey)
            if key in self.actions:
                raise ValueError(f"'{hotkey}' is assigned to more than one action in {CFG}")
            self.actions[key] = action
            self.names[key] = hotkey
        self.active = False
        self.requested = False
        self.commands = queue.Queue()
        self.held = set()
        self.grabbed = set()
        for key, action in self.actions.items():
            if action == "toggle" and not self.grab(key):
                raise ValueError(f"Cannot grab '{self.names[key]}': it is already used by another application. Change it in {CFG}")
        self.root.change_attributes(event_mask=X.KeyPressMask | X.KeyReleaseMask)

    def grab(self, key):
        keycode, mask = key
        for lock in LOCK_MASKS:
            errors = []
            self.root.grab_key(keycode, mask | lock, False, X.GrabModeAsync, X.GrabModeAsync, onerror=lambda err, req, errors=errors: errors.append(err))
            self.disp.sync()
            if errors:
                self.ungrab(key)
                return False
        self.grabbed.add(key)
        return True

    def ungrab(self, key):
        keycode, mask = key
        for lock in LOCK_MASKS:
            self.root.ungrab_key(keycode, mask | lock)
        self.disp.sync()
        self.held.discard(keycode)
        self.grabbed.discard(key)

    def set_active(self, active):
        if active != self.requested:
            self.requested = active
            self.commands.put(active)

    def apply_active(self, active):
        if active == self.active:
            return
        self.active = active
        for key, action in self.actions.items():
            if action == "toggle":
                continue
            if not active:
                self.ungrab(key)
            elif not self.grab(key):
                print(f"Cannot grab '{self.names[key]}': it is already used by another application", file=sys.stderr, flush=True)

    def run(self):
        e = None
        while True:
            while not self.commands.empty():
                self.apply_active(self.commands.get())
            if e is None:
                if not self.disp.pending_events():
                    select.select([self.disp], [], [], UI_TICK_MS / 1000)
                    continue
                e = self.disp.next_event()
            cur, e = e, None
            if cur.type == X.KeyRelease:
                time.sleep(0.03)
                if self.disp.pending_events():
                    e = self.disp.next_event()
                    if e.type == X.KeyPress and e.detail == cur.detail and e.time == cur.time:
                        e = None
                        continue
                self.held.discard(cur.detail)
            elif cur.type == X.KeyPress and cur.detail not in self.held:
                self.held.add(cur.detail)
                action = self.actions.get((cur.detail, cur.state & MOD_BITS))
                if action:
                    self.events.put(action)


def load_config():
    cfg = configparser.ConfigParser(delimiters=("=",), comment_prefixes=(), allow_no_value=True)
    cfg.optionxform = str
    try:
        cfg.read(CFG)
    except configparser.Error as e:
        raise ValueError(f"Cannot read {CFG}: {e}")
    before = {name: dict(cfg[name]) for name in cfg.sections()}
    if "dictate" not in cfg:
        cfg["dictate"] = {}
    sec = cfg["dictate"]
    if (sec.get("hotkey"), sec.get("flush_hotkey")) in (("F2", "F3"), ("Super+F5", "Super+F6"), ("F5", "F6")):
        for k in ("hotkey", "flush_hotkey", "cancel_hotkey"):
            sec.pop(k, None)
    if sec.get("cancel_hotkey") == "Esc" and "stop_hotkey" not in sec:
        sec["cancel_hotkey"] = DEFAULT_CANCEL_HOTKEY
    if (sec.get("cancel_hotkey"), sec.get("stop_hotkey")) == ("Backspace", "Esc"):
        sec["cancel_hotkey"], sec["stop_hotkey"] = DEFAULT_CANCEL_HOTKEY, DEFAULT_STOP_HOTKEY
    sec.setdefault("hotkey", DEFAULT_HOTKEY)
    sec.setdefault("flush_hotkey", DEFAULT_FLUSH_HOTKEY)
    sec.setdefault("cancel_hotkey", DEFAULT_CANCEL_HOTKEY)
    sec.setdefault("stop_hotkey", DEFAULT_STOP_HOTKEY)
    sec.setdefault("language", DEFAULT_LANGUAGE)
    if "window" not in sec:
        for line in WINDOW_COMMENTS:
            sec[line] = None
        sec["window"] = str(DEFAULT_WINDOW)
    if "languages" not in cfg:
        cfg["languages"] = DEFAULT_LANGUAGES
    if {name: dict(cfg[name]) for name in cfg.sections()} != before:
        with open(CFG, "w") as f:
            cfg.write(f)
    bindings = [("toggle", sec["hotkey"].strip()), ("flush", sec["flush_hotkey"].strip()), ("cancel", sec["cancel_hotkey"].strip()), ("stop", sec["stop_hotkey"].strip())]
    for hotkey, value in cfg["languages"].items():
        if hotkey.startswith("#"):
            continue
        bindings.append((f"lang:{norm_lang(value) or 'auto'}", hotkey.strip()))
    try:
        window = float(sec["window"])
    except ValueError:
        window = 0
    if not MIN_WINDOW <= window <= MAX_WINDOW:
        raise ValueError(f"Invalid window '{sec['window']}' in {CFG}: use a number of seconds from {MIN_WINDOW} to {MAX_WINDOW}")
    return bindings, norm_lang(sec["language"]), window


class Session(threading.Thread):
    def __init__(self, lang, window):
        super().__init__(daemon=True)
        self.initial_lang = lang
        self.window = window
        self.new_lang = None
        self.lang_changed = threading.Event()
        self.stop = threading.Event()
        self.flush = threading.Event()
        self.cancel = threading.Event()
        self.buf = bytearray()
        self.lock = threading.Lock()
        self.latest_text = ""
        self.latest_lock = threading.Lock()
        self.rec = subprocess.Popen(["arecord", "-q", "-f", "S16_LE", "-r", str(SR), "-c", "1", "-t", "raw"], stdout=subprocess.PIPE)
        self.reader = threading.Thread(target=self.read, daemon=True)
        self.reader.start()

    def read(self):
        while chunk := self.rec.stdout.read(3200):
            with self.lock:
                self.buf += chunk

    def audio(self, start):
        with self.lock:
            data = bytes(self.buf[start * 2:])
        return np.frombuffer(data, np.int16).astype(np.float32) / 32768

    def request_flush(self):
        self.flush.set()

    def request_cancel(self):
        self.cancel.set()

    def request_language(self, lang):
        self.new_lang = lang
        self.lang_changed.set()

    def set_latest(self, text):
        with self.latest_lock:
            self.latest_text = text

    def get_latest(self):
        with self.latest_lock:
            return self.latest_text

    def type_text(self, text):
        subprocess.run(["xdotool", "type", "--clearmodifiers", "--file", "-"], input=text.replace("\n", " ").encode())

    def flush_now(self):
        text = self.get_latest().strip()
        if text:
            self.type_text(text)
        with self.lock:
            return len(self.buf) // 2

    def cancel_now(self):
        with self.lock:
            return len(self.buf) // 2

    def run(self):
        try:
            self.work()
        finally:
            self.rec.terminate()
            ui.put(None)

    def work(self):
        model = WhisperModel(MODEL, device="cuda" if CUDA else "cpu", compute_type="int8_float16" if CUDA else "int8")
        chosen = lang = self.initial_lang
        start, done = 0, ""
        while not self.stop.is_set():
            if self.lang_changed.is_set():
                self.lang_changed.clear()
                chosen = lang = self.new_lang
            if self.cancel.is_set():
                self.cancel.clear()
                start = self.cancel_now()
                done, lang = "", chosen
                self.set_latest("")
            if self.flush.is_set():
                self.flush.clear()
                start = self.flush_now()
                done, lang = "", chosen
                self.set_latest("")
            a = self.audio(start)
            if len(a) < SR // 2:
                time.sleep(0.1)
                continue
            segs, info = model.transcribe(a, language=lang, beam_size=1, vad_filter=True, initial_prompt=done[-200:] or None)
            segs = list(segs)
            if lang is None and len(a) > 4 * SR and info.language_probability >= 0.85:
                lang = info.language
            if len(a) > self.window * SR and len(segs) > 1:
                done += "".join(s.text for s in segs[:-1])
                start += int(segs[-2].end * SR)
                segs = segs[-1:]
            preview = done + "".join(s.text for s in segs)
            self.set_latest(preview)
            ui.put(preview)
        self.rec.terminate()
        self.reader.join()
        a = self.audio(start)
        segs = model.transcribe(a, language=lang, vad_filter=True, initial_prompt=done[-200:] or None)[0] if len(a) > SR // 2 else []
        text = (done + "".join(s.text for s in segs)).strip()
        del model
        gc.collect()
        if text:
            self.type_text(text)


# an empty input region makes the caption invisible to the pointer, so Tk never sees it as the focus window
def make_click_through(win):
    disp = display.Display()
    try:
        if not disp.has_extension("SHAPE"):
            return
        top = disp.create_resource_object("window", win.winfo_id())
        while (tree := top.query_tree()).parent.id != tree.root.id:
            top = tree.parent
        top.shape_rectangles(shape.SO.Set, shape.SK.Input, X.Unsorted, 0, 0, [])
        disp.sync()
    finally:
        disp.close()


def main():
    global LANG, MODEL, WINDOW
    p = argparse.ArgumentParser(prog="dictate")
    p.add_argument("--language", help="e.g. pt, en, auto (default: language in ~/.dictate.conf)")
    p.add_argument("--model", help="default: large-v3 with CUDA, small on CPU")
    a = p.parse_args()
    MODEL = a.model or ("large-v3" if CUDA else "small")
    try:
        bindings, LANG, WINDOW = load_config()
        if a.language:
            LANG = norm_lang(a.language)
    except ValueError as e:
        sys.exit(str(e))
    if running():
        sys.exit(0)
    try:
        grabber = HotkeyGrabber(bindings)
    except ValueError as e:
        sys.exit(str(e))
    with open(PIDF, "w") as f:
        f.write(str(os.getpid()))
    grabber.start()

    root = tk.Tk()
    root.overrideredirect(True)
    root.attributes("-topmost", True)
    label = tk.Label(root, bg="black", fg="white", font=("Sans", 20), wraplength=1100, justify="left", padx=16, pady=10)
    label.pack()

    def on_map(e):
        if e.widget is root:
            make_click_through(root)

    root.bind("<Map>", on_map)
    root.withdraw()
    state = {"session": None, "on": False, "lang": LANG}

    def show(text):
        label.config(text=text)
        root.update_idletasks()
        x = (root.winfo_screenwidth() - root.winfo_reqwidth()) // 2
        y = root.winfo_screenheight() - root.winfo_reqheight() - 80
        root.geometry(f"+{x}+{y}")
        root.deiconify()

    def tick():
        while not grabber.events.empty():
            action = grabber.events.get()
            s = state["session"]
            if action == "toggle":
                state["on"] = not state["on"]
            elif action == "stop":
                state["on"] = False
            elif state["on"] and s is not None:
                if action == "flush":
                    s.request_flush()
                elif action == "cancel":
                    s.request_cancel()
                else:
                    state["lang"] = None if action == "lang:auto" else action[5:]
                    s.request_language(state["lang"])
                show(f"● [{state['lang'] or 'auto'}] listening")
        on = state["on"]
        grabber.set_active(on)
        s = state["session"]
        if on and s is None:
            state["lang"] = LANG
            state["session"] = Session(LANG, WINDOW)
            state["session"].start()
            show(f"● [{LANG or 'auto'}] listening")
        elif not on and s is not None:
            s.stop.set()
        while not ui.empty():
            msg = ui.get()
            if msg is None:
                root.withdraw()
                state["session"] = None
            else:
                show(f"● [{state['lang'] or 'auto'}] " + msg.strip())
        root.after(UI_TICK_MS, tick)

    root.after(UI_TICK_MS, tick)
    root.mainloop()


if __name__ == "__main__":
    main()
