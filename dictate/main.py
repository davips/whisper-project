import os, sys, sysconfig

base = os.path.join(sysconfig.get_paths()["purelib"], "nvidia")
libs = [p for p in (os.path.join(base, "cublas", "lib"), os.path.join(base, "cudnn", "lib")) if os.path.isdir(p)]
cur = os.environ.get("LD_LIBRARY_PATH", "")
if libs and not set(libs) <= set(cur.split(":")):
    os.environ["LD_LIBRARY_PATH"] = ":".join(libs + cur.split(":") if cur else libs)
    os.execv(sys.executable, [sys.executable, "-m", "dictate.main"] + sys.argv[1:])

import argparse, configparser, gc, queue, subprocess, threading, time
import numpy as np
import tkinter as tk
import ctranslate2
from faster_whisper import WhisperModel
from Xlib import XK, display

SR = 16000
PIDF = os.path.join(os.environ.get("XDG_RUNTIME_DIR", "/tmp"), "dictate.pid")
CFG = os.path.expanduser("~/.dictate.conf")
DEFAULT_HOTKEY = "F2"
CUDA = ctranslate2.get_cuda_device_count() > 0
LANG = MODEL = None
ui = queue.Queue()


def running():
    try:
        with open(PIDF) as f:
            pid = int(f.read())
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            return b"dictate" in f.read()
    except (OSError, ValueError):
        return False


class HotkeyToggle:
    def __init__(self, hotkey):
        self.disp = display.Display()
        keysym = XK.string_to_keysym(hotkey)
        if keysym == 0:
            raise ValueError(f"Invalid hotkey '{hotkey}' in {CFG}")
        self.keycode = self.disp.keysym_to_keycode(keysym)

    def pressed(self):
        keys = self.disp.query_keymap()
        return bool(keys[self.keycode // 8] & (1 << (self.keycode % 8)))

    def close(self):
        self.disp.close()


def load_hotkey():
    cfg = configparser.ConfigParser()
    if os.path.exists(CFG):
        cfg.read(CFG)
    else:
        cfg["dictate"] = {"hotkey": DEFAULT_HOTKEY}
        with open(CFG, "w") as f:
            cfg.write(f)
    hotkey = cfg.get("dictate", "hotkey", fallback=DEFAULT_HOTKEY).strip()
    if XK.string_to_keysym(hotkey) == 0:
        raise ValueError(f"Invalid hotkey '{hotkey}' in {CFG}")
    return hotkey


class Session(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.stop = threading.Event()
        self.buf = bytearray()
        self.lock = threading.Lock()
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

    def run(self):
        try:
            self.work()
        finally:
            self.rec.terminate()
            ui.put(None)

    def work(self):
        model = WhisperModel(MODEL, device="cuda" if CUDA else "cpu", compute_type="int8_float16" if CUDA else "int8")
        lang, start, done = LANG, 0, ""
        while not self.stop.is_set():
            a = self.audio(start)
            if len(a) < SR:
                time.sleep(0.2)
                continue
            segs, info = model.transcribe(a, language=lang, beam_size=1, vad_filter=True, initial_prompt=done[-200:] or None)
            segs = list(segs)
            if lang is None and len(a) > 4 * SR:
                lang = info.language
            if len(a) > 20 * SR and len(segs) > 1:
                done += "".join(s.text for s in segs[:-1])
                start += int(segs[-2].end * SR)
                segs = segs[-1:]
            ui.put(done + "".join(s.text for s in segs))
        self.rec.terminate()
        self.reader.join()
        a = self.audio(start)
        segs = model.transcribe(a, language=lang, vad_filter=True, initial_prompt=done[-200:] or None)[0] if len(a) > SR // 2 else []
        text = (done + "".join(s.text for s in segs)).strip()
        del model
        gc.collect()
        if text:
            subprocess.run(["xdotool", "type", "--clearmodifiers", "--file", "-"], input=text.encode())


def main():
    global LANG, MODEL
    p = argparse.ArgumentParser(prog="dictate")
    p.add_argument("--language", help="e.g. pt, en (default: auto-detect)")
    p.add_argument("--model", help="default: large-v3 with CUDA, small on CPU")
    a = p.parse_args()
    LANG, MODEL = a.language, a.model or ("large-v3" if CUDA else "small")
    try:
        hotkey = load_hotkey()
    except ValueError as e:
        sys.exit(str(e))
    if running():
        sys.exit(0)
    with open(PIDF, "w") as f:
        f.write(str(os.getpid()))

    root = tk.Tk()
    root.overrideredirect(True)
    root.attributes("-topmost", True)
    label = tk.Label(root, bg="black", fg="white", font=("Sans", 20), wraplength=1100, justify="left", padx=16, pady=10)
    label.pack()
    root.withdraw()
    toggle = HotkeyToggle(hotkey)
    state = {"session": None, "on": False, "prev_pressed": False}

    def show(text):
        label.config(text=text)
        root.update_idletasks()
        x = (root.winfo_screenwidth() - root.winfo_reqwidth()) // 2
        y = root.winfo_screenheight() - root.winfo_reqheight() - 80
        root.geometry(f"+{x}+{y}")
        root.deiconify()

    def tick():
        pressed = toggle.pressed()
        if pressed and not state["prev_pressed"]:
            state["on"] = not state["on"]
        state["prev_pressed"] = pressed
        on = state["on"]
        s = state["session"]
        if on and s is None:
            state["session"] = Session()
            state["session"].start()
            show("● listening")
        elif not on and s is not None:
            s.stop.set()
        while not ui.empty():
            msg = ui.get()
            if msg is None:
                root.withdraw()
                state["session"] = None
            else:
                show("● " + msg.strip())
        root.after(150, tick)

    root.after(150, tick)
    try:
        root.mainloop()
    finally:
        toggle.close()


if __name__ == "__main__":
    main()
