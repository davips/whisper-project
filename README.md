# whisper-project

Local dictation for Linux desktops using Whisper large-v3 through faster-whisper, which runs on CTranslate2 (an inference engine for transformer models). Pressing your configured hotkey (default: Alt+Super+Space) starts listening and shows a live transcription preview at the bottom of the screen. Pressing it again types the final text into the focused window, such as an email client, a browser or a chat box.

## Features

- Toggle with a global hotkey from `~/.dictate.conf` (default: Alt+Super+Space).
- Flush recognized text mid-session with another hotkey (default: Enter), without unloading the model.
- Cancel the sentence in progress and keep listening (default: Esc).
- End the session with another hotkey (default: Super); the remaining text is typed.
- Live provisional transcription in an always-on-top overlay that does not take keyboard focus.
- Final text typed directly into the focused window.
- English by default, with one key per language (F1 to F12 by default) to switch while listening. The key-to-language map is editable, and `auto` enables detection.
- GPU (graphics processing unit) inference with int8 weights, using about 1.7 GiB of VRAM (video memory).
- CPU (central processing unit) fallback with the `small` model when no GPU is available.
- Model loaded only while dictating, so VRAM is released afterwards.
- Starts automatically at login.

## Requirements

- Linux with an X11 session. Wayland is not supported.
- System Python 3.10 to 3.13.
- A microphone available through ALSA (Advanced Linux Sound Architecture), including PipeWire and PulseAudio setups.
- apt, dnf or pacman for automatic installation of system tools. Otherwise, install pipx, Python Tk, arecord and xdotool manually.
- Optional: an NVIDIA GPU with driver 525 or later, for CUDA (NVIDIA compute platform) 12 support, and about 3 GB of free VRAM for large-v3.

## Installation

`````bash
git clone git@github.com:<user>/whisper-project.git
cd whisper-project
./install.sh
`````

The script:

1. Installs the system tools.
2. Installs the package with pipx, adding the `cuda` extra when `nvidia-smi` detects a GPU.
3. Downloads the model.
4. Installs, enables and starts the systemd user service.

To update: run `git pull`, then `./install.sh`.

To uninstall: run `./install.sh --uninstall`. The downloaded model stays in `~/.cache/huggingface`.

## Usage

1. Place the cursor where the text should go.
2. Press your configured hotkey. The overlay shows "● listening", then the provisional text.
3. Press the flush hotkey to type text recognized so far and keep listening.
4. Press the cancel hotkey to discard recognized text so far and keep listening.
5. Press a language key (F1 to F12 by default) to switch language. The audio recorded so far is transcribed again in the new language.
6. Press the stop hotkey (or the toggle hotkey again) when done. The remaining text is typed.

Command-line options:

`````
dictate [--language LANG] [--model MODEL]
`````

- `--language`: a language code such as `pt` or `en`, or `auto` for detection. The default is `language` in `~/.dictate.conf` (`en`).
- `--model`: a faster-whisper model name such as `medium` or `large-v3`. The default is `large-v3` with CUDA and `small` on CPU.

To make options permanent, edit the `ExecStart=` line in `~/.config/systemd/user/dictate.service` (or run `systemctl --user edit dictate`).

To choose the hotkeys and languages, edit `~/.dictate.conf`:

`````
[dictate]
hotkey = Alt+Super+Space
flush_hotkey = Enter
cancel_hotkey = Esc
stop_hotkey = Super
language = en

[languages]
F1 = en
F2 = pt
F3 = de
F4 = es
F5 = pl
F6 = it
F7 = fr
F8 = ja
F9 = zh
F10 = ru
F11 = no
F12 = id
`````

## How it works

- `arecord` starts capturing 16 kHz mono audio as soon as the hotkey starts dictation, while the model loads.
- For the preview, the audio recorded so far is re-transcribed in a loop with greedy decoding (beam size 1).
- The flush hotkey types the text recognized up to that moment and continues listening from new audio.
- A language key sets the language for the rest of the session, including after a flush or cancel. The next session starts again with `language`.
- The cancel hotkey discards recognized text up to that moment and continues listening from new audio.
- After 20 s of audio, all segments except the latest are committed. Committed text is passed as `initial_prompt` for context, which keeps each pass short.
- When the hotkey stops dictation, the remaining audio is transcribed with beam search (beam size 5) and VAD (voice activity detection) filtering.
- The text is typed with `xdotool` into the focused window.
- The program re-launches itself once with `LD_LIBRARY_PATH` pointing to the cuBLAS and cuDNN libraries installed from pip wheels, so no system CUDA toolkit is needed.

## Notes

- The focused window must accept keyboard input.
- The toggle hotkey is grabbed globally. The flush, cancel and stop hotkeys are grabbed only while dictation is on, so Enter, Esc and Super work normally otherwise. Grabbed keys are not delivered to the focused application. Modifiers (`Ctrl`, `Alt`, `Shift`, `Super`) can be combined with `+`, for example `Ctrl+Alt+F5`. `Enter`, `Esc`, `Backspace` and bare `Super`, `Alt`, `Ctrl`, `Shift` (left key) are accepted as key names.
- If a key is already taken by another application, the program exits with an error in the log; choose another key in `~/.dictate.conf`.
- Every key must be assigned to a single action. Each line in `[languages]` is `key = language`, where the key can have modifiers (`Ctrl+F1`) and the language is a Whisper code, `auto`, or `pt-br` (same as `pt`).
- The language keys are grabbed only while dictation is on. If another application already owns one of them (for example F12 for a drop-down terminal), it is skipped and the log names it. Remap it in `[languages]`.
- After the first dictation, the process keeps a CUDA context of about 80 MiB of VRAM.
- PyAV is pinned below version 19, because PyAV 19 removed an argument used by faster-whisper 1.2.1 (SYSTRAN/faster-whisper#1492).
- If a model download hangs, set `HF_HUB_DISABLE_XET=1` to bypass the Xet transfer backend.
- Logs can be viewed with `journalctl --user -u dictate -f`.

## Development

`````bash
poetry install --extras cuda
poetry run dictate
````
