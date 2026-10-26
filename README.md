# whisper-project

Local dictation for Linux desktops using Whisper large-v3 through faster-whisper, which runs on CTranslate2 (an inference engine for transformer models). Turning Scroll Lock on starts listening and shows a live transcription preview at the bottom of the screen. Turning it off pastes the final text into the focused window, such as an email client, a browser or a chat box.

## Features

- Toggle with Scroll Lock: the lock state is the listening state.
- Live provisional transcription in an always-on-top overlay that does not take keyboard focus.
- Final text pasted through the clipboard, which handles accented characters reliably.
- Automatic language detection, fixed for the session after 4 s of audio.
- GPU (graphics processing unit) inference with int8 weights, using about 1.7 GiB of VRAM (video memory).
- CPU (central processing unit) fallback with the `small` model when no GPU is available.
- Model loaded only while dictating, so VRAM is released afterwards.
- Starts automatically at login.

## Requirements

- Linux with an X11 session. Wayland is not supported.
- System Python 3.10 to 3.13.
- A microphone available through ALSA (Advanced Linux Sound Architecture), including PipeWire and PulseAudio setups.
- apt, dnf or pacman for automatic installation of system tools. Otherwise, install pipx, Python Tk, arecord, xclip, xdotool, xset and xmodmap manually.
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
4. Makes Scroll Lock a lock key if it is not one already.
5. Creates the login autostart entry and starts the program.

To update: run `git pull`, then `./install.sh`.

To uninstall: run `./install.sh --uninstall`. The downloaded model stays in `~/.cache/huggingface`.

## Usage

1. Place the cursor where the text should go.
2. Press Scroll Lock. The overlay shows "● listening", then the provisional text.
3. Press Scroll Lock again. The final text is pasted.

Command-line options:

`````
dictate [--language LANG] [--model MODEL]
`````

- `--language`: a language code such as `pt` or `en`. The default is auto-detection.
- `--model`: a faster-whisper model name such as `medium` or `large-v3`. The default is `large-v3` with CUDA and `small` on CPU.

To make options permanent, edit the `Exec=` line in `~/.config/autostart/dictate.desktop`.

## How it works

- `arecord` starts capturing 16 kHz mono audio as soon as Scroll Lock turns on, while the model loads.
- For the preview, the audio recorded so far is re-transcribed in a loop with greedy decoding (beam size 1).
- After 20 s of audio, all segments except the latest are committed. Committed text is passed as `initial_prompt` for context, which keeps each pass short.
- When Scroll Lock turns off, the remaining audio is transcribed with beam search (beam size 5) and VAD (voice activity detection) filtering.
- The text is copied with `xclip` and pasted with `xdotool` (Ctrl+V).
- The program re-launches itself once with `LD_LIBRARY_PATH` pointing to the cuBLAS and cuDNN libraries installed from pip wheels, so no system CUDA toolkit is needed.

## Notes

- Dictation replaces the clipboard content.
- Terminals paste with Ctrl+Shift+V, so the automatic paste does not work there. The text remains in the clipboard.
- Only an off-to-on transition starts listening. If Scroll Lock is on at startup, the first press only turns it off.
- After the first dictation, the process keeps a CUDA context of about 80 MiB of VRAM.
- PyAV is pinned below version 19, because PyAV 19 removed an argument used by faster-whisper 1.2.1 (SYSTRAN/faster-whisper#1492).
- If a model download hangs, set `HF_HUB_DISABLE_XET=1` to bypass the Xet transfer backend.
- Logs go to `$XDG_RUNTIME_DIR/dictate.log` when the program is started by `install.sh`, and to `~/.xsession-errors` when it is started at login.

## Development

`````bash
poetry install --extras cuda
poetry run dictate
````
