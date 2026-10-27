#!/usr/bin/env bash
set -e
HF_HUB_DISABLE_XET=1
D="$(dirname "$(realpath "$0")")"
BIN="${PIPX_BIN_DIR:-$HOME/.local/bin}"
AUTO="$HOME/.config/autostart"
SERVICE_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"

stop_running() {
systemctl --user stop dictate.service 2>/dev/null || true
P="$(cat "${XDG_RUNTIME_DIR:-/tmp}/dictate.pid" 2>/dev/null || true)"
grep -qs dictate "/proc/$P/cmdline" && kill "$P" || true
}

if [ "$1" = "--uninstall" ]; then
systemctl --user disable --now dictate.service 2>/dev/null || true
rm -f "$SERVICE_DIR/dictate.service"
systemctl --user daemon-reload 2>/dev/null || true
stop_running
pipx uninstall whisper-project || true
rm -f "$AUTO/dictate.desktop" "$AUTO/dictate-scrolllock.desktop"
exit 0
fi

if command -v apt-get >/dev/null; then
sudo apt-get install -y pipx python3-tk alsa-utils xdotool
elif command -v dnf >/dev/null; then
sudo dnf install -y pipx python3-tkinter alsa-utils xdotool
elif command -v pacman >/dev/null; then
sudo pacman -S --needed --noconfirm python-pipx tk alsa-utils xdotool
else
echo "Unknown package manager: install pipx, Python Tk, arecord and xdotool manually."
fi

EXTRA=""
MODEL="small"
if nvidia-smi -L >/dev/null 2>&1; then
EXTRA="[cuda]"
MODEL="large-v3"
fi
PIPX_HOME_DIR="${PIPX_HOME:-$HOME/.local/pipx}"
if [ -d "$PIPX_HOME_DIR/venvs/whisper-project" ]; then
    pipx runpip whisper-project install "$D$EXTRA"
else
    pipx install "$D$EXTRA"
fi

PY="$(dirname "$(realpath "$BIN/dictate")")/python"
MODEL_CACHE="$HOME/.cache/huggingface/hub/models--Systran--faster-whisper-$MODEL"
if [ -d "$MODEL_CACHE" ]; then
    echo "Model $MODEL already cached. Skipping download."
else
    echo "Downloading model $MODEL..."
    HF_HUB_DISABLE_XET=1 "$PY" -c "from faster_whisper import download_model; download_model(\"$MODEL\")"
fi

mkdir -p "$SERVICE_DIR"
if [ "$BIN" = "$HOME/.local/bin" ]; then
    cp "$D/dictate.service" "$SERVICE_DIR/dictate.service"
else
    sed "s|%h/\.local/bin/dictate|$BIN/dictate|" "$D/dictate.service" > "$SERVICE_DIR/dictate.service"
fi
rm -f "$AUTO/dictate.desktop" "$AUTO/dictate-scrolllock.desktop"

stop_running
systemctl --user daemon-reload
systemctl --user enable dictate.service
systemctl --user restart dictate.service
echo "Installed and started dictate.service. Hotkeys in ~/.dictate.conf (defaults: Alt+Super+Space start, Enter flush, Esc cancel sentence, Super end; F1-F12 switch language)."
