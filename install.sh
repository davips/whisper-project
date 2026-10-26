#!/usr/bin/env bash
set -e
HF_HUB_DISABLE_XET=1
D="$(dirname "$(realpath "$0")")"
BIN="${PIPX_BIN_DIR:-$HOME/.local/bin}"
AUTO="$HOME/.config/autostart"
LOG="${XDG_RUNTIME_DIR:-/tmp}/dictate.log"

stop_running() {
P="$(cat "${XDG_RUNTIME_DIR:-/tmp}/dictate.pid" 2>/dev/null || true)"
grep -qs dictate "/proc/$P/cmdline" && kill "$P" || true
}

if [ "$1" = "--uninstall" ]; then
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
pipx install --force "$D$EXTRA"

echo "Downloading model $MODEL..."
PY="$(dirname "$(realpath "$BIN/dictate")")/python"
HF_HUB_DISABLE_XET=1 "$PY" -c "from faster_whisper import download_model; download_model(\"$MODEL\")"

mkdir -p "$AUTO"
rm -f "$AUTO/dictate-scrolllock.desktop"

cat > "$AUTO/dictate.desktop" << EOF
[Desktop Entry]
Type=Application
Name=Dictation
Exec=$BIN/dictate
EOF

stop_running
setsid -f "$BIN/dictate" > "$LOG" 2>&1
echo "Installed. Toggle dictation with the hotkey in ~/.dictate.conf (default: F2)."
