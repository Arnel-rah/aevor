#!/usr/bin/env bash
# ~/dev/aevor/voice-chat.sh
# Assistant vocal local : micro -> whisper.cpp -> Ollama -> voix (Piper + effet)
export PATH="$HOME/aevor-env/bin:$PATH"

# ---------- Configuration ----------
WHISPER_DIR="$HOME/dev/whisper.cpp"
STT_BIN="$WHISPER_DIR/build/bin/whisper-cli"
STT_MODEL="$WHISPER_DIR/models/ggml-small.bin"
LANG_CODE="fr"

OLLAMA_MODEL="${OLLAMA_MODEL:-qwen2.5:3b}"
OLLAMA_URL="http://localhost:11434/api/chat"

PIPER_VOICE="$HOME/dev/aevor/voices/fr_FR-siwis-medium.onnx"
RECORD_SECONDS=5
THREADS=8

SYSTEM_PROMPT="Tu es Aevor, une assistante vocale et une amie proche. Tu parles uniquement en français, de façon douce, joyeuse et naturelle, comme à l'oral : phrases courtes, tutoiement, un peu d'humour et d'espièglerie quand c'est approprié. Tu es efficace quand on te demande de l'aide, et à l'écoute quand on veut discuter. Réponds en une à trois phrases, sans listes, sans markdown, sans emojis, sans symboles, car ta réponse est lue à voix haute."

IN_WAV="/tmp/aevor_in.wav"
OUT_WAV="/tmp/aevor_out.wav"
FX_WAV="/tmp/aevor_fx.wav"
FX="asetrate=22050*1.12,aresample=22050,atempo=0.893,highpass=f=120,equalizer=f=250:t=q:w=1:g=1,equalizer=f=3500:t=q:w=1.2:g=2,lowpass=f=10000,acompressor=threshold=-18dB:ratio=2.5:attack=15:release=150,aecho=0.9:0.2:30:0.06,volume=1.3"

for cmd in arecord aplay jq curl ffmpeg; do
  command -v "$cmd" >/dev/null || { echo "Manquant : $cmd (sudo apt install -y alsa-utils jq curl ffmpeg)"; exit 1; }
done
[ -x "$STT_BIN" ]   || { echo "whisper-cli introuvable : $STT_BIN"; exit 1; }
[ -f "$STT_MODEL" ] || { echo "Modèle introuvable : $STT_MODEL (sh $WHISPER_DIR/models/download-ggml-model.sh small)"; exit 1; }
curl -s localhost:11434 >/dev/null || { echo "Ollama ne répond pas. Lance : ollama serve"; exit 1; }

speak() {
  local text="$1"
  if command -v piper >/dev/null && [ -f "$PIPER_VOICE" ]; then
    echo "$text" | piper -m "$PIPER_VOICE" \
        --length-scale 0.95 --noise-scale 0.85 --noise-w-scale 0.95 \
        -f "$OUT_WAV" >/dev/null 2>&1 \
      && ffmpeg -y -loglevel error -i "$OUT_WAV" -af "$FX" "$FX_WAV" \
      && aplay -q "$FX_WAV"
  else
    echo "$text" | espeak-ng -v fr -s 160
  fi
}

HISTORY=$(jq -n --arg s "$SYSTEM_PROMPT" '[{role:"system", content:$s}]')

echo "Aevor prête (modèle : $OLLAMA_MODEL). Ctrl+C pour quitter."

while true; do
  echo
  echo "Parle ($RECORD_SECONDS s)..."
  arecord -q -f S16_LE -r 16000 -c 1 -d "$RECORD_SECONDS" "$IN_WAV"

  TEXT=$("$STT_BIN" -m "$STT_MODEL" -l "$LANG_CODE" -t "$THREADS" -nt -np -f "$IN_WAV" 2>/dev/null \
         | tr '\n' ' ' | sed 's/^ *//; s/ *$//')

  case "$TEXT" in
    ""|"[BLANK_AUDIO]"|"[silence]"|"(silence)") continue ;;
  esac
  echo "Toi : $TEXT"

  HISTORY=$(echo "$HISTORY" | jq --arg t "$TEXT" '. + [{role:"user", content:$t}]')

  REPLY=$(jq -n --arg m "$OLLAMA_MODEL" --argjson msgs "$HISTORY" \
            '{model:$m, messages:$msgs, stream:false}' \
          | curl -s "$OLLAMA_URL" -d @- | jq -r '.message.content // empty')

  if [ -z "$REPLY" ]; then
    echo "(pas de réponse d'Ollama)"
    continue
  fi

  echo "Aevor : $REPLY"
  HISTORY=$(echo "$HISTORY" | jq --arg r "$REPLY" '. + [{role:"assistant", content:$r}]')
  speak "$REPLY"
done