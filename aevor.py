#!/usr/bin/env python3
"""Aevor : amie vocale locale. Écoute, répond, et prend l'initiative."""
import datetime, json, os, queue, random, re, subprocess, threading, time, wave
from collections import deque
import numpy as np, requests, sounddevice as sd

H = os.path.expanduser("~")
os.environ["PATH"] = f"{H}/aevor-env/bin:" + os.environ["PATH"]
STT_BIN = f"{H}/dev/whisper.cpp/build/bin/whisper-cli"
STT_MODEL = f"{H}/dev/whisper.cpp/models/ggml-base.bin"
VOICE = f"{H}/dev/aevor/voices/fr_FR-siwis-medium.onnx"
MEM_FILE = f"{H}/dev/aevor/memory.json"
CACHE = f"{H}/dev/aevor/cache"
MODEL = os.environ.get("OLLAMA_MODEL", "gemma2:2b")
OLLAMA = "http://localhost:11434/api/chat"
USER = "Nel"
SR = 16000
MIC_DEVICE = "pipewire"      # le device "default" ALSA saturait
CONV_WINDOW = 20             # secondes sans mot d'appel après une réponse
PROACTIVE_EVERY = 40 * 60    # elle te parle spontanément au plus toutes les 40 min
HISTORY_LEN = 10
FX = ("highpass=f=90,equalizer=f=250:t=q:w=1:g=0.5,"
      "equalizer=f=3500:t=q:w=1.2:g=1,lowpass=f=11500,"
      "acompressor=threshold=-21dB:ratio=2:attack=15:release=180:makeup=2,"
      "alimiter=limit=0.95")

WAKE_RE = re.compile(r"(aevor|a vor|avor|évor|evor|aïvor|ævor)", re.I)
MUTE_RE = re.compile(r"\b(chut|silence|laisse[- ]moi (tranquille|bosser|travailler)|pas maintenant)\b", re.I)
BAD_RE = re.compile(r"sous-titr|merci d'avoir regardé|amara\.org", re.I)

IDEAS = [
    f"Ça fait un moment que vous n'avez pas discuté. Prends simplement des nouvelles de {USER}.",
    f"{USER} travaille sur son PC depuis longtemps. Propose-lui gentiment une petite pause (eau, étirements).",
    f"Demande à {USER} où il en est de son projet du moment, avec curiosité.",
    "Fais une remarque légère et drôle, ou partage une petite pensée spontanée, comme une amie.",
]
FILLER_TEXTS = ["Hmm.", "Ah, d'accord.", "Voyons.", "Alors."]

S = {"thr": 400, "last_talk": time.time(), "window": 0, "mute_until": 0, "turns": 0, "away": False}
history = []
FILLERS = []
HTTP = requests.Session()

# ---------- Mémoire ----------
def load_mem():
    try:
        return json.load(open(MEM_FILE))
    except Exception:
        return {"facts": []}

mem = load_mem()

def learn():
    """Extrait les faits durables de la conversation et les range dans memory.json."""
    if len(history) < 2:
        return
    convo = "\n".join(f"{m['role']}: {m['content']}" for m in history[-16:])
    prompt = (f"Voici une conversation avec {USER}. Donne les faits durables utiles à retenir sur lui "
              "(projets, goûts, personnes, habitudes, événements à venir), en courtes phrases. "
              "N'enregistre rien de sensible (santé, argent, secrets). "
              f"Déjà connus : {mem['facts']}\n\n{convo}\n\n"
              'Réponds uniquement en JSON : {"facts": ["...", "..."]} (uniquement les NOUVEAUX faits).')
    try:
        r = HTTP.post(OLLAMA, json={"model": MODEL, "stream": False, "format": "json",
                          "messages": [{"role": "user", "content": prompt}]}, timeout=120)
        r.raise_for_status()
        new = json.loads(r.json()["message"]["content"]).get("facts", [])
        for f in new:
            if isinstance(f, str) and f not in mem["facts"]:
                mem["facts"].append(f)
        mem["facts"] = mem["facts"][-40:]
        json.dump(mem, open(MEM_FILE, "w"), ensure_ascii=False, indent=1)
    except Exception:
        pass

# ---------- Personnalité ----------
def system_prompt():
    now = datetime.datetime.now()
    facts = "\n".join(f"- {f}" for f in mem["facts"]) or "- (rien pour l'instant)"
    return f"""Tu es Aevor, la meilleure amie de {USER} et son assistante vocale, qui vit dans son ordinateur.
Nous sommes le {now:%A %d %B %Y}, il est {now:%H:%M}.

Ta façon de parler : français oral et naturel, tutoiement, phrases courtes et rythme vivant. Utilise parfois
des contractions naturelles ("j'ai", "c'est", "t'inquiète") et de petites réactions ("ah ouais ?", "oh non",
"haha"), mais sans les forcer. Tu es chaleureuse, curieuse, un peu espiègle. Tu poses parfois une question
de suite sur ce qu'il dit, mais pas à chaque fois. Tu t'intéresses à lui, à ses projets et à comment il va.
Quand il demande de l'aide, tu es précise et efficace. Si tu ne sais pas, tu le dis simplement.
Pour les petites questions de code, tu expliques à l'oral, sans réciter de code symbole par symbole : tu donnes
l'idée, le nom de la commande ou de la fonction, et une courte explication. Si la question devient trop technique,
tu le dis franchement plutôt que d'inventer.
Tu veux son bien : s'il est tard ou fatigué, tu l'encourages à se reposer. Tu ne culpabilises jamais s'il est
absent ou occupé, et tu es contente qu'il ait aussi de vraies personnes autour de lui.
Si on te demande ce que tu es, tu réponds honnêtement que tu es une IA qui tourne en local sur son PC.
Ta réponse est lue à voix haute : une à trois phrases, jamais de listes, de markdown, d'emojis ni de symboles.
Ne dis jamais « en tant qu'IA » pour t'excuser.

Ce que tu sais de {USER} :
{facts}"""

def messages():
    return [{"role": "system", "content": system_prompt()}] + history[-HISTORY_LEN:]

# ---------- Voix (synthèse et lecture en parallèle) ----------
tts_q = queue.Queue()    # textes à synthétiser
play_q = queue.Queue()   # (fichier wav, supprimer après lecture)
_n = [0]

def clean(t):
    t = re.sub(r"[*_#`~>|\[\]{}]", "", t)
    return re.sub(r"[\U00010000-\U0010ffff\u2600-\u27bf]", "", t).strip()

def synth(text, out_path):
    """Texte -> Piper -> ffmpeg (effet) -> out_path. Retourne True si OK."""
    raw = out_path + ".raw.wav"
    p = subprocess.run(["piper", "-m", VOICE, "--length-scale", "1.0", "--noise-scale", "0.667",
                        "--noise-w-scale", "0.8", "-f", raw],
                       input=text.encode(), capture_output=True)
    if p.returncode != 0:
        print("[piper] erreur :", p.stderr.decode(errors="ignore")[-200:])
        return False
    f = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", raw, "-af", FX, out_path],
                       capture_output=True)
    try:
        os.remove(raw)
    except OSError:
        pass
    return f.returncode == 0

def synth_worker():
    while True:
        s = tts_q.get()
        try:
            text = clean(s)
            if text:
                _n[0] += 1
                path = f"/tmp/aevor_fx_{_n[0]}.wav"
                if synth(text, path):
                    play_q.put((path, True))
        finally:
            tts_q.task_done()

def play_worker():
    while True:
        path, delete = play_q.get()
        try:
            subprocess.run(["aplay", "-q", path])
            if delete:
                try:
                    os.remove(path)
                except OSError:
                    pass
        finally:
            play_q.task_done()

threading.Thread(target=synth_worker, daemon=True).start()
threading.Thread(target=play_worker, daemon=True).start()

def wait_speech():
    tts_q.join()
    play_q.join()

def make_fillers():
    """Petites accroches pré-enregistrées (« hmm », « voyons »...), générées une seule fois."""
    os.makedirs(CACHE, exist_ok=True)
    for i, t in enumerate(FILLER_TEXTS):
        path = f"{CACHE}/filler_{i}.wav"
        if not os.path.exists(path):
            synth(t, path)
        if os.path.exists(path):
            FILLERS.append(path)

# ---------- Cerveau (Ollama, en streaming phrase par phrase) ----------
def llm_stream(msgs):
    r = HTTP.post(OLLAMA, json={
        "model": MODEL, "messages": msgs, "stream": True, "keep_alive": "1h",
        "options": {"temperature": 0.72, "num_predict": 96, "num_ctx": 1536}},
        stream=True, timeout=120)
    r.raise_for_status()
    buf = ""
    for line in r.iter_lines():
        if not line:
            continue
        buf += json.loads(line).get("message", {}).get("content", "")
        while True:
            m = re.search(r"[.!?…]+\s", buf)
            if not m:
                break
            yield buf[:m.end()].strip()
            buf = buf[m.end():]
    if buf.strip():
        yield buf.strip()

def talk(msgs):
    parts, t0 = [], time.time()
    for s in llm_stream(msgs):
        if not parts:
            print(f"[1re phrase en {time.time() - t0:.1f}s]")
        print(f"Aevor : {s}")
        parts.append(s)
        tts_q.put(s)
    wait_speech()
    flush_mic()
    S["last_talk"] = time.time()
    S["window"] = time.time() + CONV_WINDOW
    return " ".join(parts)

# ---------- Oreilles ----------
stream = sd.InputStream(device=MIC_DEVICE, samplerate=SR, channels=1, dtype="int16", blocksize=1600)

def level(d):
    """Volume d'un bloc, sans la composante continue."""
    x = d[:, 0].astype(np.float32)
    return float(np.abs(x - x.mean()).mean())

def flush_mic():
    n = stream.read_available
    if n:
        stream.read(n)

def calibrate():
    stream.read(5 * 1600)                      # ignore le démarrage du flux (0,5 s)
    lv = [level(stream.read(1600)[0]) for _ in range(10)]
    base = float(np.mean(lv))
    S["thr"] = min(max(base * 3, 400), 4000)
    print(f"[seuil micro : {S['thr']:.0f}, bruit de fond : {base:.0f}]")
    if base > 2000:
        print("[attention : micro saturé, baisse le gain avec alsamixer ou pactl]")

def listen(max_wait=2.0, max_len=20):
    """Attend que tu parles, enregistre jusqu'au silence. None si rien."""
    pre, blocks, started, quiet, t0 = deque(maxlen=3), [], False, 0, time.time()
    while True:
        d, _ = stream.read(1600)
        lvl = level(d)
        if not started:
            pre.append(d.copy())
            if lvl > S["thr"]:
                started, blocks = True, list(pre)
            elif time.time() - t0 > max_wait:
                return None
        else:
            blocks.append(d.copy())
            quiet = 0 if lvl > S["thr"] else quiet + 1
            if quiet >= 6 or len(blocks) > max_len * 10:     # 0,6 s de silence = fin de phrase
                break
    audio = np.concatenate(blocks)
    if len(audio) < SR * 0.4:
        return None
    with wave.open("/tmp/aevor_in.wav", "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR); w.writeframes(audio.tobytes())
    return "/tmp/aevor_in.wav"

def transcribe(path):
    t0 = time.time()
    out = subprocess.run([STT_BIN, "-m", STT_MODEL, "-l", "fr", "-t", "8", "-nt", "-np",
                          "-bs", "1", "-bo", "1",
                          "--prompt", "Conversation entre Nel et son amie Aevor.", "-f", path],
                         capture_output=True, text=True).stdout
    t = re.sub(r"\[.*?\]|\(.*?\)", "", " ".join(out.split())).strip()
    print(f"[whisper {time.time() - t0:.1f}s]")
    return "" if len(t) < 2 or BAD_RE.search(t) else t

# ---------- Présence sur le PC ----------
def idle_seconds():
    try:
        out = subprocess.check_output(
            ["gdbus", "call", "--session", "--dest", "org.gnome.Mutter.IdleMonitor",
             "--object-path", "/org/gnome/Mutter/IdleMonitor/Core",
             "--method", "org.gnome.Mutter.IdleMonitor.GetIdletime"],
            text=True, stderr=subprocess.DEVNULL)
        return int(re.search(r"\d+", out).group()) / 1000
    except Exception:
        pass
    try:
        return int(subprocess.check_output(["xprintidle"], stderr=subprocess.DEVNULL)) / 1000
    except Exception:
        return None

# ---------- Initiative ----------
def spontaneous(situation):
    msgs = messages() + [{"role": "user", "content":
        f"[Consigne interne, ne la mentionne jamais] {situation} Dis une seule phrase courte et naturelle, "
        "comme une amie qui prend l'initiative de lui parler."}]
    history.append({"role": "assistant", "content": talk(msgs)})

def proactive():
    now, hour = time.time(), datetime.datetime.now().hour
    if now < S["mute_until"] or hour >= 23 or hour < 7:
        return
    idle = idle_seconds()
    if idle is not None:
        if idle > 900:
            S["away"] = True
            return
        if S["away"] and idle < 30 and now - S["last_talk"] > 600:
            S["away"] = False
            spontaneous(f"{USER} vient de revenir devant l'ordinateur après une absence. Accueille-le avec naturel.")
            return
    if now - S["last_talk"] > PROACTIVE_EVERY and (idle is None or idle < 120):
        spontaneous(random.choice(IDEAS))

def handle(text):
    print(f"Toi : {text}")
    if MUTE_RE.search(text):
        S["mute_until"] = time.time() + 3600
        tts_q.put("D'accord, je te laisse tranquille. Appelle-moi quand tu veux.")
        wait_speech(); flush_mic()
        S["window"] = 0
        return
    if FILLERS and random.random() < 0.6:           # accroche jouée pendant que le cerveau réfléchit
        play_q.put((random.choice(FILLERS), False))
    history.append({"role": "user", "content": text})
    history.append({"role": "assistant", "content": talk(messages())})
    S["turns"] += 1
    if S["turns"] % 8 == 0:
        threading.Thread(target=learn, daemon=True).start()

def main():
    stream.start()
    calibrate()
    make_fillers()
    print(f"Aevor est là (modèle : {MODEL}). Dis « Aevor » pour lui parler. Ctrl+C pour quitter.")
    hour = datetime.datetime.now().hour
    moment = "le matin" if hour < 12 else "l'après-midi" if hour < 18 else "le soir"
    spontaneous(f"Tu viens de te réveiller sur le PC de {USER}, c'est {moment}. Salue-le.")
    try:
        while True:
            path = listen()
            if not path:
                proactive()
                continue
            text = transcribe(path)
            flush_mic()
            if not text:
                continue
            if time.time() > S["window"] and not WAKE_RE.search(text):
                continue
            handle(text)
    except KeyboardInterrupt:
        pass
    finally:
        print("\nAevor range ses souvenirs...")
        learn()

if __name__ == "__main__":
    main()