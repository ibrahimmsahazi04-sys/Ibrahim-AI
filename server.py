"""Assistant vocal : serveur Python gratuit (FastAPI + LLM compatible OpenAI + DuckDuckGo)."""
import os, re, json, shutil, threading, uuid, subprocess
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse
from datetime import datetime
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from openai import OpenAI
try:
    from ddgs import DDGS
except ImportError:
    from duckduckgo_search import DDGS

# --- Configuration (variables d'environnement) ---
# Par défaut : Ollama en local (100 % gratuit). Pour Groq (gratuit) :
#   LLM_BASE_URL=https://api.groq.com/openai/v1  LLM_MODEL=openai/gpt-oss-120b  LLM_API_KEY=gsk_...
BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")
MODEL = os.getenv("LLM_MODEL", "qwen2.5:7b")
API_KEY = os.getenv("LLM_API_KEY", "ollama")
TOKEN = os.getenv("ASSISTANT_TOKEN", "changeme")
WORKDIR = os.path.abspath(os.getenv("WORKDIR", "./workspace"))
os.makedirs(WORKDIR, exist_ok=True)

# --- Génération de vidéo (facultative) : appelle un Space Hugging Face gratuit ---
# VIDEO_SPACE=proprietaire/nom-du-space   (un Space Gradio qui prend une description en premier argument)
# VIDEO_API_NAME=/predict                  (voir « Use via API » en bas du Space)
# HF_TOKEN=hf_...                          (votre jeton Hugging Face : c'est SON quota gratuit qui est utilisé)
VIDEO_SPACE = os.getenv("VIDEO_SPACE", "")
VIDEO_API = os.getenv("VIDEO_API_NAME", "/predict")
HF_TOKEN = os.getenv("HF_TOKEN") or None
MEDIA = os.path.join(WORKDIR, "media")
os.makedirs(MEDIA, exist_ok=True)
jobs = {}

client = OpenAI(base_url=BASE_URL, api_key=API_KEY)
app = FastAPI()
sessions = {}
state_lock = threading.Lock()

SYSTEM = (
    "Tu es Saïd, un assistant vocal en français. Tu appelles toujours l'utilisateur « monsieur ». Réponds de façon courte et naturelle, car ta réponse est lue à voix haute. "
    "Utilise web_search pour toute information récente (tu peux lancer plusieurs recherches en même temps pour des sujets différents). Pour ouvrir un site ou une application (YouTube, Google Maps, WhatsApp, un appel, un SMS), utilise TOUJOURS l'outil open_url au lieu de donner seulement le lien : par exemple https://www.youtube.com/results?search_query=... pour chercher sur YouTube, tel:+269... pour appeler. Ne dis jamais que tu ne peux pas ouvrir. Pour écrire du code, donne-le dans un bloc ``` avec une phrase d'explication. "
    "Utilise run_command pour exécuter une commande shell : l'utilisateur la validera avant exécution. N'écris jamais d'astérisques ni de mise en forme markdown (pas de gras, pas de titres, pas de tableaux)."
)
SYSTEM_KM = (
    "Tu es Saïd, un assistant vocal. Tu réponds en shingazidja (le comorien de la Grande Comore), avec des phrases courtes et simples. "
    "Tu appelles l'utilisateur « bwana ». Tu connais mal cette langue : si tu hésites sur un mot, écris-le en shingazidja puis ajoute "
    "entre parenthèses sa traduction en français. Si l'utilisateur écrit en français, comprends-le mais réponds en shingazidja. "
    "Utilise web_search pour toute information récente (plusieurs recherches possibles en même temps), open_url pour ouvrir un site ou une application (jamais seulement le lien), run_command pour exécuter une commande shell (l'utilisateur la validera), "
    "et donne le code dans un bloc ``` avec une phrase d'explication. N'écris jamais d'astérisques ni de mise en forme markdown (pas de gras, pas de titres, pas de tableaux)."
)
TOOLS = [
    {"type": "function", "function": {
        "name": "web_search", "description": "Cherche sur le web",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "run_command", "description": "Exécute une commande shell dans le dossier de travail",
        "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}},
]

def clean(t):
    """Retire astérisques, titres et liens markdown (hors blocs de code)."""
    parts = t.split("```")
    for i in range(0, len(parts), 2):
        p = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", parts[i])
        p = re.sub(r"[*＊✱∗•`]+", "", p)
        parts[i] = re.sub(r"^#+\s*", "", p, flags=re.M)
    return "```".join(parts)

def _run_video(job, prompt):
    try:
        from gradio_client import Client
        c = Client(VIDEO_SPACE, token=HF_TOKEN) if HF_TOKEN else Client(VIDEO_SPACE)
        res = c.predict(prompt, api_name=VIDEO_API)
        path = res[0] if isinstance(res, (list, tuple)) else res
        for _ in range(2):
            if isinstance(path, dict):
                path = path.get("video") or path.get("path") or path.get("value")
        name = job + ".mp4"
        shutil.copy(path, os.path.join(MEDIA, name))
        jobs[job] = {"status": "done", "url": "/media/" + name}
    except Exception as e:
        jobs[job] = {"status": "error", "error": str(e)[:300]}

def start_video(prompt):
    job = uuid.uuid4().hex
    jobs[job] = {"status": "pending"}
    threading.Thread(target=_run_video, args=(job, prompt), daemon=True).start()
    return job

def web_search(query):
    try:
        res = DDGS().text(query, max_results=5)
        return "\n".join(f"- {r['title']}: {r['body']} ({r['href']})" for r in res) or "Aucun résultat."
    except Exception as e:
        return f"Erreur de recherche : {e}"

def run_command(cmd):
    try:
        p = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30, cwd=WORKDIR)
        return ((p.stdout + p.stderr) or "(aucune sortie)")[:3000]
    except subprocess.TimeoutExpired:
        return "Commande interrompue après 30 secondes."

DT = "%Y-%m-%dT%H:%M"
AGENDA_RULES = (
    "\nRègles de l'agenda : pour ajouter, modifier ou supprimer un événement, utilise add_event, update_event ou delete_event "
    "(jamais seulement du texte). Convertis « demain », « lundi prochain », « dans deux heures » en date précise à partir de la date actuelle. "
    "Si l'heure manque, demande-la. Avant de supprimer, demande confirmation sauf si l'utilisateur l'a clairement demandé. "
    "Si add_event signale un conflit d'horaire, préviens l'utilisateur. Pour lister l'agenda, lis la liste ci-dessus et résume-la brièvement."
)

def _parse(s):
    try:
        return datetime.strptime(str(s), DT)
    except Exception:
        return None

def _int(v, d):
    try:
        return int(v)
    except Exception:
        return d

def agenda_text(now, agenda):
    lines = []
    for e in (agenda or [])[:60]:
        notes = f" — {e['notes']}" if e.get("notes") else ""
        lines.append(f"- [{e.get('id')}] {str(e.get('start', '')).replace('T', ' ')} ({e.get('dur', 60)} min) {e.get('title', '')}{notes}")
    return ("\n\nDate et heure actuelles de l'utilisateur : " + (now or "inconnue") +
            ".\nAgenda de l'utilisateur (événements à venir) :\n" + ("\n".join(lines) or "(aucun événement)") + AGENDA_RULES)

def _conflicts(s, ev):
    st = _parse(ev["start"])
    en = st.timestamp() + ev["dur"] * 60
    deleted = {x["id"] for x in s.get("agenda_actions", []) if x["op"] == "delete"}
    others = [e for e in s.get("agenda_ctx", []) if e.get("id") not in deleted]
    others += [x["event"] for x in s.get("agenda_actions", []) if x["op"] == "add"]
    out = []
    for o in others:
        os_ = _parse(o.get("start"))
        if os_ and st.timestamp() < os_.timestamp() + _int(o.get("dur"), 60) * 60 and os_.timestamp() < en:
            out.append(f"{o.get('title')} ({o.get('start')})")
    return out

def agenda_tool(s, name, a):
    acts = s.setdefault("agenda_actions", [])
    known = {e.get("id") for e in s.get("agenda_ctx", [])} | {x["event"]["id"] for x in acts if x["op"] == "add"}
    if name == "delete_event":
        eid = str(a.get("id", ""))
        if eid not in known:
            return "Événement introuvable : identifiant inconnu."
        acts.append({"op": "delete", "id": eid})
        return "Événement supprimé."
    if name == "add_event":
        if not a.get("title") or not _parse(a.get("start")):
            return "Erreur : il faut un titre et une date de début au format YYYY-MM-DDTHH:MM (heure locale de l'utilisateur)."
        ev = {"id": uuid.uuid4().hex[:8], "title": str(a["title"])[:120], "start": a["start"],
              "dur": max(1, _int(a.get("duration_minutes"), 60)), "notes": str(a.get("notes") or "")[:300],
              "remind": max(0, _int(a.get("remind_minutes"), 10))}
        conflicts = _conflicts(s, ev)
        acts.append({"op": "add", "event": ev})
        msg = "Événement ajouté (enregistré sur l'appareil de l'utilisateur)."
        if conflicts:
            msg += " ATTENTION, conflit d'horaire avec : " + ", ".join(conflicts) + ". Préviens l'utilisateur."
        return msg
    eid = str(a.get("id", ""))
    if eid not in known:
        return "Événement introuvable : identifiant inconnu."
    fields = {}
    for k_in, k_out in (("title", "title"), ("start", "start"), ("notes", "notes"),
                        ("duration_minutes", "dur"), ("remind_minutes", "remind")):
        if a.get(k_in) is not None:
            fields[k_out] = a[k_in]
    if "start" in fields and not _parse(fields["start"]):
        return "Erreur : date de début au format YYYY-MM-DDTHH:MM requise."
    for k in ("dur", "remind"):
        if k in fields:
            fields[k] = _int(fields[k], 60 if k == "dur" else 10)
    acts.append({"op": "update", "id": eid, "fields": fields})
    return "Événement modifié."

ALLOWED = {"https", "http", "tel", "sms", "mailto", "geo", "whatsapp"}

def safe_url(u):
    u = (u or "").strip()
    return u if urlparse(u).scheme.lower() in ALLOWED else None

_EV = {"title": {"type": "string"}, "start": {"type": "string", "description": "Début au format YYYY-MM-DDTHH:MM, heure locale de l'utilisateur"},
       "duration_minutes": {"type": "integer"}, "notes": {"type": "string"}, "remind_minutes": {"type": "integer", "description": "Rappel X minutes avant (10 par défaut)"}}
TOOLS += [
    {"type": "function", "function": {"name": "add_event", "description": "Ajoute un événement à l'agenda de l'utilisateur.",
        "parameters": {"type": "object", "properties": _EV, "required": ["title", "start"]}}},
    {"type": "function", "function": {"name": "update_event", "description": "Modifie un événement existant (id visible dans l'agenda).",
        "parameters": {"type": "object", "properties": {"id": {"type": "string"}, **_EV}, "required": ["id"]}}},
    {"type": "function", "function": {"name": "delete_event", "description": "Supprime un événement de l'agenda.",
        "parameters": {"type": "object", "properties": {"id": {"type": "string"}}, "required": ["id"]}}},
]

TOOLS.append({"type": "function", "function": {
    "name": "open_url",
    "description": "Ouvre un site web ou une application sur l'appareil de l'utilisateur (un bouton Ouvrir s'affiche). Adresses autorisées : https, http, tel, sms, mailto, geo, whatsapp.",
    "parameters": {"type": "object", "properties": {"url": {"type": "string"}, "label": {"type": "string", "description": "Nom court affiché sur le bouton, par exemple YouTube"}}, "required": ["url"]}}})

if VIDEO_SPACE:
    TOOLS.append({"type": "function", "function": {
        "name": "generate_video",
        "description": "Génère un court clip vidéo (quelques secondes) à partir d'une description détaillée écrite en anglais. La vidéo apparaît dans le chat quand elle est prête (plusieurs minutes possibles).",
        "parameters": {"type": "object", "properties": {"prompt": {"type": "string"}}, "required": ["prompt"]}}})

def check(token):
    if token != TOKEN:
        raise HTTPException(401, "Code d'accès invalide")

def loop(sid):
    s = sessions[sid]
    try:
        for _ in range(6):
            m = client.chat.completions.create(model=MODEL, messages=s["messages"], tools=TOOLS).choices[0].message
            msg = {"role": "assistant", "content": m.content or ""}
            if m.tool_calls:
                msg["tool_calls"] = [{"id": t.id, "type": "function",
                    "function": {"name": t.function.name, "arguments": t.function.arguments}} for t in m.tool_calls]
            s["messages"].append(msg)
            if not m.tool_calls:
                return {"type": "reply", "text": clean(m.content or ""), "job": s.pop("last_job", None), "open": s.pop("opens", []), "agenda": s.pop("agenda_actions", [])}
            pending = None
            calls = [(t, json.loads(t.function.arguments or "{}")) for t in m.tool_calls]
            searches = [(t, a) for t, a in calls if t.function.name not in ("run_command", "generate_video", "open_url", "add_event", "update_event", "delete_event")]
            if searches:
                with ThreadPoolExecutor(max_workers=3) as ex:
                    results = list(ex.map(lambda ta: web_search(ta[1].get("query", "")), searches))
                for (t, _), r in zip(searches, results):
                    s["messages"].append({"role": "tool", "tool_call_id": t.id, "content": r})
            for t, args in calls:
                if t.function.name == "run_command":
                    if pending is None:
                        pending = {"id": t.id, "command": args.get("command", "")}
                    else:
                        s["messages"].append({"role": "tool", "tool_call_id": t.id, "content": "Ignoré : une commande à la fois."})
                elif t.function.name in ("add_event", "update_event", "delete_event"):
                    s["messages"].append({"role": "tool", "tool_call_id": t.id, "content": agenda_tool(s, t.function.name, args)})
                elif t.function.name == "open_url":
                    u = safe_url(args.get("url"))
                    if u:
                        s.setdefault("opens", []).append({"url": u, "label": (args.get("label") or u)[:60]})
                        content = "Le lien est affiché à l'utilisateur avec un bouton Ouvrir. Dis-lui brièvement d'appuyer sur ce bouton."
                    else:
                        content = "Lien refusé : type d'adresse non autorisé."
                    s["messages"].append({"role": "tool", "tool_call_id": t.id, "content": content})
                elif t.function.name == "generate_video":
                    s["last_job"] = start_video(args.get("prompt", ""))
                    s["messages"].append({"role": "tool", "tool_call_id": t.id, "content": "Génération lancée en arrière-plan. Dis brièvement à l'utilisateur qu'elle peut prendre plusieurs minutes et que la vidéo apparaîtra dans le chat."})
            if pending:
                s["pending"] = pending
                return {"type": "confirm", "command": pending["command"]}
        return {"type": "reply", "text": "Je n'ai pas réussi à terminer cette tâche."}
    except Exception as e:
        return {"type": "reply", "text": f"Erreur du modèle : {e}"}

class Chat(BaseModel):
    sid: str
    text: str
    lang: str = "fr"
    now: str = ""
    agenda: list = []

class Confirm(BaseModel):
    sid: str
    approve: bool

def finish(sid, res):
    """Termine une demande ; une demande parallèle est rangée dans la conversation principale."""
    s = sessions.get(sid)
    if s and s.get("parent") and res.get("type") == "reply":
        parent = sessions.get(s["parent"])
        if parent is not None:
            parent["inbox"] += [{"role": "user", "content": s["question"]},
                                {"role": "assistant", "content": res.get("text", "")}]
        sessions.pop(sid, None)
    res["sid"] = sid
    return res

@app.post("/api/chat")
def chat(b: Chat, x_token: str = Header("")):
    check(x_token)
    sysmsg = (SYSTEM_KM if b.lang == "km" else SYSTEM) + agenda_text(b.now, b.agenda)
    with state_lock:
        s = sessions.setdefault(b.sid, {"messages": [{"role": "system", "content": sysmsg}],
                                        "pending": None, "busy": False, "inbox": []})
        sid = b.sid
        if s["busy"] or s["pending"]:
            # Demande en parallèle : conversation annexe qui reprend le contexte récent.
            recent = [{"role": m["role"], "content": m["content"]} for m in s["messages"]
                      if m["role"] in ("user", "assistant") and m.get("content") and not m.get("tool_calls")][-6:]
            sid = b.sid + ":" + uuid.uuid4().hex[:6]
            s = sessions[sid] = {"messages": [{"role": "system", "content": sysmsg}] + recent,
                                 "pending": None, "busy": False, "inbox": [], "parent": b.sid, "question": b.text}
        else:
            s["messages"][0]["content"] = sysmsg
            if s["inbox"]:
                s["messages"].extend(s["inbox"])
                s["inbox"].clear()
        s["agenda_ctx"] = b.agenda
        s["messages"].append({"role": "user", "content": b.text})
        s["busy"] = True
    try:
        res = loop(sid)
    finally:
        s["busy"] = False
    return finish(sid, res)

@app.post("/api/confirm")
def confirm(b: Confirm, x_token: str = Header("")):
    check(x_token)
    s = sessions.get(b.sid)
    if not s or not s["pending"]:
        return {"type": "reply", "text": "Aucune commande en attente.", "sid": b.sid}
    p, s["pending"] = s["pending"], None
    out = run_command(p["command"]) if b.approve else "Refusé par l'utilisateur."
    s["messages"].append({"role": "tool", "tool_call_id": p["id"], "content": out})
    s["busy"] = True
    try:
        r = loop(b.sid)
    finally:
        s["busy"] = False
    r["output"] = out if b.approve else None
    return finish(b.sid, r)

@app.get("/api/video/{job}")
def video_status(job: str, x_token: str = Header("")):
    check(x_token)
    return jobs.get(job, {"status": "error", "error": "Tâche inconnue"})

app.mount("/media", StaticFiles(directory=MEDIA), name="media")

@app.get("/")
def index():
    base = os.path.dirname(os.path.abspath(__file__))
    for p in (os.path.join(base, "static", "index.html"), os.path.join(base, "index.html")):
        if os.path.exists(p):
            return FileResponse(p)
    raise HTTPException(404, "index.html introuvable")
