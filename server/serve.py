#!/usr/bin/env python3
# Persistent local chat server for the streaming-MoE engine, with saved conversations + live controls.
# Loads a model ONCE (run_qwen35.exe in SERVE mode); conversations persist server-side (shared across
# every device on the LAN). Controls: model switch (relaunch), Stop, Regenerate, temperature.
#   py server/serve.py --models "80b=PATH;35b=PATH" --model 80b --exe <exe> --builddir <build> --www <server> [--data DIR] --k 48 --ram 2048 --port 8080
import argparse, base64, json, os, subprocess, sys, threading, time, uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

ap = argparse.ArgumentParser()
ap.add_argument("--models", required=True)      # "name=path;name=path" (only existing files)
ap.add_argument("--model", default=None)        # initial model NAME (default: first)
ap.add_argument("--exe", required=True)
ap.add_argument("--builddir", required=True)
ap.add_argument("--www", required=True)
ap.add_argument("--data", default=None)
ap.add_argument("--k", type=int, default=48)
ap.add_argument("--ram", type=int, default=2048)
ap.add_argument("--port", type=int, default=8080)
A = ap.parse_args()
MODELS = {}
for part in A.models.split(";"):
    if "=" in part:
        n, p = part.split("=", 1)
        if os.path.exists(p): MODELS[n.strip()] = p.strip()
if not MODELS: sys.exit("no valid models in --models")
current_model = A.model if A.model in MODELS else next(iter(MODELS))
DATA = A.data or os.path.join(os.path.dirname(os.path.abspath(__file__)), "_chatdata")
os.makedirs(DATA, exist_ok=True)
STORE = os.path.join(DATA, "conversations.json")

env = dict(os.environ); env["SERVE"] = "1"
for v in ("CSV", "THROTTLE_MBPS", "DRYRUN", "MAXKV", "NOEOS", "GATEDUMP", "PROFILE", "MAXGEN", "TFLEN", "IOSELFTEST"):
    env.pop(v, None)

ready = threading.Event()
lock = threading.Lock()            # serializes engine generation (one in-flight turn)
io_lock = threading.Lock()         # serializes raw writes to engine stdin (SEND/RESET/LOAD/STOP/TEMP)
last_perf = {"tokens": 0, "tok_s": 0.0}
proc = None
engine_cid = None                  # which conversation's context is loaded in the engine
current_temp = 0.0

def drain_stderr(p):
    for raw in iter(p.stderr.readline, b""):
        s = raw.decode("utf-8", "replace").rstrip()
        if "[serve] ready" in s: ready.set()
        if s.startswith("[") and "tok/s]" in s:
            try:
                n = int(s[1:].split(" tokens")[0]); t = float(s.split(", ")[1].split(" tok/s")[0])
                last_perf.update(tokens=n, tok_s=t)
            except Exception: pass
        print("  engine|", s, file=sys.stderr, flush=True)

def start_engine(path):
    """(Re)launch the engine subprocess on `path`; resets context + restores temperature."""
    global proc, engine_cid
    if proc:
        try: proc.stdin.write(b"/quit\n"); proc.stdin.flush()
        except Exception: pass
        proc.terminate()
        try: proc.wait(timeout=5)
        except Exception:
            try: proc.kill()
            except Exception: pass
    ready.clear()
    proc = subprocess.Popen([A.exe, path, "chat", str(A.k), str(A.ram)],
                            cwd=A.builddir, env=env, bufsize=0,
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    engine_cid = None
    threading.Thread(target=drain_stderr, args=(proc,), daemon=True).start()

def send_engine(s: bytes):
    with io_lock:
        proc.stdin.write(s); proc.stdin.flush()

def stop_engine():
    with io_lock:
        try: proc.stdin.write(b"<<<STOP>>>\n"); proc.stdin.flush()
        except Exception: pass

def set_temp(t):
    global current_temp
    current_temp = float(t)
    with io_lock:
        try: proc.stdin.write(("<<<TEMP %.3f>>>\n" % current_temp).encode()); proc.stdin.flush()
        except Exception: pass

def wait_eot():
    while True:
        b = proc.stdout.read(1)
        if not b or b == b"\x04": break

def stream_until_eot():
    buf = bytearray()
    while True:
        b = proc.stdout.read(1)
        if not b: ready.clear(); break
        if b == b"\x04": break
        buf += b
        try:
            text = buf.decode("utf-8"); buf.clear()
            if text: yield text
        except UnicodeDecodeError:
            continue

def engine_reset():
    send_engine(b"<<<RESET>>>\n"); wait_eot()

def engine_load(messages):
    parts = [b"<<<LOAD>>>\n"]
    for m in messages:
        tag = b"U" if m["role"] == "user" else b"A"
        parts.append(tag + b" " + base64.b64encode(m["content"].encode("utf-8")) + b"\n")
    parts.append(b"<<<ENDLOAD>>>\n")
    send_engine(b"".join(parts)); wait_eot()

def ensure_loaded(cid):
    global engine_cid
    if engine_cid == cid: return
    engine_reset()
    conv = STATE["convs"].get(cid)
    if conv and conv["messages"]: engine_load(conv["messages"])
    engine_cid = cid

# ---- conversation store ----
store_lock = threading.Lock()
STATE = {"convs": {}, "order": [], "active": None}

def load_store():
    global STATE
    try:
        with open(STORE, encoding="utf-8") as f: s = json.load(f)
        STATE = {"convs": s.get("convs", {}), "order": s.get("order", []), "active": s.get("active")}
    except Exception:
        STATE = {"convs": {}, "order": [], "active": None}

def save_store():
    tmp = STORE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f: json.dump(STATE, f, ensure_ascii=False)
    os.replace(tmp, STORE)

def new_conv():
    cid = uuid.uuid4().hex[:12]
    STATE["convs"][cid] = {"id": cid, "title": "New chat", "created": time.time(), "updated": time.time(), "messages": []}
    STATE["order"].insert(0, cid); STATE["active"] = cid
    return STATE["convs"][cid]

def touch(cid):
    if cid in STATE["order"]: STATE["order"].remove(cid)
    STATE["order"].insert(0, cid)

def conv_summaries():
    out = []
    for cid in STATE["order"]:
        c = STATE["convs"].get(cid)
        if c: out.append({"id": cid, "title": c["title"], "updated": c["updated"], "count": len(c["messages"])})
    return out

load_store()
with store_lock:
    if not STATE["order"]: new_conv()
    if STATE["active"] not in STATE["convs"]: STATE["active"] = STATE["order"][0]
    save_store()

class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    def log_message(self, *a): pass

    def _json(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)

    def _body(self):
        ln = int(self.headers.get("Content-Length", 0))
        try: return json.loads(self.rfile.read(ln) if ln else b"{}")
        except Exception: return {}

    def do_GET(self):
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            try:
                with open(os.path.join(A.www, "index.html"), "rb") as f: body = f.read()
                self.send_response(200); self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)
            except OSError: self._json(500, {"error": "index.html missing"})
        elif path == "/api/status":
            self._json(200, {"ready": ready.is_set(), "model": current_model, "models": list(MODELS.keys()),
                             "temp": current_temp, "k": A.k, "ram": A.ram, "active": STATE["active"],
                             "tokens": last_perf["tokens"], "tok_s": last_perf["tok_s"]})
        elif path == "/api/conversations":
            with store_lock: self._json(200, {"conversations": conv_summaries(), "active": STATE["active"]})
        elif path == "/api/conversation":
            cid = parse_qs(urlparse(self.path).query).get("id", [None])[0]
            with store_lock:
                c = STATE["convs"].get(cid); self._json(200 if c else 404, c or {"error": "not found"})
        else:
            self._json(404, {"error": "not found"})

    def _generate(self, aid, conv, user_msg):
        """Assumes `lock` held. (Re)load context, append the user turn, stream the answer, persist."""
        ensure_loaded(aid)
        with store_lock:
            conv["messages"].append({"role": "user", "content": user_msg})
            if conv["title"] == "New chat":
                conv["title"] = (user_msg[:38] + "…") if len(user_msg) > 38 else user_msg
            touch(aid); save_store()
        t0 = time.time()
        send_engine(user_msg.encode("utf-8") + b"\n<<<SEND>>>\n")
        self.send_response(200)
        for h, v in (("Content-Type", "text/event-stream"), ("Cache-Control", "no-cache"),
                     ("X-Accel-Buffering", "no"), ("Connection", "close")): self.send_header(h, v)
        self.end_headers()
        answer = []
        try:
            for chunk in stream_until_eot():
                answer.append(chunk)
                self.wfile.write(b"data: " + json.dumps({"delta": chunk}).encode() + b"\n\n"); self.wfile.flush()
            done = {"done": True, "secs": round(time.time() - t0, 1), "tok_s": last_perf["tok_s"], "tokens": last_perf["tokens"]}
            self.wfile.write(b"data: " + json.dumps(done).encode() + b"\n\n"); self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass
        with store_lock:
            conv["messages"].append({"role": "assistant", "content": "".join(answer)})
            conv["updated"] = time.time(); save_store()

    def do_POST(self):
        global engine_cid
        path = urlparse(self.path).path
        if path == "/api/new":
            with lock, store_lock:
                c = new_conv(); engine_reset(); engine_cid = c["id"]; save_store()
            self._json(200, c); return
        if path == "/api/select":
            cid = self._body().get("id")
            with store_lock:
                if cid in STATE["convs"]: STATE["active"] = cid; save_store(); self._json(200, STATE["convs"][cid])
                else: self._json(404, {"error": "not found"})
            return
        if path == "/api/rename":
            b = self._body(); cid = b.get("id"); title = (b.get("title") or "").strip()[:80]
            with store_lock:
                if cid in STATE["convs"] and title: STATE["convs"][cid]["title"] = title; save_store(); self._json(200, {"ok": True})
                else: self._json(400, {"error": "bad request"})
            return
        if path == "/api/delete":
            cid = self._body().get("id")
            with lock, store_lock:
                if cid in STATE["convs"]:
                    del STATE["convs"][cid]
                    if cid in STATE["order"]: STATE["order"].remove(cid)
                    if engine_cid == cid: engine_cid = None
                    if STATE["active"] == cid: STATE["active"] = STATE["order"][0] if STATE["order"] else new_conv()["id"]
                    save_store()
                self._json(200, {"active": STATE["active"], "conversations": conv_summaries()})
            return
        if path == "/api/stop":
            stop_engine(); self._json(200, {"ok": True}); return
        if path == "/api/temp":
            try: t = max(0.0, min(2.0, float(self._body().get("temp", 0))))
            except Exception: t = 0.0
            set_temp(t); self._json(200, {"temp": current_temp}); return
        if path == "/api/model":
            name = self._body().get("model")
            if name not in MODELS: self._json(400, {"error": "unknown model"}); return
            with lock:
                global current_model
                if name != current_model:
                    start_engine(MODELS[name]); current_model = name
                    if not ready.wait(timeout=120): self._json(504, {"error": "model load timed out"}); return
                    if current_temp: set_temp(current_temp)
            self._json(200, {"model": current_model}); return
        if path == "/api/chat":
            if not ready.is_set(): self._json(503, {"error": "model still loading"}); return
            msg = (self._body().get("message") or "").replace("\r\n", "\n").replace("\r", "\n")
            if not msg.strip(): self._json(400, {"error": "empty"}); return
            with lock:
                with store_lock:
                    aid = STATE["active"]; conv = STATE["convs"].get(aid)
                    if conv is None: conv = new_conv(); aid = conv["id"]
                self._generate(aid, conv, msg)
            return
        if path == "/api/regenerate":
            if not ready.is_set(): self._json(503, {"error": "model still loading"}); return
            with lock:
                with store_lock:
                    aid = STATE["active"]; conv = STATE["convs"].get(aid)
                    if not conv or len(conv["messages"]) < 2 or conv["messages"][-1]["role"] != "assistant":
                        self._json(400, {"error": "nothing to regenerate"}); return
                    conv["messages"].pop()                       # drop last assistant
                    user_msg = conv["messages"].pop()["content"] # drop + reuse last user
                    save_store()
                engine_cid = None                                # force context rebuild from the trimmed history
                self._generate(aid, conv, user_msg)
            return
        self._json(404, {"error": "not found"})

start_engine(MODELS[current_model])
httpd = ThreadingHTTPServer(("0.0.0.0", A.port), H)
print(f"[serve] {current_model}: loading, then http://localhost:{A.port}  (LAN: http://<this-ip>:{A.port})", file=sys.stderr, flush=True)
threading.Thread(target=lambda: (ready.wait(), print("[serve] model ready - open the page", file=sys.stderr, flush=True)), daemon=True).start()
try:
    httpd.serve_forever()
except KeyboardInterrupt:
    pass
finally:
    try: proc.stdin.write(b"/quit\n"); proc.stdin.flush()
    except Exception: pass
    if proc: proc.terminate()
