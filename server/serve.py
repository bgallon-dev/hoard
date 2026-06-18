#!/usr/bin/env python3
# Persistent local chat server for the streaming-MoE engine.
# Loads the model ONCE (run_qwen35.exe in SERVE mode) and serves a browser chat over HTTP+SSE.
# The model stays resident and the expert cache stays warm across conversations.
#   py server/serve.py --model <gguf> --exe <run_qwen35.exe> --builddir <build> --k 48 --ram 2048 --port 8080
import argparse, json, os, subprocess, sys, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--exe", required=True)
ap.add_argument("--builddir", required=True)   # run engine from here so its ggml/llama DLLs load
ap.add_argument("--www", required=True)         # dir holding index.html
ap.add_argument("--name", default="model")
ap.add_argument("--k", type=int, default=48)
ap.add_argument("--ram", type=int, default=2048)
ap.add_argument("--port", type=int, default=8080)
A = ap.parse_args()

# ---- launch the engine once, in SERVE mode ----
env = dict(os.environ); env["SERVE"] = "1"
for v in ("CSV", "THROTTLE_MBPS", "DRYRUN", "MAXKV", "NOEOS", "Q3DBG", "Q3DUMP", "TFLEN"):
    env.pop(v, None)
proc = subprocess.Popen([A.exe, A.model, "chat", str(A.k), str(A.ram)],
                        cwd=A.builddir, env=env, bufsize=0,
                        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

ready = threading.Event()
lock = threading.Lock()            # one engine = one conversation = one in-flight request
last_perf = {"tokens": 0, "tok_s": 0.0}

def drain_stderr():
    for raw in iter(proc.stderr.readline, b""):
        s = raw.decode("utf-8", "replace").rstrip()
        if "[serve] ready" in s:
            ready.set()
        # engine prints "[N tokens, X tok/s]" at the end of each turn
        if s.startswith("[") and "tok/s]" in s:
            try:
                n = int(s[1:].split(" tokens")[0]); t = float(s.split(", ")[1].split(" tok/s")[0])
                last_perf.update(tokens=n, tok_s=t)
            except Exception:
                pass
        print("  engine|", s, file=sys.stderr, flush=True)
threading.Thread(target=drain_stderr, daemon=True).start()

def stream_until_eot():
    """Yield decoded text chunks from engine stdout until the 0x04 end-of-turn byte."""
    buf = bytearray()
    while True:
        b = proc.stdout.read(1)
        if not b:                       # engine died
            ready.clear(); break
        if b == b"\x04":
            break
        buf += b
        try:
            text = buf.decode("utf-8"); buf.clear()
            if text:
                yield text
        except UnicodeDecodeError:
            continue                    # incomplete multibyte; wait for the rest

def send_engine(s: bytes):
    proc.stdin.write(s); proc.stdin.flush()

class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass     # quiet

    def _send(self, code, ctype, body=b""):
        self.send_response(code); self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body))); self.end_headers()
        if body: self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            try:
                with open(os.path.join(A.www, "index.html"), "rb") as f: body = f.read()
                self._send(200, "text/html; charset=utf-8", body)
            except OSError:
                self._send(500, "text/plain", b"index.html missing")
        elif self.path == "/api/status":
            st = {"ready": ready.is_set(), "name": A.name, "k": A.k, "ram": A.ram,
                  "tokens": last_perf["tokens"], "tok_s": last_perf["tok_s"]}
            self._send(200, "application/json", json.dumps(st).encode())
        else:
            self._send(404, "text/plain", b"not found")

    def do_POST(self):
        ln = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(ln) if ln else b"{}"
        if self.path == "/api/reset":
            with lock:
                send_engine(b"<<<RESET>>>\n")
                while True:
                    b = proc.stdout.read(1)
                    if not b or b == b"\x04": break
            self._send(200, "application/json", b'{"ok":true}')
            return
        if self.path == "/api/chat":
            if not ready.is_set():
                self._send(503, "application/json", b'{"error":"model still loading"}'); return
            try:
                msg = json.loads(body).get("message", "")
            except Exception:
                self._send(400, "application/json", b'{"error":"bad json"}'); return
            msg = msg.replace("\r\n", "\n").replace("\r", "\n")
            with lock:
                t0 = time.time()
                send_engine(msg.encode("utf-8") + b"\n<<<SEND>>>\n")
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("X-Accel-Buffering", "no")
                self.end_headers()
                try:
                    for chunk in stream_until_eot():
                        self.wfile.write(b"data: " + json.dumps({"delta": chunk}).encode() + b"\n\n")
                        self.wfile.flush()
                    done = {"done": True, "secs": round(time.time() - t0, 1),
                            "tok_s": last_perf["tok_s"], "tokens": last_perf["tokens"]}
                    self.wfile.write(b"data: " + json.dumps(done).encode() + b"\n\n")
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    pass                # client navigated away mid-stream
            return
        self._send(404, "text/plain", b"not found")

httpd = ThreadingHTTPServer(("0.0.0.0", A.port), H)
print(f"[serve] {A.name}: loading model, then http://localhost:{A.port}  (LAN: http://<this-ip>:{A.port})", file=sys.stderr, flush=True)
threading.Thread(target=lambda: (ready.wait(), print("[serve] model ready - open the page", file=sys.stderr, flush=True)), daemon=True).start()
try:
    httpd.serve_forever()
except KeyboardInterrupt:
    pass
finally:
    try: proc.stdin.write(b"/quit\n"); proc.stdin.flush()
    except Exception: pass
    proc.terminate()
