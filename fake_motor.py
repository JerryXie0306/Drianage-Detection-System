"""Pretend to be the ESP32 motor controller, to dry-run DrainGuard without moving any hardware.

    python fake_motor.py            # listens on 127.0.0.1:8082 and prints every command it receives

Then put in motor.txt:   http://127.0.0.1:8082
"""

import argparse
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        stamp = time.strftime("%H:%M:%S")
        if u.path == "/timed":
            msg = f"RUN {q.get('dir', ['?'])[0]} for {q.get('ms', ['?'])[0]} ms (stops by itself)"
        elif u.path in ("/forward", "/reverse"):
            msg = f"RUN {u.path[1:]} CONTINUOUSLY (nothing stops it except /stop)"
        elif u.path in ("/stop", "/auto", "/status", "/"):
            msg = u.path
        else:
            self.send_response(404)
            self.end_headers()
            return
        print(f"[{stamp}] {msg}")
        body = b'{"ok":true}'
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8082)
    a = ap.parse_args()
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), H)
    srv.daemon_threads = True
    print(f"Fake motor on http://127.0.0.1:{a.port}   (Ctrl+C to stop)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
