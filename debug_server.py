import http.server
import socketserver
import os

PORT = 8501
WEB_DIR = os.path.join(os.path.dirname(__file__), "web")
ALLOWED_FILE = "test_math.html"

class TestMathHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path in ("/", "/test_math.html"):
            self.path = "/" + ALLOWED_FILE
        elif self.path != "/" + ALLOWED_FILE:
            self.send_error(404, "File not found")
            return
        return http.server.SimpleHTTPRequestHandler.do_GET(self)

    def translate_path(self, path):
        # Only allow serving the test_math.html file
        if path == "/" or path == "/" + ALLOWED_FILE:
            return os.path.join(WEB_DIR, ALLOWED_FILE)
        return os.devnull  # Block all other files

if __name__ == "__main__":
    os.chdir(WEB_DIR)
    with socketserver.TCPServer(("", PORT), TestMathHandler) as httpd:
        print(f"Serving /web/{ALLOWED_FILE} at http://localhost:{PORT}/")
        httpd.serve_forever()
