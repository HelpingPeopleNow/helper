"""HTTP server for audio transcription endpoint."""
import cgi
import json
import logging
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler

from internal.adapters.whisper_stt import transcribe_audio

logger = logging.getLogger(__name__)


class TranscribeHandler(BaseHTTPRequestHandler):
    """Handles POST /api/v1/transcribe — accepts audio, returns text."""

    def do_POST(self) -> None:
        if self.path != "/api/v1/transcribe":
            self.send_response(404)
            self.end_headers()
            return

        try:
            content_type = self.headers.get("Content-Type", "")
            if "multipart/form-data" not in content_type:
                self._json_response(400, {"error": "Expected multipart/form-data"})
                return

            # Parse multipart form data
            form = cgi.FieldStorage(
                fp=self.rfile,
                headers=self.headers,
                environ={
                    "REQUEST_METHOD": "POST",
                    "CONTENT_TYPE": content_type,
                },
            )

            file_field = form["audio"]
            if not file_field.filename:
                self._json_response(400, {"error": "No audio file provided"})
                return

            audio_bytes = file_field.file.read()
            filename = file_field.filename or "audio.webm"

            if len(audio_bytes) == 0:
                self._json_response(400, {"error": "Empty audio file"})
                return

            if len(audio_bytes) > 10 * 1024 * 1024:  # 10MB limit
                self._json_response(400, {"error": "Audio file too large (max 10MB)"})
                return

            logger.info("Transcribing: filename=%s size=%d", filename, len(audio_bytes))
            result = transcribe_audio(audio_bytes, filename)
            logger.info("Transcription done: text_len=%d lang=%s", len(result["text"]), result["language"])

            self._json_response(200, result)

        except Exception:
            logger.exception("Transcription endpoint error")
            self._json_response(500, {"error": "Transcription failed"})

    def _json_response(self, status: int, data: dict) -> None:
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        """Health check."""
        if self.path == "/health":
            body = json.dumps({"status": "ok", "service": "transcribe"}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, fmt, *args) -> None:
        logger.debug("transcribe: %s", fmt % args)


def serve_transcribe(port: int = 8085) -> HTTPServer:
    """Start the transcription HTTP server in a daemon thread."""
    server = HTTPServer(("0.0.0.0", port), TranscribeHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    logger.info("Transcribe HTTP server on :%d", port)
    return server
