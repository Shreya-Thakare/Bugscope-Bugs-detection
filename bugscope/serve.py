#!/usr/bin/env python3
"""Serve BugScope landing page."""
import http.server
import socketserver
import os
from pathlib import Path

PORT = 8080
os.chdir(Path(__file__).parent / "frontend")

print(f"BugScope landing page → http://localhost:{PORT}")
print("Dashboard: streamlit run app/app.py")
print("API:       uvicorn api:app --port 8000")
with socketserver.TCPServer(("", PORT), http.server.SimpleHTTPRequestHandler) as httpd:
    httpd.serve_forever()
