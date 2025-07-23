#!/usr/bin/env python3
import http.server
import socketserver
import json
from urllib.parse import urlparse

class MockHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        path = urlparse(self.path).path
        
        if path == '/health':
            response = {"status": "healthy", "environment": "development"}
        elif path == '/ready':
            response = {"status": "ready", "services": ["mock"]}
        elif path == '/startup':
            response = {"status": "started"}
        elif path == '/metrics':
            response = {"trades_total": 0, "portfolio_value": 1000000}
        else:
            response = {"message": "Quantum Trading System - Mock Server"}
        
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(response).encode())

if __name__ == "__main__":
    PORT = 8080
    with socketserver.TCPServer(("", PORT), MockHandler) as httpd:
        print(f"Mock server running at http://localhost:{PORT}")
        print("Press Ctrl+C to stop")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nServer stopped")
