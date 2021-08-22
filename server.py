from urllib.parse import unquote
from http.server import BaseHTTPRequestHandler, HTTPServer
from pred import spacing


hostName = "localhost"
serverPort = 3031


class MyServer(BaseHTTPRequestHandler):
    def do_GET(self):
        print(u"[START]: Received GET for %s" % (self.path))
        if self.path.startswith("/spacing?"):
            query_string = self.path.partition('?')[2]
            body = unquote(query_string.split('=')[1])
            
            self.send_response(200)
            self.send_header("Content-type", "text/plain;charset=utf-8")
            self.end_headers()
            self.wfile.write(bytes(spacing(body), "utf-8"))

    def do_POST(self):
        content_length = int(self.headers['Content-Length']) # <--- Gets the size of data
        post_data = self.rfile.read(content_length) # <--- Gets the data itself
        decoded_post_data = post_data.decode('utf-8')
        
        self.send_response(200)
        self.send_header("Content-type", "text/plain;charset=utf-8")
        # self.send_header('Access-Control-Allow-Credentials', 'true')
        # self.send_header('Access-Control-Allow-Origin', 'http://localhost:8888')
        self.end_headers()
        self.wfile.write(bytes(spacing(decoded_post_data), "utf-8"))

    def do_OPTIONS(self):
        self.send_response(200, "ok")
        self.send_header('Access-Control-Allow-Credentials', 'true')
        self.send_header('Access-Control-Allow-Origin', 'localhost')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header("Access-Control-Allow-Headers", "X-Requested-With, Content-type")


if __name__ == "__main__":
    webServer = HTTPServer((hostName, serverPort), MyServer)
    print("Server started http://%s:%s" % (hostName, serverPort))

    try:
        webServer.serve_forever()
        
    except KeyboardInterrupt:
        pass
    
    webServer.server_close()
    print("Server stopped.")
