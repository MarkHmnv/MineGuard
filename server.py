from glob import glob
from http.server import BaseHTTPRequestHandler, HTTPServer
import socket
import os


def get_local_ip() -> str:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 80))
        local_ip = s.getsockname()[0]
        s.close()
        return local_ip
    except Exception:
        return 'localhost'


def get_latest_log_file() -> str:
    log_files = glob('*_mineguard.log')
    return max(log_files, key=os.path.getctime) if log_files else ''


class Server(BaseHTTPRequestHandler):
    base_directory = os.path.dirname(__file__)

    routes = {
        '/': ('index.html', 'text/html'),
        '/map.html': ('map.html', 'text/html'),
    }

    def do_GET(self):
        if self.path == '/log':
            self.routes['/log'] = (get_latest_log_file(), 'text/plain')

        if self.path.split('?')[0] in self.routes:
            file_name, content_type = self.routes[self.path.split('?')[0]]
            self.serve_file(os.path.join(self.base_directory, file_name), content_type)
        else:
            file_path = os.path.join(self.base_directory, self.path[1:])
            if os.path.isfile(file_path):
                content_type = 'application/octet-stream'
                self.serve_file(file_path, content_type)
            else:
                self.send_error(404, 'File Not Found')

    def serve_file(self, file_path, content_type):
        try:
            with open(file_path, 'rb') as f:
                self.send_response(200)
                self.send_header('Content-type', content_type)
                self.end_headers()
                self.wfile.write(f.read())
        except FileNotFoundError:
            self.send_error(404, 'File Not Found')
        except IOError:
            self.send_error(500, 'Internal Server Error')


def run(server_class=HTTPServer, handler_class=Server, port=8000):
    local_ip = get_local_ip()
    server_address = ('', port)
    httpd = server_class(server_address, handler_class)
    print(f'Starting server on http://{local_ip}:{port}')
    httpd.serve_forever()


if __name__ == '__main__':
    run()
