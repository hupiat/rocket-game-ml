import json
from http.server import SimpleHTTPRequestHandler, HTTPServer
from main import ask_model, step_generation
from urllib.parse import urlparse, parse_qs
from genetic import Genetic
from data import Data
from collections import namedtuple

PATH_ASK = '/ask_model'
PATH_STEP = '/step_generation'
PATH_COUNT = '/count_individuals'


class Handler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        SimpleHTTPRequestHandler.end_headers(self)

    def to_data(self, json_datas):
        dictionaries = json.loads(json_datas)
        for dict in dictionaries:
            yield Data(dict['id'], dict['score'], dict['rocket_top'],
                       dict['wall_direction'], dict['wall_left'], dict['wall_length'])

    def do_GET(self):
        url = urlparse(self.path)
        if url.path == PATH_ASK or url.path == PATH_STEP:
            parsed = parse_qs(url.query)
            hasDatas = False
            try:
                datas = list(self.to_data(parsed.get('datas')[0]))
                hasDatas = datas is not None and len(datas) != 0
            except TypeError:
                pass
            self.send_response(200 if hasDatas else 404)
            if hasDatas:
                max_height = float(parsed.get('max_height')[0])
                max_width = float(parsed.get('max_width')[0])
                if url.path == PATH_ASK:
                    self.end_headers()
                    predictions = list(ask_model(datas, max_height, max_width))
                    self.wfile.write(bytes(str(predictions), 'utf-8'))
                else:
                    self.end_headers()
                    step_generation(datas)
        elif url.path == PATH_COUNT:
            self.send_response(200)
            self.end_headers()
            self.wfile.write(bytes(str(Genetic.count_individuals), 'utf-8'))
        else:
            self.send_response(404)
            self.end_headers()


PORT = 8080
server_address = ('', PORT)

httpd = HTTPServer(server_address, Handler)
print('Server listening at :', PORT)
try:
    httpd.serve_forever()
except KeyboardInterrupt:
    pass
httpd.server_close()
