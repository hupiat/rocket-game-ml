import json
from http.server import SimpleHTTPRequestHandler, HTTPServer
from main import ask_model, step_generation
from urllib.parse import urlparse, parse_qs
from genetic import Genetic
from data import Data
from collections import namedtuple

PATH_ASK = '/ask_model'
PATH_STEP = '/step_generation'
PATH_COUNT = '/count_generation'

class Handler(SimpleHTTPRequestHandler): 
  
  def to_data(self, json_datas):
    dictionaries = json.loads(json_datas)
    for dict in dictionaries:
      yield Data(dict['id'], dict['score'], dict['rocket_top'], 
        dict['rocket_left'], dict['wall_left'], dict['wall_top'])

  def do_GET(self):
    url = urlparse(self.path)
    if url.path == PATH_ASK or url.path == PATH_STEP:
      hasDatas = False
      try: 
        datas = list(self.to_data(parse_qs(url.query).get('datas')[0]))
        hasDatas = datas is not None
      except TypeError: pass
      self.send_response(200 if hasDatas else 404)
      if url.path == PATH_ASK and hasDatas:
        self.end_headers()
        predictions = list(ask_model(datas))
        self.wfile.write(bytes(str(predictions), 'utf-8'))
      elif hasDatas:
        step_generation(datas)
        self.end_headers()
    elif url.path == PATH_COUNT:
      self.send_response(200)
      self.end_headers()
      self.wfile.write(bytes(str(Genetic.generation_count), 'utf-8'))
    else:
      self.send_response(404)
      self.end_headers()
 
PORT = 8888
server_address = ('', PORT)

httpd = HTTPServer(server_address, Handler)
print('Server listening at :', PORT)
try:
  httpd.serve_forever()
except KeyboardInterrupt: pass
httpd.server_close()