import json
import ssl
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

from modules import shared
from modules.text_generation import encode, generate_reply

from extensions.api.util import build_parameters, try_start_cloudflared


class ModelHandler:
    @staticmethod
    def handle_request(handler):
        handler.send_response(200)
        handler.end_headers()
        response = json.dumps({'result': shared.model_name})
        handler.wfile.write(response.encode('utf-8'))


class GenerateHandler:
    @staticmethod
    def default_parameters():
        return {
            'max_new_tokens': 1000,
            'do_sample': True,
            'temperature': 0.1,
            'top_p': 0.1,
            'typical_p': 1,
            'repetition_penalty': 1.18,
            'top_k': 40,
            'min_length': 0,
            'no_repeat_ngram_size': 0,
            'num_beams': 1,
            'penalty_alpha': 0,
            'length_penalty': 1,
            'early_stopping': False,
            'seed': -1,
            'add_bos_token': True,
            'truncation_length': 2048,
            'ban_eos_token': False,
            'skip_special_tokens': True,
            'encoder_repetition_penalty':1,
            'custom_stopping_strings': '',  
            'stopping_strings': ["Human:"]
        }

    @staticmethod
    def extract_response(response_text, prompt):
            # Remove the prompt from the response_text
            response_text = response_text.replace(prompt, '')

            # Define identifiers
            assistant_identifier = '### Assistant:'
            human_identifier = ['### Human:', '### Human']

            # Split the text by line breaks
            lines = response_text.split('\n')

            # Initialize an empty list for the processed lines
            processed_lines = []

            # Iterate over each line
            for line in lines:
                # If the line starts with the assistant identifier, remove it
                if line.startswith(assistant_identifier):
                    line = line[len(assistant_identifier):].strip()
                # If the line starts with a human identifier, skip it
                elif any(line.startswith(identifier) for identifier in human_identifier):
                    continue

                # Append the processed line to the list
                processed_lines.append(line)

            # Join the processed lines back together and return the result
            return '\n'.join(processed_lines)

    @staticmethod
    def handle_request(handler, body):
        handler.send_response(200)
        handler.send_header('Content-Type', 'application/json')
        handler.end_headers()

        prompt = body['prompt']
        generate_params = GenerateHandler.default_parameters()
        stopping_strings = generate_params.pop('stopping_strings')

        generator = generate_reply(
            prompt, generate_params, stopping_strings=stopping_strings)

        answer = ''
        for a in generator:
            if isinstance(a, str):
                answer = a
            else:
                answer = a[0]

        extracted_text = GenerateHandler.extract_response(answer, prompt)
        response = json.dumps({'results': [{'text': extracted_text}]})
        handler.wfile.write(response.encode('utf-8'))


class TokenCountHandler:
    @staticmethod
    def handle_request(handler, body):
        handler.send_response(200)
        handler.send_header('Content-Type', 'application/json')
        handler.end_headers()

        tokens = encode(body['prompt'])[0]
        response = json.dumps({
            'results': [{
                'tokens': len(tokens)
            }]
        })
        handler.wfile.write(response.encode('utf-8'))


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/api/v1/model':
            ModelHandler.handle_request(self)
        else:
            self.send_error(404)

    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        body = json.loads(self.rfile.read(content_length).decode('utf-8'))

        if self.path == '/api/v1/generate':
            GenerateHandler.handle_request(self, body)
        elif self.path == '/api/v1/token-count':
            TokenCountHandler.handle_request(self, body)
        else:
            self.send_error(404)


def _run_server(port: int, share: bool=False):
    address = '0.0.0.0' if shared.args.listen else '127.0.0.1'
    server = ThreadingHTTPServer((address, port), Handler)
    sslctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    sslctx.load_cert_chain(certfile='cert.pem', keyfile="key.pem")
    server.socket = sslctx.wrap_socket(server.socket, server_side=True)

    def on_start(public_url: str):
        print(f'Starting non-streaming server at public url {public_url}/api')

    if share:
        try:
            try_start_cloudflared(port, max_attempts=3, on_start=on_start)
        except Exception:
            pass
    else:
        print(f'Starting API at http://{address}:{port}/api')

    server.serve_forever()


def start_server(port: int, share: bool = False):
    Thread(target=_run_server, args = [port, share], daemon=True).start()
