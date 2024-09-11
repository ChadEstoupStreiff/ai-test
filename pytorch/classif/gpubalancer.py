import json
import logging
import socket


class GPUBalancerClient:
    def __init__(
        self,
        importance: str,
        max_gpu_load: str,
        title: str,
        description: str = "No desc",
        host="gpu_balancer",
        port=80,
    ):
        self.host = host
        self.port = port
        self.importance = importance
        self.max_gpu_load = max_gpu_load
        self.title = title
        self.description = description
        self.s = None

    def request_gpu(self):
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.connect((self.host, self.port))
        request = json.dumps(
            {
                "type": "request",
                "importance": self.importance,
                "max_gpu_load": self.max_gpu_load,
                "title": self.title,
                "description": self.description,
            }
        )
        self.s.sendall(request.encode())
        self.wait_for_permission()

    def wait_for_permission(self):
        while True:
            data = self.s.recv(1024)
            response = data.decode()
            if response == "yes":
                logging.info("GPU Balancer >> Permission granted")
                break

    def done(self):
        self.s.sendall(json.dumps({"type": "done"}).encode())
        self.s.close()
        logging.info("GPU Balancer >> GPU Released")

    def __enter__(
        self,
    ):
        self.request_gpu()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.done()
        return False
