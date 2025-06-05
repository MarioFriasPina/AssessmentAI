""" Litserve server for serving AI models. """

import litserve as ls

class SimpleLitAPI(ls.LitAPI):
    def setup(self, device):
        # self.model1 = lambda x: x**2
        # self.model2 = lambda x: x**3
        pass

    def decode_request(self, request):
        return request["input"]

    def predict(self, x):
        # squared = self.model1(x)
        # cubed = self.model2(x)
        # output = squared + cubed
        return {"output": 0}

    def encode_response(self, output):
        return {"output": output}

if __name__ == "__main__":
    api = SimpleLitAPI()
    server = ls.LitServer(
        api,
        accelerator="auto",
        workers_per_device=1,   # Number of processes per device
        max_batch_size=1,       # The max number of requests to batch together.
        batch_timeout=0.5,      # The max time to wait for a batch of requests.
        stream=False,           # Whether to stream responses or not.
        callbacks=None,         # Run custom actions at specific points in the server's lifecycle, such as before or after predictions.

    )
    server.run(port=443)        # Run the server on port 443
