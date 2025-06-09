""" Litserve server for serving AI models. """

import litserve as ls
from stable_baselines3 import PPO
import gymnasium as gym

import numpy as np
import torch

class SimpleLitAPI(ls.LitAPI):
    def setup(self, device):
        self.env = gym.make("CarRacing-v3", render_mode="rgb_array")

        model_file = "checkpoints/basicPPO/final_ppo_carracing.zip"
        self.model = PPO.load(model_file, env=self.env, device=device)

    def decode_request(self, request):
        return {"obs": request["obs"]}

    def predict(self, x):
        with torch.no_grad():
            action = self.model.predict(x, deterministic=True)
        return action

    def encode_response(self, output):
        return {"action": output}

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
