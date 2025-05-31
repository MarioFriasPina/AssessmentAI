from fastapi import FastAPI, Request
import numpy as np
import cv2

app = FastAPI()

@app.post("/predict")
async def predict(request: Request):
    img_bytes = await request.body()
    nparr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    # ...run your model here...
    action = 0  # Example: nothing
    return {"action": action}