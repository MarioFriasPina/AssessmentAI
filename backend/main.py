from concurrent.futures import ThreadPoolExecutor
import asyncio
from contextlib import asynccontextmanager
import logging
import json
import aiohttp
import gymnasium as gym
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
from av import VideoFrame


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the ML model
    logger.info("Shutting down PeerConnections")
    await asyncio.gather(*[pc.close() for pc in pcs])
    pcs.clear()
    yield


app = FastAPI(lifespan=lifespan)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pc")

# Use uvloop for performance
asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())

# Thread executor for blocking tasks
executor = ThreadPoolExecutor(max_workers=2)

# Store peer connections and actions
pcs = set()
latest_actions = {}

MODEL_URL = "http://192.24.0.9:443/predict"

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.websocket("/ws")
async def websocket_handler(websocket: WebSocket):
    await websocket.accept()

    session_id = websocket.query_params.get("session_id")
    if not session_id:
        await websocket.close()
        return

    latest_actions[session_id] = None

    try:
        while True:
            data = await websocket.receive_text()
            try:
                payload = json.loads(data)
                action = payload.get("action")
                latest_actions[session_id] = action
            except json.JSONDecodeError:
                continue
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {session_id}")
    finally:
        latest_actions.pop(session_id, None)

class UserVideoTrack(VideoStreamTrack):
    def __init__(self, env, session_id):
        super().__init__()
        self.env = env
        self.session_id = session_id

    async def recv(self):
        pts, time_base = await self.next_timestamp()
        action = latest_actions.get(self.session_id)
        if action is None:
            action = self.env.action_space.sample()

        await asyncio.get_event_loop().run_in_executor(executor, self.env.step, action)
        frame = await asyncio.get_event_loop().run_in_executor(executor, self.env.render)
        video_frame = VideoFrame.from_ndarray(frame, format="rgb24")
        video_frame.pts = pts
        video_frame.time_base = time_base
        return video_frame

class RLVideoTrack(VideoStreamTrack):
    def __init__(self, env, model_url, skip=2):
        super().__init__()
        self.env = env
        self.model_url = model_url
        self.skip = skip
        self.frame_counter = 0
        self.last_obs = None
        self.last_action = None
        connector = aiohttp.TCPConnector(limit=0, keepalive_timeout=75)
        self.session = aiohttp.ClientSession(connector=connector)

    async def get_remote_action(self, frame):
        tensor = frame.astype('float32') / 255.0
        tensor = tensor.transpose(2, 0, 1)[None, ...]
        payload = tensor.tobytes()
        shape_info = json.dumps(tensor.shape)

        async with self.session.post(
            self.model_url,
            data=payload,
            headers={
                'Content-Type': 'application/octet-stream',
                'X-SHAPE': shape_info
            }
        ) as resp:
            result = await resp.json()
            return result['action']

    async def recv(self):
        pts, time_base = await self.next_timestamp()

        if self.last_obs is None:
            obs, _ = await asyncio.get_event_loop().run_in_executor(executor, self.env.reset)
            self.last_obs = obs

        frame = await asyncio.get_event_loop().run_in_executor(executor, self.env.render)

        if self.frame_counter % self.skip == 0:
            try:
                action = await self.get_remote_action(frame)
            except Exception as e:
                logger.error(f"AI inference failed: {e}")
                action = self.env.action_space.sample()
            self.last_action = action
        else:
            action = self.last_action

        await asyncio.get_event_loop().run_in_executor(executor, self.env.step, action)
        self.frame_counter += 1

        video_frame = VideoFrame.from_ndarray(frame, format="rgb24")
        video_frame.pts = pts
        video_frame.time_base = time_base
        return video_frame

@app.post("/offer")
async def offer(request: Request):
    params = await request.json()
    session_id = params.get("session_id")
    if not session_id:
        return JSONResponse(content={"error": "Missing session_id"}, status_code=400)

    offer_sdp = params["sdp"]
    offer_type = params["type"]

    _offer = RTCSessionDescription(sdp=offer_sdp, type=offer_type)
    pc = RTCPeerConnection()
    pcs.add(pc)
    logger.info(f"Created PeerConnection {pc}")

    await pc.setRemoteDescription(_offer)

    env_user = gym.make("CarRacing-v3", render_mode="rgb_array", continuous=False)
    env_user.reset()
    env_rl = gym.make("CarRacing-v3", render_mode="rgb_array", continuous=False)
    env_rl.reset()

    video_track_user = UserVideoTrack(env_user, session_id)
    video_track_rl = RLVideoTrack(env_rl, MODEL_URL)

    pc.addTrack(video_track_user)
    pc.addTrack(video_track_rl)

    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return {
        "sdp": pc.localDescription.sdp,
        "type": pc.localDescription.type
    }



