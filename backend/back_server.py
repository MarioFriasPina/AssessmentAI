import asyncio
import logging
import json
import aiohttp
import cv2
import gymnasium as gym

from av import VideoFrame
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
from aiohttp import web

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pc")

# Set this to your AI server's internal/private IP
model_url = "http://localhost:8000/predict"

pcs = set()
latest_actions = {}

async def websocket_handler(request):
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    session_id = request.query.get("session_id")
    if not session_id:
        await ws.close()
        return ws
    latest_actions[session_id] = None
    async for msg in ws:
        if msg.type == web.WSMsgType.TEXT:
            data = json.loads(msg.data)
            latest_actions[session_id] = data.get("action")
        elif msg.type == web.WSMsgType.ERROR:
            logger.error('ws connection closed with exception %s', ws.exception())
    del latest_actions[session_id]
    return ws

class RLVideoTrack(VideoStreamTrack):
    def __init__(self, env, model_url):
        super().__init__()
        self.env = env
        self.model_url = model_url
        self.last_obs = None
        self.session = aiohttp.ClientSession()

    async def get_remote_action(self, frame):
        _, img_encoded = cv2.imencode('.jpg', frame)
        img_bytes = img_encoded.tobytes()
        async with self.session.post(
            self.model_url,
            data=img_bytes,
            headers={'Content-Type': 'application/octet-stream'}
        ) as resp:
            result = await resp.json()
            return result['action']

    async def recv(self):
        pts, time_base = await self.next_timestamp()
        if self.last_obs is None:
            self.last_obs, _ = self.env.reset()
        frame = self.env.render()
        action = await self.get_remote_action(frame)
        self.last_obs, _, _, _, _ = self.env.step(action)
        video_frame = VideoFrame.from_ndarray(frame, format="rgb24")
        video_frame.pts = pts
        video_frame.time_base = time_base
        return video_frame

class GymVideoTrack(VideoStreamTrack):
    def __init__(self, env, session_id):
        super().__init__()
        self.env = env
        self.session_id = session_id

    async def recv(self):
        pts, time_base = await self.next_timestamp()
        action = latest_actions.get(self.session_id)
        if action is None:
            action = self.env.action_space.sample()
        self.env.step(action)
        frame = self.env.render()
        video_frame = VideoFrame.from_ndarray(frame, format="rgb24")
        video_frame.pts = pts
        video_frame.time_base = time_base
        return video_frame

async def offer(request):
    params = await request.json()
    session_id = params.get("session_id")
    if not session_id:
        return web.json_response({"error": "Missing session_id"}, status=400)
    _offer = RTCSessionDescription(sdp=params['sdp'], type=params['type'])
    pc = RTCPeerConnection()
    pcs.add(pc)
    logger.info("Created PeerConnection %s", pc)
    await pc.setRemoteDescription(_offer)
    env_user = gym.make("CarRacing-v3", render_mode="rgb_array", continuous=False)
    env_user.reset()
    env_rl = gym.make("CarRacing-v3", render_mode="rgb_array", continuous=False)
    env_rl.reset()
    video_track_user = GymVideoTrack(env_user, session_id)
    pc.addTrack(video_track_user)
    video_track_rl = RLVideoTrack(env_rl, model_url)
    pc.addTrack(video_track_rl)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)
    return web.json_response({
        'sdp': pc.localDescription.sdp,
        'type': pc.localDescription.type
    })

async def on_shutdown(_):
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()

if __name__ == '__main__':
    app = web.Application()
    app.on_shutdown.append(on_shutdown)
    app.router.add_post('/offer', offer)
    app.router.add_get('/ws', websocket_handler)
    web.run_app(app, port=8080)