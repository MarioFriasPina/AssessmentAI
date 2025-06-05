import asyncio
import json
import logging
import uuid
from typing import Any, Dict
from concurrent.futures import ThreadPoolExecutor

import aiohttp
import gymnasium as gym
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
from av import VideoFrame
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("webrtc-server")

# FastAPI app
app = FastAPI()

# ThreadPool executor for offloading gym.step() and gym.render()
executor = ThreadPoolExecutor()

# Global state
pcs = set()  # All active RTCPeerConnection objects

# session_data maps session_id -> {
#     "pc": RTCPeerConnection,
#     "env_user": gym.Env,
#     "env_rl": gym.Env,
#     "user_track": UserVideoTrack,
#     "rl_track": RLVideoTrack,
#     "queue": asyncio.Queue,
# }
session_data: Dict[str, Dict[str, Any]] = {}
session_data_lock = asyncio.Lock()

# Convenience: alias for the queue storage (mirrors session_data's "queue" field)
# but we keep both in sync via session_data_lock.
session_queues: Dict[str, asyncio.Queue] = {}

# Model inference server URL
MODEL_URL = "http://localhost:8001/inference"


async def cleanup(session_id: str):
    """
    Tear down everything associated with a given session_id:
      - Stop video tracks (which closes any associated aiohttp session)
      - Close the RTCPeerConnection
      - Remove from pcs set
      - Close gym environments
      - Remove session_data and session_queues entries
    """
    async with session_data_lock:
        info = session_data.pop(session_id, None)
        queue = session_queues.pop(session_id, None)

    if not info:
        return

    pc: RTCPeerConnection = info["pc"]
    env_user = info["env_user"]
    env_rl = info["env_rl"]
    user_track: "UserVideoTrack" = info["user_track"]
    rl_track: "RLVideoTrack" = info["rl_track"]

    # Stop the video tracks (this will also close aiohttp sessions in RLVideoTrack)
    try:
        await user_track.stop()
    except Exception as e:
        logger.warning(f"Error stopping UserVideoTrack for session {session_id}: {e}")
    try:
        await rl_track.stop()
    except Exception as e:
        logger.warning(f"Error stopping RLVideoTrack for session {session_id}: {e}")

    # Close the PeerConnection
    try:
        await pc.close()
    except Exception as e:
        logger.warning(f"Error closing PeerConnection for session {session_id}: {e}")
    pcs.discard(pc)

    # Close gym environments
    try:
        env_user.close()
    except Exception:
        pass
    try:
        env_rl.close()
    except Exception:
        pass

    logger.info(f"Cleaned up session {session_id}")


class UserVideoTrack(VideoStreamTrack):
    """
    Video track that steps the gym environment based on client-supplied actions.
    If no action is pending, it samples a random action.
    """

    def __init__(self, env: gym.Env, session_id: str):
        super().__init__()  # initialize base class
        self.env = env
        self.session_id = session_id

    async def recv(self) -> VideoFrame:
        pts, time_base = await self.next_timestamp()

        # Pull the most recent action from the queue (drain older ones)
        async with session_data_lock:
            queue = session_queues.get(self.session_id)
        action = None
        if queue is not None:
            try:
                while True:
                    action = queue.get_nowait()
            except asyncio.QueueEmpty:
                pass

        # If no action came from client, choose a random action
        if action is None:
            action = self.env.action_space.sample()

        # Step the environment off the event loop
        result = await asyncio.get_event_loop().run_in_executor(
            executor, self.env.step, action
        )
        obs, reward, done, info = result
        if done:
            # If episode ended, reset before rendering the next frame
            obs = await asyncio.get_event_loop().run_in_executor(
                executor, self.env.reset
            )

        # Render a frame off the event loop
        frame = await asyncio.get_event_loop().run_in_executor(
            executor, self.env.render
        )

        video_frame = VideoFrame.from_ndarray(frame, format="rgb24")
        video_frame.pts = pts
        video_frame.time_base = time_base
        return video_frame


class RLVideoTrack(VideoStreamTrack):
    """
    Video track that queries a remote model for actions every `skip` frames,
    otherwise repeats the last action.
    """

    def __init__(self, env: gym.Env, model_url: str, skip: int = 2):
        super().__init__()
        self.env = env
        self.model_url = model_url
        self.skip = skip
        self.frame_counter = 0
        self.last_action = None

        connector = aiohttp.TCPConnector(limit=0, keepalive_timeout=75)
        self.session = aiohttp.ClientSession(connector=connector)

    async def get_remote_action(self, frame) -> Any:
        # Convert frame (H x W x C) to NHWC->NCHW float32 bytes
        import numpy as np

        tensor = frame.astype("float32") / 255.0
        tensor = tensor.transpose(2, 0, 1)[None, ...]  # shape: (1, C, H, W)
        payload = tensor.tobytes()
        shape_info = json.dumps(tensor.shape)

        async with self.session.post(
            self.model_url,
            data=payload,
            headers={
                "Content-Type": "application/octet-stream",
                "X-SHAPE": shape_info,
            },
        ) as resp:
            result = await resp.json()
            return result.get("action")

    async def recv(self) -> VideoFrame:
        pts, time_base = await self.next_timestamp()

        # If first frame, reset environment
        if self.frame_counter == 0:
            obs = await asyncio.get_event_loop().run_in_executor(
                executor, self.env.reset
            )

        # Render current frame before stepping
        frame = await asyncio.get_event_loop().run_in_executor(
            executor, self.env.render
        )

        # Decide whether to query the remote model
        if self.frame_counter % self.skip == 0:
            try:
                action = await self.get_remote_action(frame)
            except Exception as e:
                logger.error(f"AI inference failed (falling back to random): {e}")
                action = self.env.action_space.sample()
            self.last_action = action
        else:
            action = self.last_action

        # Step the environment
        result = await asyncio.get_event_loop().run_in_executor(
            executor, self.env.step, action
        )
        obs, reward, done, info = result
        if done:
            obs = await asyncio.get_event_loop().run_in_executor(
                executor, self.env.reset
            )

        self.frame_counter += 1

        video_frame = VideoFrame.from_ndarray(frame, format="rgb24")
        video_frame.pts = pts
        video_frame.time_base = time_base
        return video_frame

    async def stop(self):
        # Close the aiohttp session when stopping the track
        try:
            await super().stop()
        except Exception:
            pass
        try:
            await self.session.close()
        except Exception:
            pass


@app.post("/offer")
async def offer(request: Request):
    """
    HTTP endpoint to receive a WebRTC SDP offer, create a new session with:
      - a server-generated UUID as session_id
      - two gym environments (user-controlled + RL-controlled)
      - two VideoStreamTracks
      - an RTCPeerConnection that streams both tracks
    Returns the SDP answer and the session_id.
    """
    params = await request.json()
    offer_sdp = params.get("sdp")
    offer_type = params.get("type")
    if not offer_sdp or not offer_type:
        return JSONResponse(content={"error": "Missing sdp or type"}, status_code=400)

    # Generate a unique session_id (UUID4)
    session_id = str(uuid.uuid4())
    logger.info(f"Creating new session: {session_id}")

    # Prepare PeerConnection
    pc = RTCPeerConnection()
    pcs.add(pc)
    logger.info(f"Created PeerConnection {pc} for session {session_id}")

    # Handle cleanup when the connection state changes
    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        if pc.iceConnectionState in ("failed", "disconnected", "closed"):
            logger.info(f"Connection state {pc.iceConnectionState} for session {session_id}; cleaning up")
            await cleanup(session_id)

    # Set remote description (the client's offer)
    _offer = RTCSessionDescription(sdp=offer_sdp, type=offer_type)
    await pc.setRemoteDescription(_offer)

    # Create Gym environments
    env_user = gym.make("CarRacing-v3", render_mode="rgb_array", continuous=False)
    env_user.reset()
    env_rl = gym.make("CarRacing-v3", render_mode="rgb_array", continuous=False)
    env_rl.reset()

    # Create a per-session queue for user actions
    action_queue: asyncio.Queue = asyncio.Queue()

    # Store everything under session_data
    async with session_data_lock:
        session_data[session_id] = {
            "pc": pc,
            "env_user": env_user,
            "env_rl": env_rl,
            "user_track": None,  # placeholder, will set below
            "rl_track": None,    # placeholder, will set below
            "queue": action_queue,
        }
        session_queues[session_id] = action_queue

    # Instantiate VideoStreamTrack objects
    user_track = UserVideoTrack(env=env_user, session_id=session_id)
    rl_track = RLVideoTrack(env=env_rl, model_url=MODEL_URL)

    # Save track references into session_data
    async with session_data_lock:
        session_data[session_id]["user_track"] = user_track
        session_data[session_id]["rl_track"] = rl_track

    # Add tracks to the PeerConnection
    pc.addTrack(user_track)
    pc.addTrack(rl_track)

    # Create the SDP answer
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    # Return SDP + session_id to the client
    return {
        "sdp": pc.localDescription.sdp,
        "type": pc.localDescription.type,
        "session_id": session_id,
    }


@app.websocket("/ws")
async def websocket_handler(websocket: WebSocket):
    """
    WebSocket endpoint for receiving user control actions.
    The client must connect with ?session_id=<UUID> matching one returned from /offer.
    """
    await websocket.accept()
    session_id = websocket.query_params.get("session_id")
    if not session_id:
        await websocket.close(code=4001)
        return

    # Verify that the session_id is known
    async with session_data_lock:
        if session_id not in session_queues:
            await websocket.close(code=4002)
            return
        queue = session_queues[session_id]

    logger.info(f"WebSocket connected for session {session_id}")

    try:
        while True:
            data = await websocket.receive_text()
            try:
                payload = json.loads(data)
                action = payload.get("action")
                if action is not None:
                    # Enqueue the action for the UserVideoTrack to pick up
                    await queue.put(action)
            except json.JSONDecodeError:
                continue
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for session {session_id}")
    finally:
        # Trigger cleanup (idempotent if peers already closed)
        await cleanup(session_id)


# Optionally, an endpoint to shut down all peer connections (for graceful server shutdown)
@app.post("/shutdown_all")
async def shutdown_all():
    """
    Force-close all active peer connections and clean up all sessions.
    """
    # Make a copy of session IDs to avoid modifying dict while iterating
    async with session_data_lock:
        sessions = list(session_data.keys())

    for sid in sessions:
        await cleanup(sid)

    return {"status": "all sessions cleaned up"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)