""" This module defines the WebSocket handler for the backend server."""
from concurrent.futures import ThreadPoolExecutor
import asyncio
import logging
import json
import aiohttp
import gymnasium as gym
import uvloop

from av import VideoFrame
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
from aiohttp import web

# Set the event loop policy to use uvloop, improves I/O performance
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pc")

# Global executor for running blocking calls in threads
executor = ThreadPoolExecutor(max_workers=2)

# URL for the AI server
MODEL_URL = "http://localhost:8000/predict"

pcs = set()
latest_actions = {}

async def websocket_handler(request):
    """This is the main entry point for the WebSocket endpoint. It parses the action from the WebSocket message and
    saves it in the latest_actions dictionary. It also removes the session from the latest_actions dictionary when the
    WebSocket is closed."""
    ws = web.WebSocketResponse()
    await ws.prepare(request)

    # Get the session_id from the query parameters
    session_id = request.query.get("session_id")
    if not session_id:
        await ws.close()
        return ws

    # Save the latest action for this session
    latest_actions[session_id] = None

    async for msg in ws:
        if msg.type == web.WSMsgType.TEXT:
            # Try to parse the action as JSON
            try:
                data = json.loads(msg.data)
                action = data.get("action")
                latest_actions[session_id] = action
            except json.JSONDecodeError:
                pass
        elif msg.type == web.WSMsgType.ERROR:
            logger.error("WebSocket connection closed with exception %s", ws.exception())

    # Elminate the session from latest_actions when WebSocket closes
    if session_id in latest_actions:
        del latest_actions[session_id]
    return ws

class UserVideoTrack(VideoStreamTrack):
    """
    A VideoStreamTrack that interacts with a gym environment.
    """
    def __init__(self, env, session_id):
        """
        Initialize the UserVideoTrack.

        :param env: The gym environment to use for the video track.
        :param session_id: The session id to use for the video track.
        """
        super().__init__()
        self.env = env
        self.session_id = session_id

    async def recv(self):
        """
        Return the next frame from the environment.

        This method is called by aiortc to get the next frame from the environment.

        It first checks if the frame counter is 0, if so, it resets the environment using
        env.reset() and stores the observation in self.last_obs.

        Then it renders the environment using env.render() and asks for a new action every
        self.skip frames using self.get_remote_action().

        If an error occurs while asking for a new action, it samples a random action using
        env.action_space.sample().

        Finally, it executes env.step(action) in a thread and increments the frame counter.

        :return: A VideoFrame object containing the rendered frame, the pts and
        time_base are set to the current timestamp and the time base of the stream track.
        """
        pts, time_base = await self.next_timestamp()

        # Obtain the most recent valid action
        action = latest_actions.get(self.session_id)
        if action is None:
            # If there is no valid action, sample a random action
            action = self.env.action_space.sample()

        # Execute env.step(action) in a thread
        await asyncio.get_event_loop().run_in_executor(executor, self.env.step, action)

        # Execute env.render() in a thread
        frame = await asyncio.get_event_loop().run_in_executor(executor, self.env.render)

        # Convert frame to VideoFrame
        video_frame = VideoFrame.from_ndarray(frame, format="rgb24")
        video_frame.pts = pts
        video_frame.time_base = time_base
        return video_frame

class RLVideoTrack(VideoStreamTrack):
    """
    A VideoStreamTrack that interacts with a reinforcement learning environment
    using an AI server.
    """
    def __init__(self, env, model_url, skip=2):
        """
        Initialize the RLVideoTrack.

        :param env: The gym environment to use for the RL model.
        :param model_url: The URL of the AI server.
        :param skip: The number of frames to skip action request.
        """
        super().__init__()
        self.env = env
        self.model_url = model_url

        # Last observation
        self.last_obs = None
        # Save the last action received from the model, to not have to ask every frame
        self.last_action = None
        # Counter for frames, used to skip frames
        self.frame_counter = 0
        # Number of frames to skip action request
        self.skip = skip

        # Create aiohttp ClientSession, with limit=0 to allow keep-alive
        connector = aiohttp.TCPConnector(limit=0, keepalive_timeout=75)
        self.session = aiohttp.ClientSession(connector=connector)

    async def get_remote_action(self, frame):
        """
        Asks the AI server for the next action given the current frame.

        The frame is normalized to [0,1], reshaped to [C, H, W] and serialized to bytes.
        The shape of the tensor is sent in the X-SHAPE header.

        The action is returned as a JSON response from the server.
        """
        # Normalize frame
        tensor = frame.astype('float32') / 255.0
        # Change shape to [C, H, W]
        tensor = tensor.transpose(2, 0, 1)
        # Add batch dimension
        tensor = tensor[None, ...]

        # Serialize tensor to bytes
        payload = tensor.tobytes()
        shape_info = json.dumps(tensor.shape)

        # Send the request to the inference server
        async with self.session.post(
            self.model_url,
            data=payload,
            headers={
                'Content-Type': 'application/octet-stream',
                'X-SHAPE': shape_info
            }
        ) as resp:
            result = await resp.json()
            # Return the action from the response
            return result['action']

    async def recv(self):
        """
        Return the next frame from the environment.

        This method is called by aiortc to get the next frame from the environment.

        It first checks if the frame counter is 0, if so, it resets the environment using
        env.reset() and stores the observation in self.last_obs.

        Then it renders the environment using env.render() and asks for a new action every
        self.skip frames using self.get_remote_action().

        If an error occurs while asking for a new action, it samples a random action using
        env.action_space.sample().

        Finally, it executes env.step(action) in a thread and increments the frame counter.

        :return: A VideoFrame object containing the rendered frame, the pts and
        time_base are set to the current timestamp and the time base of the stream track.
        """

        pts, time_base = await self.next_timestamp()

        # If first frame, reset the environment
        if self.last_obs is None:
            obs, _ = await asyncio.get_event_loop().run_in_executor(executor, self.env.reset)
            self.last_obs = obs

        # Execute env.render() in a thread
        frame = await asyncio.get_event_loop().run_in_executor(executor, self.env.render)

        # Only ask for a new action every `self.skip` frames
        if self.frame_counter % self.skip == 0:
            try:
                action = await self.get_remote_action(frame)
            except Exception as e:
                logger.error("Error on get_remote_action: %s", e)
                # In case of error, sample a random action
                action = self.env.action_space.sample()
            self.last_action = action
        else:
            # Retrieve the last action
            action = self.last_action

        # Execute env.step(action) in a thread
        await asyncio.get_event_loop().run_in_executor(executor, self.env.step, action)

        # Increment frame counter
        self.frame_counter += 1

        # Convert frame to VideoFrame
        video_frame = VideoFrame.from_ndarray(frame, format="rgb24")
        video_frame.pts = pts
        video_frame.time_base = time_base
        return video_frame

async def offer(request):
    """
    Handles the offer from the client, creates a PeerConnection, 2 environments
    (one for the user and one for the RL model), and 2 video tracks (one for the
    user and one for the RL model).

    :param request: The request containing the offer
    :return: The answer SDP
    """
    params = await request.json()
    session_id = params.get("session_id")
    if not session_id:
        return web.json_response({"error": "Missing session_id"}, status=400)

    _offer = RTCSessionDescription(sdp=params['sdp'], type=params['type'])
    pc = RTCPeerConnection()
    pcs.add(pc)
    logger.info("Created PeerConnection %s", pc)

    await pc.setRemoteDescription(_offer)

    # Create 2 environments: one for the user and one for the RL model
    env_user = gym.make("CarRacing-v3", render_mode="rgb_array", continuous=False)
    env_user.reset()
    env_rl = gym.make("CarRacing-v3", render_mode="rgb_array", continuous=False)
    env_rl.reset()

    # Create the video track for the user
    video_track_user = UserVideoTrack(env_user, session_id)
    pc.addTrack(video_track_user)

    # Create the video track for the RL model
    video_track_rl = RLVideoTrack(env_rl, MODEL_URL, skip=2)
    pc.addTrack(video_track_rl)

    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return web.json_response({
        'sdp': pc.localDescription.sdp,
        'type': pc.localDescription.type
    })

async def on_shutdown(_):
    """
    Shut down all peer connections and clear the set of all connections.

    This function is meant to be passed to the `on_shutdown` event of an
    aiohttp web application.
    """
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()

# if __name__ == '__main__':
#     app = web.Application()
#     app.on_shutdown.append(on_shutdown)
#     app.router.add_post('/offer', offer)
#     app.router.add_get('/ws', websocket_handler)
#     web.run_app(app, port=8080)

app = web.Application()
app.on_shutdown.append(on_shutdown)
app.router.add_post('/offer', offer)
app.router.add_get('/ws', websocket_handler)