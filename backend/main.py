from concurrent.futures import ThreadPoolExecutor
import asyncio
from contextlib import asynccontextmanager
import logging
import json
import aiohttp
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
import gymnasium as gym
from pylibsrtp import Session
import uvicorn
from fastapi import Depends, FastAPI, WebSocket, WebSocketDisconnect, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
from av import VideoFrame
from sqlalchemy import create_engine, MetaData, Column, Integer, String, Select
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from pydantic import BaseModel
import bcrypt
import jwt
import base64
from datetime import datetime, timedelta

import os
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession


# region Conf

DATABASE_URL = "sqlite+aiosqlite:///./test.db"  # Change this to your database URL

# Create the SQLAlchemy engine
engine_base = create_async_engine(DATABASE_URL, echo=True)
engine = sessionmaker(
    bind=engine_base,
    class_=AsyncSession,
    expire_on_commit=False,
)

# Create a configured "Session" class
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create a base class for declarative models
Base = declarative_base()

# Create a metadata instance
metadata = MetaData()

# end region

# region Crypt

SECRET_KEY = os.getenv('SECRET', '123')
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


def hash_password(password: str) -> str:
    # Hash the password using bcrypt
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password.encode(encoding='utf-8'), salt)
    return base64.b64encode(hashed).decode('utf-8')  # Encode as base64


def verify_password(plain_password: str, hashed_password: str) -> bool:
    # Decode the base64 encoded hashed password
    decoded_hashed = base64.b64decode(hashed_password.encode('utf-8'))
    return bcrypt.checkpw(plain_password.encode('utf-8'), decoded_hashed)


def create_access_token(data: dict, expires_delta: timedelta = None) -> str:
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now() + expires_delta
    else:
        expire = datetime.now() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def get_current_user(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        # You can add additional checks here, e.g., checking the user ID or roles
        return payload  # Return the payload if the token is valid
    except jwt.PyJWTError as e:

        raise HTTPException(
            status_code=401,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

# end region
# region Databases


async def get_db() -> AsyncSession:
    async with engine() as session:
        yield session


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    name_last = Column(String)
    email = Column(String, unique=True, index=True)
    password = Column(String, nullable=False)

    runs = Column(Integer, default=0)
    wins = Column(Integer, default=0)

    record = Column(Integer, default=0)

# end region

# region pydantic def


class NewUser(BaseModel):
    name: str
    email: str
    name_last: str
    password: str


class LoginData(BaseModel):
    username: str
    password: str


class UserData(BaseModel):
    win: int
    loss: int
    total: int
    record: float


# end region


# region Fastapi

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the ML model
    logger.info("Shutting down PeerConnections")
    await asyncio.gather(*[pc.close() for pc in pcs])
    await init_db()
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


async def init_db():
    async with engine_base.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


# end region
# region login


@app.post('/token')
async def login(form_data: OAuth2PasswordRequestForm = Depends(), db: AsyncSession = Depends(get_db)) -> str:
    query = Select(User).where(User.email == form_data.username)
    result = await db.execute(query)

    res = result.first()

    if res is None:
        raise HTTPException(403, 'Invalid username or password')

    user: User = res[0]

    if verify_password(form_data.password, user.password):

        return create_access_token(
            {
                'user': user.name,
                'email': user.email,
            }
        )

    raise HTTPException(403, 'Invalid username or password')


@app.post('/create')
async def create_user(data: NewUser, db: AsyncSession = Depends(get_db)) -> str:
    newUser = User(
        name=data.name,
        name_last=data.name_last,
        email=data.email,
        password=hash_password(data.password),
    )
    db.add(newUser)
    await db.commit()

    return "success"


@app.get('/data')
async def get_data(current_user: dict = Depends(get_current_user)):

    return UserData(
        win=0,
        loss=0,
        total=0,
        record=0,
    )


# end region
# region Video endpoints
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

    env_user = gym.make(
        "CarRacing-v3", render_mode="rgb_array", continuous=False)
    env_user.reset()
    env_rl = gym.make(
        "CarRacing-v3", render_mode="rgb_array", continuous=False)
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


# end region
