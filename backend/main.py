""" FastAPI application with WebRTC, Gym environments, and user authentication."""

from typing import Any, Dict
from concurrent.futures import ThreadPoolExecutor
import asyncio
from datetime import datetime, timedelta, timezone
import base64
import os
import uuid

import cv2
from contextlib import asynccontextmanager
import logging
import json
import aiohttp
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
import gymnasium as gym
from pylibsrtp import Session
from fastapi import Depends, FastAPI, WebSocket, WebSocketDisconnect, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from av import VideoFrame
from sqlalchemy import create_engine, MetaData, Column, Integer, String, Select, DateTime, asc, desc
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from pydantic import BaseModel
import bcrypt
import jwt
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
import uvloop


# region Conf

DATABASE_URL = "sqlite+aiosqlite:///./test.db"  # Change this to your database URL

# Create the SQLAlchemy engine
engine_base = create_async_engine(DATABASE_URL, echo=True)
engine = sessionmaker(
    bind=engine_base,  # type: ignore
    class_=AsyncSession,
    expire_on_commit=False,
)  # type: ignore

# Create a configured "Session" class
# SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create a base class for declarative models
Base = declarative_base()

# Create a metadata instance
metadata = MetaData()

# end region

# region Crypt

SECRET_KEY = os.getenv('SECRET', 'verysecure')
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


def create_access_token(data: dict, expires_delta: timedelta | None = None) -> str:
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + \
            timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
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


async def get_db() -> AsyncSession:  # type: ignore
    async with engine() as session:  # type: ignore
        yield session  # type: ignore


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

    last_game = Column(DateTime, default=datetime.now(timezone.utc))

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


class LeaderboardEntry(BaseModel):
    name: str
    score: int
    date: datetime

# end region


# region Fastapi

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize the database on startup
    await init_db()
    yield
    # On shutdown, close all PeerConnections
    logger.info("Shutting down PeerConnections")
    await asyncio.gather(*[pc.close() for pc in pcs])
    pcs.clear()

app = FastAPI(lifespan=lifespan)

# Set up logging from gunicorn access
logger = logging.getLogger("gunicorn.access")

# Use uvloop for performance
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

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
async def login(form_data: OAuth2PasswordRequestForm = Depends(), db: AsyncSession = Depends(get_db)) -> dict[str, str | int]:
    '''
    login login and get a token using the standard OAuth2PasswordBearer

    Args:
        form_data (OAuth2PasswordRequestForm, optional): The client details. Requires username and password. Defaults to Depends().
        db (AsyncSession, optional): The Db instance (auto). Defaults to Depends(get_db).

    Raises:
        HTTPException: 403 if credentials are invalid

    Returns:
        dict[str, str | int]: The json standard bearer type token
    '''
    query = Select(User).where(User.email == form_data.username)
    result = await db.execute(query)

    res = result.first()

    if res is None:
        raise HTTPException(403, 'Invalid username or password')

    user: User = res[0]

    if verify_password(form_data.password, str(user.password)):

        return {
            "access_token": create_access_token(
                {
                    'user': user.name,
                    'email': user.email,
                }
            ),
            "token_type": "bearer",
            "expires_in": ACCESS_TOKEN_EXPIRE_MINUTES
        }

    raise HTTPException(403, 'Invalid username or password')


@app.post('/create')
async def create_user(data: NewUser, db: AsyncSession = Depends(get_db)) -> str:
    '''
    create_user Create and register a new user. Use email for authentication (not username)

    Args:
        data (NewUser): The new user data.
        db (AsyncSession, optional): The Db session. Defaults to Depends(get_db).

    Returns:
        str: A success message.
    '''
    newUser = User(
        name=data.name,
        name_last=data.name_last,
        email=data.email,
        password=hash_password(data.password),
    )
    db.add(newUser)
    await db.commit()

    return "success"

# region userdata

@app.get('/data')
async def get_data(db: AsyncSession = Depends(get_db), current_user: dict = Depends(get_current_user)):
    '''
    get_data Get a user's data to display in the front-end

    Args:
        db (AsyncSession, optional): The Db session. Defaults to Depends(get_db).
        current_user (dict, optional): The required Bearer token. Defaults to Depends(get_current_user).

    Raises:
        HTTPException: 403 Unauthorized user

    Returns:
        UserData: The stats for the given user.
    '''
    query = Select(User).where(User.email == current_user['email'])
    result = await db.execute(query)

    res = result.first()

    if res is None:
        raise HTTPException(403, 'Invalid username or password')

    user: User = res[0]

    return UserData(
        win=user.wins, # type: ignore
        loss=user.runs - user.wins, # type: ignore
        total=user.runs, # type: ignore
        record=user.record, # type: ignore
    )


@app.get('/leaderboard')
async def get_leaderboard(db: AsyncSession = Depends(get_db), current_user: dict = Depends(get_current_user)) -> list[LeaderboardEntry]:
    '''
    get_leaderboard Return the leaderboard for the top 20.

    Args:
        db (AsyncSession, optional): The db session (Fastapi). Defaults to Depends(get_db).
        current_user (dict, optional): The user token. Defaults to Depends(get_current_user).

    Raises:
        HTTPException: 403 Unauthorized

    Returns:
        list[LeaderboardEntry]: A list of the leaderboard entries in order.
    '''
    query = Select(User).order_by(User.record.desc()).limit(20)
    result = await db.execute(query)

    res = result.all()

    if res is None:
        raise HTTPException(403, 'Invalid username or password')

    return [LeaderboardEntry(name=i[0].name, score=i[0].record, date=i[0].last_game) for i in res]

# end region
# region Video endpoints

# ThreadPool executor for offloading gym.step() and gym.render()
executor = ThreadPoolExecutor()

session_data: Dict[str, Dict[str, Any]] = {}
session_data_lock = asyncio.Lock()

# Convenience: alias for the queue storage (mirrors session_data's "queue" field)
# but we keep both in sync via session_data_lock.
session_queues: Dict[str, asyncio.Queue] = {}

# Model inference server URL
MODEL_URL = "https://192.24.0.9:443/predict"


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
    
    env_user = info["env_user"]
    env_rl = info["env_rl"]

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

@app.post("/offer")
async def offer(request: Request):
    """
    Create a new session for video streaming and user actions.
    Returns a session_id for the client to use in WebSocket connections.
    """
    # Generate a unique session_id (UUID4)
    session_id = str(uuid.uuid4())
    logger.info(f"Creating new session: {session_id}")

    # Create Gym environments
    env_user = gym.make("CarRacing-v3", render_mode="rgb_array", continuous=True)
    env_user.reset()
    env_rl = gym.make("CarRacing-v3", render_mode="rgb_array", continuous=True)
    env_rl.reset()

    # Create a per-session queue for user actions
    action_queue: asyncio.Queue = asyncio.Queue()

    # Store everything under session_data
    async with session_data_lock:
        session_data[session_id] = {
            "env_user": env_user,
            "env_rl": env_rl,
            "queue": action_queue,
            "user_action": [0.0, 0.0, 0.0],  # default action
        }
        session_queues[session_id] = action_queue

    return {
        "session_id": session_id
    }

@app.websocket("/video_user")
async def video_user_stream(websocket: WebSocket):
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
    env = session_data[session_id]["env_user"]
    obs = env.reset()
    action = [0.0, 0.0, 0.0]
    try:
        while True:
            # Step environment with latest action
            obs, reward, done, info = env.step(action)
            if done:
                obs = env.reset()
            frame = env.render()
            # Encode frame as JPEG
            _, buffer = cv2.imencode('.jpg', frame)
            await websocket.send_bytes(buffer.tobytes())
            # Optionally: await asyncio.sleep(1/30)  # 30 FPS
    except WebSocketDisconnect:
        env.close()

@app.websocket("/video_rl")
async def video_rl_stream(websocket: WebSocket):
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
    env = session_data[session_id]["env_rl"]
    obs = env.reset()
    try:
        while True:
            # RL logic here (call your model or random)
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            if done:
                obs = env.reset()
            frame = env.render()
            _, buffer = cv2.imencode('.jpg', frame)
            await websocket.send_bytes(buffer.tobytes())
    except WebSocketDisconnect:
        env.close()

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

# Add default path for health check


@app.get("/")
async def root():
    """
    Default path for health check.
    """
    return {"status": "ok"}

# end region
