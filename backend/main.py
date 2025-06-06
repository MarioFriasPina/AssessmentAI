""" FastAPI application with WebRTC, Gym environments, and user authentication."""

from typing import Any, Dict
from concurrent.futures import ThreadPoolExecutor
import asyncio
from datetime import datetime, timedelta, timezone
import base64
import os
import uuid

import numpy as np
import cv2
from contextlib import asynccontextmanager
import logging
import json
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
import gymnasium as gym
from fastapi import Depends, FastAPI, WebSocket, WebSocketDisconnect, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import requests
from sqlalchemy import create_engine, MetaData, Column, Integer, String, Select, DateTime, asc, desc, update
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from pydantic import BaseModel
import bcrypt
import jwt
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
import uvloop


# region Conf

DATABASE_URL = "postgresql+asyncpg://clouduser:Mnb%40sdpoi87@172.24.0.22:5432/igdrasil"  
BACKEND_URL = "https://172.24.0.9"

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

    last_game = Column(DateTime(timezone=True),default=lambda: datetime.now(timezone.utc),nullable=False)

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

class UserProfile(BaseModel):
    name: str
    name_last: str
    email: str

# end region


# region Fastapi

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize the database on startup
    await init_db()
    yield

app = FastAPI(lifespan=lifespan)

# Set up logging from gunicorn access
logger = logging.getLogger("gunicorn.access")

# Use uvloop for performance
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://10.49.12.47:9999"],
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

@app.get("/users/me", response_model=UserProfile)
async def read_users_me(
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> UserProfile:
    """
    Returns minimum profile for the current user (name, name_last, email).
    """
    user_email = current_user.get("email")
    if not user_email:
        raise HTTPException(status_code=401, detail="Invalid token or invalid email.")

    query = Select(User).where(User.email == user_email)
    result = await db.execute(query)
    row = result.first()
    if not row:
        raise HTTPException(status_code=404, detail="User not found.")

    user_obj: User = row[0]
    return UserProfile(
        name=user_obj.name, # type: ignore
        name_last=user_obj.name_last, # type: ignore
        email=user_obj.email, # type: ignore
    )

# end region
# region Video endpoints

# ThreadPool executor for offloading gym.step(), env.reset(), env.render(), and cv2.imencode
executor = ThreadPoolExecutor()

# session_data holds per-session environments, queues, and last‐seen action
session_data: Dict[str, Dict[str, Any]] = {}
session_data_lock = asyncio.Lock()

# Convenience alias to the queue within session_data; kept in sync under session_data_lock
session_queues: Dict[str, asyncio.Queue] = {}

# Model inference server URL (not used in this snippet, but kept for reference)
MODEL_URL = "https://192.24.0.9:443/predict"


async def cleanup(session_id: str):
    """
    Tear down everything associated with a given session_id:
      - Close gym environments
      - Remove session_data and session_queues entries
    """
    async with session_data_lock:
        info = session_data.pop(session_id, None)
        session_queues.pop(session_id, None)

    if not info:
        return
    
    db = await get_db()
    
    env_user = info["env_user"]
    env_rl = info["env_rl"]

    user_email = info.get("user_email")

    if not user_email:
        logger.warning(f"[cleanup] session {session_id} has no user_email, skipping DB update")
    else:
        # Add current and new stats
        query = Select(User).where(User.email == user_email)
        result = await db.execute(query)
        user = result.first()[0]
        curr_runs = user.runs + info["runs"]
        curr_wins = user.wins + info["wins"]
        curr_record = user.record

        # Update user stats
        update_query = update(User).where(User.email == user_email).values(
            wins = curr_wins,
            runs = curr_runs
        )
        async with db.begin():
            await db.execute(update_query)

            # Only update the record if the current run is a new record
            if curr_record < info["best_reward"]:
                update_query = update(User).where(User.email == user_email).values(
                    record = info["best_reward"],
                    last_game = info["last_game"]
                )
                await db.execute(update_query)

    # Close gym environments (synchronously is okay here, but could be offloaded if desired)
    await asyncio.gather(
        asyncio.get_running_loop().run_in_executor(executor, env_user.close),
        asyncio.get_running_loop().run_in_executor(executor, env_rl.close),
    )

    logger.info(f"[cleanup] Cleaned up session {session_id}")


@app.post("/offer")
async def offer(current_user: dict = Depends(get_current_user)):
    """
    Create a new session for video streaming and user actions.
    Returns a session_id for the client to use in WebSocket connections.
    """
    # Generate a unique session_id (UUID4)
    session_id = str(uuid.uuid4())
    logger.info(f"[offer] Creating new session: {session_id}")

    # Create Gym environments
    env_user = gym.make("CarRacing-v3", render_mode="rgb_array", continuous=True)
    #await asyncio.get_running_loop().run_in_executor(executor, env_user.reset)

    env_rl = gym.make("CarRacing-v3", render_mode="rgb_array", continuous=True)
    #await asyncio.get_running_loop().run_in_executor(executor, env_rl.reset)

    # Create a per-session queue for user actions
    action_queue: asyncio.Queue = asyncio.Queue()

    # Store everything under session_data
    async with session_data_lock:
        session_data[session_id] = {
            "env_user": env_user,
            "env_rl": env_rl,
            "queue": action_queue,
            "last_user_action": [0.0, 0.0, 0.0], # The action last seen by the user
            "reward_user": 0.0,
            "reward_rl": 0.0,
            "best_reward": 0.0,
            "runs" : 0,
            "wins" : 0,
            "last_game" : datetime.now(),
            "user_email" : current_user.get("email"),

            # Coordinator flags
            "needs_reset_user" : False,
            "needs_reset_rl" : False,
            "live_video_user" : True,
            "live_video_rl" : True,
        }
        session_queues[session_id] = action_queue

    return {"session_id": session_id}

async def reset_both_envs(session_id: str):
    # 1) Grab both env_user and env_rl under the lock
    async with session_data_lock:
        info = session_data.get(session_id)
        if info is None:
            # The session was cleaned up or never existed
            raise RuntimeError(f"Session {session_id} not found in reset_both_envs()")

        env_user = info["env_user"]
        env_rl   = info["env_rl"]

    # 2) Call env_user.reset() off the thread pool, then unpack safely
    result_user = await asyncio.get_running_loop().run_in_executor(executor, env_user.reset)
    if isinstance(result_user, tuple) and len(result_user) >= 1:
        # Gymnasium style: (obs_array, info_dict)
        obs_us = result_user[0]
    else:
        # Classic Gym style: just obs_array
        obs_us = result_user

    # 3) Do the same for env_rl
    result_rl = await asyncio.get_running_loop().run_in_executor(executor, env_rl.reset)
    if isinstance(result_rl, tuple) and len(result_rl) >= 1:
        obs_rl = result_rl[0]
    else:
        obs_rl = result_rl

    # 4) Now update the shared stats under the lock
    async with session_data_lock:
        # Double‐check that session still exists
        info = session_data.get(session_id)
        if info is None:
            # If it vanished while we were in the threadpool, bail out
            raise RuntimeError(f"Session {session_id} disappeared mid‐reset.")

        info["runs"] += 1
        info["best_reward"] = max(info["reward_user"], info["best_reward"])

        if info["reward_user"] > info["reward_rl"]:
            info["wins"] += 1
            info["last_game"] = datetime.now()

        # Zero out for the next run
        info["reward_user"] = 0.0
        info["reward_rl"]  = 0.0

    return obs_us, obs_rl

@app.websocket("/video_user")
async def video_user_stream(websocket: WebSocket):
    await websocket.accept()
    session_id = websocket.query_params.get("session_id")
    if not session_id:
        await websocket.close(code=4001)
        return

    # Atomically grab session info and queue
    async with session_data_lock:
        info = session_data.get(session_id)
        queue = session_queues.get(session_id)

    if not info or not queue:
        await websocket.close(code=4002)
        return

    env = info["env_user"]
    last_action = info["last_user_action"]

    # Initialize environment (already reset in /offer, but safe to reset again)
    obs = await asyncio.get_running_loop().run_in_executor(executor, env.reset)
    info["runs"] += 1

    try:
        while True:
            # Drain all pending actions to get the most recent one
            while True:
                try:
                    last_action = queue.get_nowait()
                    # Update stored last action
                    info["last_user_action"] = last_action
                except asyncio.QueueEmpty:
                    break

            # Step environment with the latest action (offloaded)
            obs, reward, term, trunc, info_step = await asyncio.get_running_loop().run_in_executor(
                executor, env.step, np.array(last_action, dtype=np.float32)
            )

            # Update reward
            async with session_data_lock:
                session_data[session_id]["reward_user"] += reward

            if term or trunc:
                async with session_data_lock:
                    session_data[session_id]["needs_reset_user"] = True
            
            # Reset environment if needed
            async with session_data_lock:
                # If either player or AI says “end‐of‐episode,” do a single combined reset
                if session_data[session_id]["needs_reset_user"] or session_data[session_id]["needs_reset_rl"]:
                    try:
                        obs_us, obs_rl = await reset_both_envs(session_id)
                    except RuntimeError:
                        # session vanished—just break
                        break
                    # Clear both flags so the other loop knows “we already reset”
                    session_data[session_id]["needs_reset_user"] = False
                    session_data[session_id]["needs_reset_rl"]   = False

            # Render (offloaded)
            frame = await asyncio.get_running_loop().run_in_executor(executor, env.render)

            # Encode frame as JPEG (offloaded)
            _, buffer = await asyncio.get_running_loop().run_in_executor(
                executor, cv2.imencode, ".jpg", frame
            )

            # Send JPEG bytes
            await websocket.send_bytes(buffer.tobytes())

            # Throttle to ~30 FPS
            await asyncio.sleep(1 / 60)

    except WebSocketDisconnect:
        logger.info(f"[video_user] WebSocket disconnected for session {session_id}")
        async with session_data_lock:
            info = session_data.get(session_id)
            if info:
                info["live_video_user"] = False
        # Do NOT cleanup() here.

    except Exception as e:
        logger.error(f"[video_user] Exception for session {session_id}: {e}")

    finally:
        to_cleanup = False
        async with session_data_lock:
            info = session_data.get(session_id)
            if info:
                info["live_video_user"] = False
                if not info.get("live_video_rl", False):
                    to_cleanup = True

        if to_cleanup:
            await cleanup(session_id)

@app.websocket("/video_rl")
async def video_rl_stream(websocket: WebSocket):
    await websocket.accept()
    session_id = websocket.query_params.get("session_id")
    if not session_id:
        await websocket.close(code=4001)
        return

    # Atomically grab session info and queue (though queue not used here)
    async with session_data_lock:
        info = session_data.get(session_id)

    if not info:
        await websocket.close(code=4002)
        return

    env = info["env_rl"]
    obs, _ = await asyncio.get_running_loop().run_in_executor(executor, env.reset)

    try:
        while True:
            # call compute
            res = requests.post(f'{BACKEND_URL}/predict',
                                verify=False, json={'obs': obs.tolist()})

            if res.status_code != 200:
                raise HTTPException(500, "AI is not available")

            data = res.json()['action']

            aiAction = np.array(data, dtype=np.float32)
            # Here you could call an external model; currently, we sample randomly

            # Step environment (offloaded)
            obs, reward, term, trunc, info_step = await asyncio.get_running_loop().run_in_executor(
                executor, env.step, aiAction
            )
            async with session_data_lock:
                session_data[session_id]["reward_rl"] += reward

            if term or trunc:
                async with session_data_lock:
                    session_data[session_id]["needs_reset_rl"] = True
            
            # Reset environment if needed
            async with session_data_lock:
                if session_data[session_id]["needs_reset_user"] or session_data[session_id]["needs_reset_rl"]:
                    try:
                        obs_us, obs = await reset_both_envs(session_id)
                    except RuntimeError:
                        # session was already deleted
                        break
                    session_data[session_id]["needs_reset_user"] = False
                    session_data[session_id]["needs_reset_rl"]   = False

            # Render (offloaded)
            frame = await asyncio.get_running_loop().run_in_executor(executor, env.render)

            # Encode frame as JPEG (offloaded)
            _, buffer = await asyncio.get_running_loop().run_in_executor(
                executor, cv2.imencode, ".jpg", frame
            )

            # Send JPEG bytes
            await websocket.send_bytes(buffer.tobytes())

            # Throttle to ~60 FPS
            await asyncio.sleep(1 / 60)

    except WebSocketDisconnect:
        logger.info(f"[video_rl] WebSocket disconnected for session {session_id}")
        # Mark RL side gone, but only if the session still exists
        async with session_data_lock:
            info = session_data.get(session_id)
            if info:
                info["live_video_rl"] = False
        # Do NOT call cleanup() here—defer to 'finally' so we only clean up once.

    except Exception as e:
        logger.error(f"[video_rl] Exception for session {session_id}: {e}")

    finally:
        # In 'finally', we decide whether BOTH sides are gone, and then cleanup exactly once.
        to_cleanup = False

        async with session_data_lock:
            info = session_data.get(session_id)
            if info:
                # First, mark RL side gone in case this was an Exception (not WebSocketDisconnect)
                info["live_video_rl"] = False

                # If the user side is already gone, *then* we know it's safe to clean up.
                if not info.get("live_video_user", False):
                    # Flag to cleanup—pop will happen inside cleanup()
                    to_cleanup = True
            # If info is None, the session was already popped by the `/video_user` side,
            # so we do nothing here.

        if to_cleanup:
            await cleanup(session_id)

@app.get("/debug")
async def uwu():
    env = gym.make("CarRacing-v3", render_mode="rgb_array", continuous=True)
    await asyncio.get_running_loop().run_in_executor(executor, env.reset)
    obs, _ = await asyncio.get_running_loop().run_in_executor(executor, env.reset)

    try:

        while True:
            # call compute
            res = requests.post(f'{BACKEND_URL}/predict', verify=False, json={'obs': obs[0].tolist()})

            if res.status_code != 200:
                raise HTTPException(500, "AI is not available")

            data = res.json()['action']

            aiAction = np.array(data, dtype=np.float32)
            # Here you could call an external model; currently, we sample randomly

            # Step environment (offloaded)
            obs, reward, term, trunc, info_step = await asyncio.get_running_loop().run_in_executor(
                executor, env.step, aiAction
            )
            if term or trunc:
                obs, _ = await asyncio.get_running_loop().run_in_executor(executor, env.reset)

            # Render (offloaded)
            frame = await asyncio.get_running_loop().run_in_executor(executor, env.render)

            # Encode frame as JPEG (offloaded)
            _, buffer = await asyncio.get_running_loop().run_in_executor(
                executor, cv2.imencode, ".jpg", frame
            )


            # Throttle to ~60 FPS
            await asyncio.sleep(1 / 60)

    except WebSocketDisconnect:
        logger.info(
            f"[video_rl] WebSocket disconnected")
    except Exception as e:
        logger.error(f"[video_rl] Exception for session {e}")
    finally:
        # Clean up session (idempotent if another handler already did it)
        pass

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

    # Atomically grab session info and queue
    async with session_data_lock:
        info = session_data.get(session_id)
        queue = session_queues.get(session_id)

    if not info or not queue:
        await websocket.close(code=4002)
        return

    logger.info(f"[ws] WebSocket connected for session {session_id}")

    try:
        while True:
            data = await websocket.receive_text()
            try:
                payload = json.loads(data)
                action = payload.get("action")
                if action is not None:
                    # Enqueue the action (and store it in session_data for fallback)
                    await queue.put(action)
                    info["last_user_action"] = action
            except json.JSONDecodeError:
                continue

    except WebSocketDisconnect:
        logger.info(f"[ws] WebSocket disconnected for session {session_id}")
    except Exception as e:
        logger.error(f"[ws] Exception for session {session_id}: {e}")
    finally:
        # Clean up this session (idempotent if others already triggered it)
        await cleanup(session_id)

@app.post("/shutdown_all")
async def shutdown_all():
    """
    Force-close all active peer connections and clean up all sessions.
    """
    # Copy session IDs under lock, then clean each
    async with session_data_lock:
        sessions = list(session_data.keys())

    for sid in sessions:
        await cleanup(sid)

    return {"status": "all sessions cleaned up"}


@app.get("/")
async def root():
    """
    Default health check endpoint.
    """
    return {"status": "ok"}

# end region
