
import asyncio
import json
import os
import secrets
import sys
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from agent_io import AWAITING_INPUT_SENTINEL

BASE_DIR = Path(__file__).resolve().parent
FRONTEND_PATH = BASE_DIR / "web_frontend.html"
INTAKE_PATH = BASE_DIR / "intake.py"


LOG_DIR = Path(os.environ.get("EXP3_LOG_DIR", str(BASE_DIR / "exp3_logs")))
LOG_DIR.mkdir(parents=True, exist_ok=True)

ACCESS_CODE = os.environ.get("ACCESS_CODE", "").strip()
IDLE_TIMEOUT_SECONDS = int(os.environ.get("IDLE_TIMEOUT_SECONDS", "1800"))   # 30 min
DONE_GRACE_SECONDS = int(os.environ.get("DONE_GRACE_SECONDS", "300"))        # 5 min
MAX_SESSIONS = int(os.environ.get("MAX_SESSIONS", "10"))
MAX_MESSAGE_CHARS = int(os.environ.get("MAX_MESSAGE_CHARS", "4000"))
REAP_INTERVAL_SECONDS = 60
ALLOWED_ORIGINS = [o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "").split(",")
                   if o.strip()]


def _file_stem(session_id: str, participant_id: Optional[str],
               vignette_id: Optional[str]) -> str:

    def clean(s: str) -> str:
        return "".join(c if c.isalnum() or c in "-_" else "_" for c in s)[:64]
    if participant_id and vignette_id:
        return f"{clean(participant_id)}__{clean(vignette_id)}__{session_id[:8]}"
    return session_id


class Session:

    def __init__(self, session_id: str, process: asyncio.subprocess.Process,
                 participant_id: Optional[str] = None,
                 vignette_id: Optional[str] = None):
        self.session_id = session_id
        self.process = process
        self.buffer: List[str] = []
        self.awaiting_input = False
        self.done = False
        self.lock = asyncio.Lock()
        self.participant_id = participant_id
        self.vignette_id = vignette_id
        self.file_stem = _file_stem(session_id, participant_id, vignette_id)
        self.started_at = time.time()
        self.last_active = self.started_at
        self.events: List[dict] = []  # [{"role": "agent"/"user", "content": str, "t": float}]

    async def read_until_input_or_exit(self) -> None:
        if self.awaiting_input or self.done:
            return
        while True:
            line_bytes = await self.process.stdout.readline()
            if not line_bytes:  # EOF -- process exited
                self.done = True
                return
            line = line_bytes.decode(errors="replace").rstrip("\n")
            if line.startswith(AWAITING_INPUT_SENTINEL):
                self.awaiting_input = True
                return
            self.buffer.append(line)

    def pop_message(self) -> str:
        msg = "\n".join(self.buffer)
        self.buffer = []
        return msg

    def log_event(self, role: str, content: str) -> None:
        now = time.time()
        self.last_active = now
        self.events.append({"role": role, "content": content, "t": now})

    async def terminate(self) -> None:
        if self.process.returncode is None:
            try:
                self.process.kill()
            except ProcessLookupError:
                pass
            try:
                await asyncio.wait_for(self.process.wait(), timeout=5)
            except asyncio.TimeoutError:
                pass

    def write_log(self) -> None:
        final_state = None
        final_state_path = LOG_DIR / f"{self.file_stem}_final_state.json"
        if final_state_path.exists():
            try:
                final_state = json.loads(final_state_path.read_text())
            except (json.JSONDecodeError, OSError):
                final_state = {"error": "final_state file existed but couldn't be read"}

        log = {
            "session_id": self.session_id,
            "participant_id": self.participant_id,
            "vignette_id": self.vignette_id,
            "started_at": self.started_at,
            "done": self.done,
            "duration_seconds": (self.events[-1]["t"] - self.started_at
                                 if self.events else 0),
            "events": self.events,
            "final_state": final_state,
        }
        (LOG_DIR / f"{self.file_stem}.json").write_text(
            json.dumps(log, indent=2, default=str))


SESSIONS: Dict[str, Session] = {}


async def _reap_idle_sessions() -> None:

    while True:
        await asyncio.sleep(REAP_INTERVAL_SECONDS)
        now = time.time()
        for sid, s in list(SESSIONS.items()):
            if s.lock.locked():
                continue
            idle = now - s.last_active
            if (s.done and idle > DONE_GRACE_SECONDS) or idle > IDLE_TIMEOUT_SECONDS:
                await s.terminate()
                try:
                    s.write_log()
                except OSError:
                    pass
                SESSIONS.pop(sid, None)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    reaper = asyncio.create_task(_reap_idle_sessions())
    try:
        yield
    finally:
        reaper.cancel()
        for s in list(SESSIONS.values()):
            await s.terminate()
            try:
                s.write_log()
            except OSError:
                pass


app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)

if ALLOWED_ORIGINS:
    app.add_middleware(
        CORSMiddleware, allow_origins=ALLOWED_ORIGINS,
        allow_methods=["GET", "POST"], allow_headers=["Content-Type", "X-Access-Code"],
    )


def require_access(x_access_code: Optional[str] = Header(default=None)) -> None:
    if ACCESS_CODE and not secrets.compare_digest(x_access_code or "", ACCESS_CODE):
        raise HTTPException(401, "Access code required.")


async def _spawn_intake_subprocess(session_id: str, participant_id: Optional[str],
                                   vignette_id: Optional[str]) -> asyncio.subprocess.Process:
    env = dict(os.environ)
    env["AGENT_WEB_MODE"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    stem = _file_stem(session_id, participant_id, vignette_id)
    env["EXP3_FINAL_STATE_PATH"] = str(LOG_DIR / f"{stem}_final_state.json")
    return await asyncio.create_subprocess_exec(
        sys.executable, "-u", str(INTAKE_PATH),
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        env=env,
        cwd=str(BASE_DIR),
    )


class StartRequest(BaseModel):
    participant_id: Optional[str] = None
    vignette_id: Optional[str] = None


class StartResponse(BaseModel):
    session_id: str
    agent_message: str
    done: bool


class MessageRequest(BaseModel):
    session_id: str
    message: str


class MessageResponse(BaseModel):
    agent_message: str
    done: bool


@app.get("/", include_in_schema=False)
async def frontend() -> FileResponse:
    return FileResponse(FRONTEND_PATH, headers={"Cache-Control": "no-store"})


@app.post("/session", response_model=StartResponse, dependencies=[Depends(require_access)])
async def start_session(req: StartRequest = StartRequest()) -> StartResponse:
    running = sum(1 for s in SESSIONS.values() if not s.done)
    if running >= MAX_SESSIONS:
        raise HTTPException(
            503, "The server is at its limit of simultaneous conversations. "
                 "Please try again in a few minutes.")

    session_id = str(uuid.uuid4())
    process = await _spawn_intake_subprocess(session_id, req.participant_id, req.vignette_id)
    session = Session(session_id, process,
                      participant_id=req.participant_id,
                      vignette_id=req.vignette_id)
    SESSIONS[session_id] = session

    async with session.lock:
        await session.read_until_input_or_exit()
        msg = session.pop_message()
        session.log_event("agent", msg)
        session.write_log()
    return StartResponse(session_id=session_id, agent_message=msg, done=session.done)


@app.post("/message", response_model=MessageResponse, dependencies=[Depends(require_access)])
async def send_message(req: MessageRequest) -> MessageResponse:

    session = SESSIONS.get(req.session_id)
    if session is None:
        raise HTTPException(404, "This conversation has expired -- click restart to begin a new one.")
    if session.done:
        raise HTTPException(400, "This session already finished.")
    if len(req.message) > MAX_MESSAGE_CHARS:
        raise HTTPException(413, f"Message too long (limit {MAX_MESSAGE_CHARS} characters).")

    message = " ".join(req.message.splitlines())

    async with session.lock:
        if not session.awaiting_input:
            raise HTTPException(
                409, "Agent isn't waiting for input right now -- wait for "
                     "the previous response before sending another message.")

        session.log_event("user", message)
        session.process.stdin.write((message + "\n").encode())
        await session.process.stdin.drain()
        session.awaiting_input = False

        await session.read_until_input_or_exit()
        msg = session.pop_message()
        session.log_event("agent", msg)
        session.write_log()
        return MessageResponse(agent_message=msg, done=session.done)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "active_sessions": sum(1 for s in SESSIONS.values() if not s.done)}
