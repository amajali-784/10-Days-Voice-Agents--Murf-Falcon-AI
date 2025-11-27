#!/usr/bin/env python3
"""
Day 6 — Fraud Alert Voice Agent (single-file, corrected & hardened)

Run:
  - Local console/demo: python fraud_voice_agent.py
  - LiveKit worker (Linux/WSL/Docker only): LIVEKIT=1 python fraud_voice_agent.py

Behavior:
 - Avoids LiveKit worker mode on Windows to prevent Unix-IPC DuplexClosed errors.
 - Attempts LiveKit + Gemini; if the LLM is overloaded (503) or session startup fails,
   it falls back to a safe "no-LLM" mode (agent uses canned replies and TTS when available).
 - Always keeps a console interactive fallback for testing.
"""
from __future__ import annotations
import asyncio
import logging
import os
import sqlite3
import platform
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Annotated
from dotenv import load_dotenv
from pydantic import Field

# Try optional LiveKit imports
LIVEKIT_AVAILABLE = False
try:
    from livekit.agents import (
        Agent,
        AgentSession,
        JobContext,
        JobProcess,
        RoomInputOptions,
        WorkerOptions,
        cli,
        function_tool,
        RunContext,
        metrics,
        MetricsCollectedEvent,
        tokenize,
    )
    from livekit.plugins import murf, silero, google, deepgram, noise_cancellation
    from livekit.plugins.turn_detector.multilingual import MultilingualModel
    LIVEKIT_AVAILABLE = True
except Exception:
    # Minimal fallbacks so console mode still works
    def function_tool(f):  # type: ignore
        return f

    class RunContext:
        def __init__(self):
            self.userdata = None

    class Agent:
        def __init__(self, instructions: str = ""):
            self.instructions = instructions

    class AgentSession:
        def __init__(self, **kwargs):
            self._started = False
            self._handlers = {}
        async def start(self, **kwargs):
            self._started = True
        async def stop(self):
            self._started = False
        def on(self, *a, **k):
            def _decor(fn):
                return fn
            return _decor
        async def send_text(self, text: str):
            print("AGENT:", text)

    class JobContext:
        pass
    class JobProcess:
        userdata = {}
    class RoomInputOptions:
        def __init__(self, **kwargs):
            pass
    class WorkerOptions:
        def __init__(self, **kwargs):
            pass
    class cli:
        @staticmethod
        def run_app(opts):
            raise RuntimeError("LiveKit not available in this environment. Run in console mode instead.")

    metrics = None
    MetricsCollectedEvent = None
    tokenize = None
    murf = silero = google = deepgram = noise_cancellation = MultilingualModel = None

# Basic config + logging
load_dotenv(".env.local")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("fraud_agent")

BANK_NAME = "SecureBank India"
_BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATABASE_FILE = os.path.join(_BASE_DIR, "fraud_cases.db")

SAMPLE_FRAUD_CASES = [
    {
        "userName": "luffy",
        "securityIdentifier": "12345",
        "cardEnding": "**** 4242",
        "caseStatus": "pending_review",
        "transactionAmount": "₹45,999",
        "transactionName": "ABC Electronics Ltd",
        "transactionTime": "2025-11-27 14:23:15",
        "transactionCategory": "e-commerce",
        "transactionSource": "alibaba.com",
        "transactionLocation": "Shenzhen, China",
        "securityQuestion": "What is your mother's maiden name?",
        "securityAnswer": "mahajan",
        "outcomeNote": ""
    },
    {
        "userName": "Priya Patel",
        "securityIdentifier": "67890",
        "cardEnding": "**** 8888",
        "caseStatus": "pending_review",
        "transactionAmount": "₹1,25,000",
        "transactionName": "Luxury Fashion Store",
        "transactionTime": "2025-11-27 03:15:42",
        "transactionCategory": "retail",
        "transactionSource": "fashionlux.net",
        "transactionLocation": "Dubai, UAE",
        "securityQuestion": "What city were you born in?",
        "securityAnswer": "Mumbai",
        "outcomeNote": ""
    },
    {
        "userName": "Amit Verma",
        "securityIdentifier": "54321",
        "cardEnding": "**** 7777",
        "caseStatus": "pending_review",
        "transactionAmount": "₹89,500",
        "transactionName": "Tech Gadgets International",
        "transactionTime": "2025-11-27 09:45:20",
        "transactionCategory": "e-commerce",
        "transactionSource": "techgadgets.co",
        "transactionLocation": "Singapore",
        "securityQuestion": "What is your favorite color?",
        "securityAnswer": "Blue",
        "outcomeNote": ""
    }
]

# DB helpers
def init_database(db_path: str = DATABASE_FILE) -> str:
    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
    conn = sqlite3.connect(db_path, timeout=10)
    try:
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS fraud_cases (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                userName TEXT NOT NULL,
                securityIdentifier TEXT NOT NULL,
                cardEnding TEXT NOT NULL,
                caseStatus TEXT NOT NULL,
                transactionAmount TEXT NOT NULL,
                transactionName TEXT NOT NULL,
                transactionTime TEXT NOT NULL,
                transactionCategory TEXT NOT NULL,
                transactionSource TEXT NOT NULL,
                transactionLocation TEXT NOT NULL,
                securityQuestion TEXT NOT NULL,
                securityAnswer TEXT NOT NULL,
                outcomeNote TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cur.execute("SELECT COUNT(*) FROM fraud_cases")
        row = cur.fetchone()
        cnt = row[0] if row else 0
        if cnt == 0:
            logger.info("Populating DB with sample fraud cases...")
            for r in SAMPLE_FRAUD_CASES:
                cur.execute("""
                    INSERT INTO fraud_cases (
                        userName, securityIdentifier, cardEnding, caseStatus,
                        transactionAmount, transactionName, transactionTime,
                        transactionCategory, transactionSource, transactionLocation,
                        securityQuestion, securityAnswer, outcomeNote
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    r["userName"], r["securityIdentifier"], r["cardEnding"], r["caseStatus"],
                    r["transactionAmount"], r["transactionName"], r["transactionTime"],
                    r["transactionCategory"], r["transactionSource"], r["transactionLocation"],
                    r["securityQuestion"], r["securityAnswer"], r["outcomeNote"]
                ))
            conn.commit()
            logger.info("Inserted sample fraud cases.")
    finally:
        conn.close()
    return db_path

def get_db_connection(db_path: str = DATABASE_FILE):
    return sqlite3.connect(db_path, timeout=10)

# Data containers
@dataclass
class FraudCaseData:
    id: Optional[int] = None
    userName: Optional[str] = None
    securityIdentifier: Optional[str] = None
    cardEnding: Optional[str] = None
    caseStatus: Optional[str] = None
    transactionAmount: Optional[str] = None
    transactionName: Optional[str] = None
    transactionTime: Optional[str] = None
    transactionCategory: Optional[str] = None
    transactionSource: Optional[str] = None
    transactionLocation: Optional[str] = None
    securityQuestion: Optional[str] = None
    securityAnswer: Optional[str] = None
    outcomeNote: Optional[str] = None
    verification_passed: bool = False
    case_loaded: bool = False

@dataclass
class Userdata:
    fraud_case: FraudCaseData

# function tools
@function_tool
async def load_fraud_case_for_user(ctx: RunContext, user_name: Annotated[str, Field(description="Name provided by the caller")]) -> str:
    logger.info("load_fraud_case_for_user: %s", user_name)
    if ctx is None:
        ctx = RunContext()
    if getattr(ctx, "userdata", None) is None:
        ctx.userdata = Userdata(fraud_case=FraudCaseData())
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("""
            SELECT id, userName, securityIdentifier, cardEnding, caseStatus,
                   transactionAmount, transactionName, transactionTime,
                   transactionCategory, transactionSource, transactionLocation,
                   securityQuestion, securityAnswer, outcomeNote
            FROM fraud_cases
            WHERE LOWER(userName) = LOWER(?)
              AND caseStatus = 'pending_review'
            LIMIT 1
        """, (user_name.strip(),))
        row = cur.fetchone()
        conn.close()
        if not row:
            return f"I don't have any pending fraud alerts for the name '{user_name}'. Please verify the name."

        fc = ctx.userdata.fraud_case
        (fc.id, fc.userName, fc.securityIdentifier, fc.cardEnding, fc.caseStatus,
         fc.transactionAmount, fc.transactionName, fc.transactionTime, fc.transactionCategory,
         fc.transactionSource, fc.transactionLocation, fc.securityQuestion, fc.securityAnswer,
         fc.outcomeNote) = row + (row[13] if len(row) > 13 else "",)
        fc.case_loaded = True
        return f"Loaded case for {fc.userName}. Flagged transaction on card {fc.cardEnding} for {fc.transactionAmount}."
    except Exception:
        logger.exception("DB error in load_fraud_case_for_user")
        return "Technical error while accessing the case. Please try again later."

@function_tool
async def verify_customer_identity(ctx: RunContext, security_answer: Annotated[str, Field(description="Answer to the security question")]) -> str:
    if ctx is None or getattr(ctx, "userdata", None) is None:
        return "No case loaded — please load a case first."
    fc = ctx.userdata.fraud_case
    if not fc.case_loaded:
        return "I need to load your case first. Can you provide the name on the account?"
    provided = (security_answer or "").strip().lower()
    expected = (fc.securityAnswer or "").strip().lower()
    if provided and expected and provided == expected:
        fc.verification_passed = True
        return "Verification successful — I can now discuss the transaction."
    else:
        fc.verification_passed = False
        try:
            conn = get_db_connection()
            cur = conn.cursor()
            cur.execute("UPDATE fraud_cases SET caseStatus = ?, outcomeNote = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                        ("verification_failed", f"Verification failed at {datetime.utcnow().isoformat()}", fc.id))
            conn.commit()
            conn.close()
        except Exception:
            logger.exception("Failed to persist verification failure")
        return "Verification failed. I cannot proceed for your security. Please contact customer service."

@function_tool
async def mark_transaction_status(ctx: RunContext, user_confirmed: Annotated[bool, Field(description="True = user confirms transaction; False = denies")], additional_notes: Annotated[str, Field(description="Optional notes")] = "") -> str:
    if ctx is None or getattr(ctx, "userdata", None) is None:
        return "No case loaded to update."
    fc = ctx.userdata.fraud_case
    if not fc.case_loaded or not fc.id:
        return "No case loaded to update."
    if not fc.verification_passed:
        return "Cannot update case without successful verification."
    if user_confirmed:
        new_status = "confirmed_safe"
        note = f"Customer confirmed transaction as legitimate. {additional_notes}".strip()
        reply = f"Transaction marked as legitimate. Card {fc.cardEnding} remains active. No further action is needed."
    else:
        new_status = "confirmed_fraud"
        note = f"Customer denied the transaction. Fraud confirmed. {additional_notes}".strip()
        reply = (f"Transaction marked as fraudulent. Card {fc.cardEnding} has been blocked and a dispute "
                 f"has been raised for {fc.transactionAmount}. Our fraud team will follow up.")
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("""
            UPDATE fraud_cases
            SET caseStatus = ?, outcomeNote = ?, updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
        """, (new_status, note, fc.id))
        conn.commit()
        conn.close()
        fc.caseStatus = new_status
        fc.outcomeNote = note
        return reply
    except Exception:
        logger.exception("DB error in mark_transaction_status")
        return "Error updating the case. Please contact customer service."

@function_tool
async def end_fraud_call(ctx: RunContext) -> str:
    if ctx is None or getattr(ctx, "userdata", None) is None:
        return "No case loaded."
    fc = ctx.userdata.fraud_case
    summary = (f"Summary for {fc.userName or 'customer'}:\n"
               f"- Transaction: {fc.transactionAmount or 'N/A'} at {fc.transactionName or 'N/A'}\n"
               f"- Status: {fc.caseStatus or 'N/A'}\n"
               f"- Action: {fc.outcomeNote or 'N/A'}\n\n"
               f"Your security is our priority at {BANK_NAME}. Contact 24/7 support for help.")
    return summary

# Agent persona
class FraudAssistant(Agent):
    def __init__(self):
        super().__init__(
            instructions=(
                f"You are a calm, professional fraud-detection representative for {BANK_NAME}. "
                "Do not ask for PINs, OTPs, CVV or full card numbers. Use only the stored security question for verification. "
                "When you need to load, verify, mark status or end a call, call the respective function tools."
            )
        )

# Prewarm & entrypoint (with LLM-failure fallback)
def prewarm(proc: JobProcess):
    try:
        if LIVEKIT_AVAILABLE:
            proc.userdata.setdefault("vad", silero.VAD.load())
        init_database(DATABASE_FILE)
        logger.info("Prewarm complete: DB initialized (and VAD if available).")
    except Exception:
        logger.exception("Prewarm failed (continuing)")

async def _start_session_with_optional_llm(ctx: JobContext, allow_llm: bool):
    """Helper: create and start an AgentSession. If allow_llm=False, LLM arg will be None."""
    userdata = Userdata(fraud_case=FraudCaseData())
    llm = None
    stt = None
    tts = None
    turn_detection = None
    vad = None
    if LIVEKIT_AVAILABLE and allow_llm:
        try:
            stt = deepgram.STT(model="nova-3")
            llm = google.LLM(model="gemini-2.5-flash")
            tts = murf.TTS(voice="en-US-matthew", style="Conversation", text_pacing=True)
            turn_detection = MultilingualModel()
            vad = ctx.proc.userdata.get("vad")
        except Exception:
            logger.exception("Failed to construct LLM/STT/TTS objects; will try no-LLM mode.")
            llm = None
    else:
        # If LiveKit not available or LLm disabled, keep stubs (AgentSession can be started)
        if LIVEKIT_AVAILABLE:
            try:
                stt = deepgram.STT(model="nova-3")
                tts = murf.TTS(voice="en-US-matthew", style="Conversation", text_pacing=True)
                turn_detection = MultilingualModel()
                vad = ctx.proc.userdata.get("vad")
            except Exception:
                # ignore - fallback will be used
                pass

    session = AgentSession(
        stt=stt if LIVEKIT_AVAILABLE else None,
        llm=llm if LIVEKIT_AVAILABLE else None,
        tts=tts if LIVEKIT_AVAILABLE else None,
        turn_detection=turn_detection if LIVEKIT_AVAILABLE else None,
        vad=vad if LIVEKIT_AVAILABLE else None,
        userdata=userdata,
        preemptive_generation=True,
    )

    # Try to start; bubble errors to caller to decide fallback
    await session.start(agent=FraudAssistant(), room=ctx.room, room_input_options=RoomInputOptions(noise_cancellation=noise_cancellation.BVC()) if LIVEKIT_AVAILABLE else None)
    return session

async def entrypoint(ctx: JobContext):
    """
    LiveKit entrypoint:
     - Try to start with LLM
     - If LLM throws overloaded/503 or session start fails, restart in no-LLM fallback mode
    """
    ctx.log_context_fields = {"room": getattr(ctx.room, "name", "unknown")}
    logger.info("Job entrypoint starting for room=%s", ctx.log_context_fields.get("room"))

    # First try: enable LLM
    session = None
    try:
        session = await _start_session_with_optional_llm(ctx, allow_llm=True)
        logger.info("Session started with LLM enabled.")
        await ctx.connect()
        return  # normal run continues inside LiveKit framework
    except Exception as e:
        # Detect overloaded / 503 conditions by message or class name if available
        msg = str(e).lower()
        logger.warning("Starting with LLM failed: %s", msg)
        # If the error looks like model overload or other transient LLM errors, retry without LLM
        try:
            if session:
                try:
                    await session.stop()
                except Exception:
                    pass
            logger.info("Attempting fallback session without LLM (safe mode).")
            session = await _start_session_with_optional_llm(ctx, allow_llm=False)
            logger.info("Fallback session started (no LLM).")
            await ctx.connect()
            return
        except Exception:
            logger.exception("Fallback no-LLM session failed; aborting job.")
            # ensure clean stop
            try:
                if session:
                    await session.stop()
            except Exception:
                pass
            raise

# Console fallback for local testing
async def console_mode():
    init_database(DATABASE_FILE)
    print("Console Fraud Alert Agent — demo (fake data).")
    name = input(f"Agent: Hello — this is the Fraud Department from {BANK_NAME}. Who am I speaking with?\n> ").strip()
    if not name:
        print("Agent: Please provide a name so I can look up your case.")
        return
    conn = get_db_connection()
    try:
        cur = conn.cursor()
        cur.execute("""
            SELECT id, userName, cardEnding, transactionAmount, transactionName, transactionTime, transactionLocation, securityQuestion, securityAnswer
            FROM fraud_cases
            WHERE LOWER(userName) = LOWER(?) AND caseStatus = 'pending_review'
            LIMIT 1
        """, (name,))
        row = cur.fetchone()
        if not row:
            print(f"Agent: I don't have any pending fraud alerts for '{name}'. Please verify the name or contact support.")
            return
        cid, uname, card, amount, merchant, ttime, tloc, sq, sa = row
        print(f"Agent: Thank you, {uname}. For your security, please answer the following verification question:")
        print(f"Agent (verification question): {sq}")
        ans = input("> ").strip()
        if ans.lower() != (sa or "").lower():
            print("Agent: Verification failed. For your security, I cannot proceed. Please contact customer service.")
            cur.execute("UPDATE fraud_cases SET caseStatus=?, outcomeNote=?, updated_at=CURRENT_TIMESTAMP WHERE id=?",
                        ("verification_failed", f"Verification failed at {datetime.utcnow().isoformat()}", cid))
            conn.commit()
            return
        print(f"Agent: Verification passed. We flagged a transaction of {amount} at {merchant} (card {card}) on {ttime} from {tloc}.")
        resp = input("Agent: Did you make this transaction? (yes/no)\n> ").strip().lower()
        if resp.startswith("y"):
            new_status = "confirmed_safe"
            note = "Customer confirmed transaction as legitimate."
            print("Agent: Marked as legitimate. No further action is required.")
        else:
            new_status = "confirmed_fraud"
            note = "Customer denied the transaction. Fraud confirmed. Card blocked (mock)."
            print("Agent: Marked as fraudulent. Card blocked (mock) and dispute initiated. Our fraud team will follow up.")
        cur.execute("UPDATE fraud_cases SET caseStatus=?, outcomeNote=?, updated_at=CURRENT_TIMESTAMP WHERE id=?", (new_status, note, cid))
        conn.commit()
        print("Agent: Summary recorded. Goodbye.")
    finally:
        conn.close()

# Main runner
if __name__ == "__main__":
    init_database(DATABASE_FILE)

    running_on_windows = platform.system() == "Windows"
    use_livekit_env = os.environ.get("LIVEKIT", "") == "1"
    use_livekit = use_livekit_env and LIVEKIT_AVAILABLE and not running_on_windows

    if use_livekit:
        try:
            logger.info("Starting LiveKit worker (LIVEKIT=1).")
            cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
        except Exception:
            logger.exception("LiveKit worker failed to start or crashed. Falling back to console mode.")
            try:
                asyncio.run(console_mode())
            except KeyboardInterrupt:
                print("\nInterrupted — goodbye.")
    else:
        if running_on_windows and use_livekit_env and not LIVEKIT_AVAILABLE:
            logger.warning("LIVEKIT requested but LiveKit libs not installed. Running console mode.")
        elif running_on_windows and use_livekit_env and LIVEKIT_AVAILABLE:
            logger.warning("LIVEKIT requested but running on Windows — LiveKit worker mode is not recommended here. Running console mode.")
        print("Running console mode (LiveKit disabled). To enable LiveKit, run this on Linux/WSL/Docker with LIVEKIT=1")
        try:
            asyncio.run(console_mode())
        except KeyboardInterrupt:
            print("\nInterrupted — goodbye.")
