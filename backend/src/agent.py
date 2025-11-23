#!/usr/bin/env python3
"""
Barista Agent — single partial file per session, stable session id derived from room
- Single partial file per session: orders_partial/partial_<session_id>.json
- All tools reuse the same Userdata stored in proc.userdata['userdata']
- do NOT reinitialize userdata with a fresh Userdata unless there's truly none
- complete_order finalizes only when order.is_complete() is True and deletes partial
- atomic writes for safety
"""

from __future__ import annotations
import argparse
import difflib
import json
import logging
import os
import re
import sys
import tempfile
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

from dotenv import load_dotenv
load_dotenv(".env.local")

# livekit imports (ensure package installed)
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    RoomInputOptions,
    WorkerOptions,
    cli,
    tokenize,
    metrics,
    MetricsCollectedEvent,
    function_tool,
    RunContext,
)
from livekit.plugins import murf, silero, google, deepgram, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

# ---------- Config / paths ----------
RECEIPT_IMAGE_PATH = "/mnt/data/68f877b0-3a28-4810-a547-0af990134ad2.png"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ORDERS_DIR = os.path.join(BASE_DIR, "orders")
PARTIALS_DIR = os.path.join(BASE_DIR, "orders_partial")
RECEIPTS_HTML_DIR = os.path.join(BASE_DIR, "receipts_html")
os.makedirs(ORDERS_DIR, exist_ok=True)
os.makedirs(PARTIALS_DIR, exist_ok=True)
os.makedirs(RECEIPTS_HTML_DIR, exist_ok=True)

# ---------- Logging ----------
logger = logging.getLogger("barista")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s %(message)s"))
logger.addHandler(handler)

# ---------- Menu / domain ----------
VALID_DRINKS = {
    "latte": "Latte (espresso + steamed milk)",
    "cappuccino": "Cappuccino (espresso + foam + milk)",
    "americano": "Americano (espresso + hot water)",
    "espresso": "Espresso (single/double shot)",
    "mocha": "Mocha (chocolate + espresso + milk)",
    "coffee": "Brewed coffee",
    "cold brew": "Cold brew",
    "matcha": "Matcha latte",
    "tea": "Assorted tea"
}
VALID_DRINK_KEYS = list(VALID_DRINKS.keys())
VALID_SIZES = ["small", "medium", "large", "extra large"]
VALID_MILKS = ["whole", "skim", "almond", "oat", "soy", "coconut", "none"]
MILK_ALIASES = {"oats": "oat", "oatmilk": "oat", "oat milk": "oat", "almondmilk": "almond"}
VALID_EXTRAS = ["sugar", "whipped cream", "caramel", "extra shot", "vanilla", "cinnamon", "honey"]

MENU_TEXT = (
    "Menu:\n"
    + "\n".join([f"{k.title()}: {v}" for k, v in VALID_DRINKS.items()])
    + "\nSizes: " + ", ".join(VALID_SIZES)
    + "\nMilks: " + ", ".join(VALID_MILKS)
    + "\nExtras: " + ", ".join(VALID_EXTRAS)
)

# ---------- Models ----------
@dataclass
class OrderState:
    drinkType: Optional[str] = None
    size: Optional[str] = None
    milk: Optional[str] = None
    extras: List[str] = field(default_factory=list)
    name: Optional[str] = None

    def is_complete(self) -> bool:
        # milk may be "none" so check is not None; extras default []
        return bool(self.drinkType and self.size and (self.milk is not None) and (self.extras is not None) and self.name)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "drinkType": self.drinkType,
            "size": self.size,
            "milk": self.milk,
            "extras": self.extras,
            "name": self.name,
        }

    def get_summary(self) -> str:
        extras_text = f" with {', '.join(self.extras)}" if self.extras else ""
        if not self.is_complete():
            return f"incomplete: {self.to_dict()}"
        return f"{self.size} {self.drinkType}{extras_text} for {self.name}"

@dataclass
class Userdata:
    order: OrderState = field(default_factory=OrderState)
    session_start: datetime = field(default_factory=datetime.utcnow)
    session_id: Optional[str] = None

# ---------- Utilities ----------

def _timestamp() -> str:
    return datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")


def normalize_list(items: Optional[List[str]]) -> List[str]:
    return [re.sub(r"\s+", " ", it.strip().lower()) for it in (items or []) if it and it.strip()]


def _atomic_write(path: str, payload: Any) -> None:
    tmp_fd, tmp_path = tempfile.mkstemp(dir=os.path.dirname(path))
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False)
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass


def _partial_path_for_session(session_id: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9_-]", "_", session_id or "anon")
    return os.path.join(PARTIALS_DIR, f"partial_{safe}.json")


def _save_partial_single(order: OrderState, session_id: str) -> str:
    path = _partial_path_for_session(session_id)
    payload = {
        "session_id": session_id,
        "saved_at": datetime.utcnow().isoformat() + "Z",
        "order": order.to_dict(),
    }
    _atomic_write(path, payload)
    logger.info("Updated partial for session %s -> %s", session_id, path)
    return path


def _save_final(order: OrderState) -> str:
    ts = _timestamp()
    filename = f"order_{ts}.json"
    path = os.path.join(ORDERS_DIR, filename)
    payload = {
        "order": order.to_dict(),
        "saved_at": datetime.utcnow().isoformat() + "Z",
        "receipt_image_path": RECEIPT_IMAGE_PATH if os.path.exists(RECEIPT_IMAGE_PATH) else None,
    }
    _atomic_write(path, payload)
    logger.info("Saved final order -> %s", path)
    return path


def _write_receipt_html(order: OrderState, json_path: str) -> Optional[str]:
    try:
        ts = _timestamp()
        filename = f"receipt_{ts}.html"
        path = os.path.join(RECEIPTS_HTML_DIR, filename)
        cup_size = {"small": 120, "medium": 160, "large": 200, "extra large": 240}.get(order.size, 160)
        extras_html = "".join(f"<li>{e}</li>" for e in (order.extras or []))
        img_tag = f'<img src="{RECEIPT_IMAGE_PATH}" alt="logo" style="max-width:120px"/>' if os.path.exists(RECEIPT_IMAGE_PATH) else ""
        html = f"""<!doctype html>
<html>
<head><meta charset="utf-8"><title>Receipt</title>
<style>body{{font-family:sans-serif;max-width:600px;margin:20px}}.header{{display:flex;gap:12px;align-items:center}}</style>
</head>
<body>
  <div class="header">{img_tag}<h2>Order Receipt</h2></div>
  <p><strong>Customer:</strong> {order.name}</p>
  <p><strong>Order:</strong> {order.size} {order.drinkType}</p>
  <p><strong>Milk:</strong> {order.milk}</p>
  <p><strong>Extras:</strong></p><ul>{extras_html}</ul>
  <div><svg width="{cup_size}" height="{cup_size}" viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
    <rect x="10" y="20" width="80" height="60" rx="12" ry="12" fill="#f3f0ec" stroke="#c9bca6"/>
    {"<ellipse cx='50' cy='22' rx='30' ry='10' fill='#fff'/>" if 'whipped cream' in (order.extras or []) else ""}
  </svg></div>
  <p style="font-size:0.9em;color:#666">Saved JSON: {json_path}</p>
</body>
</html>"""
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(html)
        logger.info("Wrote receipt HTML -> %s", path)
        return path
    except Exception:
        logger.exception("failed to write receipt html")
        return None

# ---------- Parsing helpers ----------

def fuzzy_match_drink(raw: str) -> Optional[str]:
    if not raw:
        return None
    d = raw.strip().lower()
    if d in VALID_DRINK_KEYS:
        return d
    for k in VALID_DRINK_KEYS:
        if k in d:
            return k
    tokens = re.split(r'\W+', d)
    for t in tokens:
        if not t: continue
        if t in VALID_DRINK_KEYS:
            return t
        candidate = difflib.get_close_matches(t, VALID_DRINK_KEYS, n=1, cutoff=0.8)
        if candidate:
            return candidate[0]
    candidate = difflib.get_close_matches(d, VALID_DRINK_KEYS, n=1, cutoff=0.7)
    return candidate[0] if candidate else None


def normalize_milk(token: str) -> Optional[str]:
    if not token:
        return None
    t = token.strip().lower()
    if t in VALID_MILKS:
        return t
    if t in MILK_ALIASES:
        return MILK_ALIASES[t]
    t2 = t.replace("milk", "").strip()
    if t2 in MILK_ALIASES:
        return MILK_ALIASES[t2]
    if t2 in VALID_MILKS:
        return t2
    return None


def parse_multi_fields(raw: str) -> Dict[str, Any]:
    out = {"drink": None, "size": None, "milk": None, "extras": [], "name": None}
    if not raw or not raw.strip():
        return out
    text = raw.strip()
    candidates = [p.strip() for p in re.split(r'[,\n;]+', text) if p.strip()]
    if not candidates:
        candidates = [text]
    for seg in candidates:
        seg_low = seg.lower()
        for s in VALID_SIZES:
            if s in seg_low:
                out["size"] = s
                break
        for token in re.findall(r'([a-zA-Z\s]+milk|oats|oat|almond|soy|coconut|skim|whole|none)', seg_low):
            nm = normalize_milk(token)
            if nm:
                out["milk"] = nm
                break
        for e in VALID_EXTRAS:
            if e in seg_low and e not in out["extras"]:
                out["extras"].append(e)
        nm_match = re.search(r"(?:for|name is|my name is|it's|its)\s+([A-Za-z\s]{2,30})$", seg, flags=re.IGNORECASE)
        if nm_match:
            out["name"] = nm_match.group(1).strip().title()
            continue
        drink_candidate = fuzzy_match_drink(seg)
        if drink_candidate:
            if any(k in seg_low for k in VALID_DRINK_KEYS):
                out["drink"] = seg.strip()
            else:
                out["drink"] = drink_candidate
    if not any(out.values()):
        tokens = re.split(r'\s+', text.lower())
        for t in tokens:
            if not out["drink"] and fuzzy_match_drink(t):
                out["drink"] = fuzzy_match_drink(t)
            if not out["size"] and t in VALID_SIZES:
                out["size"] = t
            if not out["milk"]:
                nm2 = normalize_milk(t)
                if nm2:
                    out["milk"] = nm2
            for e in VALID_EXTRAS:
                if e in text.lower() and e not in out["extras"]:
                    out["extras"].append(e)
        last = text.strip().split()[-1]
        if last and not out["name"] and not fuzzy_match_drink(last.lower()) and last.lower() not in VALID_SIZES and not normalize_milk(last):
            if re.match(r'^[A-Za-z]{2,30}$', last):
                out["name"] = last.title()
    out["extras"] = list(dict.fromkeys([e for e in out["extras"] if e in VALID_EXTRAS]))
    return out

# ---------- Robust ctx.proc handling ----------

def get_proc(ctx) -> JobProcess:
    """
    Return JobProcess-like object with .userdata mapping.
    Avoid accessing ctx.userdata property directly (it may be a property).
    """
    if "proc" in getattr(ctx, "__dict__", {}):
        proc = ctx.__dict__["proc"]
        if getattr(proc, "userdata", None) is None:
            proc.userdata = {}
        return proc
    if "_userdata" in getattr(ctx, "__dict__", {}):
        class DummyProc: pass
        dp = DummyProc()
        dp.userdata = ctx.__dict__["_userdata"]
        return dp
    if not hasattr(ctx, "_userdata"):
        setattr(ctx, "_userdata", {})
    class DummyProc2: pass
    dp2 = DummyProc2()
    dp2.userdata = getattr(ctx, "_userdata")
    return dp2

# ---------- Tools ----------
@function_tool
async def menu(ctx: RunContext) -> str:
    return MENU_TEXT

@function_tool
async def process_input(ctx: RunContext, text: str) -> str:
    """
    Single tool to parse multi-field utterances and update the single partial file for the session.
    """
    proc = get_proc(ctx)
    # reuse existing userdata if present; if absent create one but preserve session_id when possible
    if "userdata" not in proc.userdata or not isinstance(proc.userdata["userdata"], Userdata):
        proc.userdata["userdata"] = Userdata()

    ud: Userdata = proc.userdata["userdata"]
    # ensure stable session_id (prefer room-derived id)
    if not ud.session_id:
        room_obj = getattr(ctx, "room", None)
        room_name = getattr(room_obj, "name", None) or getattr(room_obj, "id", None)
        if room_name:
            ud.session_id = f"sid_{re.sub(r'[^A-Za-z0-9_-]', '_', str(room_name))}"
        else:
            ud.session_id = f"sid_{datetime.utcnow().strftime('%Y%m%dT%H%M%S%f')}"

    parsed = parse_multi_fields(text or "")
    saved = []

    # write parsed fields into partial (we overwrite existing values using new parsed values)
    if parsed.get("drink"):
        ud.order.drinkType = parsed["drink"]; saved.append("drinkType")
    if parsed.get("size"):
        ud.order.size = parsed["size"]; saved.append("size")
    if parsed.get("milk"):
        ud.order.milk = parsed["milk"]; saved.append("milk")
    if parsed.get("extras"):
        ud.order.extras = list(dict.fromkeys((ud.order.extras or []) + parsed["extras"])); saved.append("extras")
    if parsed.get("name"):
        ud.order.name = parsed["name"]; saved.append("name")

    proc.userdata["userdata"] = ud
    if ud.session_id:
        _save_partial_single(ud.order, ud.session_id)

    if not saved:
        return f"error: could not parse fields from '{text}'"

    logger.info("process_input saved fields: %s for session %s", saved, ud.session_id)
    return f"ok: saved {', '.join(saved)}"

# Helper to ensure we have a userdata instance and stable session id for setters

def _ensure_userdata_for_ctx(ctx) -> Userdata:
    proc = get_proc(ctx)
    if "userdata" not in proc.userdata or not isinstance(proc.userdata["userdata"], Userdata):
        proc.userdata["userdata"] = Userdata()
    ud: Userdata = proc.userdata["userdata"]
    if not ud.session_id:
        room_obj = getattr(ctx, "room", None)
        room_name = getattr(room_obj, "name", None) or getattr(room_obj, "id", None)
        if room_name:
            ud.session_id = f"sid_{re.sub(r'[^A-Za-z0-9_-]', '_', str(room_name))}"
        else:
            ud.session_id = f"sid_{datetime.utcnow().strftime('%Y%m%dT%H%M%S%f')}"
    return ud

@function_tool
async def set_drink_type(ctx: RunContext, drink: str) -> str:
    ud = _ensure_userdata_for_ctx(ctx)
    parsed = parse_multi_fields(drink or "")
    if parsed.get("drink"):
        ud.order.drinkType = parsed["drink"]
    else:
        fm = fuzzy_match_drink(drink or "")
        if fm:
            ud.order.drinkType = fm
    proc = get_proc(ctx)
    proc.userdata["userdata"] = ud
    _save_partial_single(ud.order, ud.session_id)
    logger.info("set_drink_type -> %s (session=%s)", ud.order.drinkType, ud.session_id)
    return "ok: saved drinkType"

@function_tool
async def set_size(ctx: RunContext, size: str) -> str:
    ud = _ensure_userdata_for_ctx(ctx)
    s = (size or "").strip().lower()
    if s not in VALID_SIZES:
        return f"error: invalid size '{size}'. Valid: {', '.join(VALID_SIZES)}"
    ud.order.size = s
    proc = get_proc(ctx)
    proc.userdata["userdata"] = ud
    _save_partial_single(ud.order, ud.session_id)
    logger.info("set_size -> %s (session=%s)", ud.order.size, ud.session_id)
    return "ok: saved size"

@function_tool
async def set_milk(ctx: RunContext, milk: str) -> str:
    ud = _ensure_userdata_for_ctx(ctx)
    nm = normalize_milk(milk or "")
    if nm is None:
        return f"error: invalid milk '{milk}'. Valid: {', '.join(VALID_MILKS)}"
    ud.order.milk = nm
    proc = get_proc(ctx)
    proc.userdata["userdata"] = ud
    _save_partial_single(ud.order, ud.session_id)
    logger.info("set_milk -> %s (session=%s)", ud.order.milk, ud.session_id)
    return "ok: saved milk"

@function_tool
async def set_extras(ctx: RunContext, extras: Optional[List[str]] = None) -> str:
    ud = _ensure_userdata_for_ctx(ctx)
    known = [e for e in normalize_list(extras or []) if e in VALID_EXTRAS]
    ud.order.extras = list(dict.fromkeys((ud.order.extras or []) + known))
    proc = get_proc(ctx)
    proc.userdata["userdata"] = ud
    _save_partial_single(ud.order, ud.session_id)
    logger.info("set_extras -> %s (session=%s)", ud.order.extras, ud.session_id)
    return "ok: saved extras"

@function_tool
async def set_name(ctx: RunContext, name: str) -> str:
    ud = _ensure_userdata_for_ctx(ctx)
    n = (name or "").strip()
    if not n:
        return "error: empty name"
    ud.order.name = n.title()
    proc = get_proc(ctx)
    proc.userdata["userdata"] = ud
    _save_partial_single(ud.order, ud.session_id)
    logger.info("set_name -> %s (session=%s)", ud.order.name, ud.session_id)
    return "ok: saved name"

@function_tool
async def get_order_status(ctx: RunContext) -> Dict[str, Any]:
    proc = get_proc(ctx)
    ud: Userdata = proc.userdata.get("userdata") or Userdata()
    return ud.order.to_dict()

@function_tool
async def complete_order(ctx: RunContext) -> str:
    proc = get_proc(ctx)
    if "userdata" not in proc.userdata or not isinstance(proc.userdata["userdata"], Userdata):
        return "error: no active session data to finalize"
    ud: Userdata = proc.userdata["userdata"]
    order = ud.order

    if not order.is_complete():
        missing = []
        if not order.drinkType: missing.append("drinkType")
        if not order.size: missing.append("size")
        if order.milk is None: missing.append("milk")
        if order.extras is None: missing.append("extras")
        if not order.name: missing.append("name")
        logger.info("complete_order called but missing: %s (session=%s)", missing, ud.session_id)
        return f"missing: {', '.join(missing)}"

    json_path = _save_final(order)
    html_path = _write_receipt_html(order, json_path)

    sid = ud.session_id
    if sid:
        try:
            p = _partial_path_for_session(sid)
            if os.path.exists(p):
                os.remove(p)
                logger.info("Removed partial for session %s", sid)
        except Exception:
            logger.exception("failed to remove partial")

    # reset userdata but set session_id to same room-derived id if possible
    new_ud = Userdata()
    room_obj = getattr(ctx, "room", None)
    room_name = getattr(room_obj, "name", None) or getattr(room_obj, "id", None)
    if room_name:
        new_ud.session_id = f"sid_{re.sub(r'[^A-Za-z0-9_-]', '_', str(room_name))}"
    proc.userdata["userdata"] = new_ud

    logger.info("complete_order finalized -> %s (session=%s)", json_path, sid)
    return json.dumps({"json": json_path, "html_receipt": html_path})

@function_tool
async def repeat_last_order(ctx: RunContext) -> str:
    files = sorted([f for f in os.listdir(ORDERS_DIR) if f.startswith("order_")])
    if not files:
        return "error: no previous orders to repeat"
    latest = files[-1]
    src = os.path.join(ORDERS_DIR, latest)
    with open(src, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    order_payload = payload.get("order") or payload
    order = OrderState(**order_payload)
    new_json = _save_final(order)
    new_html = _write_receipt_html(order, new_json)
    logger.info("repeat_last_order created -> %s", new_json)
    return json.dumps({"json": new_json, "html_receipt": new_html})

# ---------- Prewarm ----------

def prewarm(proc: JobProcess):
    try:
        proc.userdata["vad"] = silero.VAD.load()
        logger.info("Prewarmed VAD (silero).")
    except Exception:
        logger.exception("failed to prewarm VAD; continuing")
        proc.userdata["vad"] = None

    # initialize userdata map if missing and do not replace an existing Userdata
    if "userdata" not in proc.userdata or not isinstance(proc.userdata["userdata"], Userdata):
        proc.userdata["userdata"] = Userdata()
    # keep a placeholder id; entrypoint will set stable room-derived id
    if not proc.userdata["userdata"].session_id:
        proc.userdata["userdata"].session_id = f"sid_prewarm_{datetime.utcnow().strftime('%Y%m%dT%H%M%S%f')}"
    logger.info("prewarm complete (session_id=%s)", proc.userdata["userdata"].session_id)

# ---------- Entrypoint (voice) ----------

async def entrypoint(ctx: JobContext):
    ctx.log_context_fields = {"room": getattr(ctx.room, "name", "unknown")}

    # find/create proc userdata map safely (do NOT access ctx.userdata property)
    if "proc" in getattr(ctx, "__dict__", {}):
        proc = ctx.__dict__["proc"]
        if getattr(proc, "userdata", None) is None:
            proc.userdata = {}
        if "userdata" not in proc.userdata or not isinstance(proc.userdata["userdata"], Userdata):
            proc.userdata["userdata"] = Userdata()
        userdata_map = proc.userdata
    else:
        if not hasattr(ctx, "_userdata"):
            ctx._userdata = {}
        if "userdata" not in ctx._userdata or not isinstance(ctx._userdata["userdata"], Userdata):
            ctx._userdata["userdata"] = Userdata()
        userdata_map = ctx._userdata

    # Create a stable session_id from the room name (fall back to room id, then to existing id)
    room_name = getattr(ctx.room, "name", None) or getattr(ctx.room, "id", None)
    if room_name:
        stable_sid = f"sid_{re.sub(r'[^A-Za-z0-9_-]', '_', str(room_name))}"
    else:
        stable_sid = userdata_map["userdata"].session_id or f"sid_{datetime.utcnow().strftime('%Y%m%dT%H%M%S%f')}"
    userdata_map["userdata"].session_id = stable_sid

    # small FS test (non-fatal)
    try:
        _ = _save_final(OrderState(drinkType="latte", size="medium", milk="oat", extras=["vanilla"], name="Test"))
    except Exception:
        logger.exception("filesystem test failed")

    # Pass the userdata mapping into AgentSession so ctx.userdata property works
    session = AgentSession(
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(model="gemini-2.5-flash"),
        tts=murf.TTS(
            voice="en-US-matthew",
            style="Conversation",
            tokenizer=tokenize.basic.SentenceTokenizer(min_sentence_len=2),
            text_pacing=True,
        ),
        turn_detection=MultilingualModel(),
        vad=userdata_map.get("vad"),
        preemptive_generation=True,
        userdata=userdata_map,   # ensure AgentSession.userdata is set to our map
    )

    usage_collector = metrics.UsageCollector()
    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        usage_collector.collect(ev.metrics)

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info("usage summary: %s", summary)

    ctx.add_shutdown_callback(log_usage)

    barista_instructions = (
        "You are a friendly barista. Prefer using the `process_input` tool when the user gives several fields in one utterance "
        "(e.g. 'Latte, small, oat milk, add caramel, Rohit'). For single answers calling the dedicated setter (e.g. set_size) is fine. "
        "After each tool call, confirm what's saved. When all fields are saved, call complete_order explicitly to finalize."
    )

    await session.start(
        agent=Agent(instructions=barista_instructions, tools=[
            menu, process_input, set_drink_type, set_size, set_milk, set_extras, set_name, get_order_status, complete_order, repeat_last_order
        ]),
        room=ctx.room,
        room_input_options=RoomInputOptions(noise_cancellation=noise_cancellation.BVC())
    )

    await ctx.connect()

# ---------- Debug / text harness ----------

def run_text_mode_interactive():
    print("Starting debug/text-mode barista (no STT/TTS). Type 'exit' to quit.")
    print(MENU_TEXT)
    ud = Userdata()
    ud.session_id = f"sid_debug_{datetime.utcnow().strftime('%Y%m%dT%H%M%S%f')}"
    while True:
        if not ud.order.is_complete():
            raw = input("\n(You can say 'menu' or give multiple fields: 'Latte, small, oat milk, add caramel, Rohit')\nWhat would you like? ").strip()
            if raw.lower() in ("exit", "quit"): break
            if raw.lower() == "menu":
                print(MENU_TEXT); continue
            parsed = parse_multi_fields(raw)
            any_set = False
            if parsed.get("drink"):
                ud.order.drinkType = parsed["drink"]; any_set = True
            if parsed.get("size"):
                ud.order.size = parsed["size"]; any_set = True
            if parsed.get("milk"):
                ud.order.milk = parsed["milk"]; any_set = True
            if parsed.get("extras"):
                ud.order.extras = list(dict.fromkeys((ud.order.extras or []) + parsed["extras"])); any_set = True
            if parsed.get("name"):
                ud.order.name = parsed["name"]; any_set = True
            if any_set:
                _save_partial_single(ud.order, ud.session_id)
                print("Saved (partial) ->", _partial_path_for_session(ud.session_id))
                print("Current partial:", ud.order.to_dict())
            else:
                print("I couldn't parse anything. Try 'Latte, small, oat milk, add caramel, Rohit' or say 'menu'.")
        if ud.order.is_complete():
            print("\nOrder summary:", ud.order.get_summary())
            ok = input("Confirm and save? (yes/no/repeat/abort): ").strip().lower()
            if ok in ("yes", "y"):
                json_path = _save_final(ud.order)
                html_path = _write_receipt_html(ud.order, json_path)
                print("Order saved:", json_path)
                if html_path: print("Receipt HTML:", html_path)
                try:
                    p = _partial_path_for_session(ud.session_id)
                    if os.path.exists(p): os.remove(p)
                except Exception:
                    pass
                ud = Userdata(); ud.session_id = f"sid_debug_{datetime.utcnow().strftime('%Y%m%dT%H%M%S%f')}"
            elif ok in ("no", "n", "abort"):
                print("Order discarded.")
                try:
                    p = _partial_path_for_session(ud.session_id)
                    if os.path.exists(p): os.remove(p)
                except Exception:
                    pass
                ud = Userdata(); ud.session_id = f"sid_debug_{datetime.utcnow().strftime('%Y%m%dT%H%M%S%f')}"
            elif ok == "repeat":
                files = sorted([f for f in os.listdir(ORDERS_DIR) if f.startswith("order_")])
                if not files: print("No previous order to repeat.")
                else:
                    with open(os.path.join(ORDERS_DIR, files[-1]), "r", encoding="utf-8") as fh:
                        payload = json.load(fh)
                    order_payload = payload.get("order") or payload
                    order = OrderState(**order_payload)
                    new_json = _save_final(order); new_html = _write_receipt_html(order, new_json)
                    print("Repeated last saved order:", new_json)
                    if new_html: print("Receipt HTML:", new_html)
                ud = Userdata(); ud.session_id = f"sid_debug_{datetime.utcnow().strftime('%Y%m%dT%H%M%S%f')}"
            else:
                print("Unknown response; starting over.")
                ud = Userdata(); ud.session_id = f"sid_debug_{datetime.utcnow().strftime('%Y%m%dT%H%M%S%f')}"

# ---------- CLI entry ----------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Barista Agent - debug or voice")
    parser.add_argument("--debug", action="store_true", help="Run interactive text-mode debug harness")
    parser.add_argument("mode", nargs="?", choices=["dev", "debug", "voice"], default=None, help="Optional positional mode: 'dev'/'debug' or 'voice'")
    args = parser.parse_args()

    mode_is_debug = args.debug or (args.mode in ("dev", "debug"))
    if mode_is_debug:
        run_text_mode_interactive()
        sys.exit(0)

    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
