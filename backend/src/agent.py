#!/usr/bin/env python3
"""
Single-file SDR Voice Agent (corrected & enhanced)

Drop into your environment and run:
  python sdr_voice_agent.py

Requires:
  - livekit.agents + livekit plugins your environment already uses
  - .env.local with required keys (if any)

Files written:
  ./sdr_data/leads.json
  ./sdr_data/meetings.json
  ./sdr_data/notes.json
  ./sdr_data/email_drafts.json
  ./sdr_data/company_faq.json  (only created with sample if missing)
"""
import asyncio
import os
import json
import re
import logging
from datetime import datetime, timedelta
from difflib import get_close_matches

from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    MetricsCollectedEvent,
    RoomInputOptions,
    WorkerOptions,
    cli,
    metrics,
    tokenize,
)
from livekit.plugins import murf, silero, google, deepgram, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("sdr_agent")
logger.setLevel(logging.INFO)
# simple console handler
ch = logging.StreamHandler()
ch.setFormatter(logging.Formatter("%(asctime)s %(levelname)6s %(name)s %(message)s"))
logger.addHandler(ch)

load_dotenv(".env.local")

# --------------------------
# Configuration & Storage
# --------------------------
DATA_DIR = os.getenv("SDR_DATA_DIR", "./sdr_data")
os.makedirs(DATA_DIR, exist_ok=True)
LEADS_FILE = os.path.join(DATA_DIR, "leads.json")
MEETINGS_FILE = os.path.join(DATA_DIR, "meetings.json")
NOTES_FILE = os.path.join(DATA_DIR, "notes.json")
EMAIL_DRAFTS_FILE = os.path.join(DATA_DIR, "email_drafts.json")
FAQ_FILE = os.path.join(DATA_DIR, "company_faq.json")

# default sample company if user doesn't provide their own FAQ
SAMPLE_COMPANY = {
    "company": "AcmeCloud",
    "tagline": "Simple cloud infra for startups",
    "pricing": "Free tier (1 project, 1GB storage). Paid plans start at INR 999 / month for 5 projects and 50GB storage.",
    "audience": "Startups, SMBs, dev teams wanting managed infra.",
    "faq": [
        {"q": "What does AcmeCloud do?", "a": "AcmeCloud provides managed cloud infrastructure for small teams: hosting, CI/CD integration, and simple dashboarding."},
        {"q": "Do you have a free tier?", "a": "Yes, we have a Free tier with 1 project and 1GB storage. Paid plans start at INR 999 / month."},
        {"q": "Who is AcmeCloud for?", "a": "Startups, indie devs, and small engineering teams who want managed infrastructure without DevOps overhead."},
        {"q": "How does pricing work?", "a": "Pricing is tiered by projects and storage. Add-ons include enterprise support and managed migrations."},
        {"q": "How to get started?", "a": "Sign up on our website and follow the onboarding wizard to create your first project."}
    ]
}

# --------------------------
# Utility helpers
# --------------------------
def load_json(path, default):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return default
    except json.JSONDecodeError:
        logger.warning("JSON decode error for %s. Using default.", path)
        return default


def save_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def append_json_list(path, item):
    arr = load_json(path, [])
    arr.append(item)
    save_json(path, arr)


def validate_company_data(d):
    if not isinstance(d, dict):
        return False
    if "company" not in d:
        return False
    if "faq" not in d or not isinstance(d["faq"], list):
        d["faq"] = []
    return True


def keyword_search_faq(faq_list, text, top_k=2):
    # Very simple search: look for overlapping words; fallback to difflib similarity on questions
    if not faq_list:
        return []
    text = (text or "").lower()
    tokens = set(re.findall(r"\w+", text))

    scored = []
    for item in faq_list:
        q = item.get("q", "").lower()
        a = item.get("a", "")
        q_tokens = set(re.findall(r"\w+", q))
        score = len(tokens & q_tokens)
        # also check answer text for keywords
        a_tokens = set(re.findall(r"\w+", a.lower()))
        score += len(tokens & a_tokens) * 0.5
        scored.append((score, item))

    scored.sort(key=lambda x: x[0], reverse=True)
    results = [it for s, it in scored if s > 0]

    if not results:
        # fallback: closest matching FAQ question using difflib
        questions = [it.get("q", "") for it in faq_list]
        matches = get_close_matches(text, questions, n=top_k, cutoff=0.4)
        for m in matches:
            for it in faq_list:
                if it.get("q") == m:
                    results.append(it)

    return results[:top_k]


# Simple persona heuristic
def infer_persona_from_text(text):
    t = (text or "").lower()
    if any(w in t for w in ["deploy", "ci", "devops", "docker", "k8s", "backend"]):
        return "Developer"
    if any(w in t for w in ["roadmap", "user story", "prioritize", "pm", "product manager", "product"]):
        return "Product Manager"
    if any(w in t for w in ["founder", "co-founder", "start up", "pitch", "fundraise"]):
        return "Founder"
    if any(w in t for w in ["marketing", "campaign", "growth"]):
        return "Marketer"
    return "Other"


# Qualification scoring heuristic
def compute_fit_score(lead: dict, notes: dict) -> int:
    score = 50
    timeline = (lead.get("timeline") or "").lower()
    if "now" in timeline:
        score += 20
    elif "soon" in timeline:
        score += 10

    team_size = lead.get("team_size")
    try:
        team = int(team_size)
        if team >= 5:
            score += 10
        elif team == 1:
            score -= 10
    except Exception:
        pass

    if notes.get("budget_mentioned"):
        score += 10
    if notes.get("decision_maker") == "yes":
        score += 10

    return max(0, min(100, score))


# --------------------------
# SDR Assistant class
# --------------------------
class SDRAssistant(Agent):
    def __init__(self, company_data):
        super().__init__(
            instructions=(
                "You are an SDR assistant named AcmeSDR. You must behave like a helpful sales rep: greet users warmly, "
                "ask discovery questions, answer product questions strictly from the loaded FAQ, and gather lead details. "
                "Keep replies concise and helpful."
            )
        )
        self.company = company_data
        self.faq = company_data.get("faq", [])
        self.state = {}

        # lead fields we want to collect
        self.lead_template = {
            "name": None,
            "company": None,
            "email": None,
            "role": None,
            "use_case": None,
            "team_size": None,
            "timeline": None,
            "notes": None,
            "created_at": None,
        }

    # Helper: find answer from FAQ; returns (found, answer_text)
    def answer_from_faq(self, question_text):
        # primary: exact/keyword FAQ search
        matches = keyword_search_faq(self.faq, question_text, top_k=2)
        if matches:
            answers = [m.get("a") for m in matches if m.get("a")]
            reply = " ".join(answers[:2])
            return True, reply

        # second: if user asked about product/company/pricing, use company-level fields
        qt = (question_text or "").lower()
        if re.search(r"what does (your )?product do|what does (your )?company do|what do you do", qt):
            return True, self.company.get("tagline") + " " + (self.company.get("company") or "")
        if re.search(r"price|cost|free tier|pricing|paid", qt):
            return True, self.company.get("pricing") or "Pricing details are available on our site."
        if re.search(r"who .* for|who should|ideal customer|target audience", qt):
            return True, self.company.get("audience") or "We serve a range of customers."
        return False, None

    # Collect a lead field if detected in user's message
    def extract_lead_fields(self, text, lead):
        if not text:
            return lead
        # email
        if not lead.get("email"):
            m = re.search(r"([\w\.-]+@[\w\.-]+\.\w+)", text)
            if m:
                lead["email"] = m.group(1)
        # name (heuristic)
        if not lead.get("name"):
            m = re.search(r"\bmy name is ([A-Z][a-z]+(?: [A-Z][a-z]+)*)", text)
            if m:
                lead["name"] = m.group(1)
            else:
                m = re.search(r"\bi('?m|am)\s+([A-Z][a-z]+(?: [A-Z][a-z]+)*)", text)
                if m:
                    lead["name"] = m.group(2)
        # company
        if not lead.get("company"):
            m = re.search(r"\bcompany (?:is|:)\s*([\w\s&.\-]{2,60})", text, re.I)
            if m:
                lead["company"] = m.group(1).strip()
        # role
        if not lead.get("role"):
            m = re.search(r"\bi am (?:a |an )?([\w\s/\-]{2,40})", text, re.I)
            if m:
                lead["role"] = m.group(1).strip()
        # team size
        if not lead.get("team_size"):
            m = re.search(r"team(?: size)?(?: is)?(?: around| about)?\s*(\d{1,4})", text, re.I)
            if m:
                lead["team_size"] = m.group(1)
        # timeline
        if not lead.get("timeline"):
            if any(w in text.lower() for w in ["next week", "next month", "soon", "in a month", "in a week"]):
                lead["timeline"] = "soon"
            elif any(w in text.lower() for w in ["now", "today", "immediately"]):
                lead["timeline"] = "now"
            elif any(w in text.lower() for w in ["later", "later this year", "not now"]):
                lead["timeline"] = "later"
        # use_case (longer text)
        if not lead.get("use_case"):
            if len(text) > 40:
                lead["use_case"] = text.strip()
        return lead

    # Called when a new conversation item arrives (user said something)
    async def handle_user_message(self, session, message_text, session_state):
        """
        - Answer from FAQ when possible
        - Otherwise gather lead info
        - Detect bookings and end-of-call
        - Return-visitor recognition when email is provided
        """
        logger.info("Handling user message: %s", message_text)
        s = session_state
        if "lead" not in s or s["lead"] is None:
            s["lead"] = self.lead_template.copy()
            s["lead"]["created_at"] = datetime.utcnow().isoformat()
            s["stage"] = "open"
            s["notes"] = {"budget_mentioned": False, "decision_maker": "unknown", "pain_points": []}
            s["greeted_returning"] = False

        lead = s["lead"]

        # 1) Try FAQ answer first
        found, ans = self.answer_from_faq(message_text)
        if found:
            await send_agent_text(session, ans)
            return

        # 2) extract lead fields from message
        prev_email = lead.get("email")
        lead = self.extract_lead_fields(message_text, lead)

        # 2.1) Return-visitor recognition (if email found and not already greeted)
        if lead.get("email") and not s.get("greeted_returning", False):
            # check existing leads for email
            existing = load_json(LEADS_FILE, [])
            found_lead = None
            for e in existing:
                if e.get("email") and e.get("email").lower() == lead.get("email").lower():
                    found_lead = e
                    break
            if found_lead:
                s["greeted_returning"] = True
                await send_agent_text(session, f"Welcome back {found_lead.get('name') or ''}. I found your previous interest: {found_lead.get('use_case') or 'N/A'}. Would you like to continue where we left off?")
                return

        # 3) persona-aware quick pitch if user asked who is this for
        if re.search(r"who .* for|who should|ideal customer|target audience", message_text, re.I):
            persona = infer_persona_from_text(message_text)
            pitch = self.generate_pitch_for_persona(persona)
            await send_agent_text(session, pitch)
            return

        # 4) pricing question branch
        if re.search(r"price|cost|free tier|pricing|paid", message_text, re.I):
            # prefer FAQ pricing entry
            found, ans = self.answer_from_faq("pricing")
            if found:
                await send_agent_text(session, ans)
                return

        # 5) booking / demo
        if re.search(r"demo|book.*meeting|schedule.*demo|call me|book a demo", message_text, re.I):
            slot_text = suggest_meeting_slots()
            await send_agent_text(session, "Sure — I can help schedule a quick demo. " + slot_text)
            s["awaiting_booking_confirmation"] = True
            return

        if s.get("awaiting_booking_confirmation") and re.search(r"(yes|book|pick|slot|1|2|3|confirm|confirm it|ok)", message_text, re.I):
            slot_choice = None
            m = re.search(r"\b(\d)\b", message_text)
            if m:
                slot_choice = int(m.group(1))
            else:
                if "first" in message_text.lower():
                    slot_choice = 1
            if slot_choice:
                meeting = book_mock_meeting(lead, slot_choice)
                append_json_list(MEETINGS_FILE, meeting)
                await send_agent_text(session, f"All set — booked {meeting['date']} at {meeting['time']}. I'll send an email confirmation to {lead.get('email') or 'the address you provide'}.")
                s["awaiting_booking_confirmation"] = False
                return

        # 6) end-of-call detection
        if re.search(r"(that's all|that is all|thanks|thank you|i'm done|im done|bye|goodbye|talk later|that's enough)", message_text, re.I):
            notes = generate_crm_notes(lead, s.get("notes", {}))
            fit = compute_fit_score(lead, notes)
            notes["fit_score"] = fit
            append_json_list(LEADS_FILE, lead)
            append_json_list(NOTES_FILE, {"lead": lead, "notes": notes})
            email = draft_followup_email(lead, notes, self.company)
            store_draft = {"lead": lead, "email_draft": email, "created_at": datetime.utcnow().isoformat()}
            append_json_list(EMAIL_DRAFTS_FILE, store_draft)
            summary = (
                f"Thanks {lead.get('name') or ''}. I've saved your details. Quick summary: {lead.get('company') or 'Company unknown'}, "
                f"role {lead.get('role') or 'unknown'}, timeline {lead.get('timeline') or 'unspecified'}. Fit score: {fit}. "
                "I've created notes and a follow-up email draft."
            )
            await send_agent_text(session, summary)
            return

        # 7) fallback: ask next discovery question
        follow_up = choose_discovery_question(lead)
        await send_agent_text(session, follow_up)

    def generate_pitch_for_persona(self, persona):
        base = f"{self.company.get('company')} is built for {self.company.get('audience') or 'small teams'}. "
        if persona == "Developer":
            return base + "Developers love it because it removes ops work and integrates with CI/CD pipelines."
        if persona == "Product Manager":
            return base + "PMs get a simple dashboard and analytics to track releases and usage."
        if persona == "Founder":
            return base + "Founders can launch prototypes fast without hiring DevOps."
        if persona == "Marketer":
            return base + "Marketing teams can quickly spin up landing pages and track campaigns."
        return base + "It's a simple way to manage infrastructure for small teams."


# --------------------------
# Meeting scheduler (mock)
# --------------------------
def suggest_meeting_slots():
    now = datetime.utcnow() + timedelta(hours=5, minutes=30)  # IST offset display
    day1 = (now + timedelta(days=1)).strftime("%Y-%m-%d")
    day2 = (now + timedelta(days=2)).strftime("%Y-%m-%d")
    slots = [
        {"id": 1, "date": day1, "time": "10:00 AM IST"},
        {"id": 2, "date": day1, "time": "3:00 PM IST"},
        {"id": 3, "date": day2, "time": "11:00 AM IST"},
    ]
    text = "Here are some slots I have available:\n"
    for s in slots:
        text += f"{s['id']}. {s['date']} at {s['time']}\n"
    text += "Which one works for you? Reply with the number."
    return text


def book_mock_meeting(lead, slot_choice):
    now = datetime.utcnow() + timedelta(hours=5, minutes=30)
    day1 = (now + timedelta(days=1)).strftime("%Y-%m-%d")
    day2 = (now + timedelta(days=2)).strftime("%Y-%m-%d")
    slots = {
        1: {"date": day1, "time": "10:00 AM IST"},
        2: {"date": day1, "time": "3:00 PM IST"},
        3: {"date": day2, "time": "11:00 AM IST"},
    }
    slot = slots.get(slot_choice, slots[1])
    meeting = {
        "lead_email": lead.get("email"),
        "lead_name": lead.get("name"),
        "date": slot["date"],
        "time": slot["time"],
        "created_at": datetime.utcnow().isoformat(),
    }
    return meeting


# --------------------------
# CRM notes generation
# --------------------------
def generate_crm_notes(lead, partial_notes):
    notes = {
        "pain_points": list(partial_notes.get("pain_points", [])),
        "budget_mentioned": partial_notes.get("budget_mentioned", False),
        "decision_maker": partial_notes.get("decision_maker", "unknown"),
    }
    if lead.get("team_size"):
        notes["pain_points"].append(f"Team size {lead.get('team_size')}")
    if lead.get("use_case"):
        notes["pain_points"].append(lead.get("use_case")[:200])
    return notes


# --------------------------
# Follow-up email draft
# --------------------------
def draft_followup_email(lead, notes, company):
    subj = f"Quick follow-up from {company.get('company')}"
    body = (
        f"Hi {lead.get('name') or ''},\n\n"
        f"Thanks for taking the time to speak with me. From our conversation, I understand you're interested in {lead.get('use_case') or 'evaluating ' + company.get('company')} and your timeline is {lead.get('timeline') or 'unspecified'}. "
        f"Our team thinks {company.get('company')} could help by {company.get('tagline')}.\n\n"
        "Would you like to confirm the demo we discussed? I'll be happy to send a calendar invite.\n\n"
        "Best regards,\nAcmeSDR"
    )
    return {"subject": subj, "body": body}


# --------------------------
# Simple discovery question chooser
# --------------------------
def choose_discovery_question(lead):
    if not lead.get("use_case"):
        return "Can you tell me briefly what you want to use our product for?"
    if not lead.get("team_size"):
        return "How large is your engineering team or the team that will use this product?"
    if not lead.get("timeline"):
        return "What's your expected timeline to start or launch? (now/soon/later)"
    if not lead.get("email"):
        return "What's the best email to reach you?"
    return "Thanks — anything else you'd like me to note?"


# --------------------------
# Sending helper (adapter to livekit session)
# --------------------------
async def send_agent_text(session, text):
    """
    Wrapper to send text back to the room/session. Depending on your SDK version, you might need to adapt this.
    """
    try:
        if hasattr(session, "send_text"):
            await session.send_text(text)
            return
        if hasattr(session, "send_agent_message"):
            await session.send_agent_message(text)
            return
        # recent versions often expose `emit` or similar
        if hasattr(session, "emit"):
            await session.emit("agent_message", {"text": text})
            return
        # fallback: log
        logger.info("Agent reply (no session send available): %s", text)
    except Exception:
        logger.exception("Failed to send agent text")


# --------------------------
# Entrypoint & LiveKit session setup
# --------------------------
def prewarm(proc: JobProcess):
    proc.userdata.setdefault("vad", silero.VAD.load())


async def entrypoint(ctx: JobContext):
    ctx.log_context_fields = {"room": ctx.room.name}

    # LLM / STT / TTS pipeline (you can swap models)
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
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=True,
    )

    # metrics collector like your original example
    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: {summary}")

    ctx.add_shutdown_callback(log_usage)

    # Load company FAQ or sample (log result)
    company_data = load_json(FAQ_FILE, SAMPLE_COMPANY)
    if not validate_company_data(company_data):
        logger.warning("Invalid or empty company data in %s. Using SAMPLE_COMPANY.", FAQ_FILE)
        company_data = SAMPLE_COMPANY
    else:
        logger.info("Loaded company data from %s: company=%s, faq_count=%d", FAQ_FILE, company_data.get("company"), len(company_data.get("faq", [])))
    assistant = SDRAssistant(company_data)

    # Conversation state per-room (kept in memory)
    session_state = {}

    # Handler when a new user message arrives.
    # livekit.event_emitter requires sync callback; schedule async work with create_task
    @session.on("conversation_item_added")
    def on_conversation_item(ev):
        text = None
        try:
            if isinstance(ev, dict):
                # common shapes
                text = ev.get("text") or ev.get("content") or ev.get("message")
                # Sometimes event has nested item: {"item": {"text": "..."}}
                if not text and "item" in ev and isinstance(ev["item"], dict):
                    text = ev["item"].get("text") or ev["item"].get("content")
            else:
                text = getattr(ev, "text", None) or getattr(ev, "content", None) or getattr(ev, "message", None)
                # nested property fallback
                item = getattr(ev, "item", None)
                if not text and item:
                    text = getattr(item, "text", None) or getattr(item, "content", None)
        except Exception:
            text = None

        if not text:
            return

        # ensure a place for state
        session_state.setdefault("lead", None)
        # schedule the async handler (do not await here)
        try:
            asyncio.create_task(assistant.handle_user_message(session, text, session_state))
        except Exception:
            logger.exception("Failed to schedule handler task")

    # Start the session and connect
    await session.start(agent=assistant, room=ctx.room, room_input_options=RoomInputOptions(noise_cancellation=noise_cancellation.BVC()))
    await ctx.connect()


if __name__ == "__main__":
    # ensure data files exist (do NOT overwrite existing)
    for p, d in [(LEADS_FILE, []), (MEETINGS_FILE, []), (NOTES_FILE, []), (EMAIL_DRAFTS_FILE, []), (FAQ_FILE, SAMPLE_COMPANY)]:
        if not os.path.exists(p):
            save_json(p, d)
            logger.info("Created %s (initial)", p)
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
