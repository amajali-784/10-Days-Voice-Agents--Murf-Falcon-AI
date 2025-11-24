import asyncio
import logging
import json
import os
from datetime import datetime, timezone
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
from livekit.agents import UserInputTranscribedEvent

logger = logging.getLogger("agent")
logging.basicConfig(level=logging.INFO)

load_dotenv(".env.local")


class WellnessAssistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="""
You are CalmCompanion, a friendly, grounded health & wellness voice companion.
Your role is to check in briefly each day with the user about mood, energy, stress,
and 1–3 simple objectives. Keep questions short and supportive. Offer small, actionable,
non-medical suggestions (e.g., "try a 5-minute walk", "break the task into one small step").
Do not diagnose or give medical advice. Be warm but concise. Listen actively and validate
their feelings without being preachy. Persist each check-in to a JSON log and
refer to previous entries briefly when the user returns to show continuity.
""",
        )


def prewarm(proc: JobProcess):
    # Preload voice activity detection model once per process
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    ctx.log_context_fields = {"room": ctx.room.name}

    # Configure an agent session using the same plugin stack as the starter template.
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

    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: {summary}")

    ctx.add_shutdown_callback(log_usage)

    # determine a reliable path for the JSON file next to this agent file
    BASE_DIR = os.path.dirname(__file__)
    DATA_DIR = os.path.join(BASE_DIR, "data")
    LOG_FILE = os.path.join(DATA_DIR, "wellness_log.json")

    def ensure_data_dir():
        os.makedirs(DATA_DIR, exist_ok=True)
        if not os.path.exists(LOG_FILE):
            with open(LOG_FILE, "w", encoding="utf-8") as fh:
                json.dump([], fh, indent=2)

    def load_log():
        ensure_data_dir()
        try:
            with open(LOG_FILE, "r", encoding="utf-8") as fh:
                data = json.load(fh)
                if isinstance(data, list):
                    return data
                return []
        except Exception:
            logger.exception("Failed to read wellness log")
            return []

    def save_entry(entry: dict):
        """
        Normalize entry to canonical schema, set timestamps to UTC ISO if missing,
        append to file and write pretty JSON.
        """
        ensure_data_dir()
        now = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")

        normalized = {
            "timestamp": entry.get("timestamp") or now,
            "mood": (entry.get("mood") or "").strip(),
            "energy": (entry.get("energy") or "").strip(),
            "stress": (entry.get("stress") or "none").strip(),
            "goals": entry.get("goals") if isinstance(entry.get("goals"), list) else (entry.get("goals") and [entry.get("goals")] or []),
            "summary": (entry.get("summary") or "").strip(),
            "finished_at": entry.get("finished_at") or now,
        }
        try:
            log = load_log()
            log.append(normalized)
            with open(LOG_FILE, "w", encoding="utf-8") as fh:
                json.dump(log, fh, indent=2, ensure_ascii=False)
            logger.info(f"Saved wellness entry at {normalized['timestamp']}")
            return True
        except Exception:
            logger.exception("Failed to save wellness entry")
            return False

    def get_recent_context(past_entries, limit=3):
        """Extract insights from recent check-ins for more personalized greetings."""
        if not past_entries:
            return None

        recent = past_entries[-limit:]
        moods = [e.get("mood", "").lower() for e in recent if e.get("mood")]
        energies = [e.get("energy", "").lower() for e in recent if e.get("energy")]

        context = {
            "last_entry": recent[-1],
            "recent_moods": moods,
            "recent_energies": energies,
            "total_checkins": len(past_entries),
        }
        return context

    def init_wellness_state(s):
        # Create a fresh in-memory session state
        state = {
            "timestamp": None,
            "mood": None,
            "energy": None,
            "stress": None,
            "goals": [],
            "summary": None,
            "finished_at": None,
        }
        setattr(s, "wellness_state", state)
        setattr(s, "wellness_step", 0)
        return state

    def generate_suggestion(state):
        """Generate a personalized suggestion based on the user's current state."""
        energy = (state.get("energy") or "").lower()
        stress = (state.get("stress") or "none").lower()
        num_goals = len(state.get("goals") or [])

        suggestions = []

        # Energy-based suggestions
        if "low" in energy or "tired" in energy:
            suggestions.append("Since energy is low, maybe pick just one goal to focus on")
        elif "high" in energy:
            suggestions.append("Great energy today — ride that momentum")

        # Stress-based suggestions
        if stress and stress != "none":
            suggestions.append("When stressed, try breaking tasks into 5-minute chunks")
            suggestions.append("Consider a quick walk or breathing break between tasks")

        # Goal-based suggestions
        if num_goals > 2:
            suggestions.append("You've got several things on your plate — tackle them one at a time")

        # Default suggestions
        default_suggestions = [
            "Try starting with the easiest task to build momentum",
            "Remember to take short breaks — even 5 minutes helps",
            "One small step forward is still progress",
        ]

        if suggestions:
            return suggestions[0] + "."
        else:
            return default_suggestions[0] + "."

    async def process_wellness_turn(s, transcript: str):
        """
        A simple step-driven handler that collects:
          - mood (text)
          - energy (text or scale)
          - stress (text)
          - goals (comma-separated or short list)
        Then produces a short grounding suggestion and a recap, saves to JSON.
        """
        state = getattr(s, "wellness_state", None)
        step = getattr(s, "wellness_step", 0)
        if state is None:
            state = init_wellness_state(s)
            step = 0

        text = (transcript or "").strip()

        def none_answer(t):
            return t.strip().lower() in ("no", "none", "not really", "nah", "nope", "n", "nothing")

        # Step 0: capture mood
        if step == 0:
            state["mood"] = text
            setattr(s, "wellness_step", 1)
            return "Thanks for sharing. What's your energy level today — would you say high, medium, or low?"

        # Step 1: capture energy
        if step == 1:
            state["energy"] = text
            setattr(s, "wellness_step", 2)
            return "Got it. Is anything stressing you out right now? It's okay to say no if things feel manageable."

        # Step 2: capture stress
        if step == 2:
            if none_answer(text):
                state["stress"] = "none"
            else:
                state["stress"] = text
            setattr(s, "wellness_step", 3)
            return "Okay, I hear you. What are one to three things you'd like to accomplish today?"

        # Step 3: capture goals and provide recap
        if step == 3:
            parts = [p.strip() for p in text.replace(" and ", ",").split(",") if p.strip()]
            if not parts:
                # If user gives an empty answer, prompt gently
                await session.say(
                    "That's alright. Is there even one small thing you'd like to focus on today?",
                    allow_interruptions=True,
                )
                setattr(s, "wellness_step", 3)
                return None  # Wait for next transcription

            state["goals"] = parts[:3]  # limit to 3

            # Generate contextual summary based on their responses
            mood_part = f"You're feeling {state['mood'] or 'uncertain'}"
            energy_part = f"with {state['energy'] or 'moderate'} energy"

            if state.get("stress") and state["stress"] != "none":
                stress_part = f"and mentioned some stress around {state['stress']}"
            else:
                stress_part = "and things feel manageable stress-wise"

            goals_part = f"Your focus today: {', '.join(state['goals'])}"

            # Offer personalized, actionable suggestion based on their state
            suggestion = generate_suggestion(state)

            recap = f"{mood_part} {energy_part} {stress_part}. {goals_part}. {suggestion}"

            state["summary"] = recap
            now = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
            state["timestamp"] = state.get("timestamp") or now
            state["finished_at"] = now

            saved = save_entry(state)
            setattr(s, "wellness_step", 4)

            confirmation = " Does this sound about right?"
            return recap + confirmation

        # Step 4: confirmation after recap
        if step == 4:
            # User has confirmed or responded to recap
            setattr(s, "wellness_step", 5)
            return "Great. I've saved this check-in. Remember, small steps add up. Feel free to come back anytime."

        # Step 5: already finished
        if step >= 5:
            return "We've completed today's check-in. If you'd like to start a new one, just say 'new check-in'."

        return "I didn't quite catch that. Could you say it again briefly?"

    async def handle_transcription(event: UserInputTranscribedEvent):
        """
        The transcription handler schedules the step-driven process and handles
        'new check-in' command to reset state.
        """
        if not getattr(event, "is_final", True):
            return

        user_text = getattr(event, "transcript", "").strip()
        if not user_text:
            return

        logger.info(f"User said: {user_text}")

        # Commands to restart the check-in
        restart_commands = ["new check-in", "new checkin", "new check", "start over", "restart", "check in"]
        if user_text.lower() in restart_commands:
            init_wellness_state(session)
            await session.say("Sure, let's start fresh. How are you feeling right now?", allow_interruptions=True)
            return

        # Handle view history command
        if "history" in user_text.lower() or "past check" in user_text.lower():
            past = load_log()
            if len(past) > 0:
                recent = past[-3:]
                summary = f"You've done {len(past)} check-ins. "
                if recent:
                    summary += f"Recently, you've been feeling {recent[-1].get('mood', 'varied')}. "
                await session.say(summary, allow_interruptions=True)
            else:
                await session.say("This is your first check-in with me.", allow_interruptions=True)
            return

        # If no wellness_state, initialize and start the flow
        if not hasattr(session, "wellness_state") or getattr(session, "wellness_state") is None:
            init_wellness_state(session)
            # process the user's first utterance as mood
            reply = await process_wellness_turn(session, user_text)
            if reply:
                await session.say(reply, allow_interruptions=True)
            return

        # Otherwise continue the step flow
        reply = await process_wellness_turn(session, user_text)
        # process_wellness_turn may return None if it asked a clarifying question and expects next input
        if reply:
            await session.say(reply, allow_interruptions=True)

    @session.on("user_input_transcribed")
    def on_user_input_transcribed(event: UserInputTranscribedEvent):
        try:
            asyncio.create_task(handle_transcription(event))
        except RuntimeError:
            loop = None
            try:
                loop = asyncio.get_event_loop()
            except Exception:
                loop = None
            if loop and loop.is_running():
                loop.create_task(handle_transcription(event))
            else:
                asyncio.run(handle_transcription(event))

    # Start the session with the wellness assistant
    await session.start(
        agent=WellnessAssistant(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    # Initialize state and greet the user with context from previous check-ins
    init_wellness_state(session)
    past = load_log()
    context = get_recent_context(past)

    if context and context["last_entry"]:
        last = context["last_entry"]
        greet = (
            f"Welcome back! Last time you were feeling {last.get('mood', 'uncertain')} "
            f"with {last.get('energy', 'moderate')} energy. How's today comparing?"
        )
    else:
        greet = "Hi, I'm CalmCompanion. Let's do a quick check-in. How are you feeling today?"

    await session.say(greet, allow_interruptions=True)

    # connect to signalling (blocks until shutdown)
    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
