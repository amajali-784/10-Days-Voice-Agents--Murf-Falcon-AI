# day4_teach_the_tutor.py
"""
Day 4 — Teach-the-Tutor: Active Recall Coach (robust, single-file)

Modes:
 - learn      -> explain a concept (Murf Falcon voice: Matthew -> "en-US-matthew")
 - quiz       -> ask a question (Murf Falcon voice: Alicia  -> "en-US-alicia")
 - teach_back -> ask user to explain back and give feedback (Murf Falcon voice: Ken    -> "en-US-ken")

Creates:
 - shared-data/day4_tutor_content.json
 - shared-data/day4_mastery.json

Run like your other LiveKit examples.
"""
import asyncio
import json
import logging
import os
import re
from typing import Optional, Any
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

logger = logging.getLogger("day4")
logging.basicConfig(level=logging.INFO)

load_dotenv(".env.local")


class TutorAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="""
You are TutorCompanion, an active-recall voice tutor.
Three modes: learn, quiz, teach_back. Be concise and encouraging.
"""
        )


def prewarm(proc: JobProcess):
    # Preload a VAD once per process to reduce cold start time
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    ctx.log_context_fields = {"room": ctx.room.name}

    # AgentSession: default TTS voice = Matthew (learn)
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
        logger.info("Usage: %s", summary)

    ctx.add_shutdown_callback(log_usage)

    # --- file paths ---
    BASE_DIR = os.path.dirname(__file__)
    CONTENT_DIR = os.path.join(BASE_DIR, "shared-data")
    os.makedirs(CONTENT_DIR, exist_ok=True)
    TUTOR_CONTENT_FILE = os.path.join(CONTENT_DIR, "day4_tutor_content.json")
    MASTERY_FILE = os.path.join(CONTENT_DIR, "day4_mastery.json")

    # --- create sample content if missing ---
    if not os.path.exists(TUTOR_CONTENT_FILE):
        sample_content = [
            {
                "id": "variables",
                "title": "Variables",
                "summary": "Variables store values so you can reuse them later. They have names and can hold different data types.",
                "sample_question": "What is a variable and why is it useful?",
                "keywords": ["value", "name", "store", "reuse", "variable", "data", "type"],
            },
            {
                "id": "loops",
                "title": "Loops",
                "summary": "Loops let you repeat an action multiple times. For loops iterate a fixed number of times; while loops continue while a condition holds.",
                "sample_question": "Explain the difference between a for loop and a while loop.",
                "keywords": ["repeat", "for", "while", "condition", "iteration", "loop", "count"],
            },
        ]
        with open(TUTOR_CONTENT_FILE, "w", encoding="utf-8") as fh:
            json.dump(sample_content, fh, indent=2, ensure_ascii=False)
        logger.info("Created sample tutor content")

    def load_tutor_content():
        try:
            with open(TUTOR_CONTENT_FILE, "r", encoding="utf-8") as fh:
                return {c["id"]: c for c in json.load(fh)}
        except Exception:
            logger.exception("Failed to load tutor content")
            return {}

    tutor_content = load_tutor_content()

    # --- mastery persistence helpers ---
    def load_mastery():
        if not os.path.exists(MASTERY_FILE):
            return {}
        try:
            with open(MASTERY_FILE, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except Exception:
            logger.exception("Failed to load mastery")
            return {}

    def save_mastery(m: dict):
        try:
            with open(MASTERY_FILE, "w", encoding="utf-8") as fh:
                json.dump(m, fh, indent=2, ensure_ascii=False)
            logger.debug("Saved mastery")
        except Exception:
            logger.exception("Failed to save mastery")

    persisted_mastery = load_mastery()

    # --- state helpers ---
    def init_tutor_state(s):
        ts = {
            "mode": None,  # 'learn' | 'quiz' | 'teach_back'
            "current_concept": None,
            "awaiting_response": False,
            "mastery": persisted_mastery.copy() if persisted_mastery else {},
        }
        setattr(s, "tutor_state", ts)
        return ts

    def set_mode(s, mode: Optional[str]):
        ts = getattr(s, "tutor_state", None) or init_tutor_state(s)
        ts["mode"] = mode
        ts["awaiting_response"] = False
        return ts

    def choose_concept(s, cid: str):
        ts = getattr(s, "tutor_state", None) or init_tutor_state(s)
        if cid not in tutor_content:
            return False
        ts["current_concept"] = cid
        ts["mastery"].setdefault(cid, {
            "times_explained": 0,
            "times_quizzed": 0,
            "times_taught_back": 0,
            "last_score": None,
            "avg_score": None,
            "updated_at": None,
        })
        return True

    def summary_mastery(ts):
        parts = []
        for cid, st in ts.get("mastery", {}).items():
            title = tutor_content.get(cid, {}).get("title", cid)
            avg = st.get("avg_score")
            parts.append(f"{title} ({cid}): avg {avg if avg is not None else 'N/A'}")
        return " — ".join(parts) if parts else "No mastery data yet."

    # --- simple keyword-based fallback evaluator ---
    import re as _re

    def keyword_evaluator(text: str, concept: dict):
        words = set(_re.findall(r"\w+", (text or "").lower()))
        keywords = set((concept.get("keywords") or []) + _re.findall(r"\w+", concept.get("summary", "").lower()))
        if not keywords:
            sc = min(100, max(0, len(words) * 2))
            return {"score": sc, "feedback": "Good attempt — try to include more concept terms."}
        overlap = words.intersection(keywords)
        frac = len(overlap) / max(1, len(keywords))
        score = int(round(frac * 100))
        if score > 85:
            fb = "Nice — you covered the main ideas clearly."
        elif score > 50:
            fb = "Solid explanation. Try to include one or two more core terms."
        elif score > 20:
            fb = "A start — aim to mention the key concepts more explicitly."
        else:
            fb = "I couldn't find core terms — include the main ideas from the summary."
        return {"score": score, "feedback": fb, "matched": list(overlap)}

    # --- LLM-based evaluator with robust invocation & fallback ---
    async def llm_evaluate(user_text: str, concept: dict, timeout_s: float = 6.0) -> dict:
        sys_prompt = (
            "You are an automated rater that scores short student explanations from 0 to 100. "
            "Return only a JSON object with numeric field 'score' (0-100) and a short 'feedback' string (1-2 sentences). "
            "Be concise and objective; base the score on how well the explanation covers the core idea in the concept summary."
        )
        user_prompt = (
            f"Concept summary: {concept.get('summary')}\n\n"
            f"Student explanation: {user_text}\n\n"
            "Return JSON: {\"score\": <0-100>, \"feedback\": \"...\"}"
        )

        # Try to call the LLM wrapper in a few common ways
        try:
            # look for common callables (generate, create, __call__)
            llm = session.llm

            # prefer generate()
            gen_call = None
            if hasattr(llm, "generate"):
                gen_call = lambda: llm.generate(
                    messages=[{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_prompt}],
                    max_output_tokens=120,
                )
            elif hasattr(llm, "create"):
                gen_call = lambda: llm.create(
                    messages=[{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_prompt}],
                    max_output_tokens=120,
                )
            elif callable(llm):
                # some wrappers let you call the object
                gen_call = lambda: llm(
                    messages=[{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_prompt}],
                    max_output_tokens=120,
                )

            if gen_call is not None:
                task = gen_call()
                res = await asyncio.wait_for(task, timeout=timeout_s)
            else:
                # no known entrypoint — skip LLM
                raise RuntimeError("No LLM entrypoint (generate/create/call) available on session.llm")

            # extract text content from typical response shapes
            text = None
            try:
                # common patterns:
                # - res.output_text
                # - res.text
                # - res.generations (list)
                # - str(res)
                if hasattr(res, "output_text"):
                    text = getattr(res, "output_text")
                elif hasattr(res, "text"):
                    text = getattr(res, "text")
                elif hasattr(res, "generations"):
                    g = getattr(res, "generations")
                    # generations may be nested
                    if isinstance(g, list) and g:
                        first = g[0]
                        if isinstance(first, list) and first:
                            cand = first[0]
                        else:
                            cand = first
                        text = getattr(cand, "text", None) or getattr(cand, "content", None) or str(cand)
                if not text:
                    # fallback to string representation
                    text = str(res)
            except Exception:
                text = str(res)

            # extract JSON substring
            m = re.search(r"(\{[\s\S]*\})", text)
            if m:
                jtxt = m.group(1)
                try:
                    parsed = json.loads(jtxt)
                    score = int(parsed.get("score", 0))
                    feedback = str(parsed.get("feedback", "")).strip()
                    return {"score": max(0, min(100, score)), "feedback": feedback or "Good effort — keep practicing."}
                except Exception:
                    logger.debug("LLM JSON parse failed; text was: %s", text)

            # heuristic: find first 0-100 integer
            mscore = re.search(r"\b([0-9]{1,3})\b", text)
            if mscore:
                score = int(mscore.group(1))
                # clamp
                score = max(0, min(100, score))
                feedback = text.strip()
                return {"score": score, "feedback": feedback}
        except asyncio.TimeoutError:
            logger.debug("LLM evaluation timed out; falling back to keyword evaluator")
        except Exception:
            logger.exception("LLM evaluation failed; falling back to keyword evaluator")

        # fallback to keyword evaluator
        return keyword_evaluator(user_text, concept)

    # --- safe speak helper (prefer voice kwarg, else default) ---
    async def speak(text: str, voice_name: Optional[str] = None, allow_interruptions: bool = True):
        if not voice_name:
            try:
                await session.say(text, allow_interruptions=allow_interruptions)
            except Exception:
                logger.exception("session.say(default) failed")
            return
        # try voice kwarg if supported by wrapper
        try:
            await session.say(text, voice=voice_name, allow_interruptions=allow_interruptions)
            return
        except TypeError:
            # wrapper doesn't accept voice param
            pass
        except Exception:
            logger.debug("session.say(voice=...) failed; falling back", exc_info=True)
        # fallback
        try:
            await session.say(text, allow_interruptions=allow_interruptions)
        except Exception:
            logger.exception("Fallback session.say failed")

    # --- mode flows ---
    async def mode_learn_flow(s):
        ts = getattr(s, "tutor_state", None) or init_tutor_state(s)
        cid = ts.get("current_concept")
        if not cid:
            await speak("No concept selected. Say 'list concepts' then 'choose <concept_id>'.", "en-US-matthew")
            return
        concept = tutor_content.get(cid)
        if not concept:
            await speak("I couldn't find that concept.", "en-US-matthew")
            return
        stats = ts["mastery"].setdefault(cid, {})
        stats["times_explained"] = stats.get("times_explained", 0) + 1
        stats["updated_at"] = datetime.now(timezone.utc).isoformat()
        save_mastery(ts["mastery"])
        text = f"{concept.get('title')}. {concept.get('summary')}"
        await speak(text, "en-US-matthew")
        ts["awaiting_response"] = False

    async def mode_quiz_flow(s):
        ts = getattr(s, "tutor_state", None) or init_tutor_state(s)
        cid = ts.get("current_concept")
        if not cid:
            await speak("No concept selected for quiz. Say 'list concepts' then 'choose <concept_id>'.", "en-US-alicia")
            return
        concept = tutor_content.get(cid)
        if not concept:
            await speak("I couldn't find that concept.", "en-US-alicia")
            return
        stats = ts["mastery"].setdefault(cid, {})
        stats["times_quizzed"] = stats.get("times_quizzed", 0) + 1
        stats["updated_at"] = datetime.now(timezone.utc).isoformat()
        save_mastery(ts["mastery"])
        question = concept.get("sample_question", "Explain this concept briefly.")
        await speak(question, "en-US-alicia")
        ts["awaiting_response"] = True

    async def mode_teach_back_prompt(s):
        ts = getattr(s, "tutor_state", None) or init_tutor_state(s)
        cid = ts.get("current_concept")
        if not cid:
            await speak("No concept selected. Say 'list concepts' then 'choose <concept_id>'.", "en-US-ken")
            return
        concept = tutor_content.get(cid)
        if not concept:
            await speak("I couldn't find that concept.", "en-US-ken")
            return
        stats = ts["mastery"].setdefault(cid, {})
        stats["times_taught_back"] = stats.get("times_taught_back", 0) + 1
        stats["updated_at"] = datetime.now(timezone.utc).isoformat()
        save_mastery(ts["mastery"])
        prompt = f"Please explain {concept.get('title')} back to me in your own words."
        await speak(prompt, "en-US-ken")
        ts["awaiting_response"] = True

    # --- transcription handler ---
    async def handle_transcription(event: UserInputTranscribedEvent):
        if not getattr(event, "is_final", True):
            return
        user_text = getattr(event, "transcript", "") or ""
        user_text = user_text.strip()
        if not user_text:
            return
        logger.info("[user] %s", user_text)

        lower = user_text.lower()
        ts = getattr(session, "tutor_state", None) or init_tutor_state(session)

        # mode switch
        m = re.search(r"\b(?:mode|switch to|go to)\s+(learn|quiz|teach[_\s-]*back|teachback|teach back)\b", lower)
        if m:
            raw = m.group(1).replace(" ", "_").replace("-", "_")
            chosen = "teach_back" if "teach" in raw else raw
            set_mode(session, chosen)
            if chosen == "learn":
                await speak("Switched to Learn mode. Say 'list concepts' or 'choose <concept_id>'.", "en-US-matthew")
            elif chosen == "quiz":
                await speak("Switched to Quiz mode. Say 'list concepts' or 'choose <concept_id>'.", "en-US-alicia")
            else:
                await speak("Switched to Teach Back mode. Say 'list concepts' or 'choose <concept_id>'.", "en-US-ken")
            return

        # basic commands
        if lower in ("enter tutor", "start tutor", "open tutor", "tutor"):
            await speak("Welcome to Teach-the-Tutor. Which mode would you like: learn, quiz, or teach back?", "en-US-matthew")
            return

        if lower in ("list concepts", "show concepts"):
            if not tutor_content:
                await speak("No tutor content available.", "en-US-matthew")
            else:
                lines = [f"{c['id']}: {c.get('title')}" for c in tutor_content.values()]
                await speak("Available concepts: " + " ; ".join(lines), "en-US-matthew")
            return

        choose = re.search(r"\bchoose\s+([a-z0-9_\-]+)\b", lower)
        if choose:
            cid = choose.group(1)
            if cid in tutor_content:
                choose_concept(session, cid)
                await speak(f"Selected {tutor_content[cid]['title']}. Now say 'mode learn', 'mode quiz', or 'mode teach back' to begin.", "en-US-matthew")
            else:
                await speak("I couldn't find that concept id. Say 'list concepts' to hear options.", "en-US-matthew")
            return

        if lower in ("show mastery", "mastery", "show my mastery"):
            await speak(summary_mastery(ts), "en-US-matthew")
            return

        if lower in ("exit tutor", "leave tutor", "done", "stop"):
            set_mode(session, None)
            ts["awaiting_response"] = False
            await speak("Exited tutor mode. Say 'enter tutor' to come back.", "en-US-matthew")
            return

        # If waiting for an answer (quiz or teach_back), evaluate it
        if ts.get("awaiting_response") and ts.get("mode") and ts.get("current_concept"):
            cid = ts["current_concept"]
            concept = tutor_content.get(cid)
            if not concept:
                await speak("No concept selected to evaluate.", "en-US-matthew")
                ts["awaiting_response"] = False
                return

            # LLM-based scoring with fallback to keyword_evaluator
            eval_result = await llm_evaluate(user_text, concept)
            stats = ts["mastery"].setdefault(cid, {
                "times_explained": 0,
                "times_quizzed": 0,
                "times_taught_back": 0,
                "last_score": None,
                "avg_score": None,
                "updated_at": None,
            })

            if ts["mode"] == "quiz":
                stats["times_quizzed"] = stats.get("times_quizzed", 0) + 1
                prev_avg = stats.get("avg_score") or 0
                count = stats["times_quizzed"]
                stats["last_score"] = eval_result["score"]
                stats["avg_score"] = round(((prev_avg * (count - 1)) + eval_result["score"]) / count, 1)
                stats["updated_at"] = datetime.now(timezone.utc).isoformat()
                await speak(f"I scored your answer {eval_result['score']} out of 100. {eval_result['feedback']}", "en-US-alicia")
            elif ts["mode"] == "teach_back":
                stats["times_taught_back"] = stats.get("times_taught_back", 0) + 1
                prev_avg = stats.get("avg_score") or 0
                count = stats["times_taught_back"]
                stats["last_score"] = eval_result["score"]
                stats["avg_score"] = round(((prev_avg * (count - 1)) + eval_result["score"]) / count, 1)
                stats["updated_at"] = datetime.now(timezone.utc).isoformat()
                await speak(f"I gave that a {eval_result['score']} out of 100. {eval_result['feedback']}", "en-US-ken")
            else:
                # unexpected: treat like general evaluation
                stats["last_score"] = eval_result["score"]
                stats["avg_score"] = eval_result["score"]
                await speak(f"I scored that {eval_result['score']} out of 100. {eval_result['feedback']}", "en-US-matthew")

            # persist mastery
            save_mastery(ts["mastery"])
            ts["awaiting_response"] = False
            return

        # If in a mode but not awaiting a response, start the mode's flow
        if ts.get("mode"):
            if ts["mode"] == "learn":
                await mode_learn_flow(session)
                return
            if ts["mode"] == "quiz":
                await mode_quiz_flow(session)
                return
            if ts["mode"] == "teach_back":
                await mode_teach_back_prompt(session)
                return

        # fallback help
        await speak(
            "I didn't understand that. Say 'enter tutor', 'list concepts', 'choose <concept_id>', 'mode learn/quiz/teach back', or 'show mastery'.",
            "en-US-matthew",
        )

    @session.on("user_input_transcribed")
    def on_user_input_transcribed(event: UserInputTranscribedEvent):
        # schedule transcription handler safely
        try:
            asyncio.create_task(handle_transcription(event))
        except RuntimeError:
            # fallback when loop isn't running
            loop = None
            try:
                loop = asyncio.get_event_loop()
            except Exception:
                loop = None
            if loop and loop.is_running():
                loop.create_task(handle_transcription(event))
            else:
                asyncio.run(handle_transcription(event))

    # initialize tutor state BEFORE starting the session to avoid race
    init_tutor_state(session)

    # Start session (wrap in try/except to surface errors)
    try:
        await session.start(
            agent=TutorAgent(),
            room=ctx.room,
            room_input_options=RoomInputOptions(
                noise_cancellation=noise_cancellation.BVC(),
            ),
        )
    except Exception:
        logger.exception("session.start failed — check LiveKit worker/job assignment and network")

    # greet user
    greeting = (
        "Hi — I'm TutorCompanion. I support three modes: learn (explain), quiz (ask), and teach back (you explain). "
        "Say 'enter tutor' to begin, or 'list concepts' to hear available topics."
    )
    await speak(greeting, "en-US-matthew")

    # block until shutdown
    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
