import logging
import json
import random
import asyncio
from dotenv import load_dotenv

from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    JobProcess as Proc,
    JobContext as JCtx,
    JobProcess,
    MetricsCollectedEvent,
    RoomInputOptions,
    WorkerOptions,
    cli,
    metrics,
    tokenize,
    function_tool,
    RunContext,
)
from livekit.plugins import murf, silero, google, deepgram, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("agent")
load_dotenv(".env.local")


class Assistant(Agent):
    """
    Game Master assistant persona:
    - Runs a single-universe D&D-style voice adventure.
    - Keeps responses concise and always ends with a player prompt ("What do you do?").
    - When a randomness/skill check is required, it should call the provided `roll_dice` tool.
    - The Python-side tools maintain a JSON world state (characters, player, inventory, events).
    """

    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are a Game Master (GM) running a single-player D&D-style voice adventure. "
                "Universe: Classic Fantasy. Tone: adventurous with light humor. "
                "Your job: describe scenes concisely, track continuity, reference the JSON world state "
                "managed by tools, and always end each reply with a direct question prompting the player (e.g., "
                "'What do you do?').\n\n"
                "When a skill check or uncertain outcome is needed, call the tool `roll_dice(sides:int, modifier:int)` "
                "which returns a numeric roll and logs the event to the world state. Use the tool results to decide outcomes. "
                "Do not call external web services. Keep messages short and voice-friendly."
            )
        )


# Tools exposed to the LLM / agent runtime -------------------------------------------------
@function_tool
async def roll_dice(context: RunContext, sides: int = 20, modifier: int = 0):
    """
    Roll a dice (1..sides), apply modifier, record to the session world_state, and return a short result string.
    This tool is intended to be called by the LLM when it needs to resolve an uncertain outcome.

    Args:
        context: RunContext (gives access to the current process/session userdata)
        sides: number of faces on the die (default 20)
        modifier: integer modifier to add to the roll
    Returns:
        dict with roll, total, detail message (tools return serializable JSON-friendly values)
    """
    proc = context.proc  # JobProcess-like object where userdata is stored
    # Ensure world_state exists
    ws = proc.userdata.setdefault("world_state", {})
    events = ws.setdefault("events", [])

    raw = random.randint(1, max(1, int(sides)))
    total = raw + int(modifier)
    entry = {
        "type": "dice_roll",
        "sides": int(sides),
        "modifier": int(modifier),
        "raw": raw,
        "total": total,
        "by": "gm_tool",
        "timestamp": context.now_iso() if hasattr(context, "now_iso") else None,
    }
    events.append(entry)
    proc.userdata["world_state"] = ws

    # A short, LLM-friendly tool output
    return {
        "raw": raw,
        "total": total,
        "detail": f"Rolled 1d{sides} => {raw} + {modifier} = {total}",
    }


@function_tool
async def save_world_state(context: RunContext):
    """
    Return the current world_state as a JSON string so the GM or caller can save it externally.
    """
    proc = context.proc
    ws = proc.userdata.get("world_state", {})
    return json.dumps(ws, indent=2)


@function_tool
async def load_world_state(context: RunContext, state_json: str):
    """
    Load/replace the current world_state from a JSON string. Use to 'resume' a saved game.
    """
    proc = context.proc
    try:
        ws = json.loads(state_json)
    except Exception as e:
        return {"error": f"failed to parse json: {e}"}
    proc.userdata["world_state"] = ws
    return {"ok": True}


# Prewarm loads the VAD model once per process ----------------------------------------------
def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()
    # initialize an empty world state template if not already present
    ws = proc.userdata.setdefault(
        "world_state",
        {
            "universe": "Classic Fantasy",
            "player": {
                "name": "Adventurer",
                "hp": 10,
                "status": "Healthy",
                "attributes": {"strength": 2, "dexterity": 1, "intelligence": 1},
                "inventory": ["torch", "rusty dagger"],
            },
            "npcs": {},
            "location": {
                "name": "Village Edge",
                "description": "A dusty road with the silhouette of a ruined watchtower to the north.",
            },
            "events": [],
            "turn_count": 0,
        },
    )


# Entrypoint: constructs the voice pipeline + GM session ------------------------------------
async def entrypoint(ctx: JobContext):
    # add room name to logs
    ctx.log_context_fields = {"room": ctx.room.name}

    # Build the session: STT, LLM, TTS, VAD, and turn detector
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

    # Metrics handling (unchanged)
    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: {summary}")

    ctx.add_shutdown_callback(log_usage)

    # Ensure world state exists in process userdata (prewarm should have set it, but double-check)
    ws = ctx.proc.userdata.setdefault(
        "world_state",
        {
            "universe": "Classic Fantasy",
            "player": {
                "name": "Adventurer",
                "hp": 10,
                "status": "Healthy",
                "attributes": {"strength": 2, "dexterity": 1, "intelligence": 1},
                "inventory": ["torch", "rusty dagger"],
            },
            "npcs": {},
            "location": {
                "name": "Village Edge",
                "description": "A dusty road with the silhouette of a ruined watchtower to the north.",
            },
            "events": [],
            "turn_count": 0,
        },
    )

    # Provide a small helper for the LLM to inspect world_state when the session starts.
    # Many LLMs can't "call" Python directly unless via tools; we rely on the tools + instructions.
    # Start the session with the GM assistant persona
    await session.start(
        agent=Assistant(),
        room=ctx.room,
        room_input_options=RoomInputOptions(noise_cancellation=noise_cancellation.BVC()),
    )

    # Connect to the user's room (joins and starts relaying audio)
    await ctx.connect()

    # Optional: Announce the initial scene using a short system-driven message sent as GM.
    # We send a system message as the assistant's first message to kick off the adventure.
    initial_scene = (
        "You find yourself on the Village Edge at dusk. A thin moon hangs over a crooked watchtower to the north. "
        "There are hoofprints leading down the road and the wind carries the smell of smoke. "
        "A child named Mira waves from the path and calls for help — something is wrong at her family's farm. "
        "What do you do?"
    )

    # We can send an assistant message into the session to ensure the player hears the opening scene.
    # Many livekit AgentSession implementations provide a method to send messages / speak as the agent.
    # The following uses an abstraction `session.speak` if available; otherwise sending via session.notify may vary.
    # We'll attempt to use session.speak(), catching if it doesn't exist.
    try:
        await session.speak(initial_scene)
    except Exception:
        # Fallback: if speak isn't available, log and rely on the LLM to produce the first prompt on player join.
        logger.info("session.speak not available; initial scene logged and world_state seeded.")
        logger.info(initial_scene)

    # Keep the entrypoint alive while connected. Typical pattern: wait until disconnect event.
    try:
        while not ctx.should_stop:
            # Increment a simple turn counter occasionally (used for metrics/world pacing).
            ws = ctx.proc.userdata.get("world_state", {})
            ws["turn_count"] = ws.get("turn_count", 0)
            await asyncio.sleep(1)
    except asyncio.CancelledError:
        logger.info("Entrypoint cancelled, shutting down voice GM.")


if __name__ == "__main__":
    # Run the worker with our entrypoint and prewarm functions
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
