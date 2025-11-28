# day7_voice_agent_fixed.py
import json
import logging
import os
import uuid
from datetime import datetime
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
    function_tool,
    RunContext,
)
from livekit.plugins import murf, silero, google, deepgram, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("day7_agent")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

load_dotenv(".env.local")

# Filenames (change if you want)
CATALOG_FILE = "catalog.json"
ORDERS_FILE = "orders.json"

# Sample catalog used for Day 7 — created automatically if missing
SAMPLE_CATALOG = [
    {"id": "bread_wholewheat", "name": "Whole Wheat Bread - 400g", "category": "Groceries", "price": 40, "tags": ["vegan"]},
    {"id": "milk_1l", "name": "Milk 1L (pack)", "category": "Groceries", "price": 55, "tags": ["dairy"]},
    {"id": "eggs_12", "name": "Eggs - 12 pcs", "category": "Groceries", "price": 110, "tags": ["protein"]},
    {"id": "peanut_butter", "name": "Peanut Butter 350g", "category": "Groceries", "price": 200, "tags": ["vegan", "spread"]},
    {"id": "pasta_500g", "name": "Pasta 500g", "category": "Groceries", "price": 95, "tags": ["vegan"]},
    {"id": "pasta_sauce", "name": "Pasta Sauce 400g", "category": "Groceries", "price": 130, "tags": ["vegan"]},
    {"id": "chips", "name": "Potato Chips 150g", "category": "Snacks", "price": 45, "tags": ["snack"]},
    {"id": "sandwich_ready", "name": "Club Sandwich (ready)", "category": "Prepared Food", "price": 180, "tags": ["prepared"]},
    {"id": "apple_kg", "name": "Apples 1kg", "category": "Groceries", "price": 160, "tags": ["fruit"]},
    {"id": "banana_kg", "name": "Bananas 1kg", "category": "Groceries", "price": 80, "tags": ["fruit"]},
    {"id": "butter_100g", "name": "Butter 100g", "category": "Groceries", "price": 90, "tags": ["dairy"]},
]

def ensure_catalog():
    if not os.path.exists(CATALOG_FILE):
        with open(CATALOG_FILE, "w", encoding="utf-8") as f:
            json.dump(SAMPLE_CATALOG, f, indent=2, ensure_ascii=False)
        logger.info("Created sample catalog.json")
    else:
        logger.debug("catalog.json already exists")

def load_catalog():
    ensure_catalog()
    with open(CATALOG_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def read_orders():
    if not os.path.exists(ORDERS_FILE):
        with open(ORDERS_FILE, "w", encoding="utf-8") as f:
            json.dump([], f)
        return []
    with open(ORDERS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def write_orders(orders):
    with open(ORDERS_FILE, "w", encoding="utf-8") as f:
        json.dump(orders, f, indent=2, ensure_ascii=False)

class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="""You are a helpful food & grocery ordering voice assistant.
You can read and call the available tools to manage a catalog, add/remove items to a cart, list the cart, map recipe-style requests to items, and place an order which is persisted to disk as JSON.
Keep responses concise and conversational. Confirm cart changes and final order totals when placing an order.""",
        )
        # simple in-memory carts keyed by room (persisted only when order placed)
        self.carts = {}  # { room_name: {item_id: qty, ...} }

    # Utility to get room key from RunContext or JobContext
    def _room_key(self, ctx: RunContext):
        # Prefer room name if present, else fallback to "default_room"
        try:
            if ctx is None:
                return "default_room"
            room = getattr(ctx, "room", None)
            if room is None:
                # try for ctx.job.room or similar shapes
                maybe_room = getattr(ctx, "job", None)
                if maybe_room:
                    return str(getattr(maybe_room, "room", "default_room"))
                return "default_room"
            return str(getattr(room, "name", "default_room"))
        except Exception:
            return "default_room"

    @function_tool
    async def list_catalog(self, context: RunContext):
        """
        Return a short list of catalog entries (id, name, price).
        """
        cat = load_catalog()
        lines = [f"{item['id']}: {item['name']} - {item['price']}" for item in cat]
        return "\n".join(lines)

    @function_tool
    async def add_to_cart(self, context: RunContext, item_id: str, qty: int = 1):
        """
        Add an item to the cart for this room.
        """
        catalog = load_catalog()
        item_map = {it["id"]: it for it in catalog}
        if item_id not in item_map:
            return f"Item {item_id} not found in catalog."
        room = self._room_key(context)
        cart = self.carts.setdefault(room, {})
        cart[item_id] = cart.get(item_id, 0) + max(1, int(qty))
        item = item_map[item_id]
        return f"Added {cart[item_id]} x {item['name']} to your cart."

    @function_tool
    async def remove_from_cart(self, context: RunContext, item_id: str, qty: int = None):
        """
        Remove qty of an item from the cart, or remove entirely if qty is None.
        """
        room = self._room_key(context)
        cart = self.carts.setdefault(room, {})
        if item_id not in cart:
            return f"{item_id} is not in your cart."
        if qty is None:
            del cart[item_id]
            return f"Removed {item_id} from your cart."
        else:
            qty = int(qty)
            if qty >= cart[item_id]:
                del cart[item_id]
                return f"Removed {item_id} from your cart."
            else:
                cart[item_id] -= qty
                return f"Decreased {item_id} quantity to {cart[item_id]}."

    @function_tool
    async def show_cart(self, context: RunContext):
        """
        Show the current cart contents with totals.
        """
        catalog = load_catalog()
        item_map = {it["id"]: it for it in catalog}
        room = self._room_key(context)
        cart = self.carts.get(room, {})
        if not cart:
            return "Your cart is empty."
        lines = []
        total = 0.0
        for iid, q in cart.items():
            item = item_map.get(iid, {"name": iid, "price": 0})
            line_total = item.get("price", 0) * q
            total += line_total
            lines.append(f"{q} x {item.get('name')} - {item.get('price')} each => {line_total}")
        lines.append(f"Total: {total}")
        return "\n".join(lines)

    @function_tool
    async def place_order(self, context: RunContext, customer_name: str = None, customer_address: str = None):
        """
        Place the current cart as an order, persist to orders.json, clear the cart, and return order summary.
        """
        catalog = load_catalog()
        item_map = {it["id"]: it for it in catalog}
        room = self._room_key(context)
        cart = self.carts.get(room, {})
        if not cart:
            return "Cannot place order: your cart is empty."
        order_items = []
        total = 0.0
        for iid, qty in cart.items():
            item = item_map.get(iid, {"name": iid, "price": 0})
            price = item.get("price", 0)
            order_items.append({"id": iid, "name": item.get("name"), "qty": qty, "unit_price": price, "line_total": price * qty})
            total += price * qty

        order = {
            "order_id": str(uuid.uuid4()),
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "room": room,
            "customer": {"name": customer_name or "anonymous", "address": customer_address or ""},
            "items": order_items,
            "total": total,
            "status": "received",
        }

        orders = read_orders()
        orders.append(order)
        write_orders(orders)

        # clear cart
        self.carts[room] = {}

        return f"Order placed. ID {order['order_id']}. Total {total}. I saved it to orders.json and set status to received."

    @function_tool
    async def ingredients_for(self, context: RunContext, dish: str):
        """
        Map a dish name to a small list of items and add them to the cart.
        This is a simple hard-coded mapping for demo purposes.
        """
        # small recipe mapping (dish -> list of item_ids and quantities)
        recipes = {
            "peanut butter sandwich": [("bread_wholewheat", 2), ("peanut_butter", 1)],
            "pasta for two": [("pasta_500g", 1), ("pasta_sauce", 1)],
            "fruit pack": [("apple_kg", 1), ("banana_kg", 1)],
            "simple breakfast": [("bread_wholewheat", 2), ("butter_100g", 1), ("milk_1l", 1)],
        }
        dish_key = dish.strip().lower()
        if dish_key not in recipes:
            return f"I don't have a recipe for {dish}. Try something like 'peanut butter sandwich' or 'pasta for two'."

        added = []
        for iid, qty in recipes[dish_key]:
            # call add_to_cart logic
            await self.add_to_cart(context, iid, qty)
            added.append(f"{qty} x {iid}")
        return f"Added ingredients for {dish}: {', '.join(added)}."

    @function_tool
    async def list_orders(self, context: RunContext, last_n: int = 5):
        """
        Return the most recent N orders from orders.json (global history).
        """
        orders = read_orders()
        if not orders:
            return "No orders found."
        last = orders[-int(last_n):]
        lines = []
        for o in last:
            lines.append(f"{o['order_id']} - {o['timestamp']} - {o['total']} - {o['status']}")
        return "\n".join(lines)

# Prewarm to load VAD model into process userdata
def prewarm(proc: JobProcess):
    try:
        proc.userdata["vad"] = silero.VAD.load()
    except Exception as e:
        logger.warning("Failed to warm VAD model: %s", e)
    ensure_catalog()
    # ensure orders file exists
    if not os.path.exists(ORDERS_FILE):
        with open(ORDERS_FILE, "w", encoding="utf-8") as f:
            json.dump([], f)
    logger.info("Prewarm complete")

async def entrypoint(ctx: JobContext):
    # Logging context fields for clarity (guarded)
    try:
        room_name = getattr(ctx, "room", None)
        if room_name is not None:
            ctx.log_context_fields = {"room": getattr(ctx.room, "name", str(ctx.room))}
    except Exception:
        # not critical; continue
        pass

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
        vad=ctx.proc.userdata.get("vad"),
        preemptive_generation=True,
    )

    # Metrics collection
    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage summary: {summary}")

    ctx.add_shutdown_callback(log_usage)

    # Start the session with the Assistant that exposes the ordering tools
    await session.start(
        agent=Assistant(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    await ctx.connect()

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
