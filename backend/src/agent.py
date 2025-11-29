# agent_ecommerce.py
import logging
import json
import uuid
import sys
from datetime import datetime
from typing import List, Optional, Dict, Any

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

logger = logging.getLogger("agent")
logging.basicConfig(level=logging.INFO)
load_dotenv(".env.local")

# Optional FastAPI imports (lazy)
HAS_FASTAPI = True
try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
except Exception:
    HAS_FASTAPI = False

# ----------------------------
# Catalog + persistence
# ----------------------------

PRODUCTS = [
    {
        "id": "mug-001",
        "name": "Stoneware Coffee Mug",
        "description": "12oz stoneware mug, dishwasher safe",
        "price": 800,
        "currency": "INR",
        "category": "mug",
        "color": "white",
        "sizes": None,
    },
    {
        "id": "hoodie-001",
        "name": "Classic Hoodie",
        "description": "Cotton blend hoodie, unisex",
        "price": 1800,
        "currency": "INR",
        "category": "apparel",
        "color": "black",
        "sizes": ["S", "M", "L", "XL"],
    },
    {
        "id": "tee-001",
        "name": "Graphic T-Shirt",
        "description": "Printed tee, 100% cotton",
        "price": 599,
        "currency": "INR",
        "category": "apparel",
        "color": "blue",
        "sizes": ["S", "M", "L"],
    },
    {
        "id": "mug-002",
        "name": "Blue Ceramic Mug",
        "description": "11oz ceramic mug in blue",
        "price": 750,
        "currency": "INR",
        "category": "mug",
        "color": "blue",
        "sizes": None,
    },
]

ORDERS_FILE = "orders.json"

# Process-scoped caches (simple, not distributed)
LAST_SHOWN_PRODUCTS: List[Dict[str, Any]] = []

def _load_orders() -> List[Dict[str, Any]]:
    try:
        with open(ORDERS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return []
    except Exception as e:
        logger.exception("Failed to load orders.json: %s", e)
        return []

def _append_order(order: Dict[str, Any]):
    try:
        orders = _load_orders()
        orders.append(order)
        with open(ORDERS_FILE, "w", encoding="utf-8") as f:
            json.dump(orders, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.exception("Failed to append order: %s", e)

def _find_product(pid: str) -> Optional[Dict[str, Any]]:
    for p in PRODUCTS:
        if p["id"] == pid:
            return p
    return None

# ----------------------------
# Optional FastAPI ACP-style API
# ----------------------------
if HAS_FASTAPI:
    app = FastAPI(title="Lite ACP Merchant (Day 9)")

    class OrderItemIn(BaseModel):
        product_id: str
        quantity: int = 1
        attributes: Optional[Dict[str, Any]] = None

    class CreateOrderIn(BaseModel):
        items: List[OrderItemIn]
        buyer: Optional[Dict[str, str]] = None

    @app.get("/acp/catalog")
    def acp_catalog():
        return {"products": PRODUCTS}

    @app.post("/acp/orders")
    def acp_create_order(payload: CreateOrderIn):
        line_items = []
        total = 0
        for it in payload.items:
            prod = _find_product(it.product_id)
            if not prod:
                raise HTTPException(status_code=400, detail=f"product {it.product_id} not found")
            unit = prod["price"]
            qty = max(1, int(it.quantity))
            total += unit * qty
            line_items.append({
                "product_id": prod["id"],
                "name": prod["name"],
                "quantity": qty,
                "unit_amount": unit,
                "currency": prod["currency"],
                "attributes": it.attributes or {},
            })
        order = {
            "id": str(uuid.uuid4()),
            "items": line_items,
            "total": total,
            "currency": "INR" if line_items else None,
            "buyer": payload.buyer or {},
            "status": "CONFIRMED",
            "created_at": datetime.utcnow().isoformat() + "Z",
        }
        _append_order(order)
        return order
else:
    app = None  # FastAPI not installed

# ----------------------------
# LiveKit Agent + function tools
# ----------------------------

class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are a helpful voice AI shopping assistant. The user is interacting via voice. "
                "Keep responses concise and friendly. Use the available tools to fetch products and create orders. "
                "When the user asks to buy or to browse, call the appropriate tool rather than inventing catalog details."
            )
        )

    # FIX: accept a single filters dict so partial args from the LLM validate cleanly
    @function_tool
    async def list_products(self, ctx: RunContext, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        filters: optional dict with any of:
          - category: str
          - max_price: number
          - color: str
          - text: str  (free-text search on name/description)
        Returns filtered product summaries and updates LAST_SHOWN_PRODUCTS so that
        references like "the second one" can be resolved.
        Defensive: returns [] on error.
        """
        try:
            if filters is None:
                filters = {}
            # extract safely
            category = filters.get("category") if isinstance(filters, dict) else None
            max_price = filters.get("max_price") if isinstance(filters, dict) else None
            color = filters.get("color") if isinstance(filters, dict) else None
            text = filters.get("text") if isinstance(filters, dict) else None

            results = PRODUCTS
            if category:
                try:
                    results = [p for p in results if p["category"].lower() == str(category).lower()]
                except Exception:
                    results = [p for p in results if p["category"] == category]
            if max_price is not None:
                try:
                    maxp = float(max_price)
                    results = [p for p in results if p["price"] <= maxp]
                except Exception:
                    # ignore invalid max_price format
                    pass
            if color:
                results = [p for p in results if p.get("color", "").lower() == str(color).lower()]
            if text:
                txt = str(text).lower()
                results = [p for p in results if txt in p["name"].lower() or txt in p.get("description", "").lower()]

            # Save lightweight summary to process-scoped cache
            global LAST_SHOWN_PRODUCTS
            LAST_SHOWN_PRODUCTS = [
                {
                    "id": p["id"],
                    "name": p["name"],
                    "description": p.get("description"),
                    "price": p["price"],
                    "currency": p["currency"],
                    "category": p["category"],
                    "color": p.get("color"),
                    "sizes": p.get("sizes"),
                }
                for p in results
            ]
            logger.debug("list_products -> filters: %s results: %d", filters, len(LAST_SHOWN_PRODUCTS))
            return LAST_SHOWN_PRODUCTS
        except Exception as e:
            logger.exception("Error in list_products: %s", e)
            return []

    @function_tool
    async def create_order(self, ctx: RunContext, line_items: List[Dict[str, Any]], buyer: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Create and persist an order.
        line_items: [{ "product_id": "...", "quantity": 1, "attributes": {}} ...]
        Returns order object or {"error": "..."} on failure.
        """
        try:
            items = []
            total = 0
            for it in line_items:
                pid = it.get("product_id")
                qty = max(1, int(it.get("quantity", 1)))
                prod = _find_product(pid)
                if not prod:
                    return {"error": f"product {pid} not found"}
                unit = prod["price"]
                items.append({
                    "product_id": prod["id"],
                    "name": prod["name"],
                    "quantity": qty,
                    "unit_amount": unit,
                    "currency": prod["currency"],
                    "attributes": it.get("attributes", {}),
                })
                total += unit * qty

            order = {
                "id": str(uuid.uuid4()),
                "items": items,
                "total": total,
                "currency": items[0]["currency"] if items else "INR",
                "buyer": buyer or {},
                "status": "CONFIRMED",
                "created_at": datetime.utcnow().isoformat() + "Z",
            }
            _append_order(order)
            logger.info("Order created: %s total=%s", order["id"], order["total"])
            return order
        except Exception as e:
            logger.exception("Error in create_order: %s", e)
            return {"error": "internal error creating order"}

    @function_tool
    async def get_last_order(self, ctx: RunContext) -> Optional[Dict[str, Any]]:
        """Return the most recent order (if any)."""
        try:
            orders = _load_orders()
            if not orders:
                return None
            return orders[-1]
        except Exception as e:
            logger.exception("Error in get_last_order: %s", e)
            return None

    @function_tool
    async def resolve_indexed_reference(self, ctx: RunContext, index: int) -> Optional[Dict[str, Any]]:
        """
        index: 1-based index referring to the last list_products() result.
        Returns the product dict or None.
        """
        try:
            if index < 1 or index > len(LAST_SHOWN_PRODUCTS):
                return None
            return LAST_SHOWN_PRODUCTS[index - 1]
        except Exception as e:
            logger.exception("Error in resolve_indexed_reference: %s", e)
            return None

def prewarm(proc: JobProcess):
    # Load VAD model into proc.userdata so entrypoint can access it
    try:
        proc.userdata["vad"] = silero.VAD.load()
    except Exception as e:
        # Log but continue; entrypoint should handle missing vad if necessary
        logger.exception("Failed to prewarm VAD: %s", e)
        proc.userdata["vad"] = None

async def entrypoint(ctx: JobContext):
    ctx.log_context_fields = {"room": ctx.room.name}
    vad = ctx.proc.userdata.get("vad", None)

    # Build AgentSession; if vad is None, still attempt to start (some environments handle it)
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
        vad=vad,
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

    # Start the voice session (agent will be available to join rooms)
    await session.start(
        agent=Assistant(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    await ctx.connect()

# ----------------------------
# Run as script
# ----------------------------
if __name__ == "__main__":
    # If started with "serve-api" arg, run FastAPI for local testing
    if "serve-api" in sys.argv:
        if not HAS_FASTAPI:
            print("\nERROR: FastAPI (and pydantic) are required to run the API server.")
            print("Install with: pip install \"fastapi[all]\" uvicorn\n")
            sys.exit(1)
        import uvicorn
        print("Serving ACP-lite API on http://127.0.0.1:8000")
        uvicorn.run(app, host="127.0.0.1", port=8000)
    else:
        # run the livekit worker (voice agent)
        cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
