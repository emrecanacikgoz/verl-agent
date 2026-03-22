"""
DB Sampler for TOD-Zero (tau-bench) — verl version.

Samples real user/reservation/order data from the tau2-bench database so the
Challenger can generate feasible tasks grounded in actual DB entities.

Ported from the TRL self-play implementation.
"""

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

# verl-agent/tau2-bench/data/tau2/domains/
TAU2_DATA_DIR = Path(__file__).parent.parent.parent.parent.parent / "tau2-bench" / "data" / "tau2" / "domains"


def _load_db(domain: str) -> Dict[str, Any]:
    db_path = TAU2_DATA_DIR / domain / "db.json"
    with open(db_path) as f:
        return json.load(f)


def _load_policy(domain: str) -> str:
    policy_path = TAU2_DATA_DIR / domain / "policy.md"
    with open(policy_path) as f:
        return f.read()


# ── Tool schemas per domain (static, for Challenger prompt) ─────────────────

AIRLINE_TOOLS = [
    {"name": "get_user_details", "type": "READ",
     "description": "Get user details by user id.",
     "parameters": {"user_id": "str"}},
    {"name": "get_reservation_details", "type": "READ",
     "description": "Get reservation details by reservation id.",
     "parameters": {"reservation_id": "str"}},
    {"name": "search_direct_flight", "type": "READ",
     "description": "Search for direct flights between two airports on a given date.",
     "parameters": {"origin": "str", "destination": "str", "date": "str"}},
    {"name": "search_onestop_flight", "type": "READ",
     "description": "Search for one-stop flights between two airports on a given date.",
     "parameters": {"origin": "str", "destination": "str", "date": "str"}},
    {"name": "get_flight_status", "type": "READ",
     "description": "Get the status of a flight on a specific date.",
     "parameters": {"flight_number": "str", "date": "str"}},
    {"name": "list_all_airports", "type": "READ",
     "description": "List all available airports.",
     "parameters": {}},
    {"name": "book_reservation", "type": "WRITE",
     "description": "Book a new reservation.",
     "parameters": {"user_id": "str", "origin": "str", "destination": "str",
                    "flight_type": "str", "cabin": "str", "flights": "list[dict]",
                    "passengers": "list[dict]", "payment_methods": "list[dict]",
                    "total_baggages": "int", "nonfree_baggages": "int", "insurance": "str"}},
    {"name": "cancel_reservation", "type": "WRITE",
     "description": "Cancel an existing reservation.",
     "parameters": {"reservation_id": "str"}},
    {"name": "update_reservation_flights", "type": "WRITE",
     "description": "Update the flights of a reservation (change dates/flights).",
     "parameters": {"reservation_id": "str", "cabin": "str", "flights": "list[dict]", "payment_id": "str"}},
    {"name": "update_reservation_passengers", "type": "WRITE",
     "description": "Update passenger information on a reservation.",
     "parameters": {"reservation_id": "str", "passengers": "list[dict]"}},
    {"name": "update_reservation_baggages", "type": "WRITE",
     "description": "Update baggage information on a reservation.",
     "parameters": {"reservation_id": "str", "total_baggages": "int",
                    "nonfree_baggages": "int", "payment_id": "str"}},
    {"name": "send_certificate", "type": "WRITE",
     "description": "Send a travel certificate to a user.",
     "parameters": {"user_id": "str", "amount": "float"}},
    {"name": "calculate", "type": "GENERIC",
     "description": "Calculate the result of a mathematical expression.",
     "parameters": {"expression": "str"}},
    {"name": "transfer_to_human_agents", "type": "GENERIC",
     "description": "Transfer the user to a human agent with a summary.",
     "parameters": {"summary": "str"}},
]

RETAIL_TOOLS = [
    {"name": "find_user_id_by_name_zip", "type": "READ",
     "description": "Find user id by first name, last name, and zip code.",
     "parameters": {"first_name": "str", "last_name": "str", "zip": "str"}},
    {"name": "find_user_id_by_email", "type": "READ",
     "description": "Find user id by email address.",
     "parameters": {"email": "str"}},
    {"name": "get_user_details", "type": "READ",
     "description": "Get user details by user id.",
     "parameters": {"user_id": "str"}},
    {"name": "get_order_details", "type": "READ",
     "description": "Get order details by order id (e.g. '#W0000000').",
     "parameters": {"order_id": "str"}},
    {"name": "get_product_details", "type": "READ",
     "description": "Get product details by product id.",
     "parameters": {"product_id": "str"}},
    {"name": "list_all_product_types", "type": "READ",
     "description": "List all available product types in the store.",
     "parameters": {}},
    {"name": "cancel_pending_order", "type": "WRITE",
     "description": "Cancel a pending order. Reason must be 'no longer needed' or 'ordered by mistake'.",
     "parameters": {"order_id": "str", "reason": "str"}},
    {"name": "exchange_delivered_order_items", "type": "WRITE",
     "description": "Exchange items in a delivered order for new variants of the same product.",
     "parameters": {"order_id": "str", "item_ids": "list[str]",
                    "new_item_ids": "list[str]", "payment_method_id": "str"}},
    {"name": "modify_pending_order_items", "type": "WRITE",
     "description": "Modify items in a pending order to new variants of the same product.",
     "parameters": {"order_id": "str", "item_ids": "list[str]",
                    "new_item_ids": "list[str]", "payment_method_id": "str"}},
    {"name": "modify_pending_order_address", "type": "WRITE",
     "description": "Modify the shipping address of a pending order.",
     "parameters": {"order_id": "str", "address1": "str", "address2": "str",
                    "city": "str", "state": "str", "country": "str", "zip": "str"}},
    {"name": "modify_pending_order_payment", "type": "WRITE",
     "description": "Modify the payment method of a pending order.",
     "parameters": {"order_id": "str", "payment_method_id": "str"}},
    {"name": "modify_user_address", "type": "WRITE",
     "description": "Modify the default address of a user.",
     "parameters": {"user_id": "str", "address1": "str", "address2": "str",
                    "city": "str", "state": "str", "country": "str", "zip": "str"}},
    {"name": "return_delivered_order_items", "type": "WRITE",
     "description": "Return items from a delivered order.",
     "parameters": {"order_id": "str", "item_ids": "list[str]", "payment_method_id": "str"}},
    {"name": "calculate", "type": "GENERIC",
     "description": "Calculate the result of a mathematical expression.",
     "parameters": {"expression": "str"}},
    {"name": "transfer_to_human_agents", "type": "GENERIC",
     "description": "Transfer the user to a human agent with a summary.",
     "parameters": {"summary": "str"}},
]

DOMAIN_TOOLS: Dict[str, List[Dict]] = {
    "airline": AIRLINE_TOOLS,
    "retail": RETAIL_TOOLS,
}


# ── Sampling functions ────────────────────────────────────────────────────────

def sample_airline_context(db: Dict[str, Any]) -> Dict[str, Any]:
    users = db["users"]
    reservations = db["reservations"]

    user_ids_with_res = [
        uid for uid, u in users.items()
        if u.get("reservations") and len(u["reservations"]) > 0
    ]
    uid = random.choice(user_ids_with_res)
    user = users[uid]
    res_id = random.choice(user["reservations"])
    reservation = reservations[res_id]

    user_context = {
        "user_id": user["user_id"],
        "name": user["name"],
        "address": user["address"],
        "email": user["email"],
        "dob": user.get("dob"),
        "membership": user.get("membership"),
        "payment_methods": list(user.get("payment_methods", {}).keys()),
    }
    res_context = {
        "reservation_id": reservation["reservation_id"],
        "origin": reservation.get("origin"),
        "destination": reservation.get("destination"),
        "flight_type": reservation.get("flight_type"),
        "cabin": reservation.get("cabin"),
        "flights": reservation.get("flights", []),
        "passengers": reservation.get("passengers", []),
        "status": reservation.get("status", "active"),
        "created_at": reservation.get("created_at"),
        "insurance": reservation.get("insurance", "no"),
        "payment_history": reservation.get("payment_history", []),
    }
    return {"user": user_context, "reservation": res_context}


def sample_retail_context(db: Dict[str, Any]) -> Dict[str, Any]:
    users = db["users"]
    orders = db["orders"]
    products = db["products"]

    user_ids_with_orders = [
        uid for uid, u in users.items()
        if u.get("orders") and len(u["orders"]) > 0
    ]
    uid = random.choice(user_ids_with_orders)
    user = users[uid]
    order_id = random.choice(user["orders"])
    order = orders[order_id]

    item_products = {}
    for item in order.get("items", []):
        pid = item.get("product_id")
        if pid and pid in products:
            prod = products[pid]
            item_products[pid] = {
                "name": prod["name"],
                "product_id": pid,
                "variants": {
                    vid: {"item_id": vid, "price": v.get("price"), "options": v.get("options", {})}
                    for vid, v in list(prod.get("variants", {}).items())[:6]
                },
            }

    user_context = {
        "user_id": user["user_id"],
        "name": user["name"],
        "address": user["address"],
        "email": user["email"],
        "payment_methods": list(user.get("payment_methods", {}).keys()),
    }
    order_context = {
        "order_id": order["order_id"],
        "status": order["status"],
        "items": order.get("items", []),
        "payment_history": order.get("payment_history", []),
    }
    return {"user": user_context, "order": order_context, "products": item_products}


def sample_context(domain: str) -> Dict[str, Any]:
    db = _load_db(domain)
    if domain == "airline":
        return sample_airline_context(db)
    elif domain == "retail":
        return sample_retail_context(db)
    else:
        raise ValueError(f"Unknown domain: {domain}")


def get_tools(domain: str) -> List[Dict[str, Any]]:
    return DOMAIN_TOOLS[domain]


def get_policy(domain: str) -> str:
    return _load_policy(domain)


def format_tools_for_prompt(tools: List[Dict[str, Any]]) -> str:
    lines = []
    for t in tools:
        params = ", ".join(f"{k}: {v}" for k, v in t["parameters"].items())
        lines.append(f"- {t['name']}({params})")
        lines.append(f"  [{t['type']}] {t['description']}")
    return "\n".join(lines)


def format_context_for_prompt(context: Dict[str, Any]) -> str:
    return json.dumps(context, indent=2, default=str)
