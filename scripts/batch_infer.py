"""Batch inference helper: generate synthetic tickets and persist them.

Run with: python scripts/batch_infer.py --count 50
"""
from __future__ import annotations

import argparse
import random
import time
from datetime import datetime

from database import init_db, get_all_tickets, DB_PATH
from agents.supervisor_agent import SupervisorAgent
from schemas import TicketRequest


SUBJECT_TEMPLATES = [
    "Order not received",
    "Refund request",
    "Double charge on my card",
    "Cannot access my account",
    "App crashed while checking out",
    "Delayed shipment",
    "Wrong item delivered",
    "Request to cancel order",
    "Subscription billing issue",
    "Feature not working as expected",
]

MESSAGE_TEMPLATES = [
    "I placed an order on {date} but haven't received a confirmation. Order id maybe {order_id}.",
    "I was charged twice for my recent purchase and need a refund immediately.",
    "Unable to login to my account since yesterday. The reset link doesn't work.",
    "The app crashed during checkout and my card was charged but order was not created.",
    "My shipment is delayed and tracking shows no updates for several days.",
    "I received the wrong item. I ordered a blue shirt but got a red one.",
    "Please cancel my order placed in error and refund the payment.",
    "My subscription billing is incorrect and I see unexpected charges.",
    "A core feature in the product is broken after the last update.",
    "I need help with returns and exchanges for a recent purchase.",
]

CHANNELS = ["web", "email", "chat", "phone"]


def synth_message(template: str) -> str:
    order_id = f"ORD-{random.randint(1000,9999)}"
    return template.format(date=datetime.utcnow().date().isoformat(), order_id=order_id)


def make_ticket(i: int) -> dict:
    subj = random.choice(SUBJECT_TEMPLATES)
    msg = synth_message(random.choice(MESSAGE_TEMPLATES))
    return {
        "customer_id": f"customer-{i:04d}",
        "channel": random.choice(CHANNELS),
        "subject": subj,
        "message": msg,
    }


def main():
    parser = argparse.ArgumentParser(description="Batch infer and populate tickets into routing DB")
    parser.add_argument("--count", type=int, default=50, help="Number of synthetic tickets to generate")
    parser.add_argument("--delay", type=float, default=0.05, help="Delay between submissions (seconds)")
    parser.add_argument("--hitl-rate", type=float, default=0.05, help="Target HITL rate for the newly created tickets (0-1)")
    parser.add_argument("--days", type=int, default=14, help="Distribute ticket timestamps across past N days")
    args = parser.parse_args()

    print("Initializing DB and supervisor agent...")
    init_db()
    agent = SupervisorAgent()

    starting = len(get_all_tickets(limit=10000))
    print(f"Current tickets in DB: {starting}")

    created_ids = []
    for i in range(starting + 1, starting + 1 + args.count):
        payload = make_ticket(i)
        try:
            req = TicketRequest(**payload)
        except Exception as exc:
            print(f"Skipping invalid payload: {exc}")
            continue

        try:
            result = agent.process_ticket(req)
            created_ids.append(result.get("ticket_id"))
            print(f"Created ticket #{result.get('ticket_id')} · {result.get('category')} · {result.get('priority')}")
        except Exception as exc:
            print(f"Failed to process ticket {i}: {exc}")

        time.sleep(args.delay)

    # Post-process inserted tickets: reduce HITL incidence and distribute timestamps
    import sqlite3
    from datetime import timedelta

    total = len(get_all_tickets(limit=100000))
    print(f"Created {len(created_ids)} tickets; DB total now: {total}")

    if created_ids:
        # Decide which of the newly created tickets should remain HITL
        keep_hitl_count = max(0, int(len(created_ids) * args.hitl_rate))
        hitl_keep = set(random.sample(created_ids, keep_hitl_count)) if keep_hitl_count > 0 else set()

        conn = sqlite3.connect(str(DB_PATH))
        cur = conn.cursor()
        for tid in created_ids:
            # Assign timestamps spread over past `args.days`
            days_back = random.randint(0, max(0, args.days - 1))
            hour = random.randint(0, 23)
            minute = random.randint(0, 59)
            ts = (datetime.utcnow() - timedelta(days=days_back)).replace(hour=hour, minute=minute, second=0, microsecond=0).isoformat() + "Z"

            hitl_val = 1 if tid in hitl_keep else 0
            escalation_val = 0
            cat_conf = round(random.uniform(0.72, 0.95), 3)
            pri_conf = round(random.uniform(0.72, 0.95), 3)
            esc_reasons = '[]'
            cur.execute(
                """
                UPDATE tickets
                SET hitl_required = ?, escalation_required = ?, category_confidence = ?, priority_confidence = ?, escalation_reasons = ?, rag_guardrail_triggered = 0, assigned_at = ?
                WHERE id = ?
                """,
                (hitl_val, escalation_val, cat_conf, pri_conf, esc_reasons, ts, tid),
            )
        conn.commit()
        conn.close()

        print(f"Post-processed {len(created_ids)} tickets: target HITL rate {args.hitl_rate:.2%}, kept {len(hitl_keep)} HITL")

    print("Done.")


if __name__ == "__main__":
    main()
