"""Interactive demo (Gradio). Runs locally or as a Hugging Face Space.

python app.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import gradio as gr

from fraudguard.data import CATEGORIES
from fraudguard.model import MODEL_PATH, score_transaction

if not MODEL_PATH.exists():  # first run (e.g. a fresh Space): train the model, about 30 seconds
    from fraudguard.train import main as train

    train()

EXAMPLES = [
    # amount, hour, category, distance, online, new device, new category, avg spend, txns 1h, txns 24h, mins since last
    [38.20, 13, "grocery", 3.0, False, False, False, 45.0, 0, 2, 240],
    [899.00, 3, "electronics", 820.0, True, True, True, 40.0, 2, 4, 6],
    [250.00, 1, "gift_cards", 12.0, True, True, False, 55.0, 3, 3, 9],
    [1200.00, 10, "travel", 15.0, True, False, True, 60.0, 0, 1, 600],
]


def predict(amount, hour, category, distance, online, new_device, new_category, avg_spend, txns_1h, txns_24h, mins_since_last):
    txn = {
        "amount": amount,
        "timestamp": f"2026-04-15T{int(hour):02d}:30:00",
        "merchant_category": category,
        "distance_from_home_km": distance,
        "is_online": online,
        "is_new_device": new_device,
        "is_new_category": new_category,
        "customer_avg_amount": avg_spend,
        "txns_last_1h": int(txns_1h),
        "txns_last_24h": int(txns_24h),
        "seconds_since_last_txn": mins_since_last * 60,
        "customer_history_length": 150,
    }
    result = score_transaction(txn)
    verdict = "🚩 FLAG for review" if result["flagged"] else "✅ Approve"
    return {"fraud": result["fraud_probability"], "legitimate": 1 - result["fraud_probability"]}, (
        f"**{verdict}** (alert threshold {result['threshold']:.2f}, model: {result['model'].replace('_', ' ')})"
    )


with gr.Blocks(title="FraudGuard") as demo:
    gr.Markdown(
        "# FraudGuard: card transaction fraud detection\n"
        "Describe a transaction and the customer's recent behaviour, and the model scores its fraud risk. "
        "Trained on simulated transactions; see the [GitHub repo](https://github.com/YatorSheldon/transaction-fraud-detection) for how it works."
    )
    with gr.Row():
        with gr.Column():
            amount = gr.Number(label="Amount ($)", value=120)
            hour = gr.Slider(0, 23, step=1, label="Hour of day", value=14)
            category = gr.Dropdown(CATEGORIES, label="Merchant category", value="retail")
            distance = gr.Number(label="Distance from home (km)", value=5)
            online = gr.Checkbox(label="Online purchase")
            new_device = gr.Checkbox(label="New device for this customer")
            new_category = gr.Checkbox(label="First purchase in this category")
        with gr.Column():
            avg_spend = gr.Number(label="Customer's average spend ($)", value=45)
            txns_1h = gr.Number(label="Transactions in the last hour", value=0, precision=0)
            txns_24h = gr.Number(label="Transactions in the last 24 hours", value=2, precision=0)
            mins_since = gr.Number(label="Minutes since last transaction", value=180)
            btn = gr.Button("Score transaction", variant="primary")
            label = gr.Label(label="Fraud risk")
            decision = gr.Markdown()
    inputs = [amount, hour, category, distance, online, new_device, new_category, avg_spend, txns_1h, txns_24h, mins_since]
    btn.click(predict, inputs=inputs, outputs=[label, decision])
    gr.Examples(EXAMPLES, inputs=inputs)

if __name__ == "__main__":
    demo.launch()
