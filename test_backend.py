import time
import requests
import uvicorn
import threading
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from app import app

def run_server():
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")

t = threading.Thread(target=run_server, daemon=True)
t.start()

time.sleep(3)

# Test endpoints
try:
    r = requests.get("http://127.0.0.1:8000/api/products")
    print("GET /api/products status:", r.status_code, "Count:", len(r.json()))

    r = requests.get("http://127.0.0.1:8000/api/cart")
    print("GET /api/cart status:", r.status_code, "Body:", r.json())

    r = requests.post("http://127.0.0.1:8000/api/cart/add", json={"product_id": "prod_001", "quantity": 2})
    print("POST /api/cart/add status:", r.status_code, "Total:", r.json()["cart"]["total"])

    r = requests.post("http://127.0.0.1:8000/api/cart/update", json={"product_id": "prod_001", "quantity_delta": -1})
    print("POST /api/cart/update status:", r.status_code, "Items:", len(r.json()["items"]))

    r = requests.post("http://127.0.0.1:8000/api/checkout")
    print("POST /api/checkout status:", r.status_code, "Order:", r.json()["order_id"])

    print("ALL TESTS PASSED!")
except Exception as e:
    print("TEST FAILED:", e)
    sys.exit(1)
