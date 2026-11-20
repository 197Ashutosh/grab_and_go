from starlette.testclient import TestClient
import sys, os, numpy as np, cv2, base64
sys.path.insert(0, r"c:\Users\197as\Documents\Document\7th SEM\Final Year Project\grab_and_go")
from app import app

client = TestClient(app)
with client:
    r = client.get('/api/products')
    assert r.status_code == 200
    prods = r.json()
    print("Clean Grocery Catalog Count:", len(prods))
    for p in prods:
        print(f"  {p['brand']} - {p['name']} (Rs.{p['price']})")
        assert "laptop" not in p["name"].lower()

    with client.websocket_connect('/ws/detect') as ws:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        _, buf = cv2.imencode('.jpg', frame)
        b64 = 'data:image/jpeg;base64,' + base64.b64encode(buf).decode('utf-8')
        
        ws.send_text(b64)
        resp = ws.receive_json()
        print("Detections on blank/background frame:", resp.get("detections"))
        assert len(resp["detections"]) == 0, "Should have ZERO detections on blank background!"

print("ALL V2 VISION ENGINE TESTS PASSED PERFECTLY!")
