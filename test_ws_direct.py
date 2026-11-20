import asyncio
import websockets
import json
import base64
import cv2
import numpy as np

async def test_ws():
    # Create a dummy image
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(img, "TEST", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    _, buffer = cv2.imencode('.jpg', img)
    b64 = "data:image/jpeg;base64," + base64.b64encode(buffer).decode('utf-8')

    # Note: we need the server running
    # Let's test with a direct function call or check against app
    print("Testing frame encoding length:", len(b64))

asyncio.run(test_ws())
