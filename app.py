from __future__ import annotations

import base64
import datetime
import json
import logging
import uuid
from pathlib import Path
from typing import Any

import clip
import cv2
import numpy as np
import pyzbar.pyzbar as pyzbar
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from PIL import Image

import config

_orig_load = torch.load
def _patch_load(*args: Any, **kwargs: Any) -> Any:
    kwargs.setdefault("weights_only", False)
    return _orig_load(*args, **kwargs)
torch.load = _patch_load

device = "cuda" if torch.cuda.is_available() else "cpu"

logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("grab_and_go")

RETAIL_CATALOG: list[dict[str, Any]] = [
    {
        "id": "prod_lays_blue",
        "name": "Lays India's Magic Masala",
        "brand": "PepsiCo",
        "price": 20,
        "category": "Chips & Snacks",
        "emoji": "🥔",
        "image": "/static/img/chips.svg",
        "prompts": [
            "a photo of a blue packet of Lays India's Magic Masala potato chips",
            "a blue bag of Lays potato chips snack being held",
            "holding a blue Lays chips packet in front of webcam"
        ],
    },
    {
        "id": "prod_lays_yellow",
        "name": "Lays Classic Salted",
        "brand": "PepsiCo",
        "price": 20,
        "category": "Chips & Snacks",
        "emoji": "🥔",
        "image": "/static/img/chips.svg",
        "prompts": [
            "a photo of a yellow packet of Lays classic salted potato chips",
            "a yellow bag of Lays potato chips",
            "holding a yellow Lays chips packet in front of camera"
        ],
    },
    {
        "id": "prod_kurkure",
        "name": "Kurkure Masala Munch",
        "brand": "PepsiCo",
        "price": 20,
        "category": "Chips & Snacks",
        "emoji": "🌽",
        "image": "/static/img/snack.svg",
        "prompts": [
            "a photo of an orange packet of Kurkure Masala Munch",
            "an orange packet of Kurkure snack",
            "holding a Kurkure spicy snack packet"
        ],
    },
    {
        "id": "prod_hide_seek",
        "name": "Hide & Seek Chocolate Chip",
        "brand": "Parle",
        "price": 30,
        "category": "Biscuits & Cookies",
        "emoji": "🍪",
        "image": "/static/img/biscuit.svg",
        "prompts": [
            "a photo of a brown packet of Parle Hide and Seek biscuits",
            "Hide and Seek chocolate chip cookies packet",
            "holding a packet of Hide & Seek biscuits"
        ],
    },
    {
        "id": "prod_oreo",
        "name": "Oreo Vanilla Creme",
        "brand": "Cadbury",
        "price": 35,
        "category": "Biscuits & Cookies",
        "emoji": "🍪",
        "image": "/static/img/biscuit.svg",
        "prompts": [
            "a photo of a blue packet of Oreo biscuits",
            "Oreo chocolate cookies pack",
            "holding an Oreo biscuit pack"
        ],
    },
    {
        "id": "prod_silk",
        "name": "Dairy Milk Silk",
        "brand": "Cadbury",
        "price": 80,
        "category": "Chocolates",
        "emoji": "🍫",
        "image": "/static/img/chocolate.svg",
        "prompts": [
            "a photo of a purple packet of Cadbury Dairy Milk Silk chocolate",
            "a Cadbury Dairy Milk chocolate bar",
            "holding a chocolate bar packet"
        ],
    },
    {
        "id": "prod_coke",
        "name": "Coca-Cola 500ml",
        "brand": "Coca-Cola",
        "price": 40,
        "category": "Beverages",
        "emoji": "🥤",
        "image": "/static/img/bottle.svg",
        "prompts": [
            "a photo of a bottle or can of Coca-Cola cold drink",
            "a plastic bottle of cold drink soda",
            "holding a bottle of cold drink or Coke"
        ],
    },
    {
        "id": "prod_maggi",
        "name": "Maggi 2-Minute Noodles",
        "brand": "Nestle",
        "price": 14,
        "category": "Instant Food",
        "emoji": "🍜",
        "image": "/static/img/cup.svg",
        "prompts": [
            "a photo of a yellow packet of Nestle Maggi 2-Minute noodles",
            "a pack of Maggi instant noodles",
            "holding a yellow Maggi noodles packet"
        ],
    },
    {
        "id": "prod_phone",
        "name": "Smartphone Device",
        "brand": "Electronics",
        "price": 14999,
        "category": "Electronics",
        "emoji": "📱",
        "image": "/static/img/phone.svg",
        "prompts": [
            "a close-up photo of a smartphone mobile phone screen held up to camera",
            "showing a mobile phone screen directly into the camera"
        ],
    },
]

BACKGROUND_PROMPTS = [
    "a webcam photo of a person sitting in their room with no grocery items",
    "a person looking at their laptop screen or webcam, empty hands",
    "an empty room background with wall, chair, desk, clothes, no retail products",
    "human face, hair, clothing, room background without products",
    "a laptop computer keyboard and monitor screen, desk environment",
]

BARCODE_CATALOG: dict[str, dict[str, Any]] = {
    "8901491101837": {
        "id": "prod_lays_blue", "name": "Lays India's Magic Masala",
        "brand": "PepsiCo", "price": 20, "category": "Chips & Snacks", "emoji": "🥔",
    },
    "8901491001007": {
        "id": "prod_kurkure", "name": "Kurkure Masala Munch",
        "brand": "PepsiCo", "price": 20, "category": "Chips & Snacks", "emoji": "🌽",
    },
    "8901719101038": {
        "id": "prod_hide_seek", "name": "Hide & Seek Chocolate Chip",
        "brand": "Parle", "price": 30, "category": "Biscuits & Cookies", "emoji": "🍪",
    },
    "8901719114137": {
        "id": "prod_hide_seek_fab", "name": "Hide & Seek Fab Vanilla",
        "brand": "Parle", "price": 30, "category": "Biscuits & Cookies", "emoji": "🍪",
    },
    "8901764012228": {
        "id": "prod_coke", "name": "Coca-Cola 500ml",
        "brand": "Coca-Cola", "price": 40, "category": "Beverages", "emoji": "🥤",
    },
    "8901764022227": {
        "id": "prod_thumsup", "name": "Thums Up 500ml",
        "brand": "Coca-Cola", "price": 40, "category": "Beverages", "emoji": "🥤",
    },
    "7622201755104": {
        "id": "prod_oreo", "name": "Oreo Vanilla Creme",
        "brand": "Cadbury", "price": 35, "category": "Biscuits & Cookies", "emoji": "🍪",
    },
    "7622201140023": {
        "id": "prod_silk", "name": "Dairy Milk Silk",
        "brand": "Cadbury", "price": 80, "category": "Chocolates", "emoji": "🍫",
    },
    "8901058852370": {
        "id": "prod_maggi", "name": "Maggi 2-Minute Noodles",
        "brand": "Nestle", "price": 14, "category": "Instant Food", "emoji": "🍜",
    },
}

cart: dict[str, dict[str, Any]] = {}

app = FastAPI(
    title="Grab & Go — Autonomous Retail System",
    version="2.0.0",
    description="Next-Gen AI Autonomous Checkout with Vision-Language Model & Barcode Verification.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

config.STATIC_DIR.mkdir(parents=True, exist_ok=True)
(config.STATIC_DIR / "img").mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(config.STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(config.TEMPLATES_DIR))

clip_model = None
clip_preprocess = None
class_features = None

@app.on_event("startup")
async def _init_vision_engine() -> None:
    global clip_model, clip_preprocess, class_features

    weights_file = Path(r"c:\Users\197as\Documents\Document\7th SEM\Final Year Project\ai_vision_engine\weights\clip\ViT-B-32.pt")
    logger.info("Initializing Vision-Language Model on %s from %s...", device, weights_file)

    try:
        clip_model, clip_preprocess = clip.load(str(weights_file), device=device)
        clip_model.eval()

        all_features = []

        for prod in RETAIL_CATALOG:
            toks = clip.tokenize(prod["prompts"]).to(device)
            with torch.no_grad():
                f = clip_model.encode_text(toks)
                f /= f.norm(dim=-1, keepdim=True)
                mean_f = f.mean(dim=0, keepdim=True)
                mean_f /= mean_f.norm(dim=-1, keepdim=True)
                all_features.append(mean_f)

        bg_toks = clip.tokenize(BACKGROUND_PROMPTS).to(device)
        with torch.no_grad():
            bg_f = clip_model.encode_text(bg_toks)
            bg_f /= bg_f.norm(dim=-1, keepdim=True)
            bg_mean = bg_f.mean(dim=0, keepdim=True)
            bg_mean /= bg_mean.norm(dim=-1, keepdim=True)
            all_features.append(bg_mean)

        class_features = torch.cat(all_features, dim=0)
        logger.info(
            "Vision Engine online: %d retail products + negative background filter ready.",
            len(RETAIL_CATALOG),
        )
    except Exception as exc:
        logger.error("Failed to load Vision Engine: %s", exc)
        raise RuntimeError(f"Vision Engine failure: {exc}") from exc

class CartAddRequest(BaseModel):
    product_id: str
    quantity: int = 1

class CartRemoveRequest(BaseModel):
    product_id: str

class CartUpdateRequest(BaseModel):
    product_id: str
    quantity_delta: int = 0

def _lookup_product(product_id: str) -> dict[str, Any] | None:
    for p in RETAIL_CATALOG:
        if p["id"] == product_id:
            return p
    for p in BARCODE_CATALOG.values():
        if p["id"] == product_id:
            return p
    return None

def _build_cart_response() -> dict[str, Any]:
    items: list[dict[str, Any]] = []
    total = 0.0
    for pid, entry in cart.items():
        subtotal = entry["price"] * entry["quantity"]
        total += subtotal
        items.append({
            "product_id": pid,
            "name": entry["name"],
            "brand": entry["brand"],
            "price": entry["price"],
            "category": entry.get("category", ""),
            "quantity": entry["quantity"],
            "subtotal": subtotal,
            "emoji": entry.get("emoji", "📦"),
            "image": entry.get("image", ""),
        })
    return {"items": items, "total": round(total, 2), "item_count": len(items)}

def _decode_base64_frame(data: str) -> np.ndarray:
    data = data.strip()
    if data.startswith("{") and data.endswith("}"):
        try:
            parsed = json.loads(data)
            if isinstance(parsed, dict):
                data = parsed.get("frame") or parsed.get("image") or parsed.get("data") or ""
        except Exception:
            pass

    if "," in data:
        data = data.split(",", 1)[1]

    raw_bytes = base64.b64decode(data)
    np_arr = np.frombuffer(raw_bytes, dtype=np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Invalid image bytes")
    return frame

def classify_retail_crop(crop_bgr: np.ndarray) -> tuple[dict[str, Any] | None, float]:
    if clip_model is None or class_features is None or crop_bgr is None or crop_bgr.size < 2000:
        return None, 0.0

    try:
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        t_img = clip_preprocess(pil_img).unsqueeze(0).to(device)

        with torch.no_grad():
            im_feat = clip_model.encode_image(t_img)
            im_feat /= im_feat.norm(dim=-1, keepdim=True)

            probs = (100.0 * im_feat @ class_features.T).softmax(dim=-1)[0].cpu().numpy()

        best_idx = int(np.argmax(probs))
        confidence = float(probs[best_idx])

        if best_idx == len(RETAIL_CATALOG):
            return None, confidence

        if confidence < 0.38:
            return None, confidence

        product = RETAIL_CATALOG[best_idx]
        return product, confidence
    except Exception as e:
        logger.warning("VLM inference exception: %s", e)
        return None, 0.0

def decode_barcode(frame_bgr: np.ndarray) -> dict[str, Any] | None:
    try:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        barcodes = pyzbar.decode(gray)

        if not barcodes:
            clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            barcodes = pyzbar.decode(enhanced)

        for b in barcodes:
            code = b.data.decode("utf-8").strip()
            poly = [[int(p.x), int(p.y)] for p in b.polygon] if b.polygon else []
            if code in BARCODE_CATALOG:
                prod = BARCODE_CATALOG[code]
                return {
                    "code": code,
                    "product": prod,
                    "product_id": prod["id"],
                    "name": prod["name"],
                    "brand": prod["brand"],
                    "price": prod["price"],
                    "emoji": prod.get("emoji", "🏷️"),
                    "points": poly,
                }
            elif len(code) >= 8:
                return {
                    "code": code,
                    "product": None,
                    "product_id": f"bc_{code}",
                    "name": f"Verified Product ({code})",
                    "brand": "Scan Ground Truth",
                    "price": 25,
                    "emoji": "🏷️",
                    "points": poly,
                }
    except Exception as bc_err:
        logger.warning("Barcode decode error: %s", bc_err)
    return None

@app.get("/", response_class=HTMLResponse)
async def page_index(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")

@app.get("/checkout", response_class=HTMLResponse)
async def page_checkout(request: Request):
    return templates.TemplateResponse(request=request, name="checkout.html")

@app.get("/payment-success", response_class=HTMLResponse)
async def page_payment_success(request: Request):
    return templates.TemplateResponse(request=request, name="payment_success.html")

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return Response(status_code=204)

@app.get("/api/products")
async def api_get_products():
    return JSONResponse(content=RETAIL_CATALOG)

@app.post("/api/cart/add")
async def api_cart_add(body: CartAddRequest):
    product = _lookup_product(body.product_id)
    if product is None:
        return JSONResponse(status_code=404, content={"error": f"Product '{body.product_id}' not found."})

    if body.quantity < 1:
        return JSONResponse(status_code=400, content={"error": "Quantity must be at least 1."})

    pid = body.product_id
    if pid in cart:
        cart[pid]["quantity"] += body.quantity
    else:
        cart[pid] = {
            "name": product["name"],
            "brand": product["brand"],
            "price": product["price"],
            "category": product.get("category", ""),
            "emoji": product.get("emoji", "📦"),
            "image": product.get("image", ""),
            "quantity": body.quantity,
        }

    cart_data = _build_cart_response()
    logger.info("Cart add: %s ×%d (Total: ₹%.2f)", product["name"], body.quantity, cart_data["total"])
    return JSONResponse(content={"message": "Item added.", "cart": cart_data, "items": cart_data["items"]})

@app.post("/api/cart/remove")
@app.delete("/api/cart/remove/{product_id}")
@app.post("/api/cart/remove/{product_id}")
async def api_cart_remove(product_id: str | None = None, body: CartRemoveRequest | None = None):
    target_id = product_id or (body.product_id if body else None)
    if not target_id or target_id not in cart:
        return JSONResponse(status_code=404, content={"error": f"Product '{target_id}' not in cart."})

    removed = cart.pop(target_id)
    cart_data = _build_cart_response()
    logger.info("Cart remove: %s", removed["name"])
    return JSONResponse(content={"message": "Item removed.", "cart": cart_data, "items": cart_data["items"]})

@app.post("/api/cart/update")
async def api_cart_update(body: CartUpdateRequest):
    pid = body.product_id
    if pid not in cart:
        return JSONResponse(status_code=404, content={"error": f"Product '{pid}' not in cart."})

    new_qty = cart[pid]["quantity"] + body.quantity_delta
    if new_qty <= 0:
        cart.pop(pid)
    else:
        cart[pid]["quantity"] = new_qty

    cart_data = _build_cart_response()
    return JSONResponse(content={"message": "Cart updated.", "cart": cart_data, "items": cart_data["items"]})

@app.get("/api/cart")
async def api_cart_get():
    return JSONResponse(content=_build_cart_response())

@app.post("/api/cart/clear")
async def api_cart_clear():
    cart.clear()
    cart_data = _build_cart_response()
    logger.info("Cart cleared.")
    return JSONResponse(content={"message": "Cart cleared.", "cart": cart_data, "items": []})

@app.post("/api/checkout")
async def api_checkout():
    if not cart:
        return JSONResponse(status_code=400, content={"error": "Cart is empty."})

    cart_data = _build_cart_response()
    order = {
        "order_id": f"ORD-{uuid.uuid4().hex[:8].upper()}",
        "items": cart_data["items"],
        "total": cart_data["total"],
        "item_count": cart_data["item_count"],
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "status": "completed",
    }
    cart.clear()
    logger.info("Order finalized: %s — ₹%.2f", order["order_id"], order["total"])
    return JSONResponse(content=order)

@app.websocket("/ws/detect")
async def ws_detect(websocket: WebSocket):
    await websocket.accept()
    client = websocket.client
    logger.info("WebSocket connected: %s", client)

    try:
        while True:
            data = await websocket.receive_text()

            try:
                frame = _decode_base64_frame(data)
            except Exception:
                await websocket.send_json({"error": "Invalid frame"})
                continue

            fh, fw = frame.shape[:2]
            detections: list[dict[str, Any]] = []

            barcode_res = decode_barcode(frame)

            rx1, ry1 = int(fw * 0.20), int(fh * 0.15)
            rx2, ry2 = int(fw * 0.80), int(fh * 0.85)
            scan_crop = frame[ry1:ry2, rx1:rx2]

            product, conf = classify_retail_crop(scan_crop)

            if product is not None:
                detections.append({
                    "label": product["name"].lower(),
                    "product_id": product["id"],
                    "name": product["name"],
                    "brand": product["brand"],
                    "price": product["price"],
                    "emoji": product["emoji"],
                    "category": product["category"],
                    "confidence": round(conf, 2),
                    "bbox": [rx1, ry1, rx2, ry2],
                })

            await websocket.send_json({
                "detections": detections,
                "barcode": barcode_res,
            })

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected: %s", client)
    except Exception as ws_err:
        logger.error("WebSocket error: %s", ws_err)
        try:
            await websocket.close(code=1011, reason=str(ws_err))
        except Exception:
            pass

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Grab & Go server on %s:%s", config.HOST, config.PORT)
    uvicorn.run("app:app", host=config.HOST, port=config.PORT, reload=config.DEBUG)
