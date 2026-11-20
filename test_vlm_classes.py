import torch
import clip
from PIL import Image
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'
clip_model, clip_preprocess = clip.load(r'c:\Users\197as\Documents\Document\7th SEM\Final Year Project\ai_vision_engine\weights\clip\ViT-B-32.pt', device=device)

# Canonical Product definitions with descriptive prompts
PRODUCTS = [
    {
        "id": "prod_lays_blue",
        "name": "Lays India's Magic Masala",
        "brand": "PepsiCo",
        "price": 20,
        "emoji": "🥔",
        "category": "Chips",
        "prompts": [
            "a photo of a blue packet of Lays India's Magic Masala potato chips",
            "a blue bag of Lays potato chips snack",
            "holding a blue Lays chips packet in front of camera"
        ]
    },
    {
        "id": "prod_lays_yellow",
        "name": "Lays Classic Salted",
        "brand": "PepsiCo",
        "price": 20,
        "emoji": "🥔",
        "category": "Chips",
        "prompts": [
            "a photo of a yellow packet of Lays classic salted potato chips",
            "a yellow bag of Lays potato chips",
            "holding a yellow Lays chips packet"
        ]
    },
    {
        "id": "prod_hide_seek",
        "name": "Hide & Seek Chocolate Chip",
        "brand": "Parle",
        "price": 30,
        "emoji": "🍪",
        "category": "Biscuits",
        "prompts": [
            "a photo of a packet of Parle Hide and Seek biscuits",
            "a brown packet of Hide & Seek chocolate chip cookies",
            "holding a Hide and Seek biscuit packet"
        ]
    },
    {
        "id": "prod_kurkure",
        "name": "Kurkure Masala Munch",
        "brand": "PepsiCo",
        "price": 20,
        "emoji": "🌽",
        "category": "Snacks",
        "prompts": [
            "a photo of an orange packet of Kurkure Masala Munch",
            "an orange bag of Kurkure snack",
            "holding a Kurkure packet"
        ]
    },
    {
        "id": "prod_coke",
        "name": "Coca-Cola 500ml",
        "brand": "Coca-Cola",
        "price": 40,
        "emoji": "🥤",
        "category": "Drinks",
        "prompts": [
            "a photo of a bottle or can of Coca-Cola cold drink",
            "a plastic bottle of cold drink soda",
            "holding a bottle of cold drink or Coke"
        ]
    },
    {
        "id": "prod_silk",
        "name": "Dairy Milk Silk",
        "brand": "Cadbury",
        "price": 80,
        "emoji": "🍫",
        "category": "Chocolate",
        "prompts": [
            "a photo of a purple packet of Cadbury Dairy Milk Silk chocolate",
            "a Cadbury chocolate bar packet",
            "holding a Dairy Milk chocolate"
        ]
    },
    {
        "id": "prod_maggi",
        "name": "Maggi 2-Minute Noodles",
        "brand": "Nestle",
        "price": 14,
        "emoji": "🍜",
        "category": "Instant Food",
        "prompts": [
            "a photo of a yellow packet of Nestle Maggi 2-Minute noodles",
            "a yellow pack of Maggi instant noodles",
            "holding a Maggi noodles packet"
        ]
    },
    {
        "id": "prod_phone",
        "name": "Smartphone Device",
        "brand": "Electronics",
        "price": 14999,
        "emoji": "📱",
        "category": "Electronics",
        "prompts": [
            "a close-up photo of a smartphone mobile phone screen held up to camera",
            "showing a mobile phone screen to camera"
        ]
    },
    {
        "id": None,
        "name": "Background / Person",
        "brand": "None",
        "price": 0,
        "emoji": "",
        "category": "None",
        "prompts": [
            "a webcam photo of a person sitting in their room with no grocery items",
            "a person looking at their laptop screen or webcam, empty hands",
            "an empty room background with wall, chair, desk, clothes, no retail products",
            "human face, hair, clothing, room background without products"
        ]
    }
]

# Calculate mean feature vector for each class
class_features = []
for p in PRODUCTS:
    toks = clip.tokenize(p["prompts"]).to(device)
    with torch.no_grad():
        f = clip_model.encode_text(toks)
        f /= f.norm(dim=-1, keepdim=True)
        mean_f = f.mean(dim=0, keepdim=True)
        mean_f /= mean_f.norm(dim=-1, keepdim=True)
        class_features.append(mean_f)

class_features = torch.cat(class_features, dim=0)
print(f"Precomputed {len(PRODUCTS)} classes. Shape: {class_features.shape}")

# Test on blank image / room image
blank = Image.fromarray(np.zeros((300, 300, 3), dtype=np.uint8))
blank_t = clip_preprocess(blank).unsqueeze(0).to(device)
with torch.no_grad():
    im_f = clip_model.encode_image(blank_t)
    im_f /= im_f.norm(dim=-1, keepdim=True)
    probs = (100.0 * im_f @ class_features.T).softmax(dim=-1)[0].cpu().numpy()

for prod, prob in zip(PRODUCTS, probs):
    print(f"  {prod['name']:30s}: {prob*100:5.1f}%")

best = PRODUCTS[int(np.argmax(probs))]
print("Winning class on blank:", best["name"])
