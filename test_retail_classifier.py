import cv2
import torch
import clip
from PIL import Image
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'
clip_model, clip_preprocess = clip.load(r'c:\Users\197as\Documents\Document\7th SEM\Final Year Project\ai_vision_engine\weights\clip\ViT-B-32.pt', device=device)

# Target retail products with distinctive descriptive prompts
RETAIL_PRODUCTS = [
    {
        'id': 'prod_lays',
        'name': "Lays India's Magic Masala",
        'brand': 'PepsiCo',
        'price': 20,
        'category': 'Chips',
        'emoji': '🥔',
        'prompts': [
            'a packet of Lays potato chips',
            'a blue packet of Lays Magic Masala chips',
            'a yellow packet of Lays classic salted chips',
            'a bag of Lays chips snack',
            'potato chips snack packet'
        ]
    },
    {
        'id': 'prod_biscuit',
        'name': 'Hide & Seek Chocolate Chip',
        'brand': 'Parle',
        'price': 30,
        'category': 'Biscuits',
        'emoji': '🍪',
        'prompts': [
            'a packet of Parle Hide and Seek biscuits',
            'Hide and Seek chocolate chip cookies packet',
            'brown packet of chocolate cookies',
            'biscuit packet packaging'
        ]
    },
    {
        'id': 'prod_kurkure',
        'name': 'Kurkure Masala Munch',
        'brand': 'PepsiCo',
        'price': 20,
        'category': 'Snacks',
        'emoji': '🌽',
        'prompts': [
            'an orange packet of Kurkure snack',
            'Kurkure Masala Munch packet',
            'crispy corn puff snack packet'
        ]
    },
    {
        'id': 'prod_coke',
        'name': 'Cold Drink / Cola Bottle',
        'brand': 'Coca-Cola',
        'price': 40,
        'category': 'Drinks',
        'emoji': '🥤',
        'prompts': [
            'a red bottle or can of Coca-Cola',
            'a plastic bottle of cold drink soda',
            'a can of cold drink or soft drink',
            'a bottle of water or soda'
        ]
    },
    {
        'id': 'prod_choc',
        'name': 'Dairy Milk Silk',
        'brand': 'Cadbury',
        'price': 80,
        'category': 'Chocolate',
        'emoji': '🍫',
        'prompts': [
            'a Cadbury Dairy Milk chocolate bar',
            'a purple packet of chocolate bar',
            'chocolate candy bar packet'
        ]
    },
    {
        'id': 'prod_maggi',
        'name': 'Maggi 2-Minute Noodles',
        'brand': 'Nestle',
        'price': 14,
        'category': 'Instant Food',
        'emoji': '🍜',
        'prompts': [
            'a yellow packet of Maggi 2-Minute noodles',
            'Nestle Maggi noodles packet',
            'instant noodles food packet'
        ]
    },
    {
        'id': 'prod_phone',
        'name': 'Smartphone Device',
        'brand': 'Electronics',
        'price': 14999,
        'category': 'Electronics',
        'emoji': '📱',
        'prompts': [
            'a smartphone mobile phone screen',
            'holding an electronic mobile phone',
            'iPhone or Android mobile device'
        ]
    },
    {
        'id': None,
        'name': 'Background / Person',
        'brand': None,
        'price': 0,
        'category': None,
        'emoji': '',
        'prompts': [
            'a person sitting in a room looking at camera without products',
            'an empty room wall or indoor background with no grocery items',
            'empty human hands with nothing in them'
        ]
    }
]

# Precompute text embeddings once
all_prompts = []
prompt_to_prod = []
for p_idx, prod in enumerate(RETAIL_PRODUCTS):
    for prompt_text in prod['prompts']:
        all_prompts.append(prompt_text)
        prompt_to_prod.append(p_idx)

tokens = clip.tokenize(all_prompts).to(device)
with torch.no_grad():
    text_features = clip_model.encode_text(tokens)
    text_features /= text_features.norm(dim=-1, keepdim=True)

print(f"Precomputed {len(all_prompts)} prompts across {len(RETAIL_PRODUCTS)} classes.")

def classify_crop(crop_bgr):
    """Classify a cropped region using CLIP."""
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(crop_rgb)
    t_img = clip_preprocess(pil_img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        im_feat = clip_model.encode_image(t_img)
        im_feat /= im_feat.norm(dim=-1, keepdim=True)
        sim = (100.0 * im_feat @ text_features.T).softmax(dim=-1)[0]
    
    # Max score per product class
    scores = [0.0] * len(RETAIL_PRODUCTS)
    for p_i, score in enumerate(sim):
        prod_i = prompt_to_prod[p_i]
        scores[prod_i] = max(scores[prod_i], score.item())
    
    best_prod_idx = int(np.argmax(scores))
    best_score = scores[best_prod_idx]
    best_prod = RETAIL_PRODUCTS[best_prod_idx]
    
    # If the winner is 'Background' or score is too low, return None
    if best_prod['id'] is None or best_score < 0.28:
        return None, best_score
    
    return best_prod, best_score

# Test on 3 images
for name in ['grocery_item_0.jpg', 'grocery_item_1.jpg', 'grocery_item_10.jpg']:
    p = f"c:\\Users\\197as\\Documents\\Document\\7th SEM\\Final Year Project\\ai_vision_engine\\custom_dataset\\raw_images\\{name}"
    img = cv2.imread(p)
    prod, conf = classify_crop(img)
    if prod:
        print(f"{name} -> [{prod['brand']}] {prod['name']} (conf: {conf*100:.1f}%)")
    else:
        print(f"{name} -> No product (conf: {conf*100:.1f}%)")
