import cv2
import torch
import clip
from PIL import Image
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'
clip_model, clip_preprocess = clip.load(r'c:\Users\197as\Documents\Document\7th SEM\Final Year Project\ai_vision_engine\weights\clip\ViT-B-32.pt', device=device)

RETAIL_PRODUCTS = [
    {
        'id': 'prod_lays',
        'name': "Lays India's Magic Masala",
        'brand': 'PepsiCo',
        'price': 20,
        'category': 'Chips',
        'emoji': '🥔',
        'prompts': ['a packet of Lays potato chips', 'a bag of Lays chips', 'potato chips packet']
    },
    {
        'id': 'prod_biscuit',
        'name': 'Hide & Seek Chocolate Chip',
        'brand': 'Parle',
        'price': 30,
        'category': 'Biscuits',
        'emoji': '🍪',
        'prompts': ['a packet of Parle Hide and Seek biscuits', 'chocolate cookies biscuit packet', 'biscuit packet']
    },
    {
        'id': 'prod_kurkure',
        'name': 'Kurkure Masala Munch',
        'brand': 'PepsiCo',
        'price': 20,
        'category': 'Snacks',
        'emoji': '🌽',
        'prompts': ['a packet of Kurkure snack', 'Kurkure Masala Munch bag']
    },
    {
        'id': 'prod_coke',
        'name': 'Cold Drink / Cola Bottle',
        'brand': 'Coca-Cola',
        'price': 40,
        'category': 'Drinks',
        'emoji': '🥤',
        'prompts': ['a bottle of Coca-Cola or cold drink', 'a plastic bottle of cold drink soda', 'a can of cold drink']
    },
    {
        'id': 'prod_choc',
        'name': 'Dairy Milk Silk',
        'brand': 'Cadbury',
        'price': 80,
        'category': 'Chocolate',
        'emoji': '🍫',
        'prompts': ['a Cadbury Dairy Milk chocolate bar', 'chocolate candy bar']
    },
    {
        'id': 'prod_maggi',
        'name': 'Maggi 2-Minute Noodles',
        'brand': 'Nestle',
        'price': 14,
        'category': 'Instant Food',
        'emoji': '🍜',
        'prompts': ['a yellow packet of Maggi 2-Minute noodles', 'instant noodles food packet']
    },
    {
        'id': 'prod_phone',
        'name': 'Smartphone Device',
        'brand': 'Electronics',
        'price': 14999,
        'category': 'Electronics',
        'emoji': '📱',
        'prompts': ['a smartphone mobile phone screen', 'holding a mobile phone']
    },
    {
        'id': None,
        'name': 'Background',
        'brand': None,
        'price': 0,
        'category': None,
        'emoji': '',
        'prompts': ['an empty room background with no products', 'a person with no grocery items']
    }
]

# Compute mean class embeddings (canonical zero-shot representation)
class_feats = []
for c in RETAIL_PRODUCTS:
    tokens = clip.tokenize(c['prompts']).to(device)
    with torch.no_grad():
        f = clip_model.encode_text(tokens)
        f /= f.norm(dim=-1, keepdim=True)
        mean_f = f.mean(dim=0, keepdim=True)
        mean_f /= mean_f.norm(dim=-1, keepdim=True)
        class_feats.append(mean_f)
class_feats = torch.cat(class_feats, dim=0)

print(f"Class features shape: {class_feats.shape}")

for name in ['grocery_item_0.jpg', 'grocery_item_1.jpg', 'grocery_item_10.jpg', 'grocery_item_101.jpg']:
    p = rf"c:\Users\197as\Documents\Document\7th SEM\Final Year Project\ai_vision_engine\custom_dataset\raw_images\{name}"
    img = Image.open(p)
    img_t = clip_preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        im_feat = clip_model.encode_image(img_t)
        im_feat /= im_feat.norm(dim=-1, keepdim=True)
        probs = (100.0 * im_feat @ class_feats.T).softmax(dim=-1)[0].cpu().numpy()
    
    best_idx = int(np.argmax(probs))
    best_p = RETAIL_PRODUCTS[best_idx]
    print(f"{name}: [{best_p['brand']}] {best_p['name']} ({probs[best_idx]*100:.1f}%)")
