import cv2
import torch
import clip
from PIL import Image
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'
clip_model, clip_preprocess = clip.load(r'c:\Users\197as\Documents\Document\7th SEM\Final Year Project\ai_vision_engine\weights\clip\ViT-B-32.pt', device=device)

RETAIL_CLASSES = [
    {
        'id': 'prod_lays', 'name': "Lays India's Magic Masala", 'brand': 'PepsiCo', 'price': 20, 'category': 'Chips', 'emoji': '🥔',
        'prompts': ['a packet of Lays potato chips', 'a bag of yellow or blue Lays chips', 'Lays chips packet', 'potato chips packet']
    },
    {
        'id': 'prod_biscuit', 'name': 'Hide and Seek Chocolate Chip', 'brand': 'Parle', 'price': 30, 'category': 'Biscuits', 'emoji': '🍪',
        'prompts': ['a packet of Hide and Seek biscuits', 'Parle Hide and Seek biscuit pack', 'chocolate chip cookies pack', 'biscuit packet']
    },
    {
        'id': 'prod_kurkure', 'name': 'Kurkure Masala Munch', 'brand': 'PepsiCo', 'price': 20, 'category': 'Snacks', 'emoji': '🌽',
        'prompts': ['a packet of Kurkure snack', 'Kurkure Masala Munch bag', 'corn puff snack packet']
    },
    {
        'id': 'prod_coke', 'name': 'Coca-Cola 500ml', 'brand': 'Coca-Cola', 'price': 40, 'category': 'Drinks', 'emoji': '🥤',
        'prompts': ['a bottle of Coca-Cola or cold drink', 'a can of soda or soft drink', 'cold drink plastic bottle']
    },
    {
        'id': 'prod_choc', 'name': 'Dairy Milk Silk', 'brand': 'Cadbury', 'price': 80, 'category': 'Chocolate', 'emoji': '🍫',
        'prompts': ['a Cadbury Dairy Milk chocolate bar', 'purple chocolate bar packet', 'chocolate bar']
    },
    {
        'id': 'prod_maggi', 'name': 'Maggi 2-Minute Noodles', 'brand': 'Nestle', 'price': 14, 'category': 'Instant Food', 'emoji': '🍜',
        'prompts': ['a yellow packet of Maggi 2-Minute noodles', 'Maggi noodles packet', 'instant ramen noodles packet']
    },
    {
        'id': 'prod_phone', 'name': 'Smartphone Device', 'brand': 'Electronics', 'price': 14999, 'category': 'Electronics', 'emoji': '📱',
        'prompts': ['a smartphone or mobile phone screen', 'holding a cell phone']
    },
    {
        'id': None, 'name': 'Background', 'brand': None, 'price': 0, 'category': None, 'emoji': '',
        'prompts': ['a person sitting in a room with no products', 'an empty room or background wall with no grocery items', 'human hands with nothing in them']
    }
]

prompt_list = []
prompt_to_class = []
for c_idx, c_info in enumerate(RETAIL_CLASSES):
    for p in c_info['prompts']:
        prompt_list.append(p)
        prompt_to_class.append(c_idx)

tokens = clip.tokenize(prompt_list).to(device)
with torch.no_grad():
    text_feat = clip_model.encode_text(tokens)
    text_feat /= text_feat.norm(dim=-1, keepdim=True)

# Test on grocery_item_0.jpg
img = Image.open(r'c:\Users\197as\Documents\Document\7th SEM\Final Year Project\ai_vision_engine\custom_dataset\raw_images\grocery_item_0.jpg')
t_img = clip_preprocess(img).unsqueeze(0).to(device)
with torch.no_grad():
    im_feat = clip_model.encode_image(t_img)
    im_feat /= im_feat.norm(dim=-1, keepdim=True)
    sim = (100.0 * im_feat @ text_feat.T).softmax(dim=-1)[0]

class_scores = [0.0] * len(RETAIL_CLASSES)
for p_idx, score in enumerate(sim):
    c_idx = prompt_to_class[p_idx]
    class_scores[c_idx] = max(class_scores[c_idx], score.item())

best_c = int(np.argmax(class_scores))
print('Best detected product:', RETAIL_CLASSES[best_c]['name'], f'Confidence: {class_scores[best_c]*100:.1f}%')
