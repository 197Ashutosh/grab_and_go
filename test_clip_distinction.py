import cv2
import torch
import clip
from PIL import Image
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'
clip_model, clip_preprocess = clip.load(r'c:\Users\197as\Documents\Document\7th SEM\Final Year Project\ai_vision_engine\weights\clip\ViT-B-32.pt', device=device)

LABELS = [
    ("a webcam photo of a person sitting in their room, background wall, laptop, face, no grocery product", None),
    ("a close-up photo of a blue or yellow packet of Lays potato chips", "prod_lays"),
    ("a close-up photo of a packet of Parle Hide and Seek chocolate chip biscuits", "prod_hide_seek"),
    ("a close-up photo of an orange packet of Kurkure snack", "prod_kurkure"),
    ("a close-up photo of a cold drink bottle or soda can", "prod_coke"),
    ("a close-up photo of a Cadbury Dairy Milk chocolate bar", "prod_silk"),
    ("a close-up photo of a yellow packet of Maggi noodles", "prod_maggi"),
    ("a close-up photo of a smartphone mobile screen being shown", "prod_phone"),
]

prompts = [l[0] for l in LABELS]
tokens = clip.tokenize(prompts).to(device)
with torch.no_grad():
    text_features = clip_model.encode_text(tokens)
    text_features /= text_features.norm(dim=-1, keepdim=True)

# Test on 3 images
for name in ['grocery_item_0.jpg', 'grocery_item_1.jpg', 'grocery_item_10.jpg']:
    p = rf"c:\Users\197as\Documents\Document\7th SEM\Final Year Project\ai_vision_engine\custom_dataset\raw_images\{name}"
    img = Image.open(p)
    img_t = clip_preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        im_feat = clip_model.encode_image(img_t)
        im_feat /= im_feat.norm(dim=-1, keepdim=True)
        probs = (100.0 * im_feat @ text_features.T).softmax(dim=-1)[0].cpu().numpy()
    
    best_idx = int(np.argmax(probs))
    print(f"{name} -> Best: '{prompts[best_idx]}' ({probs[best_idx]*100:.1f}%) | Product: {LABELS[best_idx][1]}")
