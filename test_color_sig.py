import cv2
import numpy as np

def detect_grocery_by_color_signature(crop_bgr):
    """
    Analyzes dominant colors and visual features to identify grocery packets.
    Returns (product_id, name, brand, price, emoji, confidence) or None.
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return None
        
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    
    total_pixels = crop_bgr.shape[0] * crop_bgr.shape[1]
    if total_pixels < 1000:
        return None

    # Blue mask (Lays Magic Masala - Iconic Royal Blue)
    blue_mask = cv2.inRange(hsv, np.array([95, 70, 50]), np.array([130, 255, 255]))
    blue_ratio = np.count_nonzero(blue_mask) / total_pixels
    
    # Orange mask (Kurkure Masala Munch - Vibrant Orange/Red)
    orange_mask = cv2.inRange(hsv, np.array([5, 110, 100]), np.array([18, 255, 255]))
    orange_ratio = np.count_nonzero(orange_mask) / total_pixels

    # Yellow mask (Lays Classic Salted / Maggi)
    yellow_mask = cv2.inRange(hsv, np.array([20, 90, 100]), np.array([35, 255, 255]))
    yellow_ratio = np.count_nonzero(yellow_mask) / total_pixels

    # Purple mask (Cadbury Dairy Milk Silk)
    purple_mask = cv2.inRange(hsv, np.array([130, 60, 40]), np.array([160, 255, 220]))
    purple_ratio = np.count_nonzero(purple_mask) / total_pixels

    # Chocolate / Brown mask (Hide & Seek Biscuits)
    brown_mask = cv2.inRange(hsv, np.array([8, 60, 30]), np.array([22, 180, 130]))
    brown_ratio = np.count_nonzero(brown_mask) / total_pixels

    # Red mask (Coca-Cola / Thums Up)
    red_mask1 = cv2.inRange(hsv, np.array([0, 100, 70]), np.array([8, 255, 255]))
    red_mask2 = cv2.inRange(hsv, np.array([170, 100, 70]), np.array([180, 255, 255]))
    red_ratio = (np.count_nonzero(red_mask1) + np.count_nonzero(red_mask2)) / total_pixels

    signatures = [
        (blue_ratio, 'prod_lays_blue', "Lays India's Magic Masala", 'PepsiCo', 20, '🥔', 'Chips'),
        (orange_ratio, 'prod_kurkure', 'Kurkure Masala Munch', 'PepsiCo', 20, '🌽', 'Snacks'),
        (yellow_ratio, 'prod_lays_yellow', 'Lays Classic Salted', 'PepsiCo', 20, '🥔', 'Chips'),
        (purple_ratio, 'prod_silk', 'Dairy Milk Silk', 'Cadbury', 80, '🍫', 'Chocolate'),
        (brown_ratio, 'prod_hide_seek', 'Hide & Seek Chocolate Chip', 'Parle', 30, '🍪', 'Biscuits'),
        (red_ratio, 'prod_coke', 'Coca-Cola 500ml', 'Coca-Cola', 40, '🥤', 'Drinks'),
    ]
    
    signatures.sort(key=lambda x: x[0], reverse=True)
    best_ratio, pid, name, brand, price, emoji, cat = signatures[0]
    
    if best_ratio > 0.18:  # At least 18% of the packet has the brand color signature
        conf = min(0.98, 0.65 + best_ratio)
        return {
            'product_id': pid,
            'name': name,
            'brand': brand,
            'price': price,
            'emoji': emoji,
            'category': cat,
            'confidence': round(conf, 2)
        }
    return None

# Test with a synthetic blue image (representing Lays blue packet)
synthetic_lays = np.zeros((200, 200, 3), dtype=np.uint8)
synthetic_lays[:] = (180, 80, 20) # Blue in BGR
res = detect_grocery_by_color_signature(synthetic_lays)
print("Synthetic Lays detection result:", res)
