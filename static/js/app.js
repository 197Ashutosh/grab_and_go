/* ============================================================
   Grab & Go - Smart Retail Checkout
   Main Application JavaScript
   ============================================================ */

// ===== STATE MANAGEMENT =====

/** Global application state */
const AppState = {
    cart: [],                   // Array of { product, quantity }
    products: [],               // All products from API
    ws: null,                   // WebSocket connection instance
    video: null,                // HTMLVideoElement reference
    mediaStream: null,          // MediaStream from getUserMedia
    detectionLoop: null,        // Interval ID for frame sending loop
    isDetecting: false,         // Whether detection loop is active
    detectionCooldowns: {},     // { productId: lastDetectedTimestamp }
    COOLDOWN_MS: 3000,          // 3-second cooldown per item type
    FRAME_INTERVAL_MS: 500,     // Send frame every 500ms
    frameWidth: 640,            // Default frame width (updated on camera init)
    frameHeight: 480,           // Default frame height (updated on camera init)
    reconnectAttempts: 0,       // WebSocket reconnection attempt count
    MAX_RECONNECT_ATTEMPTS: 10, // Max auto-reconnect tries
    reconnectTimeout: null,     // Timeout ID for reconnection delay
    audioCtx: null,             // Web Audio API context (lazy init)
};


// ===== CAMERA MODULE =====

/**
 * Initialize the webcam camera feed.
 * Requests user permission, starts the video stream, and sets up the
 * canvas overlay dimensions to match the video.
 */
const initCamera = async () => {
    try {
        const video = document.getElementById('camera-video');
        if (!video) {
            console.error('Camera video element not found');
            return;
        }

        const stream = await navigator.mediaDevices.getUserMedia({
            video: {
                width: { ideal: 1280 },
                height: { ideal: 720 },
                facingMode: 'environment'
            },
            audio: false
        });

        video.srcObject = stream;
        await video.play();

        AppState.video = video;
        AppState.mediaStream = stream;

        // Wait for video metadata to load to get actual dimensions
        video.addEventListener('loadedmetadata', () => {
            AppState.frameWidth = video.videoWidth;
            AppState.frameHeight = video.videoHeight;

            // Size the detection overlay canvas to match
            const overlay = document.getElementById('detection-overlay');
            if (overlay) {
                overlay.width = video.videoWidth;
                overlay.height = video.videoHeight;
            }
        });

        // Update UI elements
        const placeholder = document.querySelector('.camera-feed__placeholder');
        if (placeholder) placeholder.style.display = 'none';

        const cameraStatus = document.querySelector('.camera-status');
        if (cameraStatus) cameraStatus.style.display = 'flex';

        showToast('Camera connected successfully', 'success');
        console.log('[Camera] Stream started');

    } catch (err) {
        console.error('[Camera] Failed to initialize:', err);
        showToast('Camera access denied. Please allow camera permissions.', 'error');
    }
};

/**
 * Stop the camera video stream and release all tracks.
 */
const stopCamera = () => {
    if (AppState.mediaStream) {
        AppState.mediaStream.getTracks().forEach(track => track.stop());
        AppState.mediaStream = null;
    }

    if (AppState.video) {
        AppState.video.srcObject = null;
    }

    stopDetectionLoop();

    // Show placeholder again
    const placeholder = document.querySelector('.camera-feed__placeholder');
    if (placeholder) placeholder.style.display = 'flex';

    const cameraStatus = document.querySelector('.camera-status');
    if (cameraStatus) cameraStatus.style.display = 'none';

    // Clear detection overlay
    const overlay = document.getElementById('detection-overlay');
    if (overlay) {
        const ctx = overlay.getContext('2d');
        ctx.clearRect(0, 0, overlay.width, overlay.height);
    }

    console.log('[Camera] Stream stopped');
};

/**
 * Capture the current video frame as a base64-encoded JPEG string.
 * @returns {string|null} Base64-encoded image data (without prefix), or null on failure.
 */
const captureFrame = () => {
    const video = AppState.video;
    if (!video || video.readyState < 2) return null; // HAVE_CURRENT_DATA

    const canvas = document.createElement('canvas');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0);

    // Encode as JPEG with 70% quality for reasonable bandwidth
    const base64 = canvas.toDataURL('image/jpeg', 0.7).split(',')[1];
    return base64;
};

/**
 * Start the detection loop that sends frames to the server via WebSocket
 * every FRAME_INTERVAL_MS milliseconds.
 */
const startDetectionLoop = () => {
    if (AppState.isDetecting) return;

    AppState.isDetecting = true;
    AppState.detectionLoop = setInterval(() => {
        captureAndSend();
    }, AppState.FRAME_INTERVAL_MS);

    // Update start/stop button UI
    const toggleBtn = document.getElementById('btn-toggle-detection');
    if (toggleBtn) {
        toggleBtn.classList.add('active');
        toggleBtn.title = 'Stop Detection';
    }

    console.log('[Detection] Loop started');
};

/**
 * Stop sending frames to the server.
 */
const stopDetectionLoop = () => {
    if (AppState.detectionLoop) {
        clearInterval(AppState.detectionLoop);
        AppState.detectionLoop = null;
    }
    AppState.isDetecting = false;

    // Update start/stop button UI
    const toggleBtn = document.getElementById('btn-toggle-detection');
    if (toggleBtn) {
        toggleBtn.classList.remove('active');
        toggleBtn.title = 'Start Detection';
    }

    // Clear detection overlay
    const overlay = document.getElementById('detection-overlay');
    if (overlay) {
        const ctx = overlay.getContext('2d');
        ctx.clearRect(0, 0, overlay.width, overlay.height);
    }

    // Hide the detected-item preview
    const preview = document.querySelector('.detected-item-preview');
    if (preview) preview.classList.remove('show');

    console.log('[Detection] Loop stopped');
};

/**
 * Capture the current frame and send it to the server over WebSocket.
 */
const captureAndSend = async () => {
    const base64 = captureFrame();
    if (!base64) return;

    if (AppState.ws && AppState.ws.readyState === WebSocket.OPEN) {
        AppState.ws.send(JSON.stringify({ frame: base64 }));
    }
};

/**
 * Draw bounding boxes and labels on the detection overlay canvas.
 * @param {Array} detections - Array of detection objects from the server.
 *   Each: { bbox: [x1,y1,x2,y2], confidence: float, label: string, product: {...} }
 */
const drawDetections = (detections) => {
    const overlay = document.getElementById('detection-overlay');
    if (!overlay) return;

    const ctx = overlay.getContext('2d');
    ctx.clearRect(0, 0, overlay.width, overlay.height);

    if (!detections || detections.length === 0) return;

    // Scale bounding box coordinates to match the overlay canvas size
    const scaleX = overlay.width / AppState.frameWidth;
    const scaleY = overlay.height / AppState.frameHeight;

    detections.forEach(det => {
        const [x1, y1, x2, y2] = det.bbox;
        const sx1 = x1 * scaleX;
        const sy1 = y1 * scaleY;
        const sw = (x2 - x1) * scaleX;
        const sh = (y2 - y1) * scaleY;

        // Draw bounding box
        ctx.strokeStyle = '#00d4aa';
        ctx.lineWidth = 2;
        ctx.strokeRect(sx1, sy1, sw, sh);

        // Compose label text
        const productName = det.product ? det.product.name : det.label;
        const confidencePercent = Math.round(det.confidence * 100);
        const labelText = `${productName} (${confidencePercent}%)`;

        // Measure label for background rect
        ctx.font = '14px Segoe UI';
        const textMetrics = ctx.measureText(labelText);
        const labelWidth = textMetrics.width + 20;
        const labelHeight = 25;

        // Draw label background
        ctx.fillStyle = 'rgba(0, 212, 170, 0.8)';
        ctx.fillRect(sx1, sy1 - labelHeight, labelWidth, labelHeight);

        // Draw label text
        ctx.fillStyle = '#000';
        ctx.font = '14px Segoe UI';
        ctx.fillText(labelText, sx1 + 5, sy1 - 7);
    });
};


// ===== WEBSOCKET MODULE =====

/**
 * Establish a WebSocket connection to the detection endpoint.
 */
const connectWebSocket = () => {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws/detect`;

    console.log(`[WebSocket] Connecting to ${wsUrl}...`);

    try {
        AppState.ws = new WebSocket(wsUrl);

        AppState.ws.onopen = () => {
            console.log('[WebSocket] Connected');
            AppState.reconnectAttempts = 0;
            updateConnectionStatus(true);
            showToast('Connected to detection server', 'success');
        };

        AppState.ws.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                handleDetectionResult(data);
            } catch (err) {
                console.error('[WebSocket] Failed to parse message:', err);
            }
        };

        AppState.ws.onclose = (event) => {
            console.log(`[WebSocket] Disconnected (code: ${event.code})`);
            updateConnectionStatus(false);
            reconnectWebSocket();
        };

        AppState.ws.onerror = (err) => {
            console.error('[WebSocket] Error:', err);
            updateConnectionStatus(false);
        };

    } catch (err) {
        console.error('[WebSocket] Failed to create connection:', err);
        updateConnectionStatus(false);
        reconnectWebSocket();
    }
};

/**
 * Handle a detection result received from the WebSocket server.
 * @param {Object} data - The server response: { detections: [...], frame_id, timestamp }
 */
const handleDetectionResult = (data) => {
    if (!data.detections || data.detections.length === 0) {
        // No detections – clear overlay and hide preview
        drawDetections([]);
        const preview = document.querySelector('.detected-item-preview');
        if (preview) preview.classList.remove('show');
        return;
    }

    // Draw bounding boxes
    drawDetections(data.detections);

    // Process each detection for auto-add-to-cart
    data.detections.forEach(det => {
        if (!det.product) return;

        const productId = det.product.id;
        const now = Date.now();

        // Check cooldown – avoid re-adding the same item too frequently
        if (AppState.detectionCooldowns[productId] &&
            (now - AppState.detectionCooldowns[productId]) < AppState.COOLDOWN_MS) {
            return; // Still in cooldown
        }

        // Update cooldown timestamp
        AppState.detectionCooldowns[productId] = now;

        // Auto-add to cart
        addToCart(det.product);

        // Play detection beep
        playDetectionBeep();

        // Show detected item preview
        showDetectedItemPreview(det.product, det.confidence);

        // Highlight product in catalog
        highlightProductCard(productId);
    });
};

/**
 * Attempt to reconnect the WebSocket with exponential backoff.
 */
const reconnectWebSocket = () => {
    if (AppState.reconnectAttempts >= AppState.MAX_RECONNECT_ATTEMPTS) {
        console.log('[WebSocket] Max reconnection attempts reached');
        showToast('Lost connection to server. Please refresh the page.', 'error');
        return;
    }

    const delay = Math.min(1000 * Math.pow(2, AppState.reconnectAttempts), 30000);
    AppState.reconnectAttempts++;

    console.log(`[WebSocket] Reconnecting in ${delay}ms (attempt ${AppState.reconnectAttempts})...`);

    if (AppState.reconnectTimeout) {
        clearTimeout(AppState.reconnectTimeout);
    }

    AppState.reconnectTimeout = setTimeout(() => {
        connectWebSocket();
    }, delay);
};


// ===== CART MODULE =====

/**
 * Add a product to the shopping cart.
 * If the product already exists, its quantity is incremented.
 * Sends a REST API call to sync with the backend.
 * @param {Object} product - The product object { id, name, price, image, category, ... }
 */
const addToCart = async (product) => {
    try {
        // API call to add item server-side
        const response = await fetch('/api/cart/add', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                product_id: product.id,
                quantity: 1
            })
        });

        if (!response.ok) {
            const err = await response.json();
            throw new Error(err.detail || 'Failed to add item');
        }

        // Update local cart state
        const existing = AppState.cart.find(item => item.product.id === product.id);
        if (existing) {
            existing.quantity += 1;
        } else {
            AppState.cart.push({ product, quantity: 1 });
        }

        // Update UI
        updateCartUI();
        showToast(`${product.name} added to cart`, 'success');

    } catch (err) {
        console.error('[Cart] Add failed:', err);
        showToast(`Failed to add item: ${err.message}`, 'error');
    }
};

/**
 * Remove a product entirely from the shopping cart.
 * @param {string|number} productId - The ID of the product to remove.
 */
const removeFromCart = async (productId) => {
    try {
        const response = await fetch(`/api/cart/remove/${productId}`, {
            method: 'DELETE'
        });

        if (!response.ok) {
            const err = await response.json();
            throw new Error(err.detail || 'Failed to remove item');
        }

        // Update local state
        AppState.cart = AppState.cart.filter(item => item.product.id !== productId);

        updateCartUI();
        showToast('Item removed from cart', 'info');

    } catch (err) {
        console.error('[Cart] Remove failed:', err);
        showToast(`Failed to remove item: ${err.message}`, 'error');
    }
};

/**
 * Update the quantity of a product in the cart.
 * @param {string|number} productId - Product ID.
 * @param {number} delta - Change in quantity (+1 or -1).
 */
const updateCartQuantity = async (productId, delta) => {
    const item = AppState.cart.find(i => i.product.id === productId);
    if (!item) return;

    const newQty = item.quantity + delta;

    if (newQty <= 0) {
        await removeFromCart(productId);
        return;
    }

    try {
        const response = await fetch('/api/cart/update', {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                product_id: productId,
                quantity: newQty
            })
        });

        if (!response.ok) {
            const err = await response.json();
            throw new Error(err.detail || 'Failed to update quantity');
        }

        item.quantity = newQty;
        updateCartUI();

    } catch (err) {
        console.error('[Cart] Update quantity failed:', err);
        showToast(`Failed to update quantity: ${err.message}`, 'error');
    }
};

/**
 * Clear all items from the shopping cart.
 */
const clearCart = async () => {
    try {
        const response = await fetch('/api/cart/clear', {
            method: 'DELETE'
        });

        if (!response.ok) {
            const err = await response.json();
            throw new Error(err.detail || 'Failed to clear cart');
        }

        AppState.cart = [];
        updateCartUI();
        showToast('Cart cleared', 'info');

    } catch (err) {
        console.error('[Cart] Clear failed:', err);
        showToast(`Failed to clear cart: ${err.message}`, 'error');
    }
};

/**
 * Re-render the cart panel UI based on current AppState.cart.
 */
const updateCartUI = () => {
    const cartItemsContainer = document.getElementById('cart-items');
    const cartEmpty = document.getElementById('cart-empty');
    const cartFooter = document.getElementById('cart-footer');
    const cartCountBadge = document.getElementById('cart-count');
    const cartSubtotalEl = document.getElementById('cart-subtotal');
    const cartTotalEl = document.getElementById('cart-total');
    const checkoutBtn = document.getElementById('btn-checkout');

    if (!cartItemsContainer) return;

    const totalItems = AppState.cart.reduce((sum, item) => sum + item.quantity, 0);
    const subtotal = getCartTotal();

    // Update badge
    if (cartCountBadge) {
        cartCountBadge.textContent = totalItems;
        cartCountBadge.classList.toggle('show', totalItems > 0);
    }

    // Show/hide empty state
    if (AppState.cart.length === 0) {
        cartItemsContainer.innerHTML = '';
        if (cartEmpty) cartEmpty.style.display = 'flex';
        if (cartFooter) cartFooter.style.display = 'none';
        return;
    }

    if (cartEmpty) cartEmpty.style.display = 'none';
    if (cartFooter) cartFooter.style.display = 'block';

    // Render cart items
    cartItemsContainer.innerHTML = AppState.cart.map(item => `
        <div class="cart-item" data-product-id="${item.product.id}">
            <div class="cart-item__image">
                ${item.product.image
                    ? `<img src="${item.product.image}" alt="${item.product.name}">`
                    : `<span>${item.product.emoji || '📦'}</span>`
                }
            </div>
            <div class="cart-item__details">
                <div class="cart-item__name">${item.product.name}</div>
                <div class="cart-item__price">${formatPrice(item.product.price)}</div>
            </div>
            <div class="cart-item__quantity">
                <button class="cart-item__qty-btn" onclick="updateCartQuantity('${item.product.id}', -1)" title="Decrease">−</button>
                <span class="cart-item__qty-value">${item.quantity}</span>
                <button class="cart-item__qty-btn" onclick="updateCartQuantity('${item.product.id}', 1)" title="Increase">+</button>
            </div>
            <button class="cart-item__remove" onclick="removeFromCart('${item.product.id}')" title="Remove">✕</button>
        </div>
    `).join('');

    // Update summary
    if (cartSubtotalEl) cartSubtotalEl.textContent = formatPrice(subtotal);
    if (cartTotalEl) cartTotalEl.textContent = formatPrice(subtotal);

    // Enable/disable checkout button
    if (checkoutBtn) checkoutBtn.disabled = AppState.cart.length === 0;
};

/**
 * Calculate the total price of all items in the cart.
 * @returns {number} Total amount.
 */
const getCartTotal = () => {
    return AppState.cart.reduce((total, item) => {
        return total + (item.product.price * item.quantity);
    }, 0);
};


// ===== PRODUCT CATALOG =====

/**
 * Fetch all products from the API and render them in the product grid.
 */
const loadProducts = async () => {
    try {
        const response = await fetch('/api/products');
        if (!response.ok) throw new Error('Failed to fetch products');

        const data = await response.json();
        AppState.products = data.products || data;

        renderProductGrid();

    } catch (err) {
        console.error('[Products] Load failed:', err);
        showToast('Failed to load products', 'error');

        // Render empty state
        const grid = document.getElementById('product-grid');
        if (grid) {
            grid.innerHTML = `
                <div style="grid-column: 1/-1; text-align: center; padding: 3rem; color: var(--text-secondary);">
                    <p>Unable to load products. Please try again.</p>
                    <button class="btn btn--secondary mt-2" onclick="loadProducts()">Retry</button>
                </div>
            `;
        }
    }
};

/**
 * Render the entire product grid from the loaded products.
 */
const renderProductGrid = () => {
    const grid = document.getElementById('product-grid');
    if (!grid) return;

    if (AppState.products.length === 0) {
        grid.innerHTML = `
            <div style="grid-column: 1/-1; text-align: center; padding: 3rem; color: var(--text-secondary);">
                <p>No products available.</p>
            </div>
        `;
        return;
    }

    grid.innerHTML = AppState.products.map(product => renderProductCard(product)).join('');

    // Update product count display
    const countEl = document.querySelector('.catalog-section__count');
    if (countEl) {
        countEl.textContent = `${AppState.products.length} items`;
    }
};

/**
 * Render a single product card HTML string.
 * @param {Object} product - Product object.
 * @returns {string} HTML string for the product card.
 */
const renderProductCard = (product) => {
    return `
        <div class="product-card" data-product-id="${product.id}" id="product-${product.id}">
            <div class="product-card__image">
                ${product.image
                    ? `<img src="${product.image}" alt="${product.name}" loading="lazy">`
                    : `<span>${product.emoji || '📦'}</span>`
                }
            </div>
            <div class="product-card__body">
                <div class="product-card__category">${product.category || 'General'}</div>
                <div class="product-card__name">${product.name}</div>
                <div class="product-card__price-row">
                    <span class="product-card__price">${formatPrice(product.price)}</span>
                    <button class="product-card__add-btn" onclick="addToCart(${JSON.stringify(product).replace(/"/g, '&quot;')})" title="Add to cart">
                        +
                    </button>
                </div>
            </div>
        </div>
    `;
};


// ===== CHECKOUT =====

/**
 * Process the checkout by sending the cart to the server.
 */
const processCheckout = async () => {
    if (AppState.cart.length === 0) {
        showToast('Your cart is empty', 'warning');
        return;
    }

    const paymentMethodEl = document.querySelector('.payment-method.selected');
    const paymentMethod = paymentMethodEl
        ? paymentMethodEl.dataset.method
        : 'cash';

    showLoading();

    try {
        const response = await fetch('/api/checkout', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                items: AppState.cart.map(item => ({
                    product_id: item.product.id,
                    quantity: item.quantity,
                    price: item.product.price
                })),
                total: getCartTotal(),
                payment_method: paymentMethod
            })
        });

        if (!response.ok) {
            const err = await response.json();
            throw new Error(err.detail || 'Checkout failed');
        }

        const result = await response.json();

        // Clear cart
        AppState.cart = [];
        updateCartUI();

        hideLoading();

        // Navigate to success page or show success UI
        showPaymentSuccess(result.order_id || result.id || 'ORD-' + Date.now());

    } catch (err) {
        hideLoading();
        console.error('[Checkout] Failed:', err);
        showToast(`Checkout failed: ${err.message}`, 'error');
    }
};

/**
 * Toggle payment form fields based on selected payment method.
 * @param {string} method - Payment method identifier (e.g., 'card', 'upi', 'cash').
 */
const showPaymentForm = (method) => {
    // Update selected state on payment method options
    document.querySelectorAll('.payment-method').forEach(el => {
        el.classList.toggle('selected', el.dataset.method === method);
    });

    // Show/hide relevant form fields
    const cardFields = document.getElementById('card-fields');
    const upiFields = document.getElementById('upi-fields');

    if (cardFields) cardFields.style.display = method === 'card' ? 'block' : 'none';
    if (upiFields) upiFields.style.display = method === 'upi' ? 'block' : 'none';
};

/**
 * Display the payment success page with animation and confetti.
 * @param {string} orderId - The order ID to display.
 */
const showPaymentSuccess = (orderId) => {
    const mainContent = document.getElementById('main-content');
    if (!mainContent) return;

    mainContent.innerHTML = `
        <div class="success-page page-enter">
            <div class="success-checkmark">✓</div>
            <h1 class="success-page__title">Payment Successful!</h1>
            <p class="success-page__subtitle">Thank you for shopping with Grab & Go</p>
            <div class="success-page__order-id">Order ID: ${orderId}</div>
            <div class="success-page__actions">
                <button class="btn btn--primary btn--lg" onclick="window.location.href='/'">
                    🛒 Continue Shopping
                </button>
                <button class="btn btn--secondary btn--lg" onclick="window.location.href='/'">
                    📋 View Receipt
                </button>
            </div>
        </div>
    `;

    // Trigger confetti
    launchConfetti();
};


// ===== UI UTILITIES =====

/**
 * Show a toast notification.
 * @param {string} message - The message to display.
 * @param {string} type - Toast type: 'success', 'error', 'info', 'warning'.
 * @param {number} duration - Auto-dismiss duration in ms (default: 3500).
 */
const showToast = (message, type = 'info', duration = 3500) => {
    let container = document.querySelector('.toast-container');
    if (!container) {
        container = document.createElement('div');
        container.className = 'toast-container';
        document.body.appendChild(container);
    }

    const icons = {
        success: '✅',
        error: '❌',
        info: 'ℹ️',
        warning: '⚠️'
    };

    const toast = document.createElement('div');
    toast.className = `toast toast--${type}`;
    toast.innerHTML = `
        <span class="toast__icon">${icons[type] || icons.info}</span>
        <span class="toast__message">${message}</span>
        <button class="toast__close" onclick="this.closest('.toast').remove()">✕</button>
    `;

    container.appendChild(toast);

    // Auto-dismiss
    setTimeout(() => {
        toast.classList.add('removing');
        setTimeout(() => toast.remove(), 300);
    }, duration);
};

/**
 * Update the WebSocket connection status indicator in the header.
 * @param {boolean} connected - Whether the connection is active.
 */
const updateConnectionStatus = (connected) => {
    const dot = document.querySelector('.status-indicator__dot');
    const label = document.querySelector('.status-indicator__label');

    if (dot) {
        dot.classList.toggle('connected', connected);
    }
    if (label) {
        label.textContent = connected ? 'Connected' : 'Disconnected';
    }
};

/**
 * Format a numeric amount as Indian Rupee currency string.
 * @param {number} amount - The amount to format.
 * @returns {string} Formatted price string (e.g., "₹1,299").
 */
const formatPrice = (amount) => {
    return '₹' + Number(amount).toLocaleString('en-IN', {
        minimumFractionDigits: 0,
        maximumFractionDigits: 2
    });
};

/**
 * Show the global loading overlay.
 */
const showLoading = () => {
    let overlay = document.querySelector('.loading-overlay');
    if (!overlay) {
        overlay = document.createElement('div');
        overlay.className = 'loading-overlay';
        overlay.innerHTML = `
            <div class="spinner"></div>
            <div class="loading-overlay__text">Processing...</div>
        `;
        document.body.appendChild(overlay);
    }
    overlay.classList.add('show');
};

/**
 * Hide the global loading overlay.
 */
const hideLoading = () => {
    const overlay = document.querySelector('.loading-overlay');
    if (overlay) {
        overlay.classList.remove('show');
    }
};


// ===== DETECTION AUDIO & VISUAL FEEDBACK =====

/**
 * Play a short detection beep sound using the Web Audio API.
 * Generates a brief synthesized tone – no external audio file required.
 */
const playDetectionBeep = () => {
    try {
        // Lazy-initialize AudioContext (requires user gesture on some browsers)
        if (!AppState.audioCtx) {
            AppState.audioCtx = new (window.AudioContext || window.webkitAudioContext)();
        }

        const ctx = AppState.audioCtx;

        // Create oscillator for a pleasant, short beep
        const oscillator = ctx.createOscillator();
        const gainNode = ctx.createGain();

        oscillator.connect(gainNode);
        gainNode.connect(ctx.destination);

        oscillator.type = 'sine';
        oscillator.frequency.setValueAtTime(880, ctx.currentTime);       // A5 note
        oscillator.frequency.exponentialRampToValueAtTime(1320, ctx.currentTime + 0.05); // Sweep up to E6

        gainNode.gain.setValueAtTime(0.15, ctx.currentTime);
        gainNode.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.15);

        oscillator.start(ctx.currentTime);
        oscillator.stop(ctx.currentTime + 0.15);

    } catch (err) {
        // Audio not supported or blocked – silently ignore
        console.warn('[Audio] Beep failed:', err);
    }
};

/**
 * Show the detected item preview card below the camera feed.
 * @param {Object} product - The detected product.
 * @param {number} confidence - Detection confidence (0-1).
 */
const showDetectedItemPreview = (product, confidence) => {
    const preview = document.querySelector('.detected-item-preview');
    if (!preview) return;

    const iconEl = preview.querySelector('.detected-item-preview__icon');
    const nameEl = preview.querySelector('.detected-item-preview__name');
    const metaEl = preview.querySelector('.detected-item-preview__meta');
    const priceEl = preview.querySelector('.detected-item-preview__price');

    if (iconEl) iconEl.textContent = product.emoji || '📦';
    if (nameEl) nameEl.textContent = product.name;
    if (metaEl) metaEl.textContent = `Confidence: ${Math.round(confidence * 100)}% • Auto-added to cart`;
    if (priceEl) priceEl.textContent = formatPrice(product.price);

    preview.classList.add('show');

    // Auto-hide after 3 seconds
    clearTimeout(preview._hideTimeout);
    preview._hideTimeout = setTimeout(() => {
        preview.classList.remove('show');
    }, 3000);
};

/**
 * Briefly highlight a product card in the catalog grid to draw attention.
 * @param {string|number} productId - The product ID to highlight.
 */
const highlightProductCard = (productId) => {
    const card = document.getElementById(`product-${productId}`);
    if (!card) return;

    card.classList.add('highlighted');
    card.scrollIntoView({ behavior: 'smooth', block: 'nearest' });

    // Remove highlight after animation completes
    setTimeout(() => {
        card.classList.remove('highlighted');
    }, 2000);
};


// ===== CONFETTI =====

/**
 * Launch a confetti animation across the screen.
 * Creates multiple colored confetti pieces that fall from the top.
 */
const launchConfetti = () => {
    const container = document.createElement('div');
    container.className = 'confetti-container';
    document.body.appendChild(container);

    const colors = ['#00d4aa', '#ff6b35', '#ffaa00', '#ff4444', '#3b82f6', '#a855f7', '#ec4899'];
    const pieceCount = 60;

    for (let i = 0; i < pieceCount; i++) {
        const piece = document.createElement('div');
        piece.className = 'confetti-piece';

        const color = colors[Math.floor(Math.random() * colors.length)];
        const left = Math.random() * 100;
        const size = Math.random() * 8 + 6;
        const fallDuration = Math.random() * 2 + 2;
        const fallDelay = Math.random() * 1.5;

        piece.style.cssText = `
            left: ${left}%;
            width: ${size}px;
            height: ${size}px;
            background: ${color};
            --fall-duration: ${fallDuration}s;
            --fall-delay: ${fallDelay}s;
        `;

        container.appendChild(piece);
    }

    // Remove confetti container after all animations finish
    setTimeout(() => {
        container.remove();
    }, 5000);
};


// ===== EVENT LISTENER SETUP =====

/**
 * Attach all event listeners for camera controls, cart actions, and page interactions.
 */
const setupEventListeners = () => {
    // Camera start button
    const btnStartCamera = document.getElementById('btn-start-camera');
    if (btnStartCamera) {
        btnStartCamera.addEventListener('click', async () => {
            if (AppState.mediaStream) {
                stopCamera();
                btnStartCamera.title = 'Start Camera';
            } else {
                await initCamera();
                btnStartCamera.title = 'Stop Camera';
            }
        });
    }

    // Toggle detection button
    const btnToggleDetection = document.getElementById('btn-toggle-detection');
    if (btnToggleDetection) {
        btnToggleDetection.addEventListener('click', () => {
            if (!AppState.mediaStream) {
                showToast('Please start the camera first', 'warning');
                return;
            }

            if (AppState.isDetecting) {
                stopDetectionLoop();
                showToast('Detection paused', 'info');
            } else {
                startDetectionLoop();
                showToast('Detection started', 'success');
            }
        });
    }

    // Clear cart button
    const btnClearCart = document.getElementById('btn-clear-cart');
    if (btnClearCart) {
        btnClearCart.addEventListener('click', () => {
            if (AppState.cart.length === 0) {
                showToast('Cart is already empty', 'info');
                return;
            }
            if (confirm('Clear all items from the cart?')) {
                clearCart();
            }
        });
    }

    // Checkout button
    const btnCheckout = document.getElementById('btn-checkout');
    if (btnCheckout) {
        btnCheckout.addEventListener('click', () => {
            if (AppState.cart.length === 0) {
                showToast('Your cart is empty', 'warning');
                return;
            }
            // Navigate to checkout page or show checkout modal
            window.location.href = '/checkout';
        });
    }

    // Payment method selection (on checkout page)
    document.querySelectorAll('.payment-method').forEach(el => {
        el.addEventListener('click', () => {
            showPaymentForm(el.dataset.method);
        });
    });

    // Pay now button (on checkout page)
    const btnPayNow = document.getElementById('btn-pay-now');
    if (btnPayNow) {
        btnPayNow.addEventListener('click', (e) => {
            e.preventDefault();
            processCheckout();
        });
    }

    // Keyboard shortcut: Escape to stop detection
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && AppState.isDetecting) {
            stopDetectionLoop();
            showToast('Detection stopped', 'info');
        }
    });
};


// ===== INITIALIZATION =====

/**
 * Initialize the application when the DOM is fully loaded.
 */
document.addEventListener('DOMContentLoaded', () => {
    console.log('[App] Grab & Go initializing...');

    // Load products catalog
    loadProducts();

    // Set up all event listeners
    setupEventListeners();

    // Connect WebSocket for real-time detection
    connectWebSocket();

    // Initialize cart UI (may have items from session)
    updateCartUI();

    console.log('[App] Initialization complete');
});
