# app.py
import streamlit as st
import numpy as np
from PIL import Image, ImageOps, ImageDraw
import tensorflow as tf

# ---------------------------
# CONFIG
# ---------------------------
MODEL_PATH = "brain_tumor_classifier.keras"
IMG_SIZE = 224
CLASS_LABELS = ['pituitary', 'meningioma', 'notumor', 'glioma']

# Parameters for detection
HEATMAP_MIN_PEAK = 0.22          # Must have strong activation
THRESHOLD_REL = 0.65             # Tighter hotspot threshold
MIN_REGION_AREA_RATIO = 0.002    # Ignore tiny false spots
MAX_REGION_AREA_RATIO = 0.25     # Reject big/false regions
BOX_PADDING = 15            # pixels padding around bounding box for zoom crop
ZOOM_CROP_SIZE = (224, 224) # size to show zoomed crop

# ---------------------------
# LOAD MODEL
# ---------------------------
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

model = load_model()


# ---------------------------
# PREPROCESS
# ---------------------------
def preprocess_image(img, size=(IMG_SIZE, IMG_SIZE)):
    img = img.convert("RGB")
    img = ImageOps.fit(img, size, Image.Resampling.LANCZOS)
    arr = np.array(img).astype("float32") / 255.0
    return np.expand_dims(arr, axis=0)


# ---------------------------
# GRAD-CAM heatmap (tensor-safe)
# ---------------------------
def make_gradcam_heatmap(img_array, model, last_conv_layer_name="Conv_1"):
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [
            model.get_layer(last_conv_layer_name).output,
            model.output
        ]
    )

    with tf.GradientTape() as tape:
        conv_output, predictions = grad_model(img_array)
        # predictions may be tensor with batch dim
        predictions = tf.squeeze(predictions)
        pred_index = tf.argmax(predictions)
        pred_output = predictions[pred_index]

    grads = tape.gradient(pred_output, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_output = conv_output[0]

    # compute heatmap and normalize to [0,1]
    heatmap = tf.reduce_sum(pooled_grads * conv_output, axis=-1)
    heatmap = tf.maximum(heatmap, 0)
    denom = tf.reduce_max(heatmap) + 1e-9
    heatmap = heatmap / denom

    return heatmap.numpy()


# ---------------------------
# Connected components (pure numpy flood-fill)
# ---------------------------
def largest_connected_component(mask):
    """
    Find the largest connected component in a boolean 2D mask.
    Returns (y_min, x_min, y_max, x_max) bounding box of the largest component
    and the component area in pixels. If none, returns None.
    """
    h, w = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    largest_area = 0
    largest_bbox = None

    # 4-connectivity
    for y in range(h):
        for x in range(w):
            if mask[y, x] and not visited[y, x]:
                # flood fill stack
                stack = [(y, x)]
                visited[y, x] = True
                ys = []
                xs = []
                while stack:
                    cy, cx = stack.pop()
                    ys.append(cy)
                    xs.append(cx)
                    # neighbors
                    for ny, nx in ((cy-1, cx), (cy+1, cx), (cy, cx-1), (cy, cx+1)):
                        if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not visited[ny, nx]:
                            visited[ny, nx] = True
                            stack.append((ny, nx))
                area = len(ys)
                if area > largest_area:
                    largest_area = area
                    y_min, y_max = min(ys), max(ys)
                    x_min, x_max = min(xs), max(xs)
                    largest_bbox = (y_min, x_min, y_max, x_max)

    if largest_bbox is None:
        return None, 0
    return largest_bbox, largest_area


# ---------------------------
# Create tight red box and zoom crop
# ---------------------------
def detect_and_box(img_pil, heatmap, threshold_rel=THRESHOLD_REL):
    """
    img_pil: original PIL image (any size)
    heatmap: 2D numpy array with values in [0,1] (spatial dims possibly smaller)
    Returns:
      boxed_img_pil (PIL): original image with red rectangle (or original if no box)
      zoom_crop_pil (PIL or None): zoomed crop of boxed region resized to ZOOM_CROP_SIZE, or None
      detection_info (dict): info like area_ratio, chosen_threshold, reason for no-box
    """
    # Resize heatmap to image size using PIL (bilinear)
    heatmap_img = Image.fromarray(np.uint8(heatmap * 255))
    heatmap_resized = np.array(heatmap_img.resize(img_pil.size, resample=Image.BILINEAR)).astype("float32") / 255.0

    # peak activation
    peak = float(heatmap_resized.max())

    info = {"peak": peak, "box_drawn": False, "reason": None}

    # quick reject if peak is tiny
    if peak < HEATMAP_MIN_PEAK:
        info["reason"] = "peak_too_low"
        return img_pil, None, info

    # binary mask using relative threshold of the peak
    thresh = peak * threshold_rel
    mask = heatmap_resized >= thresh

    # find largest connected component
    bbox, area = largest_connected_component(mask)
    if bbox is None or area == 0:
        info["reason"] = "no_component"
        return img_pil, None, info

    img_area = img_pil.size[0] * img_pil.size[1]
    area_ratio = area / float(img_area)

    info["area"] = area
    info["area_ratio"] = area_ratio
    info["threshold_used"] = thresh

    # Reject if region is too small or too large
    if area_ratio < MIN_REGION_AREA_RATIO:
        info["reason"] = "region_too_small"
        return img_pil, None, info
    if area_ratio > MAX_REGION_AREA_RATIO:
        info["reason"] = "region_too_large"
        return img_pil, None, info

    # bbox is in (y_min, x_min, y_max, x_max)
    y1, x1, y2, x2 = bbox

    # Add padding and clip to image bounds
    pad = BOX_PADDING
    left = max(0, x1 - pad)
    top = max(0, y1 - pad)
    right = min(img_pil.size[0] - 1, x2 + pad)
    bottom = min(img_pil.size[1] - 1, y2 + pad)

    # Draw rectangle on a copy
    boxed = img_pil.copy()
    draw = ImageDraw.Draw(boxed)
    draw.rectangle([left, top, right, bottom], outline="red", width=4)

    # Create zoomed crop
    crop = img_pil.crop((left, top, right, bottom))
    zoom_crop = crop.resize(ZOOM_CROP_SIZE, resample=Image.Resampling.LANCZOS)

    info["box_coords"] = (left, top, right, bottom)
    info["box_drawn"] = True

    return boxed, zoom_crop, info


# ---------------------------
# Predict and localize
# ---------------------------
def predict_and_localize(img_pil):
    img_array = preprocess_image(img_pil)  # model-size input
    preds = model.predict(img_array)[0]
    pred_index = int(np.argmax(preds))
    pred_label = CLASS_LABELS[pred_index]
    pred_conf = float(preds[pred_index])

    # If predicted notumor, skip localization entirely
    if pred_label == "notumor":
        return pred_label, preds, None, None, {"reason": "predicted_notumor", "confidence": pred_conf}

    # compute grad-cam heatmap
    heatmap = make_gradcam_heatmap(img_array, model)

    # detect and draw box
    boxed_img, zoom_crop, info = detect_and_box(img_pil, heatmap)

    info["pred_confidence"] = pred_conf
    info["pred_label"] = pred_label

    return pred_label, preds, boxed_img, zoom_crop, info


# ---------------------------
# STREAMLIT UI
# ---------------------------
st.title("🧠 Brain Tumor Detection")
st.write("Upload an MRI image. The model predicts tumor type")

uploaded_file = st.file_uploader("Upload MRI Image (jpg/png)", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", width=350)

    if st.button("Predict"):
        with st.spinner("Analyzing..."):
            label, probs, boxed_img, zoom_crop, info = predict_and_localize(img)

        st.markdown(    f"""
    <h1 style="text-align:center; 
               font-size:35px; 
               font-weight:900; 
               color:#D91C1C;">
        PREDICTED CLASS: {label.upper()}<br>
        <span style="font-size:28px;">
            CONFIDENCE: {probs.max():.3f}
        </span>
    </h1>
    """,
    unsafe_allow_html=True)

        # Show probabilities
        st.subheader("Class probabilities")
        for cls, p in zip(CLASS_LABELS, probs):
            st.write(f"- {cls}: {p:.3f}")

        # If box drawn, show boxed image and zoom crop
        if boxed_img is not None and info.get("box_drawn", False):
            st.subheader("🔴 Tumor localization (red box)")
            st.image(boxed_img, caption="Tumor box", width=400)
            st.subheader("🔎 Zoomed crop")
            st.image(zoom_crop, caption="Zoomed tumor crop", width=300)

        else:
            # No valid region found
            reason = info.get("reason", "unknown")
            st.info(f"No confident tumor region detected (reason: {reason}).")
            st.write("Detection info:", info)
