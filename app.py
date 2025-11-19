# app.py - Flask app supporting index input or image upload,
# and result page with predicted class, confidence, class probabilities,
# plus showing the original Olivetti image (if index chosen) or uploaded image.
from flask import Flask, request, render_template, redirect, url_for
import joblib
import numpy as np
from PIL import Image
import io
import os
import base64
from sklearn.datasets import fetch_olivetti_faces

app = Flask(__name__)

MODEL_PATH = "savedmodel.pth"

# Load model bundle if available
model = None
if os.path.exists(MODEL_PATH):
    bundle = joblib.load(MODEL_PATH)
    model = bundle.get("model", None)
else:
    print("Warning: savedmodel.pth not found. Run `python train.py` to generate it before predictions.")

# Load Olivetti dataset once (used when user enters an index)
# This call downloads the dataset the first time; ensure internet available.
try:
    olivetti = fetch_olivetti_faces()
    olivetti_X = olivetti.data           # shape (400, 4096)
    olivetti_images = olivetti.images    # shape (400, 64, 64)
    olivetti_targets = olivetti.target   # shape (400,)
except Exception as e:
    olivetti = None
    olivetti_X = None
    olivetti_images = None
    olivetti_targets = None
    print("Could not fetch Olivetti dataset in app:", e)

def preprocess_image_bytes(file_bytes):
    """Convert uploaded bytes to Olivetti-like flattened vector (1,4096)."""
    img = Image.open(io.BytesIO(file_bytes)).convert("L")   # grayscale
    img = img.resize((64,64))
    arr = np.array(img, dtype=np.float32) / 255.0
    flat = arr.flatten().reshape(1, -1)
    return flat, img  # return PIL image for later display

def pil_image_to_base64(pil_img):
    """Encode PIL image to base64 data URL for inline display."""
    buffered = io.BytesIO()
    pil_img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{img_str}"

@app.route("/", methods=["GET"])
def index():
    # show index/upload form
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return render_template("predict.html", error="Model not available. Run python train.py first.")

    # Two possible inputs: 'index' (0-399) OR file upload 'image'
    index_value = request.form.get("index", "").strip()
    uploaded_file = request.files.get("image", None)

    X = None
    display_img_b64 = None
    chosen_source = None  # "index" or "upload"

    if index_value != "":
        # user entered an index
        try:
            idx = int(index_value)
        except ValueError:
            return render_template("predict.html", error="Index must be an integer between 0 and 399.")
        if olivetti_X is None:
            return render_template("predict.html", error="Olivetti dataset not available in this environment.")
        if idx < 0 or idx >= olivetti_X.shape[0]:
            return render_template("predict.html", error=f"Index {idx} out of range (0-{olivetti_X.shape[0]-1}).")
        X = olivetti_X[idx].reshape(1, -1)
        # convert the dataset image (64x64 array) to a PIL Image for display
        pil_img = Image.fromarray((olivetti_images[idx] * 255).astype(np.uint8)).convert("L").resize((256,256))
        display_img_b64 = pil_image_to_base64(pil_img)
        chosen_source = f"index {idx}"
    elif uploaded_file and uploaded_file.filename != "":
        # user uploaded a file
        try:
            file_bytes = uploaded_file.read()
            X, pil_img_small = preprocess_image_bytes(file_bytes)
            # Upscale for display clarity
            pil_img = pil_img_small.resize((256,256))
            display_img_b64 = pil_image_to_base64(pil_img)
            chosen_source = "uploaded image"
        except Exception as e:
            return render_template("predict.html", error=f"Could not process uploaded image: {e}")
    else:
        return render_template("predict.html", error="No input provided. Enter an index or upload an image.")

    # Run prediction & probabilities
    try:
        probs = None
        # scikit-learn DecisionTreeClassifier has predict_proba if fitted with classes
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X)[0]   # shape (n_classes,)
        else:
            # fallback: create one-hot with the predicted class
            pred = model.predict(X)[0]
            n_classes = len(model.classes_) if hasattr(model, "classes_") else 40
            probs = np.zeros(n_classes)
            probs[list(model.classes_).index(pred)] = 1.0

        pred_class = int(model.predict(X)[0])
        confidence = float(probs.max()) * 100.0
        # prepare list of (class, percent) pairs
        class_probs = [(int(cls), float(p * 100.0)) for cls, p in zip(model.classes_, probs)]
        # sort by class id just in case
        class_probs.sort(key=lambda x: x[0])
    except Exception as e:
        return render_template("predict.html", error=f"Prediction error: {e}")

    return render_template(
        "predict.html",
        predicted=pred_class,
        confidence=f"{confidence:.1f}%",
        class_probs=class_probs,
        image_data=display_img_b64,
        source=chosen_source
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
