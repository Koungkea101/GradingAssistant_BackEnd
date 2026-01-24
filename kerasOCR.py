import keras_ocr
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import base64
import io
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# ---------- LOAD MODEL ----------
pipeline = keras_ocr.pipeline.Pipeline()

# ---------- PREPROCESSING ----------
def preprocess_for_ocr(image_path=None, image_array=None):
    """
    Preprocess image for OCR. Can accept either a file path or a numpy array.
    """
    if image_array is not None:
        # If image_array is provided, use it directly
        img = image_array
    elif image_path is not None:
        # If image_path is provided, read from file
        img = cv2.imread(image_path)
    else:
        raise ValueError("Either image_path or image_array must be provided")

    img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    return rgb


def decode_base64_image(base64_string):
    """
    Decode base64 string to numpy array (OpenCV format).
    """
    # Remove data URL prefix if present (e.g., "data:image/jpeg;base64,")
    if ',' in base64_string:
        base64_string = base64_string.split(',')[1]

    # Decode base64 to bytes
    image_data = base64.b64decode(base64_string)

    # Convert bytes to PIL Image
    pil_image = Image.open(io.BytesIO(image_data))

    # Convert PIL to RGB if not already
    if pil_image.mode != 'RGB':
        pil_image = pil_image.convert('RGB')

    # Convert PIL to numpy array (OpenCV format: BGR)
    image_array = np.array(pil_image)
    image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)

    return image_array


# ---------- NEW ADD: SORT INTO LINES ----------
def sort_into_lines(results, y_threshold=20):
    lines = []

    for text, box in results:
        y_center = np.mean(box[:, 1])
        placed = False

        for line in lines:
            if abs(line["y"] - y_center) < y_threshold:
                line["items"].append((text, box))
                placed = True
                break

        if not placed:
            lines.append({
                "y": y_center,
                "items": [(text, box)]
            })

    lines = sorted(lines, key=lambda l: l["y"])

    sorted_lines = []
    for line in lines:
        items = sorted(
            line["items"],
            key=lambda x: np.min(x[1][:, 0])
        )
        sorted_lines.append(items)

    return sorted_lines


# ---------- LOAD MODEL ----------
pipeline = keras_ocr.pipeline.Pipeline()


# ---------- IMAGE PATH ----------
# image_path = "image_ocr/image1.jpg"  # Commented out - only used for testing


# ---------- PREPROCESS IMAGE ----------
# image = preprocess_for_ocr(image_path)  # Commented out - only used for testing


# ---------- OCR ----------
# Commented out the standalone script functionality
# raw_results = pipeline.recognize([image])[0]

# convert to (text, box) format
# results = [(text, box) for text, box in raw_results]


# ---------- NEW ADD: EXTRACT SENTENCES ----------
# lines = sort_into_lines(results)

# print("Final extracted text:\n")
# for line in lines:
#     sentence = " ".join([text for text, _ in line])
#     print(sentence)


@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "service": "keras-ocr"}), 200


@app.route('/extract_text', methods=['POST'])
def extract_text_endpoint():
    try:
        image_array = None
        temp_path = None

        # Check if request contains base64 image data
        if request.is_json and 'image' in request.json:
            try:
                base64_data = request.json['image']
                image_array = decode_base64_image(base64_data)
            except Exception as e:
                return jsonify({"error": f"Invalid base64 image data: {str(e)}"}), 400

        # Check if image file is in request (traditional file upload)
        elif 'image' in request.files:
            file = request.files['image']
            if file.filename == '':
                return jsonify({"error": "No image file selected"}), 400

            # Save uploaded file temporarily
            temp_path = f"/tmp/{file.filename}"
            file.save(temp_path)

        else:
            return jsonify({
                "error": "No image provided. Send either a file upload or JSON with base64 image data."
            }), 400

        try:
            # Process the image
            if image_array is not None:
                # Use base64 decoded image
                image = preprocess_for_ocr(image_array=image_array)
            else:
                # Use file path
                image = preprocess_for_ocr(image_path=temp_path)

            raw_results = pipeline.recognize([image])[0]
            results = [(text, box) for text, box in raw_results]
            lines = sort_into_lines(results)

            # Extract text as sentences
            extracted_text = []
            for line in lines:
                sentence = " ".join([text for text, _ in line])
                if sentence.strip():  # Only add non-empty sentences
                    extracted_text.append(sentence)

            # Clean up temporary file if it was created
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)

            return jsonify({
                "success": True,
                "extracted_text": extracted_text,
                "total_lines": len(extracted_text)
            }), 200

        except Exception as e:
            # Clean up temporary file in case of error
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)
            return jsonify({"error": f"OCR processing failed: {str(e)}"}), 500

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5002))
    print(f"Starting Keras OCR Flask API on port {port}")
    app.run(host='0.0.0.0', port=port, debug=True)
