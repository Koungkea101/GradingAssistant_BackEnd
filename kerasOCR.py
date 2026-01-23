import keras_ocr
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# ---------- LOAD MODEL ----------
pipeline = keras_ocr.pipeline.Pipeline()

# ---------- PREPROCESSING ----------
def preprocess_for_ocr(image_path):
    img = cv2.imread(image_path)

    img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    return rgb


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
        # Check if image file is in request
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"}), 400

        file = request.files['image']
        if file.filename == '':
            return jsonify({"error": "No image file selected"}), 400

        # Save uploaded file temporarily
        temp_path = f"/tmp/{file.filename}"
        file.save(temp_path)

        try:
            # Process the image
            image = preprocess_for_ocr(temp_path)
            raw_results = pipeline.recognize([image])[0]
            results = [(text, box) for text, box in raw_results]
            lines = sort_into_lines(results)

            # Extract text as sentences
            extracted_text = []
            for line in lines:
                sentence = " ".join([text for text, _ in line])
                if sentence.strip():  # Only add non-empty sentences
                    extracted_text.append(sentence)

            # Clean up temporary file
            os.remove(temp_path)

            return jsonify({
                "success": True,
                "extracted_text": extracted_text,
                "total_lines": len(extracted_text)
            }), 200

        except Exception as e:
            # Clean up temporary file in case of error
            if os.path.exists(temp_path):
                os.remove(temp_path)
            return jsonify({"error": f"OCR processing failed: {str(e)}"}), 500

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5002))
    print(f"Starting Keras OCR Flask API on port {port}")
    app.run(host='0.0.0.0', port=port, debug=True)
