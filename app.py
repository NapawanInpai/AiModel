from flask import Flask, request, jsonify
import numpy as np
import cv2
import base64
from io import BytesIO
from PIL import Image
import tensorflow as tf

app = Flask(__name__)

# โหลด TFLite model
interpreter = tf.lite.Interpreter(model_path="yamodel.tflite")
interpreter.allocate_tensors()

# ข้อมูล input/output
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_shape = input_details[0]['shape']

class_names = ['Decolgen', 'Fish Oil', 'Medicol', 'Tylenol', 'Vaginy']
IMG_SIZE = (150, 150)

def preprocess_image(image_b64):
    try:
        image_data = base64.b64decode(image_b64)
        image = Image.open(BytesIO(image_data)).convert('RGB')
        image = image.resize(IMG_SIZE)
        img_array = np.array(image, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        print("Error in image preprocessing:", e)
        return None

@app.route('/')
def home():
    return '🧠 TFLite Classifier API is running!'

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if 'image' not in data:
        return jsonify({'error': 'No image provided'}), 400

    img_array = preprocess_image(data['image'])
    if img_array is None:
        return jsonify({'error': 'Invalid image'}), 400

    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    class_index = np.argmax(output_data)
    confidence = float(np.max(output_data))

    result = class_names[class_index] if confidence >= 0.7 else "No"

    return jsonify({
        'result': result,
        'confidence': round(confidence, 2)
    })

if __name__ == '__main__':
    app.run(debug=True)
