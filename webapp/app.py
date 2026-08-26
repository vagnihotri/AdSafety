import os
import time
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import onnxruntime as ort
from PIL import Image
import numpy as np
import ast

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

MODEL_PATH = "model/best.onnx"
print(f"Loading ONNX model from {MODEL_PATH}")
ort_session = ort.InferenceSession(MODEL_PATH)

meta = ort_session.get_modelmeta().custom_metadata_map
names_str = meta.get('names', '{}')
names_dict = ast.literal_eval(names_str)
CLASS_NAMES = [names_dict.get(i) for i in range(len(names_dict))]

TAXONOMY = {
    "weapon": 1, "adult_explicit": 1, "alcohol_tobacco": 1, "drugs_paraphernalia": 1,
    "compliant_apparel": 0, "compliant_cutlery": 0, "compliant_beverages": 0, "compliant_general": 0
}

THRESHOLDS = {
    "block": 0.85,
    "review": 0.50
}

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def preprocess_image(image_path):
    # Load image and resize to 224x224
    img = Image.open(image_path).convert('RGB')
    img = img.resize((224, 224))
    
    # Convert to numpy array and normalize 0.0-1.0
    img_data = np.array(img).astype('float32') / 255.0
    
    # Transpose from HWC to CHW
    img_data = np.transpose(img_data, (2, 0, 1))
    
    # Add batch dimension [1, 3, 224, 224]
    img_data = np.expand_dims(img_data, axis=0)
    
    return img_data

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image part in request"}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"{int(time.time())}_{filename}")
        file.save(filepath)
        
        print(f"--- Evaluating Image: {filepath} ---")
        try:
            # 1. Preprocess
            input_tensor = preprocess_image(filepath)
            
            # 2. Inference
            input_name = ort_session.get_inputs()[0].name
            ort_inputs = {input_name: input_tensor}
            ort_outs = ort_session.run(None, ort_inputs)
            logits = ort_outs[0][0]
            
            # 3. Softmax
            probs = softmax(logits)
            
            # 4. Evaluate Policy
            top_idx = int(np.argmax(probs))
            predicted_class = CLASS_NAMES[top_idx]
            confidence = float(probs[top_idx])
            
            violation_score = 0.0
            breakdown = {}
            
            for idx, prob in enumerate(probs):
                cname = CLASS_NAMES[idx]
                prob_val = float(prob)
                breakdown[cname] = prob_val
                if TAXONOMY.get(cname, 0) == 1:
                    violation_score += prob_val
                    
            if violation_score >= THRESHOLDS["block"]:
                verdict = "AUTOMATED BLOCK"
            elif violation_score >= THRESHOLDS["review"]:
                verdict = "HUMAN REVIEW QUEUE"
            else:
                verdict = "APPROVED"
                
            response_data = {
                "verdict": verdict,
                "violation_score": float(violation_score),
                "predicted_class": predicted_class,
                "confidence": float(confidence),
                "breakdown": breakdown
            }
            
            print(f"Evaluation result for {filename}:\n{response_data}\n---------------------------------")
            return jsonify(response_data)
        except Exception as e:
            print(f"Error evaluating image: {e}")
            return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5056)
