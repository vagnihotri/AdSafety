import os
import time
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from ultralytics import YOLO
import numpy as np

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load the exported ONNX model using YOLO wrapper for handling inference properly
MODEL_PATH = "model/best.onnx"
print(f"Loading ONNX model from {MODEL_PATH}")
model = YOLO(MODEL_PATH, task="classify")

TAXONOMY = {
    "weapon": 1, "adult_explicit": 1, "alcohol_tobacco": 1, "drugs_paraphernalia": 1,
    "compliant_apparel": 0, "compliant_cutlery": 0, "compliant_beverages": 0, "compliant_general": 0
}

THRESHOLDS = {
    "block": 0.85,
    "review": 0.50
}

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
        
        # Inference using ONNX (via YOLO API)
        print(f"--- Evaluating Image: {filepath} ---")
        try:
            results = model(filepath, verbose=False)
            res = results[0]
            probs = res.probs.data.cpu().numpy()
            
            # Using model class names
            class_names = res.names
            top_idx = int(np.argmax(probs))
            predicted_class = class_names[top_idx]
            confidence = float(probs[top_idx])
            
            violation_score = 0.0
            breakdown = {}
            for idx, cname in class_names.items():
                prob_val = float(probs[idx])
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
    app.run(debug=True, host='0.0.0.0', port=5050)
