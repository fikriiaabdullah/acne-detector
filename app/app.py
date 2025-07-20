from flask import Flask, render_template, url_for, request, send_file
import io
from PIL import Image, ImageDraw, ImageFont
import os
if os.getenv("HOME") is None:
    os.environ["HOME"] = os.getenv("USERPROFILE")
from roboflow import Roboflow
import os
import random
import base64
import re
import cv2
import numpy as np

default_confidence = 25  # Lowered for better detection
max_size = (640, 640)    # Increased resolution
default_overlap = 20     # Adjusted overlap

# acne model
rf = Roboflow(api_key="QLCz3UoEIgfO52X0zObK")
project = rf.workspace().project("acne-vulgaris")
model = project.version(6).model

app = Flask(__name__)

def preprocess_image_for_detection(image):
    """Enhanced image preprocessing for better acne detection"""
    # Convert PIL to OpenCV
    cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # 1. Noise reduction with bilateral filter (preserves edges)
    denoised = cv2.bilateralFilter(cv_image, 9, 75, 75)
    
    # 2. CLAHE for better contrast in different lighting
    lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    enhanced = cv2.merge([l, a, b])
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    
    # 3. Subtle sharpening to enhance acne details
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]]) * 0.5
    sharpened = cv2.filter2D(enhanced, -1, kernel)
    
    # 4. Color balance adjustment for skin tone normalization
    sharpened = cv2.addWeighted(enhanced, 0.7, sharpened, 0.3, 0)
    
    return cv2.cvtColor(sharpened, cv2.COLOR_BGR2RGB)

def detect_face_regions(image):
    """Improved face detection with multiple cascades"""
    cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
    
    # Try multiple cascade classifiers
    face_cascades = [
        cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    ]
    
    all_faces = []
    for cascade in face_cascades:
        if cascade.empty():
            continue
            
        faces = cascade.detectMultiScale(
            gray, 
            scaleFactor=1.05, 
            minNeighbors=3, 
            minSize=(80, 80),
            maxSize=(500, 500),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        # Convert to list if it's a numpy array
        if len(faces) > 0:
            all_faces.extend(faces.tolist())
    
    if len(all_faces) == 0:
        return None
    
    # Simple approach: just return the largest face
    largest_face = max(all_faces, key=lambda face: face[2] * face[3])
    return tuple(largest_face)

def hybrid_acne_detection(image, roboflow_model, face_coords):
    """Combine Roboflow model predictions with CV-based validation"""
    
    # 1. Get Roboflow predictions with multiple confidence levels
    temp_path = 'temp_detection_image.jpg'
    image.save(temp_path, quality=95)
    
    # Try multiple confidence thresholds and combine results
    all_predictions = []
    confidence_levels = [20, 25, 30, 35]
    
    for conf in confidence_levels:
        try:
            result = roboflow_model.predict(temp_path, confidence=conf, overlap=25)
            predictions = result.json()['predictions']
            
            # Add confidence level to each prediction for filtering
            for pred in predictions:
                pred['detection_confidence'] = conf
                all_predictions.append(pred)
        except Exception as e:
            print(f"Error with confidence {conf}: {e}")
            continue
    
    # Clean up temporary file
    if os.path.exists(temp_path):
        os.remove(temp_path)
    
    # 2. Filter predictions using face region
    if face_coords:
        face_x, face_y, face_w, face_h = face_coords
        face_predictions = []
        
        for pred in all_predictions:
            pred_center_x = pred['x']
            pred_center_y = pred['y']
            
            # Check if prediction is within face region (with some padding)
            padding = 20
            if (face_x - padding <= pred_center_x <= face_x + face_w + padding and
                face_y - padding <= pred_center_y <= face_y + face_h + padding):
                face_predictions.append(pred)
        
        all_predictions = face_predictions
    
    # 3. Remove duplicates using Non-Maximum Suppression
    filtered_predictions = apply_nms(all_predictions, overlap_threshold=0.3)
    
    # 4. CV-based validation for each prediction
    validated_predictions = []
    cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    for pred in filtered_predictions:
        if validate_acne_spot(cv_image, pred):
            # Improve classification based on visual features
            improved_class = improve_acne_classification(cv_image, pred)
            pred['class'] = improved_class
            validated_predictions.append(pred)
    
    return validated_predictions

def apply_nms(predictions, overlap_threshold=0.3):
    """Apply Non-Maximum Suppression to remove duplicate detections"""
    if not predictions:
        return []
    
    # Convert to format needed for NMS
    boxes = []
    scores = []
    for pred in predictions:
        x1 = pred['x'] - pred['width'] / 2
        y1 = pred['y'] - pred['height'] / 2
        x2 = pred['x'] + pred['width'] / 2
        y2 = pred['y'] + pred['height'] / 2
        
        boxes.append([x1, y1, x2, y2])
        scores.append(pred['confidence'])
    
    boxes = np.array(boxes, dtype=np.float32)
    scores = np.array(scores, dtype=np.float32)
    
    # Apply OpenCV's NMS
    indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), 0.2, overlap_threshold)
    
    if len(indices) > 0:
        indices = indices.flatten()
        return [predictions[i] for i in indices]
    else:
        return []

def validate_acne_spot(cv_image, prediction):
    """Validate if a prediction is actually an acne spot using CV techniques"""
    try:
        # Extract region around prediction
        x, y = int(prediction['x']), int(prediction['y'])
        w, h = int(prediction['width']), int(prediction['height'])
        
        # Expand region slightly for better analysis
        padding = 10
        x1 = max(0, x - w//2 - padding)
        y1 = max(0, y - h//2 - padding)
        x2 = min(cv_image.shape[1], x + w//2 + padding)
        y2 = min(cv_image.shape[0], y + h//2 + padding)
        
        roi = cv_image[y1:y2, x1:x2]
        
        if roi.size == 0:
            return False
        
        # Convert to different color spaces for analysis
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # 1. Size validation
        area = w * h
        if area < 25 or area > 1000:
            return False
        
        # 2. Shape validation (aspect ratio)
        aspect_ratio = w / h if h > 0 else 0
        if aspect_ratio < 0.3 or aspect_ratio > 3.0:
            return False
        
        # 3. Color validation for skin-like regions
        # Check if it's in skin tone range
        skin_pixels = 0
        total_pixels = roi.shape[0] * roi.shape[1]
        
        # Vectorized approach for skin detection
        h_channel = hsv_roi[:, :, 0]
        s_channel = hsv_roi[:, :, 1]
        v_channel = hsv_roi[:, :, 2]
        
        # Create skin mask using vectorized operations
        skin_mask1 = (h_channel >= 0) & (h_channel <= 25)
        skin_mask2 = (h_channel >= 160) & (h_channel <= 180)
        skin_h_mask = skin_mask1 | skin_mask2
        
        skin_s_mask = (s_channel >= 25) & (s_channel <= 173)
        skin_v_mask = (v_channel >= 80) & (v_channel <= 255)
        
        skin_mask = skin_h_mask & skin_s_mask & skin_v_mask
        skin_pixels = np.sum(skin_mask)
        
        skin_ratio = skin_pixels / total_pixels
        if skin_ratio < 0.3:  # At least 30% should be skin-like
            return False
        
        # 4. Texture analysis - acne spots typically have different texture
        # Calculate local standard deviation
        mean_val = np.mean(gray_roi)
        std_val = np.std(gray_roi)
        
        # Acne spots usually have some texture variation
        if std_val < 10:
            return False
        
        # 5. Edge analysis - acne spots should have defined edges
        edges = cv2.Canny(gray_roi, 50, 150)
        edge_density = np.sum(edges) / (edges.shape[0] * edges.shape[1] * 255)
        
        if edge_density < 0.05:  # Should have some edge content
            return False
        
        return True
        
    except Exception as e:
        print(f"Validation error: {e}")
        return False

def improve_acne_classification(cv_image, prediction):
    """Improve acne classification based on visual features"""
    try:
        x, y = int(prediction['x']), int(prediction['y'])
        w, h = int(prediction['width']), int(prediction['height'])
        
        # Extract ROI
        x1 = max(0, x - w//2)
        y1 = max(0, y - h//2)
        x2 = min(cv_image.shape[1], x + w//2)
        y2 = min(cv_image.shape[0], y + h//2)
        
        roi = cv_image[y1:y2, x1:x2]
        
        if roi.size == 0:
            return prediction.get('class', 'papule')
        
        # Analyze color properties
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Calculate metrics
        avg_brightness = np.mean(gray_roi)
        color_variance = np.var(hsv_roi[:,:,1])  # Saturation variance
        size = w * h
        
        # Enhanced classification logic (removed comedone detection)
        if color_variance > 500 and avg_brightness > 120:  # Red/inflamed with variation
            return 'pustule'
        elif size > 200:  # Large spots
            return 'nodule'
        else:
            return 'papule'  # Default medium-sized bumps
            
    except Exception as e:
        print(f"Classification error: {e}")
        return prediction.get('class', 'papule')

# PDF report generator (single image version)
def generateReport(res, totalScore, fn, fpu, fpa, patientRegNo, patientName, patientAge, patientSex):
    from fpdf import FPDF
    from datetime import datetime, timedelta
    now = datetime.now()
    now_ist = now + timedelta(hours=5, minutes=30)
    dt_string = now_ist.strftime("%d/%m/%Y %H:%M:%S IST")
    pdf = FPDF(format='Letter')
    pdf.add_page()
    pdf.set_font('Arial', 'B', 14)
    pdf.set_fill_color(255, 255, 255)
    pdf.set_text_color(255, 0, 0)
    pdf.cell(0, 10, str('Acne Severity Analysis Report') , 0, 1, 'C')
    pdf.set_text_color(0, 0, 0)
    pdf.set_font('Arial', '', 12)
    pdf.cell(10,10, str('Date & Time: {}').format(dt_string), ln=1)
    pdf.cell(10,10, str('Patient Registration Number: {0}'.format(patientRegNo)), ln=1)
    pdf.cell(10,10, str('Patient Name: {0}'.format(patientName)), ln=1)
    pdf.cell(10,10, str('Patient Age: {0}'.format(patientAge)), ln=1)
    pdf.cell(10,10, str('Patient Sex: {0}'.format(patientSex)), ln=1)
    pdf.set_font('Arial', 'B', 13)
    pdf.cell(0, 10, str('Result: ' + res) , 0, 1, 'C')
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, str('Global Score: {}'.format(totalScore)), 0, 1, 'C')
    pdf.set_text_color(0,0,0)
    pdf.set_font('Arial', 'B', 13)
    pdf.cell(10, 10, str('Face Region'), ln=1)
    pdf.set_font('Arial', '', 12)
    pdf.cell(10, 10, str('              Nodule: {}'.format(fn)), ln=1)
    pdf.cell(10, 10, str('              Pustule: {}'.format(fpu)), ln=1)
    pdf.cell(10, 10, str('              Papule: {}'.format(fpa)), ln=1)
    pdf.image('static/images/result_of_upload_front_face.jpg', x=15, y=210,w=50)
    pdf.output('acne_report.pdf', 'F')

@app.route('/upload_front_face', methods=['POST'])
def upload_front_face():
    image_data_url = request.form['image']
    patientRegNo = request.form['patientregno']
    patientName = request.form['patientname']
    patientAge = request.form['patientage']
    patientSex = request.form['patientsex']

    if not image_data_url:
        return "No image uploaded", 400
    
    try:
        # Decode base64 image
        image_data = re.sub('^data:image/.+;base64,', '', image_data_url)
        image_bytes = base64.b64decode(image_data)
        original_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        # Preprocess image for better detection
        preprocessed_image = preprocess_image_for_detection(original_image)
        processed_pil = Image.fromarray(preprocessed_image)
        
        # Detect face region
        face_coords = detect_face_regions(processed_pil)
        
        if face_coords is None:
            return "No face detected in the image. Please ensure your face is clearly visible and try again.", 400
        
        # Perform hybrid acne detection
        acne_detections = hybrid_acne_detection(processed_pil, model, face_coords)
        
        # Convert back to OpenCV for drawing
        cv_image = cv2.cvtColor(np.array(processed_pil), cv2.COLOR_RGB2BGR)
        
        # Draw face detection rectangle
        x, y, w, h = face_coords
        cv2.rectangle(cv_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(cv_image, 'Face Detected', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Count acne types (removed comedone)
        nodule_count = len([d for d in acne_detections if d['class'].lower() in ['nodule']])
        pustule_count = len([d for d in acne_detections if d['class'].lower() in ['pustule']])
        papule_count = len([d for d in acne_detections if d['class'].lower() in ['papule']])
        
        # Draw detection boxes (removed comedone colors)
        colors = {
            'nodule': (0, 0, 139),      # Dark red
            'pustule': (0, 255, 255),   # Yellow
            'papule': (0, 0, 255),      # Red
        }
        
        for detection in acne_detections:
            class_name = detection['class'].lower()
            color = colors.get(class_name, (128, 128, 128))
            
            # Calculate bounding box
            center_x, center_y = int(detection['x']), int(detection['y'])
            width, height = int(detection['width']), int(detection['height'])
            
            x1 = center_x - width // 2
            y1 = center_y - height // 2
            x2 = center_x + width // 2
            y2 = center_y + height // 2
            
            # Draw rectangle
            cv2.rectangle(cv_image, (x1, y1), (x2, y2), color, 2)
            
            # Add label with confidence
            label = f"{detection['class']} ({detection['confidence']:.0f}%)"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            
            # Background for text
            cv2.rectangle(cv_image, (x1, y1 - label_size[1] - 8), 
                         (x1 + label_size[0] + 4, y1), color, -1)
            
            # Text
            cv2.putText(cv_image, label, (x1 + 2, y1 - 4), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Add summary text
        total_count = len(acne_detections)
        summary_text = f"Total: {total_count} acne spots detected"
        cv2.putText(cv_image, summary_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(cv_image, summary_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
        
        # Convert back to PIL Image
        result_image = Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
        
        # Encode to base64 for web display
        buffered = io.BytesIO()
        result_image.save(buffered, format="JPEG", quality=90)
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        # Save result image
        os.makedirs('static/images', exist_ok=True)
        result_image.save('static/images/result_of_upload_front_face.jpg')
        
        return render_template(
            'result.html',
            image_data=img_str,
            papula=papule_count,
            pustula=pustule_count,
            nodule=nodule_count,
            patientRegNo=patientRegNo,
            patientName=patientName,
            patientAge=patientAge,
            patientSex=patientSex
        )
        
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        return f"Error processing image: {str(e)}", 500

@app.route('/download_result')
def download_result():
    return send_file('static/images/result_of_upload_front_face.jpg', as_attachment=True)

@app.route('/')
def index():
    css_file = url_for('static', filename='css/style.css')
    return render_template('index.html', css_file=css_file)

@app.route('/instructions')
def instructions():
    css_file = url_for('static', filename='css/style.css')
    return render_template('instructions.html', css_file=css_file)

@app.route('/download')
def download():
    return "Download not implemented"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
