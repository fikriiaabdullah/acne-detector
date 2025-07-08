from flask import Flask, render_template, url_for, request, send_file
import io
from PIL import Image, ImageDraw, ImageFont
from roboflow import Roboflow
import os
import random
import base64
import re

default_confidence = 2
max_size = (416, 416)
default_overlap = 15

# acne model
rf = Roboflow(api_key="QLCz3UoEIgfO52X0zObK")
project = rf.workspace().project("acne-vulgaris")
model = project.version(6).model

app = Flask(__name__)

# PDF report generator (single image version)
def generateReport(res, totalScore, fn, fpu, fpa, fc, patientRegNo, patientName, patientAge, patientSex):
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
    pdf.cell(10, 10, str('              Comedone: {}'.format(fc)), ln=1)
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
        image_data = re.sub('^data:image/.+;base64,', '', image_data_url)
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        return f"Invalid image file: {e}", 400

    width, height = image.size

    # Simulasi hasil analisa
    papula = random.randint(1, 5)
    komedo = random.randint(1, 5)
    pustula = random.randint(1, 5)

    # Area wajah lebih sempit (tengah gambar)
    face_x_min = int(width * 0.4)
    face_x_max = int(width * 0.6) - 30
    face_y_min = int(height * 0.35)
    face_y_max = int(height * 0.75) - 30

    # Generate box random untuk setiap jenis jerawat
    boxes = []
    # Papula & Pustula (pipi kiri & kanan)
    for _ in range(papula):
        x = random.randint(face_x_min, face_x_max)
        y = random.randint(face_y_min, face_y_max)
        boxes.append({'x': x, 'y': y, 'w': 30, 'h': 30, 'type': 'Papula'})

    # Komedo (hidung)
    for _ in range(komedo):
        x = random.randint(face_x_min, face_x_max)
        y = random.randint(face_y_min, face_y_max)
        boxes.append({'x': x, 'y': y, 'w': 30, 'h': 30, 'type': 'Komedo'})

    for _ in range(pustula):
        x = random.randint(face_x_min, face_x_max)
        y = random.randint(face_y_min, face_y_max)
        boxes.append({'x': x, 'y': y, 'w': 30, 'h': 30, 'type': 'Pustula'})

    # Gambar box dan label di foto
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    for box in boxes:
        x, y, w, h = box['x'], box['y'], box['w'], box['h']
        color = {'Papula': 'red', 'Komedo': 'blue', 'Pustula': 'green'}[box['type']]
        draw.rectangle([x, y, x+w, y+h], outline=color, width=3)
        # Tambahkan label nama penyakit di atas box
        text = box['type']
        bbox = draw.textbbox((0, 0), text, font=font)
        text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.rectangle([x, y-text_h-4, x+text_w+4, y], fill=color)
        draw.text((x+2, y-text_h-2), text, fill='white', font=font)

    # Encode ke base64 untuk tampilan web
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()

    # Simpan file hasil ke static/images
    os.makedirs('static/images', exist_ok=True)
    image.save('static/images/result_of_upload_front_face.jpg')

    return render_template(
        'result.html',
        image_data=img_str,
        papula=papula,
        komedo=komedo,
        pustula=pustula,
        patientRegNo=patientRegNo,
        patientName=patientName,
        patientAge=patientAge,
        patientSex=patientSex
    )

@app.route('/download_result')
def download_result():
    return send_file('static/images/result_of_upload_front_face.jpg', as_attachment=True)

@app.route('/')
def index():
    css_file = url_for('static', filename='css/style.css')
    return render_template('index.html', css_file=css_file)

@app.route('/download')
def download():
    # Dummy PDF download (optional, or you can remove this route)
    return "Download not implemented"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

