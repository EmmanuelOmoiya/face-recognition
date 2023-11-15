from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import os
from werkzeug.utils import secure_filename
import cv2
import face_recognition
import joblib
import numpy as np

app = Flask(__name__)

# Specify the paths to the trained model and label encoder
model_path = "face_recognition_model.joblib"
label_encoder_path = "label_encoder.joblib"
trained_model, label_encoder = joblib.load(model_path), joblib.load(label_encoder_path)

# Specify the upload folder
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def recognize_faces_and_save(image_path, output_image_path, model, label_encoder):
    # Load the input image
    input_image = face_recognition.load_image_file(image_path)

    # Find all face locations in the image
    face_locations = face_recognition.face_locations(input_image)

    # Recognize faces and draw rectangles and labels
    for (top, right, bottom, left) in face_locations:
        face_image = input_image[top:bottom, left:right]

        # Resize the face image to match the trained model input size
        face_image = cv2.resize(face_image, (100, 100))

        # Encode the face
        face_encoding = face_recognition.face_encodings(face_image)
        if not face_encoding:
            continue

        # Predict the label using the trained model
        predictions = model.predict([face_encoding])
        label_idx = np.argmax(predictions)
        confidence = predictions[0][label_idx]

        # Map numerical label back to the original string label
        predicted_label = label_encoder.inverse_transform([label_idx])[0]

        # Draw rectangle and label on the image
        cv2.rectangle(input_image, (left, top), (right, bottom), (0, 255, 0), 2)
        font = cv2.FONT_HERSHEY_DUPLEX
        label_text = f'{predicted_label} ({confidence:.2f})'
        cv2.putText(input_image, label_text, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

    # Save the image with rectangles and labels
    cv2.imwrite(output_image_path, cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR))


@app.route('/')
def index():
    return render_template('app.html')

@app.route('/', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']

    if file.filename == '':
        return redirect(request.url)

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Specify the path for the output image
        output_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'output_' + filename)

        # Recognize faces, add rectangles and labels, and save the new image
        recognize_faces_and_save(file_path, output_image_path, trained_model, label_encoder)

        return render_template('app.html', filename=filename, output_filename='output_' + filename)

    return redirect(request.url)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
