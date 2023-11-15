from flask import Flask, render_template, request, flash, redirect, url_for
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Set a secret key for the flash messages

UPLOAD_FOLDER = 'data'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_person_folder(person_name):
    person_folder = os.path.join(app.config['UPLOAD_FOLDER'], person_name)
    if not os.path.exists(person_folder):
        os.makedirs(person_folder)
    return person_folder

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/upload")
def upload():
    return render_template("upload.html")

@app.route("/upload-image", methods=['POST'])
def upload_files():
    person_name = request.form.get('person_name')

    if not person_name:
        flash('Please provide a name for the person.', 'error')
        return redirect(url_for('upload'))

    uploaded_files = request.files.getlist('file')

    person_folder = get_person_folder(person_name)

    for file in uploaded_files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(person_folder, filename))
        else:
            flash('Invalid file format! Please upload only PNG or JPG files.', 'error')

    flash('Files successfully uploaded to {}'.format(person_name), 'success')
    return redirect(url_for('upload'))

if __name__ == "__main__":
    app.run(debug=True)
