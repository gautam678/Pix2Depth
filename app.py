from flask import Flask, render_template,request
from main import process_rgb,pix2depth
import json
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join('static/uploads')
@app.route("/",methods=['GET', 'POST'])
def main():
    if request.method == 'POST':
        file = request.files['image']
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(input_path)
        result_path = process_rgb(input_path)
        img_left = str(input_path)
        img_right = str(result_path)
    else:
        img_left = os.path.join(app.config['UPLOAD_FOLDER'], 'profile.png')
        img_right = os.path.join(app.config['UPLOAD_FOLDER'], 'profile.png')
        
    return render_template('client/index.html',image_left=img_left,image_right=img_right)

@app.route("/depth",methods=['GET', 'POST'])
def depth():
    if request.method == 'POST':
        file = request.files['image']
        f = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(f)
        img_left = str(f)
        img_right = str(f)
    else:
        img_left = os.path.join(app.config['UPLOAD_FOLDER'], 'profile.png')
        img_right = os.path.join(app.config['UPLOAD_FOLDER'], 'profile.png')
        
    return render_template('client/depth.html',image_left=img_left,image_right=img_right)

@app.route("/portrait",methods=['GET', 'POST'])
def potrait():
    if request.method == 'POST':
        file = request.files['image']
        f = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(f)
        # Perform depth conversion
        result_path = pix2depth(f)
        img_left = str(f)
        img_right = str(result_path)
    else:
        img_left = os.path.join(app.config['UPLOAD_FOLDER'], 'profile.png')
        img_right = os.path.join(app.config['UPLOAD_FOLDER'], 'profile.png')
        
    return render_template('client/potrait.html',image_left=img_left,image_right=img_right)


if __name__ == "__main__":
    app.run(debug=True)