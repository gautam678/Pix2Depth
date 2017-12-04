from flask import Flask, render_template,request
from main import pix2depth, portrait_mode, depth2pix
import json
import os
from config import CONFIG

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join('static/uploads')

@app.route("/",methods=['GET', 'POST'])
def main():
    if request.method == 'POST':
        file = request.files['image']
        model_name = request.form['model']
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(input_path)
        if not development:
            result_path = pix2depth(input_path,CONFIG['model'][model_name])
        else:
            result_path = str(input_path)
        img_left = str(input_path)
        img_right = str(result_path)
    else:
        img_left = os.path.join(app.config['UPLOAD_FOLDER'], 'pix.jpg')
        img_right = os.path.join(app.config['UPLOAD_FOLDER'], 'depth.jpg')
        
    return render_template('client/index.html',image_left=img_left,image_right=img_right)

@app.route("/depth",methods=['GET', 'POST'])
def depth():
    if request.method == 'POST':
        file = request.files['image']
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(input_path)
        if not development:
            result_path = depth2pix(input_path)
        else:
            result_path = str(input_path)
        img_left = str(input_path)
        img_right = str(result_path)
    else:
        img_left = os.path.join(app.config['UPLOAD_FOLDER'], 'depth.jpg')
        img_right = os.path.join(app.config['UPLOAD_FOLDER'], 'pix.jpg')
        
    return render_template('client/depth.html',image_left=img_left,image_right=img_right)

@app.route("/portrait",methods=['GET', 'POST'])
def portrait():
    if request.method == 'POST':
        file = request.files['image']
        input_path= os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(input_path)
        # Perform depth conversion
        if not development:
            result_path = portrait_mode(input_path, 'siva')
        else:
            result_path = str(input_path)
        img_left = str(input_path)
        img_right = str(result_path)
    else:
        img_left = os.path.join(app.config['UPLOAD_FOLDER'], 'pix.jpg')
        img_right = os.path.join(app.config['UPLOAD_FOLDER'], 'pix.jpg')
        
    return render_template('client/potrait.html',image_left=img_left,image_right=img_right)


if __name__ == "__main__":
    development = CONFIG['development']
    app.run(debug=CONFIG['development'], host=CONFIG['host'], port=CONFIG['port'])
