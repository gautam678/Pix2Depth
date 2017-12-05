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
        model =  CONFIG['pix2depth'][model_name]
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(input_path)
        if not development:
            result_path = pix2depth(input_path,model)
        else:
            result_path = str(input_path)
        img_left = str(input_path)
        img_right = str(result_path)
    else:
        img_left = os.path.join(app.config['UPLOAD_FOLDER'], 'pix.jpg')
        img_right = os.path.join(app.config['UPLOAD_FOLDER'], 'depth.jpg')
        
    return render_template('client/index.html',image_left=img_left,image_right=img_right,options = CONFIG['pix2depth'])

@app.route("/depth",methods=['GET', 'POST'])
def depth():
    if request.method == 'POST':
        file = request.files['image']
        model_name = request.form['model']
        model = CONFIG['depth2pix'][model_name]
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(input_path)
        if not development:
            result_path = depth2pix(input_path,model)
        else:
            result_path = str(input_path)
        img_left = str(input_path)
        img_right = str(result_path)
    else:
        img_left = os.path.join(app.config['UPLOAD_FOLDER'], 'depth.jpg')
        img_right = os.path.join(app.config['UPLOAD_FOLDER'], 'pix.jpg')
        
    return render_template('client/depth.html',image_left=img_left,image_right=img_right,options = CONFIG['depth2pix'])

@app.route("/portrait",methods=['GET', 'POST'])
def portrait():
    if request.method == 'POST':
        file = request.files['image']
        model_name = request.form['model']
        model = CONFIG['portrait'][model_name]
        input_path= os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(input_path)
        # Perform depth conversion
        if not development:
            result_path = portrait_mode(input_path, model)
        else:
            result_path = str(input_path)
        img_left = str(input_path)
        img_right = str(result_path)
    else:
        img_left = os.path.join(app.config['UPLOAD_FOLDER'], 'pix.jpg')
        img_right = os.path.join(app.config['UPLOAD_FOLDER'], 'pix.jpg')
        
    return render_template('client/potrait.html',image_left=img_left,image_right=img_right,options = CONFIG['portrait'])

@app.route("/examples",methods=['GET','POST'])
def example():
    epoch = str(22)
    if request.method == 'POST':
        epoch = request.form['epoch']
        epoch = str(min([32, int(epoch)]))
    path = 'http://www.cs.virginia.edu/~ks6cq/cyclegan-1/output/cyclegan/exp_rgb2dep/20171202-023330/imgs/'
    img = str(0)
    print epoch
    populate_page = []
    image_types = ['input','fake','cyc']
    a2b = ['A','B']
    for i in range(0,20):
        list_of_images=[]
        for img_type in image_types:
            for j in a2b:
                image = path+img_type+j+'_'+epoch+'_'+str(i)+'.jpg'
                list_of_images.append(image)
        populate_page.append(list_of_images)
    return render_template('client/example.html',path = populate_page)


if __name__ == "__main__":
    development = CONFIG['development']
    app.run(debug=CONFIG['development'], host=CONFIG['host'], port=CONFIG['port'])
