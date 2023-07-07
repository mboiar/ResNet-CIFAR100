import os
from typing import Any, Dict, Optional
import io
import base64
import json

import joblib
import torch
import numpy as np
from flask import Flask, request, render_template, redirect
from PIL import Image

from models.resnet import ResNet, preprocess, get_probs

app = Flask(__name__)
trained_model_path = os.path.join("models", "trained", "resnet4000.pth")

net = ResNet()
state_dict = torch.load(trained_model_path, torch.device('cpu'))
net.load_state_dict(state_dict)
net.eval()


# @app.route('/')
# def upload_form():
#     return render_template('home.html', output={})

@app.route("/", methods=["POST", "GET"])
def classify_data() -> Any:
    if request.method == "POST":
        # Check if a file was submitted
        if 'image' not in request.files:
            print('No file found')
            return redirect(request.url)
        file = request.files['image']
        # Check if the file has a name
        if file.filename == '':
            print('No file selected')
            return redirect(request.url)
        image_string = file.read()
        pil_img = Image.open(io.BytesIO(image_string))
        with io.BytesIO() as buf:
                pil_img.save(buf, 'jpeg')
                image_bytes = buf.getvalue()
        img_enc = base64.b64encode(image_bytes).decode('ascii')
        # img_io = io.BytesIO()
        # pil_img.save(img_io, 'jpeg', quality=100)
        # img_io.seek(0)
        # img_enc = base64.b64encode(img_io.getvalue()).decode('ascii')
        img_html = f'<img src="data:image/jpg;base64,{img_enc}" class="img-fluid" height="200px"/>'

        img = np.asarray(pil_img).swapaxes(0, 2)
        img = preprocess(img)

        output = get_probs(net, img)

        return render_template(
            "home.html", output=output, 
            user_img=img_html,
            prob_labels=json.dumps([str(i) for i in output.values()]),
            cat_labels=json.dumps(list(output.keys()))
        ), 200
    
    else:
        return render_template("home.html"), 200

if __name__ == '__main__':
    
    # net = ResNet()
    # net.load_state_dict(torch.load(trained_model_path)).cpu().eval()

    app.run(debug=True)