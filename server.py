import os
from typing import Tuple
import io
import base64
import json

import torch
import numpy as np
from flask import Flask, request, render_template, redirect
from PIL import Image

from models.resnet import ResNet, preprocess
from models.utils import get_probs
from datasets.cifar100 import CLASSES


# app initialization
app = Flask(__name__)

# model initialization
net = ResNet()
trained_model_path = os.path.join("models", "trained", "resnet.pth")
state_dict = torch.load(trained_model_path, torch.device('cpu'))
net.load_state_dict(state_dict)
net.eval()

def pil_image_to_html(pil_img: Image, height: int = 200) -> str:
    """Convert PIL image to a string embeddable in a HTML document."""

    with io.BytesIO() as buf:
            pil_img.save(buf, 'jpeg')
            image_bytes = buf.getvalue()
    img_enc = base64.b64encode(image_bytes).decode('ascii')
    return f'<img src="data:image/jpg;base64,{img_enc}" class="img-fluid" height="{height}px"/>'

@app.route("/", methods=["POST", "GET"])
def classify_data() -> Tuple[str, int]:
    if request.method == "POST":

        # Check if an image was selected
        if 'image' not in request.files:
            print('No file found')
            return redirect(request.url)
        file = request.files['image']
        if file.filename == '':
            print('No file selected')
            return redirect(request.url)

        # Load binary image and run inference
        pil_img = Image.open(io.BytesIO(file.read()))
        img_html = pil_image_to_html(pil_img)
        img = preprocess(np.array(pil_img))
        output = get_probs(net, img, add_softmax=True)
        output = {CLASSES[i[0]]:i[1] for i in output.items()}

        return render_template(
            "home.html", output=output, 
            user_img=img_html,
            prob_labels=json.dumps([str(i) for i in sorted(output.values())]),
            cat_labels=json.dumps(sorted(list(output.keys()), key=lambda item: output[item]))
        ), 200
    
    else:
        return render_template("home.html"), 200

if __name__ == '__main__':
    app.run(debug=True)
