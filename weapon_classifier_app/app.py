from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

import os
import torch
from PIL import Image
from torchvision import transforms

from model import load_model


app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = load_model(
    "model.pt",
    device
)

class_dict = {
    "Hand grenade (قنبلة يدوية)": 0,
    "Self diffence (سلاح دفاع ذاتي/ مسدس)": 1,
    "Semi automatic (سلاح نصف آلي)": 2
}

idx_to_class = {
    value: key
    for key, value in class_dict.items()
}


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None
    confidence = None
    image_path = None

    if request.method == "POST":

        if "image" not in request.files:
            return render_template(
                "index.html",
                error="No image was uploaded."
            )

        file = request.files["image"]

        if file.filename == "":
            return render_template(
                "index.html",
                error="No image was selected."
            )

        filename = secure_filename(file.filename)

        image_path = os.path.join(
            app.config["UPLOAD_FOLDER"],
            filename
        )

        file.save(image_path)

        image = Image.open(image_path).convert("RGB")

        image = transform(image)

        image = image.unsqueeze(0)

        image = image.to(device)

        with torch.inference_mode():

            logits = model(image)

            probabilities = torch.softmax(
                logits,
                dim=1
            )

            confidence, predicted = torch.max(
                probabilities,
                dim=1
            )

        prediction = idx_to_class[
            predicted.item()
        ]

        confidence = confidence.item() * 100

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        image_path=image_path
    )


if __name__ == "__main__":
    app.run(
        debug=True
    )