"""
Code explained:
This code creates a Flask web application that can predict the label of an uploaded image.

When the client sends a GET request to the endpoint "/predict_img", the application returns the "home.html" template with the value "Image".

When the client sends a POST request to the same endpoint, the code checks if the request contains a file in the "file" field. If not, the code returns the message "Image not uploaded".

If the file exists, it reads the binary content and tries to create an image object using the Image module from the Pillow library. If the content is not an image file, the code returns an error message "Not an Image, please upload the file again!".

The code converts the image to RGB format and passes it to the "predict" function from the "predict.py" file. This function returns a label, which is returned to the client as a JSON object with the key "predictions".

Finally, the code runs the Flask app in debug mode.
"""
from flask import Flask, request, render_template, jsonify
import io
from flask_cors import CORS

from predict import *

app = Flask(__name__)
CORS(app)


@app.route("/predict_img", methods=["GET", "POST"])
def predict_label():
    if request.method == "GET":
        return render_template("home.html", value="Image")
    if request.method == "POST":
        if "file" not in request.files:
            return "Image not uploaded"

        file = request.files["file"].read()

        try:
            img = Image.open(io.BytesIO(file))
        except IOError:
            return jsonify(predictions="Not an Image, please upload file a gain!")

        img = img.convert("RGB")

        label = predict(img)

        return jsonify(predictions=label)


if __name__ == "__main__":
    app.run(debug=True)

