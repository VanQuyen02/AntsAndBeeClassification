"""
1. Import required modules from "lib", "config", "utils", and "image_transform".
2. Define a list of class names (class_index) for the two classes "ants" and "bees".
3. Define a Predictor class that takes class_index as input and has a method predict_max to return the class name with the maximum predicted probability.
4. Create an instance of the Predictor class.
5. Define a predict function to make predictions for a single input image.
6. Load a pre-trained VGG16 network, and set its last layer to have 2 output classes.
7. Load the pre-trained model parameters from the save_path.
8. Apply the image transform to the input image to resize, normalize, and convert it to tensor.
9. Add a batch dimension to the input image tensor to make it (1, channel, height, width).
10. Run a forward pass through the network to get the prediction probabilities.
11. Use the Predictor's predict_max method to get the class name with the maximum prediction probability.
Return the predicted class name.
"""

from lib import *
from config import *
from utils import *
from image_transform import ImageTransform

class_index = ["ants", "bees"]

class Predictor():
    def __init__(self, class_index):
        self.clas_index = class_index

    def predict_max(self, output): # [0.9, 0.1]
        max_id = np.argmax(output.detach().numpy())
        predicted_label = self.clas_index[max_id]
        return predicted_label


predictor = Predictor(class_index)

def predict(img):
    # prepare network
    use_pretrained = True
    net = models.vgg16(pretrained=use_pretrained)
    net.classifier[6] = nn.Linear(in_features=4096, out_features=2)
    net.eval()

    # prepare model
    model = load_model(net, save_path)

    # prepare input img
    transform = ImageTransform(resize, mean, std)
    img = transform(img, phase="test")
    img = img.unsqueeze_(0) # (chan, height, width) -> (1, chan, height, width)

    # predict
    output = model(img)
    response = predictor.predict_max(output)

    return response