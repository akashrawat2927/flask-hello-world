import cv2
import joblib
import numpy as np
from flask import Flask, request, jsonify


app = Flask(__name__)
my_ml_model = joblib.load("knnmodel.joblib")

@app.route('/predict', methods=['POST'])


def predict():
    image = request.data
    
    imageArray = bytearray(image)
   
    npArray = np.asarray(imageArray, dtype=np.uint8)
    # Decode the image using OpenCV
    image = cv2.imdecode(npArray, cv2.IMREAD_COLOR)


    image_size = 32
    test_sample = cv2.resize(image, (image_size, image_size),interpolation = cv2.INTER_AREA).reshape(1,-1)



    
    # process the data and make predictions
    result = my_ml_model.predict(test_sample)
    print(result)
    #return the predictions as JSON
    return jsonify(result.tolist())


if __name__ == '__main__':
    app.run()
