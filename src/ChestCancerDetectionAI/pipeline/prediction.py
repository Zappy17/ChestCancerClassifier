import numpy as np
import tensorflow as tf
import os

class PredictionPipeline:
    def __init__(self, filename):
        self.filename = filename

    def predict(self):
        # Load model using TensorFlow
        model = tf.keras.models.load_model(os.path.join("model", "model.h5"))

        # Process the image using OpenCV (no need for keras.preprocessing)
        import cv2

        imagename = self.filename
        test_image = cv2.imread(imagename)
        test_image = cv2.resize(test_image, (224, 224))  # Resize the image to match model's expected input
        test_image = np.array(test_image)
        test_image = np.expand_dims(test_image, axis=0)  # Add batch dimension

        # Normalize the image if your model expects it (mean subtraction or scaling)
        test_image = test_image / 255.0

        # Predict the class
        result = np.argmax(model.predict(test_image), axis=1)

        if result[0] == 1:
            prediction = 'Normal'
            return [{"image": prediction}]
        else:
            prediction = 'Adenocarcinoma Cancer'
            return [{"image": prediction}]
