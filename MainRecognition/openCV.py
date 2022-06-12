import cv2
import numpy as np
from PIL import Image
from keras import models
import os
import tensorflow as tf

currentPath = os.getcwdb().decode("utf-8") 
#Load the saved model
model = models.load_model(currentPath+'\\fruit-recognition.h5')
video = cv2.VideoCapture(0)
img_height = 100
img_width = 100
class_names = ['apple', 'banana', 'lemon', 'orange']

while True:
        _, frame = video.read()

        #Convert the captured frame into RGB
        im = Image.fromarray(frame, 'RGB')

        #Resizing into 128x128 because we trained the model with this image size.
        im = im.resize((img_width,img_height))
        img_array = np.array(im)

        #Our keras model used a 4D tensor, (images x height x width x channel)
        #So changing dimension 128x128x3 into 1x128x128x3 
        img_array = np.expand_dims(img_array, axis=0)

        #Calling the predict method on model to predict 'me' on the image
        result = model.predict(img_array, verbose=0)
        prediction = int(result[0][0])
        score = tf.nn.softmax(result[0])
        if(100 * np.max(score)) > 80:
            print(class_names[np.argmax(score)] + " ("+str(100 * np.max(score))+"%)")
        #if prediction is 0, which means I am missing on the image, then show the frame in gray color.
        if prediction == 0:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        cv2.imshow("Capturing", frame)
        key=cv2.waitKey(1)
        if key == ord('q'):
                break
video.release()
cv2.destroyAllWindows()