from tabnanny import verbose
import cv2
import numpy as np
from PIL import Image
from keras import models
import keras
import os
import tensorflow as tf

currentPath = os.getcwdb().decode("utf-8") 
#Load the saved model
model = models.load_model(currentPath+'\\fruit-recognition.h5')
finalPath = os.path.join(currentPath, 'MainRecognition\\final')

img_height = 100
img_width = 100
class_names = ['apple', 'banana', 'lemon', 'orange']
vid = cv2.VideoCapture(0)
print("Camera connection successfully established")
i = 0
while(True):  
    r, frame = vid.read() 
    cv2.imshow('frame', frame)
    cv2.imwrite(finalPath+'\\'+str(i)+".jpg", frame)
    #Predecir
    test_image = keras.utils.load_img(finalPath+'\\'+str(i)+".jpg", target_size = (img_width, img_height))
    test_image = tf.keras.preprocessing.image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    result = model.predict(test_image, verbose=0)
    # print(result)
    # print(tf.argmax(result[0], axis=-1))
    score = tf.nn.softmax(result[0])
    if(100 * np.max(score)) > 80:
        print(class_names[np.argmax(score)] + " ("+str(100 * np.max(score))+"%)")
    os.remove(finalPath+'\\'+str(i)+".jpg")
    i = i + 1
    # time.sleep(2)
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break
vid.release() 
cv2.destroyAllWindows() 
