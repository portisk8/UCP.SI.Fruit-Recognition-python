import numpy as np
from keras.preprocessing import image
import keras
import cv2
import os
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

currentPath = os.getcwdb().decode("utf-8") 
trainPath = os.path.join(currentPath, 'MainRecognition\\train')
testPath = os.path.join(currentPath, 'MainRecognition\\test')
finalPath = os.path.join(currentPath, 'MainRecognition\\final')

training_set = train_datagen.flow_from_directory(trainPath,
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')
test_datagen = ImageDataGenerator(rescale = 1./255)
test_set = test_datagen.flow_from_directory(testPath,
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')
print("Image Processing.......Compleated")
cnn = tf.keras.models.Sequential()
print("Building Neural Network.....")
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[64, 64, 3]))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
cnn.add(tf.keras.layers.Flatten())
cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))
cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
print("Training cnn")
cnn.fit(x = training_set, validation_data = test_set, epochs = 25)
vid = cv2.VideoCapture(0)
print("Camera connection successfully established")
i = 0
while(True):  
    r, frame = vid.read() 
    cv2.imshow('frame', frame)
    cv2.imwrite(finalPath+'\\'+str(i)+".jpg", frame)
    test_image = keras.utils.load_img(finalPath+'\\'+str(i)+".jpg", target_size = (64, 64))
    test_image = tf.keras.preprocessing.image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    result = cnn.predict(test_image)
    print(result)
    training_set.class_indices
    if result[0][0] ==1:
        print("banana")
    if result[0][0] == 0:
        print("apple")
    os.remove(finalPath+'\\'+str(i)+".jpg")
    i = i + 1
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break
vid.release() 
cv2.destroyAllWindows() 
