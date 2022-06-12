import os
from datetime import datetime
print(os.getcwdb())

currentPath = os.getcwdb().decode("utf-8") 
trainPath = os.path.join(currentPath, 'MainRecognition\\train')
testPath = os.path.join(currentPath, 'MainRecognition\\test')
finalPath = os.path.join(currentPath, 'MainRecognition\\final')
print(trainPath)
print(testPath)
print(finalPath)
now = datetime.now()
print(datetime.now().strftime("%Y%m%d_%H%M%S")+'.jpg')
