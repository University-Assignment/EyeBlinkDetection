import glob
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split

dir = "dataset_B_Eye_Images/"
categories = ["closedRightEyes", "openRightEyes"]
X = []
Y = []

for idx, cat in enumerate(categories):
    imageDir = dir + cat
    files = glob.glob(imageDir + "/*.jpg")
    for i, f in enumerate(files):
        img = Image.open(f)
        data = np.asarray(img)
        data = data.reshape((24, 24, 1))
        X.append(data)
        Y.append(idx)
X = np.array(X)
Y = np.array(Y)

xTrain, xTest, yTrain, yTest = train_test_split(X, Y)
np.save('dataset/xTrain.npy', xTrain)
np.save('dataset/yTrain.npy', yTrain)
np.save('dataset/xTest.npy', xTest)
np.save('dataset/yTest.npy', yTest)