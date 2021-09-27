import cv2
import utils_paths
import random
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

def own_data_loader():

    # 读取数据和标签
    print("------开始读取数据------")
    data = []
    labels = []


    # 拿到图像数据路径，方便后续读取
    imagePaths = sorted(list(utils_paths.list_images('./owndata')))
    random.seed(42)
    random.shuffle(imagePaths)


    # 遍历读取数据
    for imagePath in imagePaths:
        # 读取图像数据
        image = cv2.imread(imagePath)
        image = cv2.resize(image, (64, 64))
        data.append(image)
        # 读取标签
        label = imagePath.split(os.path.sep)[-2]
        labels.append(label)

    # 对图像数据做scale操作
    data = np.array(data, dtype="uint8") 
    labels = np.array(labels)
    print(labels)

    # 数据集切分
    (trainX, testX, trainY, testY) = train_test_split(data,labels, test_size=0.25, random_state=42)


    # 转换标签为one-hot encoding格式
    lb = LabelBinarizer()
    trainY = lb.fit_transform(trainY)
    testY = lb.transform(testY)

    return (trainX, trainY), (testX, testY)







