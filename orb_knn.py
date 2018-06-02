from sklearn.neighbors import KNeighborsClassifier
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2
import os
from matplotlib import pyplot as plt
from scipy.cluster.vq import *
from sklearn.preprocessing import StandardScaler

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to input dataset")

ap.add_argument("-t", "--tdataset", required=True,
	help="path to test dataset")

args = vars(ap.parse_args())

imagePaths = list(paths.list_images(args["dataset"]))				#path to train data
timagePaths = list(paths.list_images(args["tdataset"]))				#path to test data

# Create feature extraction and keypoint detector objects

orb = cv2.ORB_create(55)

# List where all the descriptors are stored
des_list = []
labels = []
for (i, imagePath) in enumerate(imagePaths):

	image = cv2.imread(imagePath)
	label = imagePath.split(os.path.sep)[-1].split(".")[0]
	labels.append(label)
	im = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
	kp = orb.detect(im,None)
	
	kp, des = orb.compute(im, kp)
	des=des.flatten()
	des_list.append((des))
	# #Uncomment to view the detected keypoints
	# cv2.drawKeypoints(im,kp,image)
	# cv2.imshow("sift",image)
 # 	cv2.waitKey(0)
	

trainLabels = np.array(labels)
trainFeat = np.array(des_list)

model = KNeighborsClassifier(n_neighbors=1, n_jobs=-1,metric = 'braycurtis')
model.fit(trainFeat, trainLabels)


tfeatures = []
tlabels = []

for (i, imagePath) in enumerate(timagePaths):
	
	image = cv2.imread(imagePath)
	label = imagePath.split(os.path.sep)[-1].split(".")[0]
	labels.append(label)
	im = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
	kp = orb.detect(im,None)
	kp, des = orb.compute(im, kp)
	des=des.flatten()
	tfeatures.append(des)
	tlabels.append(label)

	# #uncomment if you want to check predicted output for each label
	# ttlabels = []
	# ttfeatures = []
	# ttfeatures.append(des)
	# ttlabels.append(label)
	# print ("For image "+ str(i))
	# print ("Label was: " + str(label))
	# print ("Prediction is: " + str(model.predict(ttfeatures)))

testFeat = (np.array(tfeatures))
testLabels = (np.array(tlabels))
	
acc = model.score(testFeat, testLabels)
print ("Accuracy is: "+str(acc))
