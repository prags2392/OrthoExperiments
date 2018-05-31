from sklearn.neighbors import KNeighborsClassifier
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2
import os
from matplotlib import pyplot as plt


def extract_color_histogram(image, bins=(96, 96, 96)):

	hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
	hist = cv2.calcHist([hsv], [0, 1, 2], None, bins,
		[0, 256, 0, 256, 0, 256])
 
	cv2.normalize(hist, hist)
 
	# return the flattened histogram as the feature vector
	return hist.flatten()

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to input dataset")

ap.add_argument("-t", "--tdataset", required=True,
	help="path to test dataset")

args = vars(ap.parse_args())

imagePaths = list(paths.list_images(args["dataset"]))				#path to train data
timagePaths = list(paths.list_images(args["tdataset"]))				#path to test data


features = []
labels = []
tfeatures = []
tlabels = []

colors = ('b','g','r','c','m','y','k','m-','b--','go', 'r+')

for (i, imagePath) in enumerate(imagePaths):

	image = cv2.imread(imagePath)
	label = imagePath.split(os.path.sep)[-1].split(".")[0]

	#cv2.imshow("img",image)
	#cv2.waitKey(0)

	hist = extract_color_histogram(image)
	features.append(hist)
	labels.append(label)
	plt.plot(hist, colors[i])
	plt.xlim([0, 256])
	#plt.show()			//uncomment this if you want to see histograms individually for each image
	

#plt.show()			#uncomment this if you want all histograms on one plot

trainLabels = np.array(labels)
trainFeat = np.array(features)
model = KNeighborsClassifier(n_neighbors=1, n_jobs=-1)
#model = KNeighborsClassifier(n_neighbors=1, n_jobs=-1,metric = 'braycurtis')		#gives the same accuracy but misclassifies different images, need to figure out the maths behind it
model.fit(trainFeat, trainLabels)


for (i, imagePath) in enumerate(timagePaths):
	
	image = cv2.imread(imagePath)
	label = imagePath.split(os.path.sep)[-1].split(".")[0]
	hist = extract_color_histogram(image)
	tfeatures.append(hist)
	tlabels.append(label)

	# uncomment if you want to check predicted output for each label
	ttlabels = []
	ttfeatures = []
	ttfeatures.append(hist)
	ttlabels.append(label)
	print ("For image "+ str(i))
	print ("Label was: " + str(label))
	print ("Prediction is: " + str(model.predict(ttfeatures)))

testFeat = (np.array(tfeatures))
testLabels = (np.array(tlabels))
	
acc = model.score(testFeat, testLabels)
print ("Accuracy is: "+str(acc))
