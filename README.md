# OrthoExperiments
This repository holds python codes for experiments on Ortho

File Descriptions:

knn_hist.py : Classification betwen smith and non-smith images
              Arguments Reqd: -d <path to train data> -t <path to test data>
              Hyperparameters that can be tuned: histogram_bins, num of neighbours
  
orb_knn.py: Classification between smith and non smith using orb keypoints
              Arguments Reqd: -d <path to train data> -t <path to test data>
              Hyperparameters that can be tuned: number of keypoints, num of neighbours
  
  
data: folder contains two sub-folders train and test. Train consists of smith and non-smith images to train, test is similar
