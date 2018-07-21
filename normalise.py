import cv2

image = cv2.imread('D:\knee\data\smith\smith.jpg')
image = cv2.resize(image, (800, 800))
gray_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
norm_image = cv2.normalize(gray_image,None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

cv2.imshow("img",norm_image)
cv2.waitKey(0)

