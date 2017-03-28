import cv2

img = cv2.imread('img/Architecture.jpg',3)
greyimg = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
greyimg = cv2.cvtColor(img,cv2.COLOR_BGR2LAB)
cv2.imshow('image',greyimg)

cv2.waitKey(0)
cv2.destroyAllWindows()
