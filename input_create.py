import cv2

def create_input(filename):
	img = cv2.imread(filename,3)
	height, width, channels = img.shape
	greyimg = cv2.cvtColor(cv2.resize(img, (224, 224)),cv2.COLOR_BGR2GRAY)
	colorimg = cv2.cvtColor(cv2.resize(img, (224, 224)),cv2.COLOR_BGR2LAB)

	resized_greyimg = cv2.resize(greyimg, (width,height))
	resized_colorimg = cv2.resize(colorimg, (width,height))

	cv2.imshow('greyimage',greyimg)
	cv2.imshow('colorimage',colorimg)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

if __name__ == "__main__":
	filename = 'img/Architecture.jpg'
	create_input(filename)