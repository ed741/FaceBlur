import math

from PIL import Image, ImageFilter
import numpy as np

inputLabelsFolder = "./data/FDDB-folds/"
inputImgFolder = "./data/originalPics/"
outputFolder = "./data/"
outputSize = 256


def create_circular_mask(h=outputSize, w=outputSize, center=None, radius=None):
	if center is None: # use the middle of the image
		center = (int(w/2), int(h/2))
	if radius is None: # use the smallest distance between the center and image walls
		radius = min(center[0], center[1], w-center[0], h-center[1])

	Y, X = np.ogrid[:h, :w]
	dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

	mask = dist_from_center <= radius
	return mask



for foldI in range(1, 11, 1):
	fold = "{0:02d}".format(foldI)
	print(fold)
	images = []
	xs = []
	ys = []
	with open("{}FDDB-fold-{}-ellipseList.txt".format(inputLabelsFolder, fold)) as imgLabels:
		line = imgLabels.readline()
		while line is not '':
			file = line.strip()
			faceCountLine = imgLabels.readline()
			faceCount = int(faceCountLine.strip())
			images.append((file, faceCount, []))
			print(file, faceCount)
			for faceI in range(faceCount):
				faceLine = imgLabels.readline().strip()
				data = faceLine.split(" ")
				print(data)
				x = float(data[3])
				y = float(data[4])
				r = max(float(data[0]), float(data[1]))
				images[-1][2].append((x, y, r))
			img = Image.open("{}{}.jpg".format(inputImgFolder, file))
			# img.show()
			inputSize = img.size
			ratio = outputSize/min(inputSize)
			img = img.resize((int(math.ceil(inputSize[0]*ratio)),int(math.ceil(inputSize[1]*ratio))), Image.ANTIALIAS)
			# img.show()
			RGB = np.array(img).astype('float')
			if len(RGB.shape) == 3 and RGB.shape[2] == 3:
				R, G, B = RGB[:outputSize, :outputSize, 0], RGB[:outputSize, :outputSize, 1], RGB[:outputSize, :outputSize, 2]
				L = R * 299 / 1000 + G * 587 / 1000 + B * 114 / 1000  # Convert to L (select your desired coefficients).
			else:
				L = RGB[:outputSize, :outputSize]
			L /= 255
			# Limg = Image.fromarray(np.uint8(L*255))  # Format back to PIL image
			# Limg.show()
			LM = np.zeros_like(L)
			for x,y,r in images[-1][2]:
				mask = create_circular_mask(center=(int(x*ratio), int(y*ratio)), radius=int(r*ratio))
				LM[mask] = 1
			# LMimg = Image.fromarray(np.uint8(LM*255))  # Format back to PIL image
			# LMimg.show()
			xs.append(L)
			ys.append(LM)
			line = imgLabels.readline()

		np.save("{}xs{}".format(outputFolder, fold), xs)
		np.save("{}ys{}".format(outputFolder, fold), ys)




