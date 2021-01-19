from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage

xtest = []
ytest = []
for foldI in range(9, 11, 1):
	fold = "{0:02d}".format(foldI)
	print("test fold:", fold)
	xs = np.load("./data/xs{}.npy".format(fold))
	ys = np.load("./data/ys{}.npy".format(fold))
	xtest.extend(xs)
	ytest.extend(ys)

xtest = [np.expand_dims(x, axis=2) for x in xtest]
xtest = np.array(xtest)
ytest = [np.expand_dims(y, axis=2) for y in ytest]
ytest = np.array(ytest)

print("testing data x:", xtest.shape)
print("testing data y:", ytest.shape)

model = keras.models.load_model('model')


i = np.random.randint(0,xtest.shape[0]-25)
imgtest = xtest[i:i+25,:,:,:]
imgtesty = ytest[i:i+25,:,:,:]
imgtestout = model.predict(imgtest)


plt.figure(figsize=(10,10))
for i in range(25):
	plt.subplot(5,5,i+1)
	plt.xticks([])
	plt.yticks([])
	plt.grid(False)
	plt.imshow(imgtest[i,:,:,0], cmap='gray')
	# The CIFAR labels happen to be arrays,
	# which is why you need the extra index
	plt.title("input")
plt.show()


plt.figure(figsize=(10,10))
for i in range(25):
	plt.subplot(5,5,i+1)
	plt.xticks([])
	plt.yticks([])
	plt.grid(False)
	plt.imshow(imgtestout[i,:,:,0], cmap='gray')
	# The CIFAR labels happen to be arrays,
	# which is why you need the extra index
	plt.title("output")
plt.show()


plt.figure(figsize=(10,10))
for i in range(25):
	plt.subplot(5,5,i+1)
	plt.xticks([])
	plt.yticks([])
	plt.grid(False)
	plt.imshow(imgtesty[i,:,:,0], cmap='gray')
	# The CIFAR labels happen to be arrays,
	# which is why you need the extra index
	plt.title("Y")
plt.show()

blur = np.zeros_like(imgtest)
for i in range(25):
	blur[i] = scipy.ndimage.gaussian_filter(imgtest[i], sigma=10)
imgtesthq = imgtest.copy()
mask = imgtestout > 0.4
np.putmask(imgtesthq, mask, blur)
plt.figure(figsize=(10,10))
for i in range(25):
	plt.subplot(5,5,i+1)
	plt.xticks([])
	plt.yticks([])
	plt.grid(False)
	plt.imshow(imgtesthq[i,:,:,0], cmap='gray')
	# The CIFAR labels happen to be arrays,
	# which is why you need the extra index
	plt.title("example")
plt.show()

plt.figure(figsize=(10,10))
for i in range(25):
	plt.subplot(5,5,i+1)
	plt.xticks([])
	plt.yticks([])
	plt.grid(False)
	plt.imshow(mask[i,:,:,0], cmap='gray')
	# The CIFAR labels happen to be arrays,
	# which is why you need the extra index
	plt.title("mask applied")
plt.show()