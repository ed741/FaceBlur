import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, backend
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage

xtrain = []
ytrain = []
xtest = []
ytest = []

for foldI in range(1, 9, 1):
	fold = "{0:02d}".format(foldI)
	print("train fold:" ,fold)
	xs = np.load("./data/xs{}.npy".format(fold))
	ys = np.load("./data/ys{}.npy".format(fold))
	xtrain.extend(xs)
	ytrain.extend(ys)
for foldI in range(9, 11, 1):
	fold = "{0:02d}".format(foldI)
	print("test fold:", fold)
	xs = np.load("./data/xs{}.npy".format(fold))
	ys = np.load("./data/ys{}.npy".format(fold))
	xtest.extend(xs)
	ytest.extend(ys)

xtrain = [np.expand_dims(x, axis=2) for x in xtrain]
xtrain = np.array(xtrain)
ytrain = [np.expand_dims(y, axis=2) for y in ytrain]
ytrain = np.array(ytrain)
xtest = [np.expand_dims(x, axis=2) for x in xtest]
xtest = np.array(xtest)
ytest = [np.expand_dims(y, axis=2) for y in ytest]
ytest = np.array(ytest)

print("training data x:", xtrain.shape)
print("training data y:", ytrain.shape)

print("testing data x:", xtest.shape)
print("testing data y:", ytest.shape)
# plt.figure(figsize=(10,10))
# for i in range(25):
# 	plt.subplot(5,5,i+1)
# 	plt.xticks([])
# 	plt.yticks([])
# 	plt.grid(False)
# 	# plt.imshow(ytrain[i], cmap=plt.cm.binary)
# 	plt.imshow(xtrain[i], cmap=plt.cm.binary)
# 	# The CIFAR labels happen to be arrays,
# 	# which is why you need the extra index
# plt.show()

class GuassLayer(layers.Layer):
	def __init__(self):
		super(GuassLayer, self).__init__()
		s, k = 1, 2  # generate a (2k+1)x(2k+1) gaussian kernel with mean=0 and sigma = s
		probs = [np.exp(-z * z / (2 * s * s)) / np.sqrt(2 * np.pi * s * s) for z in range(-k, k + 1)]
		kernel_weights = np.outer(probs, probs)
		kernel_weights = np.expand_dims(kernel_weights, axis=-1)
		kernel_weights = np.expand_dims(kernel_weights, axis=-1)
		self.depthwise_kernel = tf.Variable(initial_value=kernel_weights, trainable=False, dtype='float32')

	def call(self, inputs, **kwargs):
		outputs = backend.depthwise_conv2d(
			inputs,
			self.depthwise_kernel,
			strides=(1,1),
			padding='same',
			data_format='channels_last')
		return outputs


inputs = keras.Input(shape=(256,256,1))
cnn = inputs
cnn = layers.Conv2D(8, (3, 3), padding='same', use_bias='false', activation='relu', input_shape=(256, 256, 1))(cnn)
# cnn = layers.MaxPool2D((2,2))(cnn)
# cnn = layers.Conv2D(2, (1, 1), padding='same', use_bias='false', activation='relu')(cnn)
cnn = layers.Conv2D(8, (3, 3), padding='same', use_bias='false', activation='relu')(cnn)
# cnn = layers.MaxPool2D((2,2))(cnn)
# cnn = layers.Conv2D(2, (1, 1), padding='same', use_bias='false', activation='relu')(cnn)
cnn = layers.Conv2D(8, (3, 3), padding='same', use_bias='false', activation='relu')(cnn)
cnn = layers.Conv2D(1, (5, 5), padding='same', use_bias='false', activation='linear')(cnn)
cnn = layers.Activation('hard_sigmoid')(cnn)
cnn = GuassLayer()(cnn)
# cnn = layers.UpSampling2D((4,4))(cnn)
outputs = cnn

model = keras.Model(inputs=inputs, outputs=outputs, name="cnn_model")

model.summary()

model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=['accuracy'])

history = model.fit(xtrain, ytrain, epochs=10, validation_data=(xtest, ytest))
model.save("model")

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()


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