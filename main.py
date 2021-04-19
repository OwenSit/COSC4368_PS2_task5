from keras.datasets import mnist
from matplotlib import pyplot

(trainX, trainy), (testX, testy) = mnist.load_data()
print(f"Train: {trainX.shape}, {trainy.shape}")
print(f"Test: {testX.shape}, {testy.shape}")
pyplot.imshow(trainX[0], cmap='gray_r')
pyplot.show()