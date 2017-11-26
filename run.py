# run NN code

from data import load_mnist_data
import nnet
import numpy as np

from matplotlib import pyplot as plt

def show(x):
    """ visualize a single training example """
    im = plt.imshow(np.reshape(1 - x, (28, 28)))
    im.set_cmap('gray')

print("loading MNIST dataset")
(train_data, valid_data) = load_mnist_data()

# reduce data sets for faster speed:
train_data = train_data[:50000]
valid_data = valid_data[:10000]

# to see a training example, uncomment:
x, y = train_data[123]
show(x)
plt.title("label = %d" % y)

# some initial params, not necessarily good ones
#net = nnet.Network([784, 15, 10])
net = nnet.Network([784, 29, 10])

# 29 good
# try decreaseing a2 size from 70

print("training")
net.train(train_data, valid_data, epochs=10, mini_batch_size=10, alpha=0.7)
# mnini batch size 5
# try fiddling with alpha was .7
# epochs was 25

ncorrect = net.evaluate(valid_data)
print("Validation accuracy: %.3f%%" % (100 * ncorrect / len(valid_data)))
