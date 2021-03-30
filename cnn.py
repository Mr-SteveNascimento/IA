import mnist
import numpy as np
import pandas as pd
from conv import conv
from maxpool import MaxPool2
from softmax import Softmax

# We only use the first 1k examples of each set in the interest of time.
# Feel free to change this if you want.
train_images = mnist.train_images()[:1000]
train_labels = mnist.train_labels()[:1000]
test_images = mnist.test_images()[:1000]
test_labels = mnist.test_labels()[:1000]

conv = conv(8)  # 28x28x1 -> 26x26x8
pool = MaxPool2()  # 26x26x8 -> 13x13x8
softmax = Softmax(13 * 13 * 8, 10)  # 13x13x8 -> 10


def tulis_hasil(hasil, fname):
    hasil = np.array(hasil)
    pd.DataFrame({"fase": hasil[:, 0], "imagem": hasil[:, 1], "akurasi": hasil[:, 2]}).to_csv(fname, index=False, header=True)


def forward(image, label):
    """
    Completes a forward pass of the CNN and calculates the accuracy and
    cross-entropy loss.
    - image is a 2d numpy array
    - label is a digit
    """
    # We transform the image from [0, 255] to [-0.5, 0.5] to make it easier
    # to work with. This is standard practice.
    out = conv.forward((image / 255) - 0.5)

    out = pool.forward(out)
    out = softmax.forward(out)

    # Calculate cross-entropy loss and accuracy. np.log() is the natural log.
    loss = -np.log(out[label])
    acc = 1 if np.argmax(out) == label else 0

    return out, loss, acc


def train(im, label, lr=.005):
    """
    Completes a full training step on the given image and label.
    Returns the cross-entropy loss and accuracy.
    - image is a 2d numpy array
    - label is a digit
    - lr is the learning rate
    """
    # Forward
    out, loss, acc = forward(im, label)

    # Calculate initial gradient
    gradient = np.zeros(10)
    gradient[label] = -1 / out[label]

    # Backprop
    gradient = softmax.backprop(gradient, lr)
    gradient = pool.backprop(gradient)
    gradient = conv.backprop(gradient, lr)

    return loss, acc


hasil = []
print('MNIST CNN initialized!')
number_image = 0
correct = 0

# Train the CNN for 3 epochs
for epoch in range(3):
    print('--- Epoch %d ---' % (epoch + 1))

    # Shuffle the training data
    permutation = np.random.permutation(len(train_images))
    train_images = train_images[permutation]
    train_labels = train_labels[permutation]

    # Train!
    loss = 0
    num_correct = 0

    for i, (im, label) in enumerate(zip(train_images, train_labels)):
        number_image += 1
        if i > 0 and i % 100 == 99:
            print(
                '[Step %d] Past 100 steps: Average Loss %.3f | Accuracy: %d%%' %
                (i + 1, loss / 100, num_correct)
            )
            loss = 0
            num_correct = 0

        l, acc = train(im, label)
        loss += l
        num_correct += acc
        correct += acc
        hasil.append(["treino", number_image, correct/number_image])


# Test the CNN
print('\n--- Testing the CNN ---')
loss = 0
num_correct = 0
number_image = 0

for i, (im, label) in enumerate(zip(test_images, test_labels)):
    number_image += 1
    _, l, acc = forward(im, label)
    loss += l
    num_correct += acc

    hasil.append(["teste", number_image, (num_correct / number_image)*100])  # Adiciona os valores para plotar

num_tests = len(test_images)
print('Test Loss:', loss / num_tests)
print('Test Accuracy:', num_correct / num_tests)
tulis_hasil(hasil, "results/results.csv")
a = input("\n\nPressione Enter para encerrar o Software")
