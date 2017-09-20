import argparse

import chainer
import chainer.links as L
from trainer.train_mnist_cnn import CNN

import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(description='Chainer: MNIST predicting CNN')
    parser.add_argument('--unit', '-u', type=int, default=1000,
                        help='Number of units')
    args = parser.parse_args()

    print('# unit: {}'.format(args.unit))
    print('')

    model = L.Classifier(CNN())
    train, test = chainer.datasets.get_mnist(ndim=3) #(1, 28, 28)

    chainer.serializers.load_npz('result_cnn/model_epoch-20.npz', model)

    x, t = test[0]
    plt.imshow(x.reshape(28, 28), cmap='gray')
    plt.show()
    print('label:', t)

    x = x[None, ...]
    y = model.predictor(x)
    y = y.data

    print('predicted_label:', y.argmax(axis=1)[0])

if __name__ == '__main__':
    main()
