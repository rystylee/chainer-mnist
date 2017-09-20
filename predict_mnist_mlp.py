import argparse

import chainer
import chainer.links as L
from train_mnist_mlp import MLP

import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(description='Chainer: MNIST predicting MLP')
    parser.add_argument('--unit', '-u', type=int, default=1000,
                        help='Number of units')
    args = parser.parse_args()

    print('# unit: {}'.format(args.unit))
    print('')

    model = L.Classifier(MLP(n_units=args.unit, n_out=10))
    train, test = chainer.datasets.get_mnist()

    chainer.serializers.load_npz('result_mlp/model_epoch-20.npz', model)

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
