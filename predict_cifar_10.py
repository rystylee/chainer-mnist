import argparse

import chainer
import chainer.links as L
from trainer.train_cifar_10 import CNN

import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(description='Chainer: CIFAR10 predicting CNN')
    parser.add_argument('--unit', '-u', type=int, default=1000,
                        help='Number of units')
    args = parser.parse_args()

    print('# unit: {}'.format(args.unit))
    print('')

    model = L.Classifier(CNN())
    train, test = chainer.datasets.get_cifar10(withlabel=True, scale=1.) # (3, 32, 32)

    chainer.serializers.load_npz('result_cifar_10/model_epoch-20.npz', model)

    x, t = test[100]
    plt.imshow(x.transpose(1, 2, 0), cmap=None) # (3, 32, 32) -> (32, 32, 3)
    plt.show()
    print('label:', t)

    x = x[None, ...]
    y = model.predictor(x)
    y = y.data

    print('predicted_label:', y.argmax(axis=1)[0])

if __name__ == '__main__':
    main()
