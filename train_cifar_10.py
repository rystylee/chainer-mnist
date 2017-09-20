import argparse

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions


class CNN(chainer.Chain):

    def __init__(self):
        super(CNN, self).__init__()
        with self.init_scope():
            self.c1 = L.Convolution2D(in_channels=3, out_channels=32, ksize=5)
            self.c2 = L.Convolution2D(in_channels=32, out_channels=64, ksize=5)
            self.l1 = L.Linear(in_size=None, out_size=10)

    def __call__(self, x):
        h = F.max_pooling_2d(F.relu(self.c1(x)), ksize=2)
        h = F.max_pooling_2d(F.relu(self.c2(h)), ksize=2)

        return self.l1(h)


def main():

    parser = argparse.ArgumentParser(description='Chainer: CIFAR-10 training CNN')
    parser.add_argument('--batchsize', '-b', type=int, default=100,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=20,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--frequency', '-f', type=int, default=-1,
                        help='Frequency of taking a snapshot')
    parser.add_argument('--gpu', '-g', type=int, default=0,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='result_cifar_10',
                        help='Directory to output the result')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    parser.add_argument('--unit', '-u', type=int, default=1000,
                        help='Number of units')
    args = parser.parse_args()

    print('GPU: {}'.format(args.gpu))
    print('# unit: {}'.format(args.unit))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))
    print('')

    model = L.Classifier(CNN())
    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()

    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    train, test = chainer.datasets.get_cifar10(withlabel=True, scale=1.) # (3, 32, 32)

    train_iter = chainer.iterators.SerialIterator(train, args.batchsize, repeat=True, shuffle=True)
    test_iter = chainer.iterators.SerialIterator(test, args.batchsize, repeat=False, shuffle=False)

    updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

    trainer.extend(extensions.Evaluator(test_iter, model, device=args.gpu))
    trainer.extend(extensions.dump_graph('main/loss'))
    frequency = args.epoch if args.frequency == -1 else max(1, args.frequency)
    trainer.extend(extensions.snapshot(), trigger=(frequency, 'epoch'))
    trainer.extend(extensions.snapshot_object(model, filename='model_epoch-{.updater.epoch}.npz'))
    trainer.extend(extensions.LogReport())

    if extensions.PlotReport.available():
        trainer.extend(extensions.PlotReport(['main/loss', 'validation/main/loss'], 'epoch', file_name='loss.png'))
        trainer.extend(extensions.PlotReport(['main/accuracy', 'validation/main/accuracy'], 'epoch', file_name='accuracy.png'))

    trainer.extend(extensions.PrintReport(['epoch', 'main/loss', 'validation/main/loss', 'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))
    trainer.extend(extensions.ProgressBar())

    if args.resume:
        chainer.serializers.load_npz(args.resume, trainer)

    trainer.run()


if __name__ == '__main__':
    main()
