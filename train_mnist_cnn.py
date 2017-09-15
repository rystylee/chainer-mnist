import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions


class CNN(chainer.Chain):

    def __init__(self):
        super(CNN, self).__init__()
        with self.init_scope():
            self.c1 = L.Convolution2D(in_channels=1, out_channels=32, ksize=5)
            self.c2 = L.Convolution2D(in_channels=32, out_channels=64, ksize=5)
            # self.l1 = L.Linear(in_size=1024, out_size=10)
            self.l1 = L.Linear(in_size=None, out_size=10)

    def __call__(self, x):
        h = F.max_pooling_2d(F.relu(self.c1(x)), 2)
        h = F.max_pooling_2d(F.relu(self.c2(h)), 2)

        return self.l1(h)


def main():

    batchsize = 100
    epoch = 20
    frequency = -1
    gpu = 0
    out = 'result_cnn'
    resume = ''
    unit = 1000

    print('GPU: {}'.format(gpu))
    print('# unit: {}'.format(unit))
    print('# Minibatch-size: {}'.format(batchsize))
    print('# epoch: {}'.format(epoch))
    print('')

    model = L.Classifier(CNN())
    if gpu >= 0:
        chainer.cuda.get_device_from_id(gpu).use()
        model.to_gpu()

    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    train, test = chainer.datasets.get_mnist(ndim=3) #(1, 28, 28)

    train_iter = chainer.iterators.SerialIterator(train, batchsize, repeat=True, shuffle=True)
    test_iter = chainer.iterators.SerialIterator(test, batchsize, repeat=False, shuffle=False)

    updater = training.StandardUpdater(train_iter, optimizer, device=gpu)
    trainer = training.Trainer(updater, (epoch, 'epoch'), out=out)

    trainer.extend(extensions.Evaluator(test_iter, model, device=gpu))
    trainer.extend(extensions.dump_graph('main/loss'))
    frequency = epoch if frequency == -1 else max(1, frequency)
    trainer.extend(extensions.snapshot(), trigger=(frequency, 'epoch'))
    trainer.extend(extensions.LogReport())

    if extensions.PlotReport.available():
        trainer.extend(extensions.PlotReport(['main/loss', 'validation/main/loss'], 'epoch', file_name='loss.png'))
        trainer.extend(extensions.PlotReport(['main/accuracy', 'validation/main/accuracy'], 'epoch', file_name='accuracy.png'))

    trainer.extend(extensions.PrintReport(['epoch', 'main/loss', 'validation/main/loss', 'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))
    trainer.extend(extensions.ProgressBar())

    if resume:
        chainer.serializers.load_npz(resume, trainer)

    trainer.run()


if __name__ == '__main__':
    main()
