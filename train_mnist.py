import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions


class MLP(chainer.Chain):

    def __init__(self, n_units, n_out):
        super(MLP, self).__init__()
        with self.init_scope():
            self.l1 = L.Linear(in_size=784, out_size=n_units)
            self.l2 = L.Linear(in_size=n_units, out_size=n_units)
            self.l3 = L.Linear(in_size=n_units, out_size=n_out)

    def __call__(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))

        return self.l3(h2)


def main():

    batchsize = 100
    epoch = 20
    frequency = -1
    gpu = 0
    out = 'result_mlp'
    resume = ''
    unit = 1000

    print('GPU: {}'.format(gpu))
    print('# unit: {}'.format(unit))
    print('# Minibatch-size: {}'.format(batchsize))
    print('# epoch: {}'.format(epoch))
    print('')

    model = L.Classifier(MLP(n_units=unit, n_out=10))
    if gpu >= 0:
        chainer.cuda.get_device_from_id(gpu).use()
        model.to_gpu()

    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    train, test = chainer.datasets.get_mnist() #(784)

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
