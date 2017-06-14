import numpy as np
import theano
from theano import tensor as T
import cPickle as pkl
from imgaug import augmenters as iaa
from matplotlib import pyplot as plt
import time
import os
import tqdm
from scipy import misc
from random import shuffle
from theano.compile.nanguardmode import NanGuardMode
from numba import jit

import utils
import metrics
import buildmodel

seq = iaa.Sequential([
    iaa.Crop(px=(0, 16)),  # crop images from each side by 0 to 16px (randomly chosen)
    iaa.Fliplr(0.5),  # horizontally flip 50% of the images
    iaa.GaussianBlur(sigma=(0, 3.0))  # blur images with a sigma of 0 to 3.0
])


def build_vgg16(phase='train', **kwargs):
    config_file = kwargs.get('config_file')
    if phase == 'train':
        training_data = kwargs.get('training_data')
        validation_data = kwargs.get('validation_data')

        model = buildmodel.Model(config_file)
        save_file = model.checkpoint
        if not model.continue_training:
            utils.load_weights('pretrained/vgg16_weights.npz', model)

        x = T.tensor4('input', theano.config.floatX)
        y = T.bvector('output')
        step = T.scalar('step', 'int32')

        input_shape = model.input_size
        placeholder_x = theano.shared(np.zeros(input_shape, 'float32'), 'input_placeholder')
        placeholder_y = theano.shared(np.zeros((model.batch_size, ), 'int8'), 'label_placeholder')
        placeholder_lr = theano.shared(np.cast[theano.config.floatX](model.learning_rate), 'learning_rate')

        output = model.inference(x)
        cost = model.build_cost(y)
        accuracy = 100. - metrics.MeanClassificationErrors(T.cast(output >= 0.5, 'int8'), T.cast(y, 'int8')) * 100.
        updates = model.build_updates()

        train = theano.function([step], cost, updates=updates, on_unused_input='warn', allow_input_downcast=True,
                                givens={x: placeholder_x, y: placeholder_y})
        validate = theano.function([], [cost, accuracy], on_unused_input='warn', allow_input_downcast=True,
                                   givens={x: placeholder_x, y: placeholder_y})

        early_stopping = False
        vote_to_terminate = 0
        epoch = 0
        num_training_batches = training_data[0].shape[0] / model.batch_size
        num_validation_batches = validation_data[0].shape[0] / model.batch_size
        best_accuracy = 0.
        best_epoch = 0
        training_cost_to_plot = []
        validation_cost_to_plot = []
        print 'Training...'
        start_training_time = time.time()
        while epoch < model.n_epochs and not early_stopping:
            epoch += 1
            print 'Epoch %d starts...' % epoch
            print '\tlearning rate decreased to %.10f' % placeholder_lr.get_value()
            batch = generator(training_data, model.batch_size)
            training_cost = 0.
            start_epoch_time = time.time()
            for idx, b in enumerate(batch):
                buildmodel.Model.set_training_status(True)
                iteration = (epoch - 1.) * num_training_batches + idx + 1
                kwargs = {'learning_rate': placeholder_lr, 'iteration': iteration}
                model.decrease_learning_rate(**kwargs)

                x, y = b
                x = x.astype(theano.config.floatX) if np.random.randint(0, 2) \
                    else seq.augment_images(x.astype(theano.config.floatX) * 255.) / 255.
                utils.update_input((x.transpose(0, 3, 1, 2), y), (placeholder_x, placeholder_y))
                training_cost += train(epoch)
                if np.isnan(training_cost):
                    try:
                        print 'Cost is NaN. Trying to go back to the last checkpoint'
                        model = pkl.load(open(model.continue_checkpoint, 'rb'))
                        model.initial_learning_rate /= 10.
                        model.final_learning_rate /= 10.
                        placeholder_lr.set_value(model.initial_learning_rate)
                        epoch = 0
                        best_epoch = 1
                    except ValueError:
                        raise ValueError('Training failed due to NaN cost')

                if iteration % num_training_batches == 0:
                    batch_valid = generator(validation_data, model.batch_size)
                    buildmodel.Model.set_training_status(False)
                    validation_cost = 0.
                    validation_accuracy = 0.
                    for b_valid in batch_valid:
                        utils.update_input((b_valid[0].transpose(0, 3, 1, 2), b_valid[1]), (placeholder_x, placeholder_y))
                        c, a = validate()
                        validation_cost += c
                        validation_accuracy += a
                    validation_cost /= num_validation_batches
                    validation_accuracy /= num_validation_batches
                    print '\tvalidation cost: %.4f' % validation_cost
                    print '\tvalidation accuracy: %.4f' % validation_accuracy
                    if validation_accuracy > best_accuracy:
                        best_epoch = epoch
                        best_accuracy = validation_accuracy
                        vote_to_terminate = 0
                        pkl.dump(model, open(save_file, 'wb'))
                        print '\tbest validation accuracy: %.4f' % best_accuracy
                        print '\tbest model dumped to %s' % save_file
                    else:
                        vote_to_terminate += 1

                    if model.display_cost:
                        training_cost_to_plot.append(training_cost/(idx + 1))
                        validation_cost_to_plot.append(validation_cost)
                        plt.clf()
                        plt.plot(training_cost_to_plot)
                        plt.plot(validation_cost_to_plot)
                        plt.show(block=False)
                        plt.pause(1e-5)

            training_cost /= num_training_batches
            print '\tepoch %d took %.2f mins' % (epoch, (time.time() - start_epoch_time)/60.)
            print '\ttraining cost: %.4f' % training_cost

            if vote_to_terminate >= 30:
                print 'Training terminated due to no improvement!'
                early_stopping = True
        print 'Best validation accuracy: %.4f' % best_accuracy

        print 'Training the network with all available data...'
        data = (np.concatenate((training_data[0], validation_data[0])), np.concatenate((training_data[1], validation_data[1])))
        if model.continue_training:
            model = pkl.load(open(model.continue_checkpoint, 'rb'))
        else:
            utils.load_weights('pretrained/vgg16_weights.npz', model)
        placeholder_lr.set_value(np.cast[theano.config.floatX](model.learning_rate))
        buildmodel.Model.set_training_status(True)
        for i in range(best_epoch):
            print 'Epoch %d starts...' % (i+1)
            batch = generator(data, model.batch_size)
            training_cost = 0.
            for idx, b in enumerate(batch):
                iteration = i * num_training_batches + idx + 1
                kwargs = {'learning_rate': placeholder_lr, 'iteration': iteration}
                model.decrease_learning_rate(**kwargs)
                x, y = b
                x = x.astype(theano.config.floatX) if np.random.randint(0, 2) \
                    else seq.augment_images(x.astype(theano.config.floatX) * 255.) / 255.
                utils.update_input((x.transpose(0, 3, 1, 2), y), (placeholder_x, placeholder_y))
                training_cost += train(i+1)
            training_cost /= num_training_batches
            print '\ttraining cost: %.4f' % training_cost
        pkl.dump(model, open(save_file, 'wb'))
        print 'Final model dumped to %s' % save_file
        print 'Training ended after %.2f hours' % ((time.time() - start_training_time) / 3600.)
    elif phase == 'test':
        path = kwargs.get('testing_path')
        is_folder = kwargs.get('is_folder')
        testing_model = kwargs.get('testing_model')
        image_list = [path + '/' + f for f in os.listdir(path) if f.endswith('.jpg')] if is_folder else [path]
        shuffle(image_list)
        model = pkl.load(open(testing_model, 'rb'))

        x = T.tensor4('input', theano.config.floatX)
        confidence = inference(x, model)
        predictions = T.cast(confidence >= 0.5, 'int8')
        test = theano.function([x], [predictions, confidence], allow_input_downcast=True)
        buildmodel.Model.set_training_status(False)

        print('Testing %d images...' % len(image_list))
        unidentified = 0.
        time.sleep(0.1)
        for i in tqdm.tqdm(range(len(image_list)), unit='images'):
            ori_img = misc.imread(image_list[i])
            if len(ori_img.shape) < 3:
                continue
            img = misc.imresize(ori_img, (224, 224)) / 255.
            img = np.expand_dims(img, 0).transpose((0, 3, 1, 2))
            pred, c = test(img)

            title = 'File %s\n' % image_list[i]
            if 0.3 < c < 0.7:
                title += 'Unidentified animal. Probably '
                unidentified += 1.
            title += 'Dog' if pred else 'Cat'
            conf = c*100 if 'dog' in title.lower() else (1-c)*100 if 'cat' in title.lower() else c*100
            title += ' with %.2f%% confidence. Coverage: %.2f%%' % (conf, 100. - 100.*unidentified/(i+1))
            plt.figure(1)
            plt.imshow(ori_img)
            plt.title(title)
            plt.show()
        print('Testing finished!')
    else:
        raise NotImplementedError


# @jit(nopython=True, cache=True)
def generator(data, batch_size):
    x, y = data
    indices = np.random.permutation(np.asarray(x).shape[0])
    x = x[indices]
    y = y[indices]
    num_batch = np.asarray(x).shape[0] / batch_size
    for i in xrange(num_batch):
        yield (x[i*batch_size:(i+1)*batch_size], y[i*batch_size:(i+1)*batch_size])


# @jit(nopython=True, cache=True)
def inference(input, model):
    feed = input
    for layer, idx in zip(model, xrange(len(model))):
        feed = layer.get_output(feed.flatten(2))if 'fc' in layer.layer_name else layer.get_output(feed)
    return feed

if __name__ == '__main__':
    training_data = pkl.load(open('training.pkl', 'rb'))
    validation_data = pkl.load(open('validation.pkl', 'rb'))
    kwargs = {'training_data': training_data, 'validation_data': validation_data, 'config_file': 'vgg16.config',
              'testing_model': 'checkpoints/run19/model.mdl', 'testing_path': 'test', 'is_folder': True}
    build_vgg16('train', **kwargs)
