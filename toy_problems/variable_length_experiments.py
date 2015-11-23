import lstm_problems
import lasagne
import theano
import theano.tensor as T
import numpy as np
import sys
sys.path.append('..')
import layers
import os
import logging
import csv
import collections
import itertools

BATCH_SIZE = 100
MAX_BATCHES = 100000
TEST_FREQUENCY = 1000
TEST_BATCH_SIZE = 10
TEST_N_BATCHES = 100
HIDDEN_SIZE = 100
RESULTS_PATH = 'results'
MIN_SEQUENCE_LENGTH = 50
MAX_SEQUENCE_LENGTH = 10000


if __name__ == '__main__':

    if not os.path.exists(RESULTS_PATH):
        os.makedirs(RESULTS_PATH)

    # Create logger for results
    logger = logging.getLogger('')
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(os.path.join(RESULTS_PATH, 'variable_length.log'))
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(message)s')
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)
    logger.addHandler(ch)
    logger.addHandler(fh)

    # Create CSV writer for results
    results_csv = open(os.path.join(RESULTS_PATH, 'variable_length.csv'), 'wb')
    writer = csv.writer(results_csv)

    # Define experiment space
    task_options = collections.OrderedDict([
        ('add', lstm_problems.add),
        ('multiply', lstm_problems.multiply)])
    aggregation_layer_options = collections.OrderedDict([
        ('attention', layers.AttentionLayer),
        ('mean', layers.MeanLayer)])
    learning_rate_options = [.01, .003, .001, .0003]
    # Create iterator over every possible hyperparameter combination
    option_iterator = itertools.product(
        task_options, aggregation_layer_options, learning_rate_options)
    # Keep track of the smallest number of batches for a given task
    best_batches_per_task = collections.defaultdict(lambda: np.inf)
    # Iterate over hypermarameter settings
    for (task, aggregation_layer, learning_rate) in option_iterator:
        logger.info(
            '####### Learning rate: {}, aggregation: {}, '
            'task: {}'.format(
                learning_rate, aggregation_layer, task))
        # Create test set of pre-sampled batches
        test_set = [task_options[task](int(n), TEST_BATCH_SIZE, int(n))
                    for n in np.linspace(MIN_SEQUENCE_LENGTH,
                                         MAX_SEQUENCE_LENGTH,
                                         TEST_N_BATCHES)]
        # Get a dummy batch to use for statistics
        dummy_batch, dummy_targets, dummy_mask = task_options[task](
            MIN_SEQUENCE_LENGTH, 1000, MAX_SEQUENCE_LENGTH)
        # Get the input shape from the dummy batch
        input_shape = (None, dummy_batch.shape[1], dummy_batch.shape[2])
        # Construct network
        layer = lasagne.layers.InputLayer(shape=input_shape, name='Input')
        n_batch, n_seq, n_features = layer.input_var.shape
        # Store a dictionary which conveniently maps names to layers we will
        # need to access later
        layers = {'in': layer}
        # Add dense input layer
        layer = lasagne.layers.ReshapeLayer(
            layer, (n_batch*n_seq, input_shape[-1]), name='Reshape 1')
        layer = lasagne.layers.DenseLayer(
            layer, HIDDEN_SIZE, W=lasagne.init.HeNormal(), name='Input dense',
            nonlinearity=lasagne.nonlinearities.leaky_rectify)
        layer = lasagne.layers.ReshapeLayer(
            layer, (n_batch, n_seq, HIDDEN_SIZE), name='Reshape 2')
        # Add the layer to aggregate over time steps
        if aggregation_layer == 'attention':
            # We must force He initialization because Lasagne doesn't like
            # 1-dim shapes in He and Glorot initializers
            layer = aggregation_layer_options[aggregation_layer](
                layer,
                W=lasagne.init.Normal(1./np.sqrt(layer.output_shape[-1])),
                name='Attention')
        elif aggregation_layer == 'mean':
            layer = aggregation_layer_options[aggregation_layer](
                layer, name='Attention')
        else:
            raise ValueError("Unknown aggregation layer type '{}'.".format(
                aggregation_layer))
        # Add dense hidden layer
        layer = lasagne.layers.DenseLayer(
            layer, HIDDEN_SIZE, W=lasagne.init.HeNormal(), name='Out dense 1',
            nonlinearity=lasagne.nonlinearities.leaky_rectify)
        # Add final dense layer, whose bias is initialized to the target mean
        layer = lasagne.layers.DenseLayer(
            layer, 1, W=lasagne.init.HeNormal(), name='Out dense 2',
            nonlinearity=lasagne.nonlinearities.leaky_rectify)
        layer = lasagne.layers.ReshapeLayer(
            layer, (-1,))
        # Keep track of the final layer
        layers['out'] = layer
        # Symbolic variable for target values
        target = T.vector('target')
        # Retrive the symbolic expression for the network
        network_output = lasagne.layers.get_output(layers['out'])
        # Create a symbolic function for the network cost
        cost = T.mean((network_output - target)**2)
        # Retrieve all network parameters
        all_params = lasagne.layers.get_all_params(layers['out'])
        # Compute updates
        updates = lasagne.updates.adam(cost, all_params, learning_rate)
        # Compile training function
        train = theano.function([layers['in'].input_var, target],
                                cost, updates=updates)

        # Accuracy is defined as the proportion of examples whose absolute
        # error is less than .04
        accuracy = T.mean(abs(network_output - target) < .04)
        compute_accuracy = theano.function(
            [layers['in'].input_var, target], accuracy)

        # Store cost over minibatches
        cost = 0
        # Keep track of best accuracy found
        best_accuracy = 0.
        try:
            for batch_idx in range(MAX_BATCHES):
                # Choose a random sequence length
                sequence_length = np.random.random_integers(
                    MIN_SEQUENCE_LENGTH, MAX_SEQUENCE_LENGTH)
                # Generate a batch of data
                X, y, m = task_options[task](
                    sequence_length, BATCH_SIZE, sequence_length)
                cost += train(X.astype(theano.config.floatX),
                              y.astype(theano.config.floatX))
                # Quit when a non-finite value is found
                if any([not np.isfinite(cost),
                        any([not np.all(np.isfinite(p.get_value()))
                            for p in all_params])]):
                    logger.info('####### Non-finite values found, aborting')
                    break
                if not (batch_idx + 1) % TEST_FREQUENCY:
                    # Compute mean accuracy across batches from test set
                    test_accuracy = np.mean([
                        compute_accuracy(
                            X_test.astype(theano.config.floatX),
                            y_test.astype(theano.config.floatX))
                        for X_test, y_test, mask_test in test_set])
                    logger.info(
                        "Batches trained: {}, cost: {}, accuracy: {}".format(
                            batch_idx + 1, cost*BATCH_SIZE/TEST_FREQUENCY,
                            test_accuracy))
                    # Update best accuracy, if a better one was found
                    if test_accuracy > best_accuracy:
                        best_accuracy = test_accuracy
                    # Quit if we achieve perfect accuracy
                    if test_accuracy == 1.:
                        break
                    # Quit if we have exceeded the smallest number of batches
                    # for this particular task/aggregation layer/sequence len
                    if batch_idx > best_batches_per_task[
                            (task, aggregation_layer)]:
                        break
                    # Reset training cost accumulator
                    cost = 0
        # Gracefully handle Theano errors
        except (RuntimeError, MemoryError) as e:
            print "Error: {}".format(e)
            # Flag this as a failed trial by setting accuracy = -1
            best_accuracy = -1
        # If the number of batches used is smaller than the best so far...
        if batch_idx < best_batches_per_task[(task, aggregation_layer)]:
            best_batches_per_task[(task, aggregation_layer)] = batch_idx
        # Write out this result to the CSV
        writer.writerow([learning_rate, aggregation_layer,
                         task, best_accuracy, batch_idx])
    results_csv.close()
