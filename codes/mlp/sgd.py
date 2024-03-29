from __future__ import division
"""
Stochastic Gradient Descent and related functionality such as
learning rate adaptation, momentum, and Polyak averaging.
"""
__authors__ = "Ian Goodfellow"
__copyright__ = "Copyright 2010-2012, Universite de Montreal"
__credits__ = ["Ian Goodfellow, David Warde-Farley"]
__license__ = "3-clause BSD"
__maintainer__ = "Ian Goodfellow, David Warde-Farley"
__email__ = "goodfeli@iro"

import logging
import warnings
import numpy as np

from debugging import detect_nan

from theano import config
from theano import function
from theano.compat.python2x import OrderedDict
from theano.gof.op import get_debug_values
from theano import tensor as T

from pylearn2.monitor import Monitor
from pylearn2.space import CompositeSpace, NullSpace
from pylearn2.train_extensions import TrainExtension
from pylearn2.training_algorithms.training_algorithm import TrainingAlgorithm
from pylearn2.training_algorithms.learning_rule import Momentum
from pylearn2.training_algorithms.learning_rule import MomentumAdjustor as LRMomentumAdjustor
from pylearn2.utils.iteration import is_stochastic
from pylearn2.utils import py_integer_types, py_float_types
from pylearn2.utils import safe_zip
from pylearn2.utils import serial
from pylearn2.utils import sharedX
from pylearn2.utils.data_specs import DataSpecsMapping
from pylearn2.utils.timing import log_timing


log = logging.getLogger(__name__)

class SGD(TrainingAlgorithm):
    """
    Stochastic Gradient Descent

    WRITEME: what is a good reference to read about this algorithm?

    A TrainingAlgorithm that does gradient descent on minibatches.

    """
    def __init__(self, learning_rate, cost=None, batch_size=None,
                 monitoring_batches=None, monitoring_dataset=None,
                 monitor_iteration_mode='sequential',
                 monitor_grads=None,
                 termination_criterion=None, update_callbacks=None,
                 learning_rule = None, init_momentum = None, set_batch_size = False,
                 train_iteration_mode = None, batches_per_iter=None,
                 theano_function_mode = None, monitoring_costs=None,
                 seed=[2012, 10, 5]):
        """
            WRITEME

            learning_rate: The learning rate to use.
                            Train object callbacks can change the learning
                            rate after each epoch. SGD update_callbacks
                            can change it after each minibatch.
            cost: a pylearn2.costs.cost.Cost object specifying the objective
                  function to be minimized.
                  Optionally, may be None. In this case, SGD will call the model's
                  get_default_cost method to obtain the objective function.
            init_momentum: **DEPRECATED** if None, does not use momentum
                            otherwise, use momentum and initialize the
                            momentum coefficient to init_momentum.
                            Callbacks can change this over time just like
                            the learning rate.

                            If the gradient is the same on every step, then
                            the update taken by the SGD algorithm is scaled
                            by a factor of 1/(1-momentum).

                            See section 9 of Geoffrey Hinton's "A Practical
                            Guide to Training Restricted Boltzmann Machines"
                            for details.
            learning_rule: training_algorithms.learning_rule.LearningRule,
                           a learning rule computes the new parameter values given
                           old parameters and first-order gradients. If learning_rule
                           is None, sgd.SGD will update parameters according to
                           the standard SGD learning rule.
            set_batch_size: if True, and batch_size conflicts with
                            model.force_batch_size, will call
                            model.set_batch_size(batch_size) in an attempt
                            to change model.force_batch_size
            theano_function_mode: The theano mode to compile the updates function with.
                            Note that pylearn2 includes some wraplinker modes that are
                            not bundled with theano. See pylearn2.devtools. These
                            extra modes let you do things like check for NaNs at every
                            step, or record md5 digests of all computations performed
                            by the update function to help isolate problems with nondeterminism.

            Parameters are updated by the formula:

            inc := momentum * inc - learning_rate * d cost / d param
            param := param + inc
        """

        if isinstance(cost, (list, tuple, set)):
            raise TypeError("SGD no longer supports using collections of Costs to represent "
                    " a sum of Costs. Use pylearn2.costs.cost.SumOfCosts instead.")
        self.norm = None
        if init_momentum:
            warnings.warn("init_momentum interface is deprecated and will "
            "become officially unsuported as of May 9, 2014. Please use the "
            "`learning_rule` parameter instead, providing an object of type "
            "`pylearn2.training_algorithms.learning_rule.Momentum` instead")
            # Convert to new interface under the hood.
            self.learning_rule = Momentum(init_momentum)
        else:
            self.learning_rule = learning_rule

        if monitor_grads is not None:
            assert type(monitor_grads) is list, "monitor_grads parameter should be a list."

        self.monitor_grads = monitor_grads
        self.learning_rate = sharedX(learning_rate, 'learning_rate')
        self.cost = cost
        self.batch_size = batch_size
        self.set_batch_size = set_batch_size
        self.batches_per_iter = batches_per_iter
        self._set_monitoring_dataset(monitoring_dataset)
        self.monitoring_batches = monitoring_batches
        self.monitor_iteration_mode = monitor_iteration_mode
        if monitoring_dataset is None:
            if monitoring_batches is not None:
                raise ValueError("Specified an amount of monitoring batches but not a monitoring dataset.")
        self.termination_criterion = termination_criterion
        self._register_update_callbacks(update_callbacks)
        if train_iteration_mode is None:
            train_iteration_mode = 'shuffled_sequential'
        self.train_iteration_mode = train_iteration_mode
        self.first = True
        self.rng = np.random.RandomState(seed)
        self.theano_function_mode = theano_function_mode
        self.monitoring_costs = monitoring_costs

    def setup(self, model, dataset):
        if self.cost is None:
            self.cost = model.get_default_cost()

        inf_params = [ param for param in model.get_params() if np.any(np.isinf(param.get_value())) ]
        if len(inf_params) > 0:
            raise ValueError("These params are Inf: "+str(inf_params))
        if any([np.any(np.isnan(param.get_value())) for param in model.get_params()]):
            nan_params = [ param for param in model.get_params() if np.any(np.isnan(param.get_value())) ]
            raise ValueError("These params are NaN: "+str(nan_params))
        self.model = model

        batch_size = self.batch_size
        if hasattr(model, "force_batch_size"):
            if model.force_batch_size > 0:
                if batch_size is not None:
                    if batch_size != model.force_batch_size:
                        if self.set_batch_size:
                            model.set_batch_size(batch_size)
                        else:
                            raise ValueError("batch_size argument to SGD conflicts with model's force_batch_size attribute")
                else:
                    self.batch_size = model.force_batch_size
        model._test_batch_size = self.batch_size
        self.monitor = Monitor.get_monitor(model)
        self.monitor._sanity_check()

        data_specs = self.cost.get_data_specs(self.model)
        mapping = DataSpecsMapping(data_specs)
        space_tuple = mapping.flatten(data_specs[0], return_tuple=True)
        source_tuple = mapping.flatten(data_specs[1], return_tuple=True)

        # Build a flat tuple of Theano Variables, one for each space.
        # We want that so that if the same space/source is specified
        # more than once in data_specs, only one Theano Variable
        # is generated for it, and the corresponding value is passed
        # only once to the compiled Theano function.
        theano_args = []
        for space, source in safe_zip(space_tuple, source_tuple):
            name = '%s[%s]' % (self.__class__.__name__, source)
            arg = space.make_theano_batch(name=name, batch_size = self.batch_size)
            theano_args.append(arg)
        theano_args = tuple(theano_args)

        # Methods of `self.cost` need args to be passed in a format compatible
        # with data_specs
        nested_args = mapping.nest(theano_args)
        fixed_var_descr = self.cost.get_fixed_var_descr(model, nested_args)
        self.on_load_batch = fixed_var_descr.on_load_batch

        cost_value = self.cost.expr(model, nested_args,
                                    ** fixed_var_descr.fixed_vars)

        if cost_value is not None and cost_value.name is None:
            # Concatenate the name of all tensors in theano_args !?
            cost_value.name = 'objective'

        # Set up monitor to model the objective value, learning rate,
        # momentum (if applicable), and extra channels defined by
        # the cost
        learning_rate = self.learning_rate
        if self.monitoring_dataset is not None:
            self.monitor.setup(
                    dataset=self.monitoring_dataset,
                    cost=self.cost,
                    batch_size=self.batch_size,
                    num_batches=self.monitoring_batches,
                    extra_costs=self.monitoring_costs,
                    mode=self.monitor_iteration_mode
                    )
            dataset_name = self.monitoring_dataset.keys()[0]
            monitoring_dataset = self.monitoring_dataset[dataset_name]
            #TODO: have Monitor support non-data-dependent channels
            self.monitor.add_channel(name='learning_rate',
                                     ipt=None,
                                     val=learning_rate,
                                     data_specs=(NullSpace(), ''),
                                     dataset=monitoring_dataset)

            if self.learning_rule:
                self.learning_rule.add_channels_to_monitor(
                        self.monitor,
                        monitoring_dataset)

        params = list(model.get_params())
        assert len(params) > 0
        for i, param in enumerate(params):
            if param.name is None:
                param.name = 'sgd_params[%d]' % i

        grads, updates = self.cost.get_gradients(model, nested_args,
                                                 ** fixed_var_descr.fixed_vars)

        for param in grads:
            assert param in params
        for param in params:
            assert param in grads

        for param in grads:
            if grads[param].name is None and cost_value is not None:
                grads[param].name = ('grad(%(costname)s, %(paramname)s)' %
                                     {'costname': cost_value.name,
                                      'paramname': param.name})

        if self.monitor_grads:
            for mon_grad in self.monitor_grads:
                for param in params:
                    if param.name == mon_grad:
                        data_specs = model.get_monitoring_data_specs()
                        norm = T.sqrt(T.sum(T.sqr(grads[param])))
                        self.norm = norm
                        #norm_val = theano.function([
                        self.monitor.add_channel(name="%s_grad_norm" % mon_grad,
                                                 ipt=theano_args,
                                                 val=norm,
                                                 data_specs=data_specs,
                                                 dataset=monitoring_dataset)

        lr_scalers = model.get_lr_scalers()

        for key in lr_scalers:
            if key not in params:
                raise ValueError("Tried to scale the learning rate on " +\
                        str(key)+" which is not an optimization parameter.")

        log.info('Parameter and initial learning rate summary:')
        for param in params:
            param_name = param.name
            if param_name is None:
                param_name = 'anon_param'
            lr = learning_rate.get_value() * lr_scalers.get(param,1.)
            log.info('\t' + param_name + ': ' + str(lr))

        if self.learning_rule:
            updates.update(self.learning_rule.get_updates(
                learning_rate, grads, lr_scalers))
        else:
            # Use standard SGD updates with fixed learning rate.
            updates.update( dict(safe_zip(params, [param - learning_rate * \
                lr_scalers.get(param, 1.) * grads[param]
                                    for param in params])))

        for param in params:
            if updates[param].name is None:
                updates[param].name = 'sgd_update(' + param.name + ')'
        model.censor_updates(updates)
        for param in params:
            update = updates[param]
            if update.name is None:
                update.name = 'censor(sgd_update(' + param.name + '))'
            for update_val in get_debug_values(update):
                if np.any(np.isinf(update_val)):
                    raise ValueError("debug value of %s contains infs" % update.name)
                if np.any(np.isnan(update_val)):
                    raise ValueError("debug value of %s contains nans" % update.name)

        if True:
                import theano
                mode = theano.compile.MonitorMode(post_func=detect_nan)
                self.theano_function_mode = mode

        with log_timing(log, 'Compiling sgd_update'):
            self.sgd_update = function(theano_args,
                                       updates=updates,
                                       name='sgd_update',
                                       on_unused_input='ignore',
                                       mode=self.theano_function_mode)
        self.params = params

    def train(self, dataset):
        if not hasattr(self, 'sgd_update'):
            raise Exception("train called without first calling setup")

        # Make sure none of the parameters have bad values
        for param in self.params:
            value = param.get_value(borrow=True)
            if np.any(np.isnan(value)) or np.any(np.isinf(value)):
                raise Exception("NaN in " + param.name)

        self.first = False
        rng = self.rng
        if not is_stochastic(self.train_iteration_mode):
            rng = None

        data_specs = self.cost.get_data_specs(self.model)

        # The iterator should be built from flat data specs, so it returns
        # flat, non-redundent tuples of data.
        mapping = DataSpecsMapping(data_specs)
        space_tuple = mapping.flatten(data_specs[0], return_tuple=True)
        source_tuple = mapping.flatten(data_specs[1], return_tuple=True)
        if len(space_tuple) == 0:
            # No data will be returned by the iterator, and it is impossible
            # to know the size of the actual batch.
            # It is not decided yet what the right thing to do should be.
            raise NotImplementedError("Unable to train with SGD, because "
                    "the cost does not actually use data from the data set. "
                    "data_specs: %s" % str(data_specs))
        flat_data_specs = (CompositeSpace(space_tuple), source_tuple)

        iterator = dataset.iterator(mode=self.train_iteration_mode,
                batch_size=self.batch_size,
                data_specs=flat_data_specs, return_tuple=True,
                rng = rng, num_batches = self.batches_per_iter)

        on_load_batch = self.on_load_batch
        for batch in iterator:
            for callback in on_load_batch:
                callback(mapping.nest(batch))
            self.sgd_update(*batch)
            # iterator might return a smaller batch if dataset size
            # isn't divisible by batch_size
            # Note: if data_specs[0] is a NullSpace, there is no way to know
            # how many examples would actually have been in the batch,
            # since it was empty, so actual_batch_size would be reported as 0.
            actual_batch_size = flat_data_specs[0].np_batch_size(batch)
            self.monitor.report_batch(actual_batch_size)
            for callback in self.update_callbacks:
                callback(self)

        # Make sure none of the parameters have bad values
        for param in self.params:
            value = param.get_value(borrow=True)
            if np.any(np.isnan(value)) or np.any(np.isinf(value)):
                raise Exception("NaN in " + param.name)

    def continue_learning(self, model):
        if self.termination_criterion is None:
            return True
        else:
            return self.termination_criterion.continue_learning(self.model)

class EpochBasedLRScaler(TrainExtension):
    """
    TODO: write the documentation here.
    """
    def __init__(self, epoch_number, lr_scale, scale_every=False, min_lr=1e-6):
        self.__dict__.update(locals())
        del self.self
        self._scaled = False
        self._count = 0

    def on_monitor(self, model, dataset, algorithm):
        if not self._initialzed:
            self._init_lr = algorithm.learning_rate.get_value()
            if self._init_lr <= self.min_lr:
                raise ValueError("The initial learning rate is smaller than the minimum allowed learning rate.")
            self._initialized = True
        self._pre_lr = algorithm.learning_rate.get_value()
        algorithm.learning_rate.set_value(np.cast[config.floatX](self.current_lr()))
        self._count += 1

    def current_lr(self):
        scaled_lr = self._pre_lr
        if not self._scaled:
            if self._count == self.epoch_number:
                scaled_lr *= self.lr_scale
                self._scaled = True
        elif self.scale_every:
            if self._count >= self.epoch_number:
                scaled_lr *=  self.lr_scale
        return scaled_lr

class MonitorBasedLRScaler(TrainExtension):
    """
    TODO: write the documentation here.
    """
    def __init__(self, valid_obj_bound, lr_scale, channel_name=None,
        min_lr=1e-6, exceeds=False, scale_every=False):
        if channel_name is None:
            raise ValueError("You should give a valid name to the channel to monitor.")
        self.__dict__.update(locals())
        del self.self
        self._scaled = False
        self._count = 0

    def on_monitor(self, model, dataset, algorithm):
        monitor = model.monitor
        # In the case the monitor has only one channel, the channel_name can
        # be omitted and the criterion will examine the only channel
        # available. However, if the monitor has multiple channels, leaving
        # the channel_name unspecified will raise an error.
        if self._channel_name is None:
            if len(monitor.channels) != 1:
                raise ValueError("Only single-channel monitors are supported "
                                 "for channel_name == None")
            self._v = monitor.channels.values()[0].val_record
        else:
            self._v = monitor.channels[self._channel_name].val_record

        if not self._initialized:
            self._init_lr = algorithm.learning_rate.get_value()
            if self._init_lr < self.min_lr:
                raise ValueError("The initial learning rate is smaller than the minimum allowed learning rate.")
            self._initialized = True
        self._count += 1
        self._pre_lr = algorithm.learning_rate.get_value()
        algorithm.learning_rate.set_value( np.cast[config.floatX](self.current_lr()))

    def _scale_lr(self, scaled_lr):
        if not self.exceeds:
            if (self.v[-1] <= self.valid_obj_bound):
                scaled_lr *= self.lr_scale
                self._scaled = True
        else:
            if (self._v[-1] >= self._v[-2]) and (self.v[-1] >=
                self.valid_obj_bound):
                scaled_lr *= self.lr_scale
                self._scaled = True
        return scaled_lr

    def current_lr(self):
        scaled_lr = self._pre_lr
        if not self._scaled:
            scaled_lr = self._scale_lr(scaled_lr)
        elif self.scale_every:
            scaled_lr = self._scale_lr(scaled_lr)
        return scaled_lr

"""
TODO: implement Nesterov momentum. Easiest way to do it is via equivalence
between regular momentum and Nesterov momentum described in this note from
Nicolas Boulanger-Lewandowski:


Yes, I found that the following change of variable simplifies the implementation of Nesterov momentum.
It is in the same form as regular momentum in the sense that both velocity and parameter updates depend
only on the gradient at the current value of the parameters.

In short:

regular momentum:
(1) v_t = mu * v_t-1 - lr * gradient_f(params_t)
(2) params_t = params_t-1 + v_t
(3) params_t = params_t-1 + mu * v_t-1 - lr * gradient_f(params_t-1)

Nesterov momentum:
(4) v_t = mu * v_t-1 - lr * gradient_f(params_t-1 + mu * v_t-1)
(5) params_t = params_t-1 + v_t

alternate formulation for Nesterov momentum:
(6) v_t = mu * v_t-1 - lr * gradient_f(params_t-1)

"""

class MonitorBasedLRAdjuster(TrainExtension):
    """
    A TrainExtension that uses the on_monitor callback to adjust
    the learning rate on each epoch. It pulls out a channel
    from the model's monitor and adjusts the learning rate
    based on what happened to the monitoring channel on the last
    epoch. If the channel is greater than high_trigger times
    its previous value, the learning rate will be scaled by
    shrink_amt (which should be < 1 for this scheme to make
    sense). The idea is that in this case the learning algorithm
    is overshooting the bottom of the objective function.

    If the objective is less than high_trigger but
    greater than low_trigger times its previous value, the
    learning rate will be scaled by grow_amt (which should be > 1
    for this scheme to make sense). The idea is that the learning
    algorithm is making progress but at too slow of a rate.
    """

    def __init__(self, high_trigger=1., shrink_amt=.99,
                 low_trigger=.99, grow_amt=1.01,
                 min_lr = 1e-7, max_lr = 1.,
                 dataset_name=None, channel_name=None):
        self.high_trigger = high_trigger
        self.shrink_amt = shrink_amt
        self.low_trigger = low_trigger
        self.grow_amt = grow_amt
        self.min_lr = min_lr
        self.max_lr = max_lr
        if channel_name is not None:
            self.channel_name = channel_name
        else:
            if dataset_name is not None:
                self.channel_name = dataset_name + '_objective'
            else:
                self.channel_name = 'objective'

    def on_monitor(self, model, dataset, algorithm):
        # TODO: more sophisticated error checking here.
        model = algorithm.model
        lr = algorithm.learning_rate
        current_learning_rate = lr.get_value()
        assert hasattr(model, 'monitor'), ("no monitor associated with " +
                                           str(model))
        monitor = model.monitor

        try:
            v = monitor.channels[self.channel_name].val_record
        except KeyError:
            raise KeyError('There is no monitoring channel named ' + \
                    self.channel_name + '. You probably need to specify '
                    'dataset_name in the MonitorBasedLRAdjuster constructor.')

        if len(v) < 1:
            if monitor.dataset is None:
                assert len(v) == 0
                raise ValueError("""You're trying to use a monitor-based learning
                        adjustor but the monitor has no entries because you didn't
                        specify a monitoring dataset""")

            raise ValueError("""For some reason there are no monitor entries,
                    yet the MonitorBasedLRAdjuster has been called. This should NEVER happen.
                    The Train object should call the monitor once on initialization, then
                    call the callbacks.
                    It seems you are either calling the callback manually rather than as part of
                    a training algorithm, or there is a problem with the Train object.""")
        if len(v) == 1:
            #only the initial monitoring has happened
            #no learning has happened, so we can't adjust the learning rate yet
            #just do nothing
            return

        rval = current_learning_rate

        if v[-1] > self.high_trigger * v[-2]:
            rval *= self.shrink_amt
            log.info("shrinking learning rate to %f" % rval)
        elif v[-1] > self.low_trigger * v[-2]:
            rval *= self.grow_amt
            log.info("growing learning rate to %f" % rval)

        rval = max(self.min_lr, rval)
        rval = min(self.max_lr, rval)

        lr.set_value(np.cast[lr.dtype](rval))


class PatienceBasedTermCrit(object):
    """
    A monitor-based termination criterion using a geometrically increasing
    ammount of patience. If the selected channel has decreased by a certain
    proportion when comparing to the lowest value seen yet, the patience is
    set to a factor of the number of examples seen, which by default
    (patience_increase=2.) ensures the model has seen as many examples as the
    number of examples that lead to the lowest value before concluding a local
    optima has been reached.

    Note: Technically, the patience corresponds to a number of epochs to be
    independent of the size of the dataset, so be aware of that when choosing
    initial_patience.
    """
    def __init__(self, prop_decrease, initial_patience,
                 patience_increase=2., channel_name=None):
        """
        Initialize a patience-based termination criterion.

        Parameters
        ----------
        prop_decrease : float
            The factor X in the (1 - X) * best_value threshold
        initial_patience : int
            Minimal number of epochs the model has to run before it can stop
        patience_increase : float, optional
            The factor X in the patience = X * n_iter update.
        channel_name : string, optional
            Name of the channel to examine. If None and the monitor
            has only one channel, this channel will be used; otherwise, an
            error will be raised.
        """
        self._channel_name = channel_name
        self.prop_decrease = prop_decrease
        self.patience = initial_patience
        self.best_value = np.inf
        self.patience_increase = patience_increase

    def __call__(self, model):
        """
        Returns True or False depending on whether the optimization should
        stop or not. The optimization should stop if it has run for a number
        of epochs superior to the patience without any improvement.

        Parameters
        ----------
        model : Model
            The model used in the experiment and from which the monitor used
            in the termination criterion will be extracted.

        Returns
        -------
        boolean
            True or False, indicating if the optimization should stop or not.
        """
        monitor = model.monitor
        # In the case the monitor has only one channel, the channel_name can
        # be omitted and the criterion will examine the only channel
        # available. However, if the monitor has multiple channels, leaving
        # the channel_name unspecified will raise an error.
        if self._channel_name is None:
            if len(monitor.channels) != 1:
                raise ValueError("Only single-channel monitors are supported "
                                 "for channel_name == None")
            v = monitor.channels.values()[0].val_record
        else:
            v = monitor.channels[self._channel_name].val_record
        # If the channel value decrease is higher than the threshold, we
        # update the best value to this value and we update the patience.
        if v[-1] < self.best_value * (1. - self.prop_decrease):
            # Using the max between actual patience and updated patience
            # ensures that the model will run for at least the initial
            # patience and that it would behave correctly if the user
            # chooses a dumb value (i.e. less than 1)
            self.patience = max(self.patience, len(v) * self.patience_increase)
            self.best_value = v[-1]

        return len(v) < self.patience


class AnnealedLearningRate(object):
    """
    This is a callback for the SGD algorithm rather than the Train object.
    This anneals the learning rate to decrease as 1/t where t is the number
    of gradient descent updates done so far. Use OneOverEpoch as Train object
    callback if you would prefer 1/t where t is epochs.
    """
    def __init__(self, anneal_start):
        self._initialized = False
        self._count = 0
        self._anneal_start = anneal_start

    def __call__(self, algorithm):
        if not self._initialized:
            self._base = algorithm.learning_rate.get_value()
        self._count += 1
        algorithm.learning_rate.set_value(self.current_learning_rate())

    def current_learning_rate(self):
        return self._base * min(1, self._anneal_start / self._count)

class ExponentialDecay(object):
    """
    This is a callback for the SGD algorithm rather than the Train object.
    This anneals the learning rate by dividing by decay_factor after each
    gradient descent step. It will not shrink the learning rate beyond
    min_lr.
    """
    def __init__(self, decay_factor, min_lr):
        if isinstance(decay_factor, str):
            decay_factor = float(decay_factor)
        if isinstance(min_lr, str):
            min_lr = float(min_lr)
        assert isinstance(decay_factor, float)
        assert isinstance(min_lr, float)
        self.__dict__.update(locals())
        del self.self
        self._count = 0

    def __call__(self, algorithm):
        if self._count == 0:
            self._base_lr = algorithm.learning_rate.get_value()
        self._count += 1
        cur_lr = self._base_lr / (self.decay_factor ** self._count)
        new_lr = max(cur_lr, self.min_lr)
        new_lr = np.cast[config.floatX](new_lr)
        algorithm.learning_rate.set_value(new_lr)

class LinearDecay(object):
    """
    This is a callback for the SGD algorithm rather than the Train object.
    This anneals the learning rate to decay_factor times of the initial value
    during time start till saturate.
    """
    def __init__(self, start, saturate, decay_factor):
        if isinstance(decay_factor, str):
            decay_factor = float(decay_factor)
        if isinstance(start, str):
            start = float(start)
        if isinstance(saturate, str):
            saturate = float(saturate)
        assert isinstance(decay_factor, float)
        assert isinstance(start, (py_integer_types, py_float_types))
        assert isinstance(saturate, (py_integer_types, py_float_types))
        assert saturate > start
        assert start > 0
        self.__dict__.update(locals())
        del self.self
        self._count = 0

    def __call__(self, algorithm):
        if self._count == 0:
            self._base_lr = algorithm.learning_rate.get_value()
            self.step = (self._base_lr - self._base_lr * self.decay_factor) /\
                    (self.saturate - self.start + 1)
        self._count += 1
        if self._count >= self.start:
            if self._count < self.saturate:
                new_lr = self._base_lr - self._step * (self._count - self.start + 1)
            else:
                new_lr = self._base_lr * self.decay_factor
        else:
            new_lr = self._base_lr
        assert new_lr > 0
        new_lr = np.cast[config.floatX](new_lr)
        algorithm.learning_rate.set_value(new_lr)


def MomentumAdjustor(final_momentum, start, saturate):
    warnings.warn("sgd.MomentumAdjustor interface is deprecated and will "
    "become officially unsuported as of May 9, 2014. Please use "
    "`learning_rule.MomentumAdjustor` instead.")
    return LRMomentumAdjustor(final_momentum, start, saturate)


class OneOverEpoch(TrainExtension):
    """
    Scales the learning rate like one over # epochs
    """
    def __init__(self, start, half_life = None, min_lr = 1e-6):
        """
            start: the epoch on which to start shrinking the learning rate
            half_life: how many epochs after start it will take for the learning rate
                       to lose half its value for the first time
                        (to lose the next half of its value will take twice
                        as long)
            min_lr: the minimum value the learning rate can take on
        """
        self.__dict__.update(locals())
        del self.self
        self._initialized = False
        self._count = 0
        assert start >= 0
        if half_life is None:
            self.half_life = start + 1
        else:
            assert half_life > 0

    def on_monitor(self, model, dataset, algorithm):
        if not self._initialized:
            self._init_lr = algorithm.learning_rate.get_value()
            if self._init_lr < self.min_lr:
                raise ValueError("The initial learning rate is smaller than the minimum allowed learning rate.")
            self._initialized = True
        self._count += 1
        algorithm.learning_rate.set_value( np.cast[config.floatX](self.current_lr()))

    def current_lr(self):
        if self._count < self.start:
            scale = 1
        else:
            scale = float(self.half_life) / float(self._count - self.start +self.half_life)
        lr = self._init_lr * scale
        clipped = max(self.min_lr, lr)
        return clipped

class LinearDecayOverEpoch(TrainExtension):
    """
    Scales the learning rate linearly on each epochs
    """
    def __init__(self, start, saturate, decay_factor):
        """
            start: the epoch on which to start shrinking the learning rate
            saturate: the epoch to saturate the shrinkage
            decay_factor: the final value would be initial learning rate times
                decay_factor
        """
        self.__dict__.update(locals())
        del self.self
        self._initialized = False
        self._count = 0
        assert isinstance(decay_factor, float)
        assert isinstance(start, (py_integer_types, py_float_types))
        assert isinstance(saturate, (py_integer_types, py_float_types))
        assert saturate > start
        assert start >= 0
        assert saturate >= start

    def on_monitor(self, model, dataset, algorithm):
        if not self._initialized:
            self._init_lr = algorithm.learning_rate.get_value()
            self._step = (self._init_lr - self._init_lr * self.decay_factor) /\
                    (self.saturate - self.start + 1)
            self._initialized = True
        self._count += 1
        algorithm.learning_rate.set_value( np.cast[config.floatX](self.current_lr()))

    def current_lr(self):
        if self._count >= self.start:
            if self._count < self.saturate:
                new_lr = self._init_lr - self._step * (self._count - self.start + 1)
            else:
                new_lr = self._init_lr * self.decay_factor
        else:
            new_lr = self._init_lr
        assert new_lr > 0
        return new_lr

class _PolyakWorker(object):
    """
    Only to be used by the PolyakAveraging TrainingCallback below.
    Do not use directly.
    """
    def __init__(self, model):
        avg_updates = OrderedDict()
        t = sharedX(1.)
        self.param_to_mean = OrderedDict()
        for param in model.get_params():
            mean = sharedX(param.get_value())
            assert type(mean) == type(param)
            self.param_to_mean[param] = mean
            avg_updates[mean] = mean - (mean - param) / t
            avg_updates[t] = t + 1.
        self.avg = function([], updates = avg_updates)

    def __call__(self, algorithm):
        self.avg()

class PolyakAveraging(TrainExtension):
    """
    See "A Tutorial on Stochastic Approximation Algorithms
        for Training Restricted Boltzmann Machines and
        Deep Belief Nets" by Kevin Swersky et al

    Notes: this is usually used with a fixed, rather than
        annealed learning rate.
        It may be used in conjunction with momentum.

    This functionality is still a work in progress. Currently,
    your model needs to implement "add_polyak_channels" to
    use it.

    The problem is that Polyak averaging shouldn't modify
    the model parameters. It should keep a second copy
    that it averages in the background. This second copy
    doesn't get to come back in and affect the learning process
    though.

    (IG tried having the second copy get pushed back into
    the model once per epoch, but this turned out to be
    harmful, at least in limited tests)

    So we need a cleaner interface for monitoring the
    averaged copy of the parameters, and we need to make
    sure the saved model at the end uses the averaged
    parameters, not the parameters used for computing
    the gradients during training.

    TOOD: make use of the new on_save callback instead
        of duplicating Train's save_freq flag
    """

    def __init__(self, start, save_path = None, save_freq = 1):
        """
            start: the epoch after which to start averaging
            (0 = start averaging immediately)
        """
        self.__dict__.update(locals())
        del self.self
        self._count = 0
        assert isinstance(start, py_integer_types)
        assert start >= 0

    def on_monitor(self, model, dataset, algorithm):
        if self._count == self.start:
            self._worker = _PolyakWorker(model)
            algorithm.update_callbacks.append(self._worker)
            #HACK
            try:
                model.add_polyak_channels(self._worker.param_to_mean, algorithm.monitoring_dataset)
            except AttributeError:
                pass
        elif self.save_path is not None and self._count > self.start and self._count % self.save_freq == 0:
            saved_params = OrderedDict()
            for param in model.get_params():
                saved_params[param] = param.get_value()
                param.set_value(self._worker.param_to_mean[param].get_value())
            serial.save(self.save_path, model)
            for param in model.get_params():
                param.set_value(saved_params[param])
        self._count += 1


class ExhaustiveSGD(SGD): # deprecated!

    def __init__(self, * args, ** kwargs):

        warnings.warn("""
        ExhaustiveSGD is deprecated. It has been renamed to SGD.
        Modify your code to use SGD. ExhaustiveSGD may be removed
        after August 1, 2013.
        """)

        super(ExhaustiveSGD,self).__init__(*args, ** kwargs)

# These classes were moved to the new submodule, but I import
# a reference to them here to avoid breaking the old interface.
from pylearn2.termination_criteria import EpochCounter as _EpochCounter

def EpochCounter(**kwargs):
    warnings.warn("training_algorithms.sgd.EpochCounter has been moved to "
            "termination_criteria.EpochCounter. This link may be removed on "
            "or after October 3, 2013.", stacklevel=2)
    return _EpochCounter(**kwargs)

from pylearn2.termination_criteria import And as _DisjunctionCriterion

def DisjunctionCriterion(**kwargs):
    warnings.warn("training_algorithms.sgd.DisjunctionCriterion has been moved to "
            "termination_criteria.And. This link may be removed on "
            "or after October 3, 2013.")
    return _DisjunctionCriterion(**kwargs)

from pylearn2.termination_criteria import Or as _ConjunctionCriterion

def ConjuctionCriterion(**kwargs):
    warnings.warn("training_algorithms.sgd.ConjunctionCriterion has been moved to "
            "termination_criteria.Or. This link may be removed on "
            "or after October 3, 2013.")
    return _ConjunctionCriterion(**kwargs)
