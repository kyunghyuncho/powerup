from collections import OrderedDict
import numpy as np
import warnings

import theano
from theano import config
from theano.gof.op import get_debug_values
from theano.sandbox import cuda
from theano import tensor as T

from pylearn2.linear.matrixmul import MatrixMul
from pylearn2.models.mlp import Layer
from pylearn2.space import Conv2DSpace
from pylearn2.space import VectorSpace
from pylearn2.utils import py_integer_types
from pylearn2.utils import sharedX

class LpUnitCluster(Layer):

    def __init__(self,
                 layer_name,
                 num_units,
                 num_pieces,
                 batch_size,
                 pool_stride = None,
                 randomize_pools = False,
                 p_sampling_mode = "normal",
                 irange = None,
                 sparse_init = None,
                 p_mean = 2.0,
                 p_std = 0.005,
                 normalize = False,
                 power_prod=False,
                 sparse_stdev = 1.,
                 include_prob = 1.0,
                 upper_bound = None,
                 init_bias = 0.,
                 relu = False,
                 centered_bias = False,
                 power_activ = "softplus",
                 uniform_p_range = (1.2, 8.0),
                 add_noise = False,
                 post_bias = False,
                 p_lr_scale = None,
                 W_lr_scale = None,
                 b_lr_scale = None,
                 max_col_norm = None,
                 max_row_norm = None,
                 mask_weights = None,
                 min_zero = False):
        """
            layer_name: A name for this layer that will be prepended to
                        monitoring channels related to this layer.
            num_units: The number of maxout units to use in this layer.
            num_pieces: The number of linear pieces to use in each maxout
                        unit.
            pool_stride: The distance between the start of each max pooling
                        region. Defaults to num_pieces, which makes the
                        pooling regions disjoint. If set to a smaller number,
                        can do overlapping pools.
            randomize_pools: Does max pooling over randomized subsets of
                        the linear responses, rather than over sequential
                        subsets.
            irange: if specified, initializes each weight randomly in
                U(-irange, irange)
            sparse_init: if specified, irange must not be specified.
                        This is an integer specifying how many weights to make
                        non-zero. All non-zero weights will be initialized
                        randomly in N(0, sparse_stdev^2)
            include_prob: probability of including a weight element in the set
               of weights initialized to U(-irange, irange). If not included
               a weight is initialized to 0. This defaults to 1.
            init_bias: All biases are initialized to this number
            W_lr_scale: The learning rate on the weights for this layer is
                multiplied by this scaling factor
            b_lr_scale: The learning rate on the biases for this layer is
                multiplied by this scaling factor
            max_col_norm: The norm of each column of the weight matrix is
                constrained to have at most this norm. If unspecified, no
                constraint. Constraint is enforced by re-projection (if
                necessary) at the end of each update.
            max_row_norm: Like max_col_norm, but applied to the rows.
            mask_weights: A binary matrix multiplied by the weights after each
                         update, allowing you to restrict their connectivity.
            min_zero: If true, includes a zero in the set we take a max over
                    for each maxout unit. This is equivalent to pooling over
                    rectified linear units.
        """

        assert p_sampling_mode in ["uniform", "normal"]
        assert power_activ in ["rect", "exp", "softplus", "sqr"]
        assert type(uniform_p_range) == tuple

        detector_layer_dim = num_units * num_pieces
        pool_size = num_pieces

        self.normalize = normalize
        self.uniform_p_range = uniform_p_range

        self.upper_bound = upper_bound
        self.power_prod = power_prod
        self.power_activ = power_activ

        self.centered_bias = centered_bias
        self.p_sampling_mode = p_sampling_mode

        if pool_stride is None:
            pool_stride = pool_size

        self.__dict__.update(locals())

        del self.self
        self.rng = np.random.RandomState(12435)

        if self.centered_bias:
            self.c = sharedX(np.zeros((self.detector_layer_dim,)), name=layer_name + "_c")
            #self.c = sharedX(self.rng.uniform(low=-0.01, high=0.01, size=(self.detector_layer_dim,)), name=layer_name + "_c")

        if not self.post_bias:
            self.b = sharedX( np.zeros((self.detector_layer_dim,)) + init_bias, name = layer_name + '_b')
        else:
            self.b = sharedX( np.zeros((self.num_units,)) + init_bias, name = layer_name + '_b')

        if self.power_activ == "softplus":
            if self.p_sampling_mode == "uniform":
                self.p = sharedX(self.get_uniform_p_vals())
            else:
                self.p = sharedX(self.get_log_p(mean=p_mean, std=p_std))
        else:
            if self.p_sampling_mode == "uniform":
                self.p = sharedX(self.get_uniform_p_vals())
            else:
                self.p = sharedX(self.get_log_p(mean=p_mean, std=p_std))

        if max_row_norm is not None:
            raise NotImplementedError()

    def get_p_vals(self, mean=None, std=None):
        rng = np.random.RandomState(12435)
        p_vals = abs(rng.normal(loc=mean, scale=std, size=(self.num_units,)))
        return p_vals

    def get_uniform_p_vals(self, min=1.5, max=9):
        """
            Sample the values uniformly such that the initial value of
            softplus(.) + 1 is between min and max.
        """
        rng = np.random.RandomState(12435)
        if self.power_activ == "softplus":
            p_vals = np.log(np.exp(rng.uniform(low=min, high=max, size=(self.num_units,))-1)-1)
        else:
            p_vals = np.sqrt(rng.uniform(low=min, high=max, size=(self.num_units,))-1)
        return p_vals

    def get_log_p(self, mean=None, std=None):
        rng = np.random.RandomState(12435)
        assert mean >= 1.0, "Mean should be greater than 1."
        if self.power_activ == "softplus":
            p_vals = np.log(rng.normal(loc=np.exp(mean-1), scale=std, size=(self.num_units,)) - 1)
        else:
            p_vals = np.sqrt(rng.normal(loc=mean, scale=std, size=(self.num_units,))-1)
        #p_vals = np.log(np.exp(rng.normal(loc=mean, scale=std, size=(self.num_units,))-1) - 1)
        return p_vals

    def get_lr_scalers(self):
        if not hasattr(self, 'W_lr_scale'):
            self.W_lr_scale = None

        #if not hasattr(self, 'b_lr_scale'):
        #    self.b_lr_scale = None

        if not hasattr(self, 'p_lr_scale'):
            self.p_lr_scale = None

        rval = OrderedDict()

        #if self.W_lr_scale is not None:
        #    W, = self.transformer.get_params()
        #    rval[W] = self.W_lr_scale

        #if self.b_lr_scale is not None:
        #    rval[self.b] = self.b_lr_scale

        if self.p_lr_scale is not None:
            rval[self.p] = self.p_lr_scale

        return rval

    def set_input_space(self, space):
        """
        Note: this resets parameters!
        """
        self.input_space = space

        if isinstance(space, VectorSpace):
            self.requires_reformat = False
            self.input_dim = space.dim
        else:
            self.requires_reformat = True
            self.input_dim = space.get_total_dimension()
            self.desired_space = VectorSpace(self.input_dim)

        self.p.name = self.layer_name + "_p"

        if not ((self.detector_layer_dim - self.pool_size) % self.pool_stride == 0):
            if self.pool_stride == self.pool_size:
                raise ValueError("detector_layer_dim = %d, pool_size = %d. Should be divisible but remainder is %d" %
                             (self.detector_layer_dim, self.pool_size, self.detector_layer_dim % self.pool_size))
            raise ValueError()

        self.h_space = VectorSpace(self.detector_layer_dim)
        self.pool_layer_dim = (self.detector_layer_dim - self.pool_size)/ self.pool_stride + 1
        self.output_space = VectorSpace(self.pool_layer_dim)

        rng = self.mlp.rng
        if self.irange is not None:
            assert self.sparse_init is None
            W = rng.uniform(-self.irange,
                            self.irange,
                            (self.input_dim, self.detector_layer_dim)) * \
                (rng.uniform(0.,1., (self.input_dim, self.detector_layer_dim))
                 < self.include_prob)
        else:
            assert self.sparse_init is not None
            W = np.zeros((self.input_dim, self.detector_layer_dim))
            def mask_rejects(idx, i):
                if self.mask_weights is None:
                    return False
                return self.mask_weights[idx, i] == 0.
            for i in xrange(self.detector_layer_dim):
                assert self.sparse_init <= self.input_dim
                for j in xrange(self.sparse_init):
                    idx = rng.randint(0, self.input_dim)
                    while W[idx, i] != 0 or mask_rejects(idx, i):
                        idx = rng.randint(0, self.input_dim)
                    W[idx, i] = rng.randn()
            W *= self.sparse_stdev

        W = sharedX(W)
        W.name = self.layer_name + '_W'

        self.transformer = MatrixMul(W)

        W ,= self.transformer.get_params()
        assert W.name is not None

        if not hasattr(self, 'randomize_pools'):
            self.randomize_pools = False

        if self.randomize_pools:
            permute = np.zeros((self.detector_layer_dim, self.detector_layer_dim))
            for j in xrange(self.detector_layer_dim):
                i = rng.randint(self.detector_layer_dim)
                permute[i,j] = 1
            self.permute = sharedX(permute)

        if self.mask_weights is not None:
            expected_shape =  (self.input_dim, self.detector_layer_dim)
            if expected_shape != self.mask_weights.shape:
                raise ValueError("Expected mask with shape "+str(expected_shape)+" but got "+str(self.mask_weights.shape))
            self.mask = sharedX(self.mask_weights)

    def censor_updates(self, updates):
        # Patch old pickle files
        if not hasattr(self, 'mask_weights'):
            self.mask_weights = None

        if self.mask_weights is not None:
            W ,= self.transformer.get_params()
            if W in updates:
                updates[W] = updates[W] * self.mask

        if self.max_col_norm is not None:
            assert self.max_row_norm is None
            W ,= self.transformer.get_params()
            if W in updates:
                updated_W = updates[W]
                col_norms = T.sqrt(T.sum(T.sqr(updated_W), axis=0))
                desired_norms = T.clip(col_norms, 0, self.max_col_norm)
                updates[W] = updated_W * (desired_norms / (1e-7 + col_norms))

    def get_params(self):
        assert self.b.name is not None
        assert self.p.name is not None
        W ,= self.transformer.get_params()
        assert W.name is not None
        rval = self.transformer.get_params()
        assert not isinstance(rval, set)
        rval = list(rval)
        #assert self.b not in rval
        #rval.append(self.b)
        assert self.p not in rval
        rval.append(self.p)

        if self.centered_bias:
            assert self.c not in rval
            rval.append(self.c)

        return rval

    def get_weight_decay(self, coeff):
        if isinstance(coeff, str):
            coeff = float(coeff)
        assert isinstance(coeff, float) or hasattr(coeff, 'dtype')
        W ,= self.transformer.get_params()
        return coeff * T.sqr(W).sum()

    def get_p_decay(self, coeff, a):
        if isinstance(coeff, str):
            coeff = float(coeff)
        assert isinstance(coeff, float) or hasattr(coeff, 'dtype')
        return coeff * T.sqr(self.p-a).sum()

    def get_p_mean_decay(self, coeff, a):
        if isinstance(coeff, str):
            coeff = float(coeff)
        assert isinstance(coeff, float) or hasattr(coeff, 'dtype')
        return coeff * T.sqr(T.mean(self.p)-a).sum()


    def get_l1_weight_decay(self, coeff):
        if isinstance(coeff, str):
            coeff = float(coeff)
        assert isinstance(coeff, float) or hasattr(coeff, 'dtype')
        W ,= self.transformer.get_params()
        return coeff * abs(W).sum()

    def get_weights(self):
        if self.requires_reformat:
            # This is not really an unimplemented case.
            # We actually don't know how to format the weights
            # in design space. We got the data in topo space
            # and we don't have access to the dataset
            raise NotImplementedError()

        W ,= self.transformer.get_params()
        W = W.get_value()

        if not hasattr(self, 'randomize_pools'):
            self.randomize_pools = False

        if self.randomize_pools:
            warnings.warn("randomize_pools makes get_weights multiply by the permutation matrix. "
                          "If you call set_weights(W) and then call get_weights(), the return value will "
                          "WP not W.")

            P = self.permute.get_value()
            return np.dot(W,P)

        return W

    def set_power(self, p_val):
        self.p.set_value(p_val)

    def set_weights(self, weights):
        W, = self.transformer.get_params()
        W.set_value(weights)

    def set_biases(self, biases):
        self.b.set_value(biases)

    def get_biases(self):
        return self.b.get_value()

    def get_power(self):
        return self.p.get_value()

    def get_weights_format(self):
        return ('v', 'h')

    def get_weights_view_shape(self):
        total = self.detector_layer_dim
        cols = self.pool_size
        if cols == 1:
            # Let the PatchViewer decide how to arrange the units
            # when they're not pooled
            raise NotImplementedError()
        # When they are pooled, make each pooling unit have one row
        rows = total // cols
        if rows * cols < total:
            rows = rows + 1
        return rows, cols


    def get_weights_topo(self):

        if not isinstance(self.input_space, Conv2DSpace):
            raise NotImplementedError()

        # There was an implementation of this, but it was broken
        raise NotImplementedError()

    def get_monitoring_channels(self):

        W ,= self.transformer.get_params()

        assert W.ndim == 2

        sq_W = T.sqr(W)

        row_norms = T.sqrt(sq_W.sum(axis=1))
        col_norms = T.sqrt(sq_W.sum(axis=0))

        powers = self.p

        monitor_dict = OrderedDict([
                            ('power_min', powers.min()),
                            ('power_mean', powers.mean()),
                            ('power_max', powers.max()),
                            ('power_std', powers.std()),
                            ('b_min', self.b.min()),
                            ('b_mean', self.b.mean()),
                            ('b_max', self.b.max()),
                            ('row_norms_min'  , row_norms.min()),
                            ('row_norms_mean' , row_norms.mean()),
                            ('row_norms_max'  , row_norms.max()),
                            ('col_norms_min'  , col_norms.min()),
                            ('col_norms_mean' , col_norms.mean()),
                            ('col_norms_max'  , col_norms.max()),
                            ])

        if self.centered_bias:
            monitor_dict["c_min"] = self.c.min()
            monitor_dict["c_mean"] = self.c.mean()
            monitor_dict["c_max"] = self.c.max()
            monitor_dict["c_std"] = self.c.std()

        return monitor_dict

    def get_monitoring_channels_from_state(self, state):

        P = state

        rval = OrderedDict()

        if self.pool_size == 1:
            vars_and_prefixes = [ (P,'') ]
        else:
            vars_and_prefixes = [ (P, 'p_') ]

        for var, prefix in vars_and_prefixes:
            v_max = var.max(axis=0)
            v_min = var.min(axis=0)
            v_mean = var.mean(axis=0)
            v_range = v_max - v_min
            # max_x.mean_u is "the mean over *u*nits of the max over e*x*amples"
            # The x and u are included in the name because otherwise its hard
            # to remember which axis is which when reading the monitor
            # I use inner.outer rather than outer_of_inner or something like that
            # because I want mean_x.* to appear next to each other in the alphabetical
            # list, as these are commonly plotted together
            for key, val in [
                             ('max_x.max_u', v_max.max()),
                             ('max_x.mean_u', v_max.mean()),
                             ('max_x.min_u', v_max.min()),
                             ('min_x.max_u', v_min.max()),
                             ('min_x.mean_u', v_min.mean()),
                             ('min_x.min_u', v_min.min()),
                             ('range_x.max_u', v_range.max()),
                             ('range_x.mean_u', v_range.mean()),
                             ('range_x.min_u', v_range.min()),
                             ('mean_x.max_u', v_mean.max()),
                             ('mean_x.mean_u', v_mean.mean()),
                             ('mean_x.min_u', v_mean.min())
                             ]:
                rval[prefix+key] = val
        return rval

    def get_power_activ(self, power_in):
        if self.power_activ == "exp":
            pT = T.exp(power_in) + 1
        elif self.power_activ == "rect":
            pT = T.maximum(power_in, 1)
        elif self.power_activ == "softplus":
            pT = T.nnet.softplus(power_in) + 1
        elif self.power_activ == "sqr":
            pT = T.sqr(power_in) + 1
        else:
            pT = abs(power_in) + 1
        return pT

    def fprop(self, state_below):
        #Implements (\sum_i^T 1/T |W_i x|^{p_j} )^(1/p_j)
        self.input_space.validate(state_below)
        epsilon = 1e-11

        if self.requires_reformat:
            if not isinstance(state_below, tuple):
                for sb in get_debug_values(state_below):
                    if sb.shape[0] != self.dbm.batch_size:
                        raise ValueError("self.powerup.batch_size is %d but got shape of %d" % (self.mlp.batch_size, sb.shape[0]))
                    assert reduce(lambda x,y: x * y, sb.shape[1:]) == self.input_dim

            state_below = self.input_space.format_as(state_below, self.desired_space)

        if not self.post_bias:
            z = state_below
            #+ self.b
        else:
            z = state_below

        if not hasattr(self, 'randomize_pools'):
            self.randomize_pools = False

        if not hasattr(self, 'pool_stride'):
            self.pool_stride = self.pool_size

        if self.randomize_pools:
            z = T.dot(z, self.permute)

        if not hasattr(self, 'min_zero'):
            self.min_zero = False

        z = z.dimshuffle('x', 0, 1) #z.reshape((1, self.batch_size, self.pool_size))
        #z = z.dimshuffle(1, 0, 2)#z.reshape(self.batch_size, self.pool_size)
        z_pools = T.repeat(z, repeats=self.num_units, axis=0)
        z_pools = z_pools.dimshuffle(1, 0, 2)#z.reshape(self.batch_size, self.pool_size)

        #Reshape the presynaptic activation to a 3D tensor. Such that the first
        #dimension is the batch size, second dimension corresponds to number of
        #hidden units and the third dimension is for the size of the pool.
        #z_pools = z.reshape((self.num_units, self.batch_size, self.pool_size))
        #z_pools = z_pools.dimshuffle(1, 0, 2)

        #Center the pools
        if self.centered_bias:
            c = self.c.reshape((self.num_units, self.pool_size))
            c = c.dimshuffle('x', 0, 1)
            z_pools = z_pools - c

        #Dimshuffle the p_j for |W_i x|^{p_j}
        power_in = self.p.dimshuffle('x', 0, 'x')
        p_j = self.get_power_activ(power_in)

        if self.relu:
            z_pools = T.maximum(z_pools, 0)
        else:
            z_pools = abs(z_pools)
            #For numerical stability,
            z_pools = T.maximum(z_pools, epsilon)

        z_pools = z_pools**p_j

        if self.normalize:
            z_summed_pools = (1. / self.pool_size) * T.sum(z_pools, axis=2)
        else:
            z_summed_pools = T.sum(z_pools, axis=2)

        #Stabilization for the backprop
        z_summed_pools = T.maximum(z_summed_pools, epsilon)

        #Dimshuffle the p_j for 1/p_j of
        #(\sum_i^T 1/T |W_i x|^{p_j} )^(1/p_j)
        power_in = self.p.dimshuffle('x', 0)
        p_j = self.get_power_activ(power_in)

        z_summed_pools = z_summed_pools**(1./p_j)

        if self.upper_bound is not None:
            z_summed_pools = T.maximum(z_summed_pools, self.upper_bound)

        if self.power_prod:
            a = power_in * z_summed_pools
        else:
            a = z_summed_pools

        #if self.post_bias:
        #    a = a + self.b

        return a

    def stddev_bias(self, x, eps=1e-9, axis=0):
        mu = T.mean(x + eps, axis=axis)
        mu.name = "std_mean"
        var = T.mean((x - mu)**2 + eps)
        var.name = "std_variance"
        stddev = T.sqrt(var)
        return stddev

    def cost_from_cost_matrix(self, cost_matrix):
        return cost_matrix.sum(axis=1).mean()

    def cost_matrix(self, Y, Y_hat):
        return T.sqr(Y - Y_hat)

