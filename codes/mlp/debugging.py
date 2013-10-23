import numpy

import theano

def detect_nan(i, node, fn):
    for output in fn.outputs:
        if numpy.isnan(output[0]).any() or numpy.isinf(output[0]).any():
            print '*** NaN detected ***'
            theano.printing.debugprint(node)
            print 'Inputs : %s' % [input[0] for input in fn.inputs]
            print 'Outputs: %s' % [output[0] for output in fn.outputs]
            import ipdb; ipdb.set_trace()

#x = theano.tensor.dscalar('x')
#f = theano.function([x], [theano.tensor.log(x) * x],
#                    mode=theano.compile.MonitorMode(
#                        post_func=detect_nan))
#f(0)  # log(0) * 0 = -inf * 0 = NaN

# The code above will print:
#   *** NaN detected ***
#   Elemwise{Composite{[mul(log(i0), i0)]}} [@A] ''
#    |x [@B]
#   Inputs : [array(0.0)]
#   Outputs: [array(nan)]
