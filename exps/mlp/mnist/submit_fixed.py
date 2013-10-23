import os
import argparse, fnmatch
import numpy
from jobman import DD, flatten, api0, sql
from powconvnet.train_pow import experiment
from pylearn2.utils.string_utils import preprocess

def pento(n_trials):
    ri = numpy.random.random_integers

    state = DD()
    with open('mnist_powerup_temp.yaml') as ymtmp:
        state.yaml_string = ymtmp.read()

    state.powerup_nunits = 240
    state.powerup_npieces = 5
    state.W_lr_scale = 0.04
    state.p_lr_scale = 0.01
    state.lr_rate = 0.1
    state.l2_pen = 1e-5
    state.l2_pen2 = 0.0000
    state.init_mom = 0.5
    state.final_mom = 0.5
    state.decay_factor = 0.5
    state.max_col_norm = 1.9365
    state.max_col_norm2 = 1.8365
    state.batch_size = 128

    state.save_path = './'

    n_pieces = [2, 3, 4, 5, 6, 8, 10, 12, 14, 16]
    n_units = [200, 240, 280, 320, 360, 420, 480]
    batch_sizes = [128, 256, 512]

    learning_rates = numpy.logspace(numpy.log10(0.001), numpy.log10(1.0), 30)
    learning_rate_scalers = numpy.logspace(numpy.log10(0.01), numpy.log10(1), 30)
    l2_pen = numpy.logspace(numpy.log10(1e-6), numpy.log10(8*1e-3), 100)
    max_col_norms = [1.7365, 1.8365, 1.9365, 2.1365, 2.2365, 2.4365]

    ind = 0
    TABLE_NAME = "powerup_mnist_1layer_fixed"
    db = api0.open_db('postgresql://gulcehrc@opter.iro.umontreal.ca/gulcehrc_db?table=' + TABLE_NAME)

    for i in xrange(n_trials):

        state.lr_rate = learning_rates[ri(learning_rates.shape[0]) - 1]
        state.powerup_nunits = n_units[ri(len(n_units)) - 1]
        state.powerup_npieces = n_pieces[ri(len(n_pieces)) - 1]
        state.W_lr_scale = learning_rate_scalers[ri(len(learning_rate_scalers)) - 1]
        state.p_lr_scale = learning_rate_scalers[ri(len(learning_rate_scalers)) - 1]
        state.batch_size = batch_sizes[ri(len(batch_sizes)) - 1]

        state.l2_pen = l2_pen[ri(l2_pen.shape[0]) - 1]
        state.init_mom = numpy.random.uniform(low=0.3, high=0.6)
        state.final_mom = numpy.random.uniform(low=state.init_mom + 0.1, high=0.9)
        state.decay_factor = numpy.random.uniform(low=0.01, high=0.05)
        state.max_col_norm = max_col_norms[ri(len(max_col_norms)) - 1]

        alphabet = list('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUWXYZ0123456789')
        numpy.random.shuffle(alphabet)
        state.save_path = './'
        state.save_path += ''.join(alphabet[:7]) + '_'
        sql.insert_job(experiment, flatten(state), db)
        ind += 1

    db.createView(TABLE_NAME + '_view')
    print "{} jobs submitted".format(ind)

if __name__ == "__main__":
    pento(80)

