import os, shutil
import argparse
import numpy
from pylearn2.config import yaml_parse
from pylearn2.utils.string_utils import preprocess
from pylearn2.utils import serial
from jobman.tools import DD

def experiment(state, channel):

    # load and save yaml
    yaml_string = state.yaml_string % (state)
    with open(state.save_path + 'model.yaml', 'w') as fp:
        fp.write(yaml_string)

    # now run yaml file with default train.py script
    train_obj = yaml_parse.load(yaml_string)
    train_obj.main_loop()

    return channel.COMPLETE
