#!/bin/bash -x

jobdispatch --repeat_jobs=10 --torque --gpu --env=THEANO_FLAGS=device=gpu,floatX=float32,force_device=True jobman sql 'postgresql://gulcehrc@opter.iro.umontreal.ca/gulcehrc_db?table=powerup_mnist_1layer_finer_p1_v2' .
