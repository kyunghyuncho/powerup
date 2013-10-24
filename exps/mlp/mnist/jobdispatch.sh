#!/bin/bash -x

jobdispatch --repeat_jobs=40 --torque --gpu --env=THEANO_FLAGS=device=gpu,floatX=float32,force_device=True jobman sql 'postgresql://gulcehrc@opter.iro.umontreal.ca/gulcehrc_db?table=powerup_mnist_finest_large_2l' .
