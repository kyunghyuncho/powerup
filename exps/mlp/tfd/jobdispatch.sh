#!/bin/bash -x

jobdispatch --repeat_jobs=120 --torque --gpu --env=THEANO_FLAGS=device=gpu,floatX=float32,force_device=True jobman sql 'postgresql://gulcehrc@opter.iro.umontreal.ca/gulcehrc_db?table=powerup_tfd_1layer_finer_large_p1' .
