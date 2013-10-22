                #!/bin/sh

                ## Reasonable default values
                # Execute the job from the current working directory.
                #PBS -d /exec5/GROUP/lisa/gulcehre/experiments/powerup/exps/mlp/mnist

                #All jobs must be submitted with an estimated run time
                #PBS -l walltime=47:59:59

                ## Job name
                #PBS -N dbi_7e6a72c009a

                ## log out/err files
                # We cannot use output_file and error_file here for now.
                # We will use dbi_...out-id and dbi_...err-id instead
                #PBS -o /exec5/GROUP/lisa/gulcehre/experiments/powerup/exps/mlp/mnist/LOGS/jobman_sql_postgresql___gulcehrc_opter.iro.umontreal.ca_gulcehrc_db_table_powerup_mnist_1layer_finer_p1_v2_._2013-10-22_07-41-38.359565/dbi_7e6a72c009a.out
                #PBS -e /exec5/GROUP/lisa/gulcehre/experiments/powerup/exps/mlp/mnist/LOGS/jobman_sql_postgresql___gulcehrc_opter.iro.umontreal.ca_gulcehrc_db_table_powerup_mnist_1layer_finer_p1_v2_._2013-10-22_07-41-38.359565/dbi_7e6a72c009a.err


                ## Number of CPU (on the same node) per job
                #PBS -l nodes=1:ppn=1

                ## Execute as many jobs as needed
                #PBS -t 0-9

                ## Queue name
                #PBS -q @hades
export THEANO_FLAGS=device=gpu,floatX=float32,force_device=True
export OMP_NUM_THREADS=1
export GOTO_NUM_THREADS=1
export MKL_NUM_THREADS=1

                ## Variable to put into the environment
                #PBS -v THEANO_FLAGS,OMP_NUM_THREADS,GOTO_NUM_THREADS,MKL_NUM_THREADS

                ## Execute the 'launcher' script in bash
                # Bash is needed because we use its "array" data structure
                # the -l flag means it will act like a login shell,
                # and source the .profile, .bashrc, and so on
                /bin/bash -l -e /exec5/GROUP/lisa/gulcehre/experiments/powerup/exps/mlp/mnist/LOGS/jobman_sql_postgresql___gulcehrc_opter.iro.umontreal.ca_gulcehrc_db_table_powerup_mnist_1layer_finer_p1_v2_._2013-10-22_07-41-38.359565/launcher
