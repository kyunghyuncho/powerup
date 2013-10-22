                #!/bin/sh

                ## Reasonable default values
                # Execute the job from the current working directory.
                #PBS -d /exec5/GROUP/lisa/gulcehre/experiments/powerup/exps/mlp/tfd

                #All jobs must be submitted with an estimated run time
                #PBS -l walltime=47:59:59

                ## Job name
                #PBS -N dbi_d1f3b0e21b2

                ## log out/err files
                # We cannot use output_file and error_file here for now.
                # We will use dbi_...out-id and dbi_...err-id instead
                #PBS -o /exec5/GROUP/lisa/gulcehre/experiments/powerup/exps/mlp/tfd/LOGS/jobman_sql_postgresql___gulcehrc_opter.iro.umontreal.ca_gulcehrc_db_table_powerup_tfd_1layer_finer_large_no_mom_._2013-10-21_18-20-01.234775/dbi_d1f3b0e21b2.out
                #PBS -e /exec5/GROUP/lisa/gulcehre/experiments/powerup/exps/mlp/tfd/LOGS/jobman_sql_postgresql___gulcehrc_opter.iro.umontreal.ca_gulcehrc_db_table_powerup_tfd_1layer_finer_large_no_mom_._2013-10-21_18-20-01.234775/dbi_d1f3b0e21b2.err


                ## Number of CPU (on the same node) per job
                #PBS -l nodes=1:ppn=1

                ## Execute as many jobs as needed
                #PBS -t 0-119

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
                /bin/bash -l -e /exec5/GROUP/lisa/gulcehre/experiments/powerup/exps/mlp/tfd/LOGS/jobman_sql_postgresql___gulcehrc_opter.iro.umontreal.ca_gulcehrc_db_table_powerup_tfd_1layer_finer_large_no_mom_._2013-10-21_18-20-01.234775/launcher
