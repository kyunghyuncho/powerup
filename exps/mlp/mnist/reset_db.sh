#!/bin/sh -x

jobman sqlstatus --reset_prio "postgres://gulcehrc@opter.iro.umontreal.ca/gulcehrc_db?table=powerup_cifar10_bria_abs_fine2"

#In case of rsync error
jobman sqlstatus --status=5 --set_status=0 "postgres://gulcehrc@opter.iro.umontreal.ca/gulcehrc_db?table=powerup_cifar10_bria_abs_fine2"

#In case of status 3
jobman sqlstatus --status=3 --set_status=0 "postgres://gulcehrc@opter.iro.umontreal.ca/gulcehrc_db?table=powerup_cifar10_bria_abs_fine2"

#In case of incomplete job status 1
jobman sqlstatus --status=1 --set_status=0 "postgres://gulcehrc@opter.iro.umontreal.ca/gulcehrc_db?table=powerup_cifar10_bria_abs_fine2"

