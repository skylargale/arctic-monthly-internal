#!/bin/bash
# ==============================================================
# PBS batch submission script for Monthly_CNN_7090.py on Casper.
#
# Usage: this script reads MONTH and GEOMETRY as shell positional args ($1, $2),
# which PBS only forwards if you submit with -v:
#
# qsub -v MONTH=march,GEOMETRY=none submit_cnn_7090.sh
# qsub -v MONTH=march,GEOMETRY=cos submit_cnn_7090.sh
# qsub -v MONTH=may,GEOMETRY=none submit_cnn_7090.sh
# qsub -v MONTH=may,GEOMETRY=cos submit_cnn_7090.sh
#
# MONTH must be one of: march, april, may
# GEOMETRY must be one of: none, cos, sqrt_cos (defaults to none if omitted)
# 
# Check status: qstat -u $USER
#
# After complete for a month, run:
# python compare_geometry.py --month march
# ==============================================================

#PBS -N monthly_cnn_may_cos
#PBS -A P93300065
#PBS -q casper
#PBS -l select=1:ncpus=8:mem=64GB
#PBS -l walltime=04:00:00
#PBS -j oe
#PBS -M skycgale@uw.edu

module load conda
conda activate env_cnn

cd $PBS_O_WORKDIR
 
MONTH=${MONTH:?"MONTH not set -- submit with: qsub -v MONTH=april,GEOMETRY=cos submit_cnn_7090.sh"}
GEOMETRY=${GEOMETRY:-none}
 
echo "Month: $MONTH"
echo "Geometry: $GEOMETRY"
 
python Monthly_CNN_7090.py \
    --month ${MONTH} \
    --geometry ${GEOMETRY}
