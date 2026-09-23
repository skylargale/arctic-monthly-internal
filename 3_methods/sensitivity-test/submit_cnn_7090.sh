#!/bin/bash
# ==============================================================
# PBS batch submission script for Monthly_CNN_7090.py on Casper.
#
# Usage: this script reads MONTH, GEOMETRY, PADDING, and LON_PADDING as env
# vars, which PBS only forwards if you submit with -v:
#
# qsub -v MONTH=march,GEOMETRY=none submit_cnn_7090.sh
# qsub -v MONTH=march,GEOMETRY=cos submit_cnn_7090.sh
# qsub -v MONTH=may,GEOMETRY=none submit_cnn_7090.sh
# qsub -v MONTH=may,GEOMETRY=cos submit_cnn_7090.sh
# qsub -v MONTH=april,PADDING=pole submit_cnn_7090.sh
# qsub -v MONTH=april,LON_PADDING=periodic submit_cnn_7090.sh
#
# MONTH must be one of: march, april, may
# GEOMETRY must be one of: none, cos, sqrt_cos (defaults to none if omitted)
# PADDING must be one of: none, pole (defaults to none if omitted)
# LON_PADDING must be one of: zero, periodic (defaults to zero if omitted)
#
# Check status: qstat -u $USER
#
# After complete for a month, run:
# python compare_geometry.py --month march
# python compare_padding.py --month march
# python compare_lon_padding.py --month march
# ==============================================================

#PBS -N monthly_cnn_may_cos
#PBS -A P93300065
#PBS -q casper
#PBS -l select=1:ncpus=8:mem=64GB
#PBS -l walltime=04:00:00
#PBS -j oe
#PBS -M skycgale@uw.edu

module load conda
conda activate arctic-monthly-cnn

cd $PBS_O_WORKDIR
 
MONTH=${MONTH:?"MONTH not set -- submit with: qsub -v MONTH=april,GEOMETRY=cos submit_cnn_7090.sh"}
GEOMETRY=${GEOMETRY:-none}
PADDING=${PADDING:-none}
LON_PADDING=${LON_PADDING:-zero}

echo "Month: $MONTH"
echo "Geometry: $GEOMETRY"
echo "Padding: $PADDING"
echo "Lon padding: $LON_PADDING"

python Monthly_CNN_7090.py \
    --month ${MONTH} \
    --geometry ${GEOMETRY} \
    --padding ${PADDING} \
    --lon-padding ${LON_PADDING}
