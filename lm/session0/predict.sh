#!/bin/bash
#PBS -l nodes=1:ppn=20
#PBS -l walltime=48:00:00
#PBS -N session1_default
#PBS -A course
#PBS -q GpuQ

export THEANO_FLAGS=device=gpu1,floatX=float32

#source ~/.profile/.bashrc
#which python
#cd $PBS_O_WORKDIR
python2 ${CODE_DIR}lm/session0/predict_lm.py "$1" "$2"
