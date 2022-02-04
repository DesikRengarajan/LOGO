#!/bin/bash
cd ..
source activate bail
python run_logo.py --env-num 5 --seed 11  --gpu-index 1
python run_logo.py --env-num 5 --seed 12  --gpu-index 1
python run_logo.py --env-num 5 --seed 13  --gpu-index 1
python run_logo.py --env-num 5 --seed 14  --gpu-index 1
python run_logo.py --env-num 5 --seed 15  --gpu-index 1
