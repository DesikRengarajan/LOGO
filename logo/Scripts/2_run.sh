#!/bin/bash
cd ..
source activate bail
python run_logo.py --env-num 1 --seed 7  --gpu-index 0
python run_logo.py --env-num 1 --seed 7  --gpu-index 0 --init-BC
python run_logo.py --env-num 2 --seed 7  --gpu-index 0
python run_logo.py --env-num 3 --seed 7  --gpu-index 0
python run_logo.py --env-num 3 --seed 7  --gpu-index 0 --init-BC
python run_logo.py --env-num 4 --seed 7  --gpu-index 0
python run_logo.py --env-num 5 --seed 7  --gpu-index 0
python run_logo.py --env-num 5 --seed 7  --gpu-index 0 --init-BC
python run_logo.py --env-num 6 --seed 7  --gpu-index 0
python run_logo.py --env-num 7 --seed 7  --gpu-index 0
python run_logo.py --env-num 7 --seed 7  --gpu-index 0 --init-BC
python run_logo.py --env-num 8 --seed 7  --gpu-index 0