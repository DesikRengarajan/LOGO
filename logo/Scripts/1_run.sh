#!/bin/bash
cd ..
source activate bail
python run_logo.py --env-num 1 --seed 1  --gpu-index 0
python run_logo.py --env-num 1 --seed 11  --gpu-index 0
python run_logo.py --env-num 1 --seed 1  --gpu-index 0 --init-BC
python run_logo.py --env-num 2 --seed 1  --gpu-index 0
python run_logo.py --env-num 3 --seed 1  --gpu-index 0
python run_logo.py --env-num 3 --seed 2  --gpu-index 0
python run_logo.py --env-num 3 --seed 1  --gpu-index 0 --init-BC
python run_logo.py --env-num 4 --seed 1  --gpu-index 0
python run_logo.py --env-num 5 --seed 1  --gpu-index 0
python run_logo.py --env-num 5 --seed 1  --gpu-index 0 --init-BC
python run_logo.py --env-num 6 --seed 1  --gpu-index 0
python run_logo.py --env-num 7 --seed 1  --gpu-index 0
python run_logo.py --env-num 7 --seed 1  --gpu-index 0 --init-BC
python run_logo.py --env-num 8 --seed 1  --gpu-index 0