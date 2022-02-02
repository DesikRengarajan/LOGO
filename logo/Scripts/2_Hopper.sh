#!/bin/bash
cd ..
source activate bail
python run_logo.py --env-num 1 --seed 11 --init-BC --gpu-index 1
python run_logo.py --env-num 1 --seed 1  --init-BC --gpu-index 1
python run_logo.py --env-num 1 --seed 2  --init-BC --gpu-index 1
python run_logo.py --env-num 1 --seed 3  --init-BC --gpu-index 1
python run_logo.py --env-num 1 --seed 4  --init-BC --gpu-index 1
python run_logo.py --env-num 1 --seed 5  --init-BC --gpu-index 1
python run_logo.py --env-num 2 --seed 11  --gpu-index 1
python run_logo.py --env-num 2 --seed 4  --gpu-index 1
python run_logo.py --env-num 2 --seed 5  --gpu-index 1