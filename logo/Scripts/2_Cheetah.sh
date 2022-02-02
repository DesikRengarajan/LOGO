#!/bin/bash
cd ..
source activate bail
python run_logo.py --env-num 3 --seed 1  --init-BC --gpu-index 1
python run_logo.py --env-num 3 --seed 2  --init-BC --gpu-index 1
python run_logo.py --env-num 3 --seed 3  --init-BC --gpu-index 1
python run_logo.py --env-num 3 --seed 4  --init-BC --gpu-index 1
python run_logo.py --env-num 3 --seed 5  --init-BC --gpu-index 1
python run_logo.py --env-num 4 --seed 4  --gpu-index 1
python run_logo.py --env-num 4 --seed 5  --gpu-index 1