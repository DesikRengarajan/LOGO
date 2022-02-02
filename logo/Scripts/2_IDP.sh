#!/bin/bash
cd ..
source activate bail
python run_logo.py --env-num 7 --seed 1  --init-BC --gpu-index 1
python run_logo.py --env-num 7 --seed 2  --init-BC --gpu-index 1
python run_logo.py --env-num 7 --seed 3  --init-BC --gpu-index 1
python run_logo.py --env-num 7 --seed 4  --init-BC --gpu-index 1
python run_logo.py --env-num 7 --seed 5  --init-BC --gpu-index 1
python run_logo.py --env-num 8 --seed 4  --gpu-index 1
python run_logo.py --env-num 8 --seed 5  --gpu-index 1