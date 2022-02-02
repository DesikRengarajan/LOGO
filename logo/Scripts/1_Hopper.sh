#!/bin/bash
cd ..
source activate bail
python run_logo.py --env-num 1 --seed 11 --gpu-index 0
python run_logo.py --env-num 1 --seed 1  --gpu-index 0
python run_logo.py --env-num 1 --seed 2	 --gpu-index 0
python run_logo.py --env-num 1 --seed 3  --gpu-index 0
python run_logo.py --env-num 1 --seed 4  --gpu-index 0
python run_logo.py --env-num 1 --seed 5  --gpu-index 0
python run_logo.py --env-num 2 --seed 1  --gpu-index 0
python run_logo.py --env-num 2 --seed 2  --gpu-index 0
python run_logo.py --env-num 2 --seed 3  --gpu-index 0