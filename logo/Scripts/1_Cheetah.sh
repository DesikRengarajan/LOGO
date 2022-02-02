#!/bin/bash
cd ..
source activate bail
python run_logo.py --env-num 3 --seed 1  --gpu-index 0
python run_logo.py --env-num 3 --seed 2	 --gpu-index 0
python run_logo.py --env-num 3 --seed 3  --gpu-index 0
python run_logo.py --env-num 3 --seed 4  --gpu-index 0
python run_logo.py --env-num 3 --seed 5  --gpu-index 0
python run_logo.py --env-num 4 --seed 1  --gpu-index 0
python run_logo.py --env-num 4 --seed 2  --gpu-index 0
python run_logo.py --env-num 4 --seed 3  --gpu-index 0