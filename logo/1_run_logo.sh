#!/bin/bash
source activate bail
python run_logo.py --env-num 1 --high-kl 0.01 --seed 1
python run_logo.py --env-num 1 --high-kl 0.01 --seed 2
python run_logo.py --env-num 1 --high-kl 0.01 --seed 11
python run_logo.py --env-num 5 --high-kl 0.01 --seed 1
python run_logo.py --env-num 5 --high-kl 0.01 --seed 2
