#!/bin/bash
set -e
python data_gen.py
python train.py --model lstm
python evaluate.py
echo 'Done. Check outputs/'
