#!/usr/bin/env bash

pwd

cd "/home/ji-ung.lee/tt-tom/ttt/"

pip freeze

python --version

python inference_all.py --model_name "/home/ji-ung.lee/models/gpt-oss-120b" --task_name "tomi"

