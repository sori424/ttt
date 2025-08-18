#!/usr/bin/env bash

pwd

cd "/home/ji-ung.lee/tt-tom/ttt/"

pip freeze

python --version

python inference_all.py --model_name "/home/ji-ung.lee/models/gpt-oss-20b" --task_name "bigtom" --use-tt

