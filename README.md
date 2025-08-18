Code repository for the theory theory theory of mind

### Experiment
#### Fantom

Load data 

`python python fantom/fantom_load.py`


### Prompt Engineering
#### Dspy

(1) Submit the job with the `.sh` file below
```
#!/usr/bin/env bash

# run setup
source /nethome/soyoung/setup-dspy.sh
# run misc. stuff
nvidia-smi
echo $CUDA_VISIBLE_DEVICES
echo $HOSTNAME

conda activate fv

which python

python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --dtype float16 --port 8000
```
(2) Open terminal and do it on your local

`ssh -L 8000:hopper-3.coli.uni-saarland.de:8000 <user_id>@login.lst.uni-saarland.de`

(3) Then open jupyternotebook on your local 

`testdspy.ipynb`
