# Do GFlowNets Transfer? Case Study on the Game of 24/42

Code for the preprint https://arxiv.org/abs/2503.01819

## Setup Instructions
- Make executable using `chmod +x install.sh`
- `./install.sh`
- `conda env create -f environment.yaml -n FoR`
- Activate conda env using `conda activate FoR`
- Install some dependencies `pip install tensorboard`
- Setup HF credentials using `huggingface-cli login --token <TOKEN>`
- Run script using `nohup python3 main.py` or `nohup python3 prontoqa_train.py`
- If facing rope scaling error `pip install -U transformers`

- For evaluation change the variables `do_train=False` and `do_test=True` in the scripts.