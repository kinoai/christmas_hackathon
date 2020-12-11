<div align="center">    
 
# Christmas Hackermoon Hackathon    

</div>

## Description   
What it does   

## How to run
First, install dependencies
```bash
# clone project
git clone https://github.com/YourGithubName/your-repo-name
cd your-repo-name

# optionally create conda environment
conda update conda
conda env create -f conda_env.yaml -n your_env_name
conda activate your_env_name

# install requirements
pip install -r requirements.txt
```

Next, you can train model without logging
```bash
# train model without Weights&Biases
# choose run config from project/run_configs.yaml
cd project
python train.py --no_wandb --run_config MNIST_CLASSIFIER_V1
```

Or you can train model with Weights&Biases logging
```yaml
# set project and entity names in project/project_config.yaml
loggers:
    wandb:
        project: "your_project_name"
        entity: "your_wandb_username_or_team"
```
```bash
# train model with Weights&Biases
# choose run config from project/run_configs.yaml
cd project
python train.py --run_config MNIST_CLASSIFIER_V1
```

Optionally you can install project as package with [setup.py](setup.py)
```bash
pip install -e .
```
<br>


#### PyCharm setup
- open this repository as PyCharm project
- set project interpreter:<br> 
`Ctrl + Shift + A -> type "Project Interpreter"`
- mark folder "project" as sources root:<br>
`right click on directory -> "Mark Directory as" -> "Sources Root"`
- set terminal emulation:<br> 
`Ctrl + Shift + A -> type "Edit Configurations..." -> select "Emulate terminal in output console"`
- run training:<br>
`right click on train.py file -> "Run 'train'"`

#### VS Code setup
- TODO
