# CNN and RNN project

This repository contains code for the course projects of Deep Learning in Taltech in autumn 2024, see also <https://github.com/mikk-kruusalu/deep_learning_project2>. The CNN was trained on [Lightning AI](lightning.ai) free studio, which allows to use a GPU.

In order to use this code, one needs to create a `conda` environment with

```bash
conda env create -f environment.yaml
```

Notice that if your computer does not include a cuda installation, you need to replace `pytorch-cuda=11.8` with `cpuonly` in the enivronment.yaml file.
Also, [Weights & Biases](wandb.ai) account is optional and one needs to login on the command line with `wandb login` before running the code.

Each model's folder includes a dataset folder which exports functions `load_data()` and `get_dataloaders()`. The dataset itself is also downloaded to this folder. The root of the model's folder includes definitions of the PyTorch models and scripts for exploring data or extra tasks in the course.

The `train.py` file is used to trigger all training. It takse one required argument `-c` that is a path to the config file. The config files are all located in the configs directory. The config files should be self-explanatory. Everything under hyperparams keyword is logged by wandb. The model section under the hyperparams keyword is passed to the model definition.

The `evaluate.py` script is used for evaluating the models if they are not trained using wandb.

Please find the reports for the course are in the root of this repository.
