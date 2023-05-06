# joint-lrsm
Joint graph learning and model fitting in Laplacian Regularized Stratified Models

This code repo implements the algorithms and baselines in our paper [Joint Graph Learning and Model Fitting in Laplacian Regularized Stratified Models](https://arxiv.org/abs/2305.02573).

## Running the experiments
```
python main.py --config [config file]
```

Here ```config file``` is the directory of some YAML file in ```configs/```.

For example, to reproduce experiments on concrete dataset, just run
```
python main.py --config concrete.yml
```

## Citing
If you wish to cite us, please use the following BibTex entry:
```
@article{jointlrsm2023,
  author = {Ziheng Cheng and Junzi Zhang and Akshay Agrawal and Stephen Boyd},
  title = {Joint Graph Learning and Model Fitting in Laplacian Regularized Stratified Models},
  year = {2023},
  journal = {arXiv preprint arXiv:2305.02573},
}
```

