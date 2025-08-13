Modelling compounding across languages with analogy and composition
======

This repository contains replication code and data that accompany the followiing paper: Xu, A., Kemp, C., Frermann, L., and Xu, Y. (2025) Modelling compounding across languages with analogy and composition. Proceedings of the 47th Annual Meeting of the Cognitive Science Society.

## Dependencies

### Software

Required `jupyter`
Required `python>=3.7`
Required packages:
```
numpy
pytorch
scipy
pandas
seaborn
matplotlib
statsmodel
```

### Data

Please see main text for information on datasets used in this project. Preprocessing code can be found under the current directory.

## Execution

Code for data analysis can be found under `./analysis`. To replicate the results in the paper, first run `combination_pred_split.py` and `subpattern_analysis.py`. The following lines provide examples for using these scripts: 

```
python -u combination_pred_split.py --model_type 1 --total_splits 5 --test_split 0 --val_split 1 --test 0 --gamma 0.5 0.5 0.5 --theta 1.0 1.0 1.0
python -u subpattern_analysis.py --gamma 0.3 0.2 0.3 --total_splits 5 --test_split 0 --val_split 1
```

After running these scripts with the chosen model parameters, the final statistics can be generated using `analysis.ipynb`.
