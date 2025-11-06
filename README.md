Our code forked from https://github.com/aHuiWang/CIKM2020-S3Rec

We provide the replication code for all experimental results. The specific steps are as follows:

# Replicating Results of Enhanced Loss Function on CIKM2020-S3Rec

## Preparation 
```python
pip install -r requirements.txt
```

## Replicating Results
### Introduction img result.
![introduction](introduction.svg)
We provide a simple script file for one-step reproduction of the experimental results presented in the introduction section.

```python
python run_introduction_experiment.py
```

Result save at `./output/introduction/All_result.txt`, and the training log save at  `./output/introduction/.*`

### Experiment table result.
![Table2](table2.jpg)

We provide a convenient way to reproduce the experimental results in the table:

```python
sh table2_results.sh
```

Result will save at `output/` folder, and our training log already save at `our_output/` folder.

# Replicating Results of Enhanced Loss Function on [Aprec repo](https://github.com/asash/bert4rec_repro).

We did not fork the [Aprec repo](https://github.com/asash/bert4rec_repro) at this repo. But we provide code and tutorials to implement our enhanced  loss in Aprec repo.

## [Turorials for Aprec](./Aprec_change/README.md)

### Note
If you have any question please leave message at ISSUE.

### Cite
If you find the our codes and datasets useful for your research or development, please cite our paper:

```
@misc{li2023improving,
      title={Improving Sequential Recommendation Models with an Enhanced Loss Function},
      author={Fangyu Li and Shenbao Yu and Feng Zeng and Fang Yang},
      year={2023},
      eprint={2301.00979},
      archivePrefix={arXiv},
      primaryClass={cs.IR}
}
```
