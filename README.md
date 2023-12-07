# FLShield
In this repository, code is for our IEEE S&P 2024 paper [FLShield: A Validation Based Federated Learning Framework to Defend Against Poisoning Attacks](https://arxiv.org/pdf/2308.05832.pdf)

## Installation
Recommended python version: 3.8.x, 3.9.x

Create a virtual environment and install the dependencies listed in `requirements.txt`:
```
pip install -r requirements.txt
```

## Usage
The basic usage is as follows:
```
python main.py --aggregation_methods=X --attack_methods=Y --type=Z
```
where `X` is the aggregation method and can take values in 'mean', 'geom_median','flame', 'flshield', 'afa', 'foolsgold'

`Y` is the poisoning attack method and can take values in - 'targeted_label_flip', 'dba', 'inner_product_manipulation', 'attack_of_the_tails', 'semantic_attack'

`Z` is the dataset and can take values in 'emnist', 'fmnist', 'cifar', 'loan'

Note: in order to run FLShield with bijective version, `--bijective_flshield` should be added to the command line.

Different scenarios require different parameters which is listed in `utils/jinja.yaml`
Some of them are changable from the command line, for example adding `--noniid=one_class_expert` will run the experiment with one class expert data distribution.





## Citation
If you find our work useful in your research, please consider citing:
```
@article{kabir2023flshield,
  title={FLShield: A Validation Based Federated Learning Framework to Defend Against Poisoning Attacks},
  author={Kabir, Ehsanul and Song, Zeyu and Rashid, Md Rafi Ur and Mehnaz, Shagufta},
  journal={arXiv preprint arXiv:2308.05832},
  year={2023}
}
```
## Acknowledgement 
- [AI-secure/DBA](https://github.com/AI-secure/DBA)
- [ebagdasa/backdoor_federated_learning](https://github.com/ebagdasa/backdoor_federated_learning)
- [krishnap25/RFA](https://github.com/krishnap25/RFA)
- [DistributedML/FoolsGold](https://github.com/DistributedML/FoolsGold)
