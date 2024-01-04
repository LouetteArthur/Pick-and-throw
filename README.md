# RL Pick-and-Throw
## Project Description

This project models an ABB FlexPicker with PyBullet in order to throw scraps into designated buckets.

In this repository, we provide a modelisation of the FlexPicker robot in PyBullet. We use RL in order to learn efficient policies to control the FlexPicker, modelled by a cartesian robot.

## Installation
You need to install Python 3.8 and the requirements provided in the requirement file.
Use the command below to download the required packages in order to run the project:

```
pip install -r requirements.txt
```

## Usage

To run the simulation, move to the code directory and use the following command:

``` 
python3 main.py [params]
```
where `params` are the parameters of the simulation. The available parameters are:

| Parameter   | Short name  | Description | Values |
| ----------- | ----------- | ----------- | ------ |
| --agent     | -a          | The agent to use | goToBucket, optim, sac, td3, ppo |
| --episodes  | -e          | The number of episodes to run | int |
| --gui       | -g          | Whether to use the GUI or not | True, False |
| --verbose   | -v          | Whether to print the logs or not | True, False |
| --reward    | -r          | The reward function to use | success, success_and_time, success_time_and_distance|
| --model     | -m          | The directory with the model to use | str |
| --seed      | -s          | The seed to use | int |
| --save_data | -d          | The directory where you want to save your experiments | str |

## Training

To train an agent, move to the code directory and use the following command:

```
python3 train_agent.py [params]
```
where `params` are the parameters of the training. The available parameters are:

| Parameter     | Short name  | Description | Values |
| --------------| ----------- | ----------- | ------ |
| --agent       | -a          | The agent to use | ppo, sac, td3 |
| --episodes    | -e          | The number of episodes to run | int |
| --save        | -s          | The path to save the model | str |
| --model       | -m          | The directory with the pretrained model to use | str |
| --reward      | -r          | The reward function to use | neural_net, weighted, success, lin_reg |
| --hyperparams | -hp         | The directory to the yaml file with the hyperparameters | str |

To train a new estimator of the PaP for the reward function, move to the code directory and use the following command:

```
python3 train_reward.py [params]
```
where `params` are the parameters of the training. The available parameters are:

| Parameter         | Short name  | Description | Values |
| ------------------| ----------- | ----------- | ------ |
| --episodes        | -e          | The number of episodes to run | int |
| --save_path       | -s          | The path to save the model | str |
| --pretrained_path | -m          | The directory with the pretrained model to use | str |

## Demo

you can test the repo with the following instructions:
```
python3 main.py -a td3 -m models/SAC_1M_opt -r success_and_time -v 1 -e 10 -g 1 -hp hyperparams/sac_opt.yaml

python3 train_agent.py -a sac -e 10000 -r success_and_time -s models/SAC_demo

python3 main.py -a goToBucket -r success_and_time -v 1 -e 10 -g 1

python3 main.py -a goToBucket -r success_and_time -v 0 -e 10000 -g 0
```
