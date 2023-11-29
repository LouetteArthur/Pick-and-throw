# PyBulletWasteThrower
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
| --agent     | -a          | The agent to use | random, human, goToBin, ppo, sac, ddpg, td3 |
| --episodes  | -e          | The number of episodes to run | int |
| --gui       | -g          | Whether to use the GUI or not | True, False |
| --verbose   | -v          | Whether to print the logs or not | True, False |
| --reward    | -r          | The reward function to use | neural_net, weighted, success, lin_reg |
| --model     | -m          | The model to use | str |
| --seed      | -s          | The seed to use | int |
| --save_data | -d          | The directory where you want to save your experiments | str |

## Training

To train a model, move to the code directory and use the following command:

```
python3 train.py [params]
```
where `params` are the parameters of the training. The available parameters are:

| Parameter   | Short name  | Description | Values |
| ----------- | ----------- | ----------- | ------ |
| --agent     | -a          | The agent to use | ppo, sac, ddpg, td3 |
| --episodes  | -e          | The number of episodes to run | int |
| --save      | -s          | The path to save the model | str |
| --model     | -m          | The pretrained model to use | str |
| --reward    | -r          | The reward function to use | neural_net, weighted, success, lin_reg |

## Demo

you can test the repo with the following instructions:
```
python3 main.py -a td3 -m models/SAC_1M_opt -r neural_net -v 1 -e 10 -g 1

python3 train_agent.py -a sac -e 10000 -r neural_net -s models/SAC_demo

python3 main.py -a goToBucket -r neural_net -v 1 -e 10 -g 1

python3 main.py -a goToBucket -r neural_net -v 0 -e 10000 -g 0
```
