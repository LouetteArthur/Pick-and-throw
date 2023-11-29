from utils import PickAndPlaceReward
from environment import TossingFlexpicker
import torch

def main(episodes=100000, pretrain_path=None, save_path="models/PickAndPlaceReward.pt"):
    env = TossingFlexpicker(GUI=False)
    estimator = PickAndPlaceReward()
    if pretrain_path is not None:
        estimator.load_state_dict(torch.load(pretrain_path))
    estimator.train(env, episodes, save_path=save_path)

if __name__ == "__main__":
    # parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=100000)
    parser.add_argument("--pretrain_path", type=str, default=None)
    parser.add_argument("--save_path", type=str, default="models/PickAndPlaceReward.pt")
    args = parser.parse_args()
    main(episodes=args.episodes, pretrain_path=args.pretrain_path, save_path=args.save_path)
