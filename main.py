import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gym
import time, os
from tensorboardX import SummaryWriter

from common.utils import create_log_dir, print_args, set_global_seeds
from common.wrappers import FrameStack, MaxAndSkipEnv, ImageToPyTorch, WarpFrame, make_atari, wrap_atari_dqn
from arguments import get_args
from train import train
from test import test

import wimblepong
from model import Rainbow
from agent import Agent

def main():
    args = get_args()
    print_args(args)

    log_dir = create_log_dir(args)
    if not args.evaluate:
        writer = SummaryWriter(log_dir)

    # Make the environment
    env = gym.make("WimblepongVisualMultiplayer-v0")
    env.unwrapped.scale = 2 #args.scale
    env.unwrapped.fps = 30 #args.fps
    env = make_atari(env)
    env = wrap_atari_dqn(env, args)
    # env = WarpFrame(env)
    # env = FrameStack(env, 4)  
    # env = MaxAndSkipEnv(env, skip=4)  # # it has its own lol
    # env = ImageToPyTorch(env)

    set_global_seeds(args.seed)
    env.seed(args.seed)

    # Model
    model = Rainbow(env, args).to(args.device)

    player_id = 1
    opponent_id = 3 - player_id
    opponent = wimblepong.SimpleAi(env, opponent_id)
    player = Agent(env, model, player_id=player_id)

    # Set the names for both SimpleAIs
    env.set_names(player.get_name(), opponent.get_name())

    if args.evaluate:
        test(env, args)
        env.close()
        return

    train(env, args, writer)

    writer.export_scalars_to_json(os.path.join(log_dir, "all_scalars.json"))
    writer.close()
    env.close()


if __name__ == "__main__":
    main()
