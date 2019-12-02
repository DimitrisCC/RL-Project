from wimblepong import Wimblepong
import numpy as np
import gym
import random
from common.utils import epsilon_scheduler


def prepro(I):
    """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
    I = I[::2,::2,0] # downsample by factor of 2
    I[I == 43] = 0 # erase background (background type 1)
    I[I != 0] = 1 # everything else (paddles, ball) just set to 1
    return I.astype(np.float).ravel()

def get_epsilon_by_frame():
    return epsilon_scheduler(1.0, 0.01, 30000)

class Agent:
    def __init__(self, env, model, player_id=1):
        self.env = env
        # Set the player id that determines on which side the ai is going to play
        self.player_id = player_id  
        # Ball prediction error, introduce noise such that the agent reflects not
        # only in straight lines
        self.bpe = 4                
        self.name = "Agent Rainbow"
        self.model = model


    def load_model(self):
        self.model = pickle.load(open(self.model_file, 'rb'))

    def get_name(self):
        """
        Interface function to retrieve the agents name
        """
        return self.name

    def get_action(self, ob=None):
        """
        Interface function that returns the action that the agent took based
        on the observation ob
        """
        observation = prepro(ob)

        model.act(observation, )
        # Get the player id from the environmen
        player = self.env.player1 if self.player_id == 1 else self.env.player2
        # Get own position in the game arena
        my_y = player.y
        # Get the ball position in the game arena
        ball_y = self.env.ball.y + (random.random()*self.bpe-self.bpe/2)

        # Compute the difference in position and try to minimize it
        y_diff = my_y - ball_y
        if abs(y_diff) < 2:
            action = 0  # Stay
        else:
            if y_diff > 0:
                action = self.env.MOVE_UP  # Up
            else:
                action = self.env.MOVE_DOWN  # Down

        return action

    def reset(self):
        # Nothing to done for now...
        return