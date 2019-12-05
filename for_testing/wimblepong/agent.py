from wimblepong import Wimblepong
import numpy as np
import gym
import random

class Agent:
    def __init__(self, env, player_id=1):
        if type(env) is not Wimblepong:
            raise TypeError("I've trained all my life to play Wimblepong and you give me this? Please reconsider your actions,")
        self.env = env
        # Set the player id that determines on which side the ai is going to play
        self.player_id = player_id  
        # Ball prediction error, introduce noise such that the agent reflects not
        # only in straight lines
        self.bpe = 4                
        self.name = "Agent 007"

    def preprocess(frame):
        """ prepro 210x160x3 uint8 frame into 6000 (75x80) 1D float vector """
        I = I[35:185] # crop - remove 35px from start & 25px from end of image in x, to reduce redundant parts of image (i.e. after ball passes paddle)
        I = I[::2,::2,0] # downsample by factor of 2.
        I[I == 144] = 0 # erase background (background type 1)
        I[I == 109] = 0 # erase background (background type 2)
        I[I != 0] = 1 # everything else (paddles, ball) just set to 1. this makes the image grayscale effectively
        return I.astype(np.float).ravel() # ravel flattens an array and collapses it into a column vector

    #### env.

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