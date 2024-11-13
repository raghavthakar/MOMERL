import torch
import torch.nn as nn
class DDPG:
    class Critic(nn.Module):
        def __init__(self):
            pass
        pass

    def __init__(self, ):
        # read all 8 critics configs from yaml and instatiate them
        # init target critic as deepcopy of main critics from above

        # instantiate a random target policy using mha configs

        # mini batch size (how many transitions to take from each replay buffer)
        pass
    
    def update_params(self, replay_buffers, mha, active_indices):
        # replay buffers is a dict where key is agent ind containing that agent's replay buffer
        pass
    pass