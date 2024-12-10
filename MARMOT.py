import sys
import yaml
import numpy as np
import torch
import random
from datetime import datetime

import ddpg
import nsgaii
import MORoverInterface
import ReplayBuffer
from utils import DataLogger

np.random.seed(2024)
torch.manual_seed(2024)
random.seed(2024)

class MARMOT:
    def __init__(self, alg_config_filename, rover_config_filename, data_filename):
        self.config_filename = alg_config_filename
        self.rover_config_filename = rover_config_filename
        self._read_config()

        # Set up a data logger
        self.data_logger = DataLogger.DataLogger(data_filename)
        self.data_logger.straight_write('marmot_config', self.config_data)
        self.data_logger.straight_write('env_config', self.rover_config_data)

        # Load up as many replay buffers as agents on a roster
        self.rep_buffs = [ReplayBuffer.ReplayBuffer(alg_config_filename) for _ in range(self.roster_size)] # NOTE: both NSGA and DDPG mutate the replay buffers

        # Init the RL and EA instances
        self.EA = nsgaii.NSGAII(alg_config_filename=alg_config_filename, rover_config_filename=rover_config_filename, 
                                replay_buffers=self.rep_buffs)
        pop, roster_wise_team_combs, roster_wise_team_fits, chosen_roster, champion_indices = self.EA.evolve()
        self.RL = ddpg.DDPG(alg_config_filename=alg_config_filename, rover_config_filename=rover_config_filename, 
                            init_target_policy=chosen_roster, replay_buffers=self.rep_buffs)
        print(pop)
        updated_roster = self.RL.update_params(chosen_roster, champion_indices)
        self.EA.insert_new_to_pop(updated_roster)

    def _read_config(self):
        """Read and load MARMOT configuration from the YAML file."""
        with open(self.config_filename, 'r') as config_file:
            self.config_data = yaml.safe_load(config_file)
            print('[MARMOT]: YAML config read.')
        
        with open(self.rover_config_filename, 'r') as config_file:
            self.rover_config_data = yaml.safe_load(config_file)
            print('[MARMOT]: YAML rover config read.')

        self._load_config()
    
    def _load_config(self):
        self.roster_size = self.config_data["Shared"]["roster_size"]
        self.num_gens = self.config_data["Meta"]["num_gens"]
    
    def run(self):
        for gen in range(self.num_gens):
            print("Generation:", gen)
            pop, roster_wise_team_combs, roster_wise_team_fits, chosen_roster, champion_indices = self.EA.evolve()
            self.data_logger.straight_write('Population', {
                "generation" : gen,
                "fitnesses" : roster_wise_team_fits,
            })
            updated_roster = self.RL.update_params(chosen_roster, champion_indices)
            self.EA.insert_new_to_pop(updated_roster)

if __name__ == "__main__":
    now = datetime.now()
    marmot = MARMOT('/home/raghav/Research/IJCAI25/MOMERL/config/MARMOTConfig.yaml',
                    '/home/raghav/Research/IJCAI25/MOMERL/config/MORoverEnvConfig.yaml',
                    '/home/raghav/Research/IJCAI25/MOMERL/experiments/data/'+now.strftime("%Y-%m-%d %H:%M:%S")+'.ndjson')
    marmot.run()