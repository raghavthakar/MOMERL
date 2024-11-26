import sys
import yaml

import ddpg
import nsga2
import MORoverInterface
import ReplayBuffer

class MARMOT:
    def __init__(self, alg_config_filename, rover_config_filename):
        self.config_filename = alg_config_filename
        self._read_config()

        # Load up as many replay buffers as agents on a roster
        self.rep_buffs = [ReplayBuffer(alg_config_filename) for _ in range(self.roster_size)] # NOTE: both NSGA and DDPG mutate the replay buffers

        # Init the RL and EA instances
        self.RL = ddpg.DDPG(alg_config_filename=alg_config_filename, rover_config_filename=rover_config_filename, replay_buffers=self.rep_buffs)
        self.EA = nsga2.NSGAII(alg_config_filename=alg_config_filename, rover_config_filename=rover_config_filename, replay_buffers=self.rep_buffs)

    def _read_config(self):
        """Read and load MARMOT configuration from the YAML file."""
        with open(self.config_filename, 'r') as config_file:
            self.config_data = yaml.safe_load(config_file)
            print('[MARMOT]: YAML config read.')

        self._load_config()
    
    def _load_config(self):
        self.roster_size = self.config_data["Shared"]["roster_size"]
        self.num_gens = self.config_data["Meta"]["num_gens"]

if __name__ == "__main__":
    marmot = MARMOT(sys.argv[1], sys.argv[2])