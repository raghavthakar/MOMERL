import random
import numpy as np
import yaml

class ReplayBuffer:
    def __init__(self, alg_config_filename):
        self.config_filename = alg_config_filename
        self._read_config()
        self.experiences = []
    
    def _read_config(self):
        """Read and load ReplayBuffer configuration from the YAML file."""
        with open(self.config_filename, 'r') as config_file:
            self.config_data = yaml.safe_load(config_file)
            print('[ReplayBuffer]: YAML config read.')

        self._load_config()
    
    def _load_config(self):
        self.buff_size = self.config_data["ReplayBuffer"]["buffer_size"]
    
    def add(self, transition, random_discard=True):
        if len(self.experiences) < self.buff_size:
            self.experiences.append(transition)
        else:
            if random_discard:
                i = random.randrange(len(self.experiences)) # get random index
                self.experiences[i], self.experiences[0] = self.experiences[0], self.experiences[i] # swap with the first element

            self.experiences.pop(0)
            self.experiences.append(transition)
    
    def sample_transitions(self, num_samples=1000):
        if num_samples > len(self.experiences):
            return None
        sampled_transitions = np.random.choice(self.experiences, size=num_samples, replace=False)
        return sampled_transitions