import random

class ReplayBuffer:
    def __init__(self, buff_size=10000):
        self.experiences = []
        self.buff_size = buff_size
    
    def add(self, transition, random_discard=True):
        if len(self.experiences) < self.buff_size:
            self.experiences.append(transition)
        else:
            if random_discard:
                i = random.randrange(len(self.experiences)) # get random index
                self.experiences[i], self.experiences[0] = self.experiences[0], self.experiences[i] # swap with the first element

            self.experiences.pop(0)
            self.experiences.append(transition)