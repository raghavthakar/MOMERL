import pygmo as pg
import multiheaded_actor as mha
import torch
import MORoverInterface
import random
import copy
import numpy as np

class NSGAII:
    def __init__(self, state_size=10, num_actions=2, hidden_size=4, popsize=10, num_heads=5, team_size=3, noise_std=0.5, noise_mean=0, num_objs=2):
        """
        Parameters:
        - state_size (int): Size of input to neural network policy, which is the number of states
        - num_actions (int): Size out output of neural network policy, which is the number of actions
        - hidden_size (int): Number of nodes in the 1 hidden layer of the neural network
        - popsize (int): Total population size of parent + offspring size, must be an even number
        - num_heads (int): Number of heads of the Multiheaded Actor
        - team_size (int): Number of agents on a team
        - noise_std (float or int): Standard deviation value for noise added during mutation
        - noise_mean (float or int): Mean value for noise added during mutation 
        - num_objs (int): The number of objectives in the problem
        """
        # MERL hidden size is 100
        # TODO: fix next_id for multihead actors
        # TODO: add code to insert traj into replay buffer
        # TODO: make fitnesses into a dict where the mha id is the key

        assert num_heads >= team_size, "number of heads of MHA must be gte the number of agents on a team"
        assert popsize % 2 == 0, "population size should be even"

        self.popsize = popsize
        self.noise_std = noise_std
        self.noise_mean = noise_mean
        self.state_size = state_size
        self.num_actions = num_actions
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.team_size = team_size
        self.num_objs = num_objs

        self.parent = [mha.MultiHeadActor(state_size, num_actions, hidden_size, num_heads, mha_id) for mha_id in range(self.popsize // 2)]
        self.offspring = None
        self.next_id = self.parent[-1].id + 1

    def mutate_policy(self, policy):
        """
        Mutates a given policy by adding noise to weights according to the std dev + mean noise params

        Parameters:
        - policy (MultiHeadActor): Neural network policy
        """
        noise_weight = [] # for debugging
        noise_bias = [] # for debugging
        with torch.no_grad():
            for layer in policy.children():
                if hasattr(layer, "weight"):
                    noise = self.noise_mean + torch.randn_like(layer.weight) * self.noise_std
                    noise_weight.append(noise)
                    layer.weight.data += noise
                
                if hasattr(layer, "bias"):
                    noise_b = self.noise_mean + torch.randn_like(layer.bias) * self.noise_std
                    noise_bias.append(noise_b)
                    layer.bias.data += noise_b
        policy.id = self.next_id
        self.next_id += 1
    
    def evaluate_fitnesses(self, r_set):
        """
        Calls function to perform rollout then assigns fitness to each index

        Parameters:
        - r_set (list of MultiHeadActors): Parent and offspring populations combined together

        Returns:
        - fitnesses (list of lists): List of fitness vectors, where each vector contains n values for n objectives
        """
        # can use multiprocessing for this later

        # TODO: Don't use key-1 for determining the index of objective

        fitnesses = [None] * len(r_set)
        for ind, policy in enumerate(r_set):
            traj, global_reward = MORoverInterface.MORoverInterface("/Users/sidd/Desktop/ijcai25/new_momerl/MOMERL/config/MORoverEnvConfig.yaml").rollout(policy, [0, 1, 2])
            # global_reward is a dict where the key is the objective and value is the reward of that objective -> {2:98, 1:45}

            g_list = [None] * self.num_objs
            # converting the dict to a list
            for key in global_reward:
                g_list[key - 1] = -global_reward[key] # Need to do -1 because the objectives in the dict start at 1
            
            assert None not in g_list, "One of the objectives was not found in the global_reward dict"
            
            fitnesses[ind] = g_list
        
        return fitnesses
    
    def make_new_pop(self, sorted_policies):
        """
        Creates mutated offspring population from the parent. Deepcopy when using this as it mutates the input

        Parameters:
        - sorted_policies (list of (MultiHeadActors, fitness vector)): List containing the policies and their associated fitnesses

        Returns:
        - new_pop (list of MultiHeadActors): List of perturbed MultiHeadActors
        """
        new_pop = []

        for team in sorted_policies:
            self.mutate_policy(team)
            new_pop.append(team)
        return new_pop
    
    def form_teams_for_mha(self, num_teams):

        return [np.random.choice(self.num_heads, size=self.team_size, replace=False).tolist() for _ in range(num_teams)]
        # number of teams to form = (popsize / 2) / 
        # this should return [[indices for team 1], [indices for team 2]...[indices for team n]] -> all for 1 mha
        
    def evolve_pop(self):
        """
        Completes one generation of NSGA2 (combine offspring + parent, sort, retain best, and create offspring)

        Returns:
        - parent (list of MultiHeadActors): List of the parent population's policies for the next generation
        """
        r_set = self.parent + (self.offspring or []) # or statement for if first generation

        # assigning id to each multiheaded actor
        counter = 0
        for mha in r_set:
            mha.id = counter
            counter += 1   

        # print("length of parent", len(self.parent))
        # if(self.offspring is None):
        #     print("offspring is none")
        # else:
        #     print("length of offspring", len(self.offspring))
        # print("length of r_set", len(r_set))

        fitnesses = self.evaluate_fitnesses(r_set)
        ndf, dl, dc, ndr = pg.fast_non_dominated_sorting(points=fitnesses)

        sorted_policies = []

        for ind in ndf:
            sorted_policies.append([(r_set[i], fitnesses[i]) for i in ind]) # tuple of (policy, fitness vector)
        
        print("sorted policies")
        print(sorted_policies)
        next_pop = []
        curr_front = 0
        if(self.offspring is None):
            # first generation, adding all points from r_set (since it's only popsize / 2)
            mha_policies = [team[0] for front in sorted_policies for team in front]
            mutated_pop = self.make_new_pop(copy.deepcopy(mha_policies)) # this does mutate the param

            self.parent = mha_policies
            self.offspring = mutated_pop

            return self.parent
        else:
            
            while((curr_front < len(sorted_policies)) and len(next_pop) + len(sorted_policies[curr_front]) <= (self.popsize // 2)):
                # We can add all the points from this front
                for pol in sorted_policies[curr_front]:
                    next_pop.append(pol[0])
                curr_front += 1
            
            
            # we have added all the pareto fronts we can, now we can only add some from the next rank
            remaining_size = (self.popsize // 2) - len(next_pop)
            if(remaining_size > 0 and curr_front < len(sorted_policies)): # true if the current front index is valid, unless all were added in the previous step
                # sort the current front based on crowding distance
                sorted_crowd_front = sorted_policies[curr_front]
                sorted_crowd_front = list(sorted(sorted_crowd_front, key=lambda x: pg.sort_population_mo([x[1]]))) # x[1] is the objectives     

                for i in range(remaining_size):
                    pol = copy.deepcopy(sorted_crowd_front[i][0])
                    self.mutate_policy(pol)
                    next_pop.append(pol) # getting the policy, mutating it, then adding  it to population
            
            self.parent = next_pop
            self.offspring = self.make_new_pop(copy.deepcopy(next_pop))

            return self.parent

if __name__ == "__main__":
    evo = NSGAII()

    for i in range(100):
        print("Gen", i)
        evo.evolve_pop()
    print("done")