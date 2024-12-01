import yaml
import numpy as np
import random
import copy
import torch
import pygmo as pg

import multiheaded_actor as mha
import MORoverInterface
import itertools
import ReplayBuffer

torch.manual_seed(2024)
np.random.seed(2024)
random.seed(2024)

class Roster_Ind:
    def __init__(self, state_size=10, num_actions=2, hidden_size=80, num_heads=4, roster=None):
        if roster is None:
            self.roster = mha.MultiHeadActor(state_size, num_actions, hidden_size, num_heads)
        else:
            self.roster = copy.deepcopy(roster)
        self.reset_borda_points()
    
    def reset_borda_points(self):
        self.borda_points = -1

class NSGAII:
    def __init__(self, alg_config_filename, rover_config_filename, replay_buffers):
        self.interface = MORoverInterface.MORoverInterface(rover_config_filename)
        self.config_filename = alg_config_filename
        self._read_config()

        self.team_size = self.interface.get_team_size()
        assert self.roster_size >= self.team_size, "Roster size must be gte the number of agents on a team"
        assert self.pop_size % 2 == 0, "Population size should be even"

        self.num_objs = self.interface.get_num_objs()
        self.state_size = self.interface.get_state_size()
        self.num_actions = self.interface.get_action_size()
        self.pop = [Roster_Ind(state_size=self.state_size, num_actions=self.num_actions, hidden_size=self.hidden_size, num_heads=self.roster_size) for _ in range(self.pop_size)]
        self.replay_buffers = replay_buffers

        # Store all possible team combinations that are legal from the roster size.
        self.all_team_combos = list(itertools.combinations(range(self.roster_size), self.team_size))
        assert len(self.all_team_combos) >= self.num_teams_formed_each_roster, "Cannot form emough teams with this roster size"
    
    def _read_config(self):
        """Read and load NSGAII configuration from the YAML file."""
        with open(self.config_filename, 'r') as config_file:
            self.config_data = yaml.safe_load(config_file)
            print('[NSGAII]: YAML config read.')

        self._load_config()
    
    def _load_config(self):
        """Load internal NSGAII configuration."""
        self.pop_size = self.config_data["NSGAII"]["pop_size"]
        self.policy_mutation_noise_std = self.config_data["NSGAII"]["policy_mutation_noise_std"]
        self.policy_mutation_noise_noise_mean = self.config_data["NSGAII"]["policy_mutation_noise_mean"]
        self.hidden_size = self.config_data["MHA"]["hidden_size"]
        self.roster_size = self.config_data["Shared"]["roster_size"]
        self.num_teams_formed_each_roster = self.config_data["NSGAII"]["num_teams_formed_each_MHA"]
        self.crossover_eta = 5
    
    def _crossover_policies(self, x1, x2):
        if not isinstance(x1, mha.MultiHeadActor) or not isinstance(x2, mha.MultiHeadActor):
            raise ValueError("Parents must be MultiHeadActors for crossover!")
        
        # Create deep copies of the parent policies to serve as offspring
        y1 = copy.deepcopy(x1)
        y2 = copy.deepcopy(x2)

        with torch.no_grad():
            # Iterate over parameters (weights and biases) of the policies
            for param_x1, param_x2, param_y1, param_y2 in zip(
                x1.parameters(), x2.parameters(), y1.parameters(), y2.parameters()
            ):
                p1 = param_x1.data
                p2 = param_x2.data

                # Generate random numbers u between 0 and 1
                u = torch.rand_like(p1)

                # Compute beta_q using the SBX formula
                beta_q = torch.where(
                    u <= 0.5,
                    (2 * u) ** (1.0 / (self.crossover_eta + 1)),
                    (1 / (2 * (1 - u))) ** (1.0 / (self.crossover_eta + 1)),
                )

                # Generate offspring parameters
                child1 = 0.5 * ((1 + beta_q) * p1 + (1 - beta_q) * p2)
                child2 = 0.5 * ((1 - beta_q) * p1 + (1 + beta_q) * p2)

                # Assign the new parameters to the offspring policies
                param_y1.data.copy_(child1)
                param_y2.data.copy_(child2)

        return y1, y2
    
    def _mutate_policy_in_place(self, policy):
        """
        Mutates a given policy in place by adding noise to weights according to the std dev + mean noise params

        Parameters:
        - policy (MultiHeadActor): Neural network policy
        """
        policy.mutate(self.policy_mutation_noise_noise_mean, self.policy_mutation_noise_std)
    
    def evolve(self):
        roster_wise_team_combinations = [None for _ in range(self.pop_size)] # Will store all sampled team combinations for all rosters
        roster_wise_team_fitnesses = [None for _ in range(self.pop_size)] # Will store all team fitnesses for all rosters 
        # Assign fitness to each Roster
        for ind_idx, ind in enumerate(self.pop):
            team_combinations = []
            team_fitnesses = []
            # Sample teams from this roster
            for _ in range(self.num_teams_formed_each_roster):
                sampled_team = random.choice(self.all_team_combos) # randomly sampled team tuple
                # Evaluate this team
                fitness = [0 for _ in range(self.num_objs)] # Will store this team's fitness
                _, fitness_dict = self.interface.rollout(ind.roster, sampled_team)
                # Store fitness
                for f in fitness_dict:
                    fitness[f] = -fitness_dict[f] # NOTE: The fitness sign is flipped to match Pygmo convention
                # Push the team and evaluation into lists
                team_combinations.append(sampled_team)
                team_fitnesses.append(fitness)
            # If team combinations and fitnesses are not equal in length
            assert len(team_combinations) == len(team_fitnesses), "Problem"
            # Insert the team combination and team fitness
            # NOTE: Index at which these are added is the index of their roster in the population
            # print("------", len(self.pop), self.pop_size)
            roster_wise_team_combinations[ind_idx] = team_combinations
            roster_wise_team_fitnesses[ind_idx] = team_fitnesses
        # Flatten the fitnesses for pygmo
        roster_wise_team_fitnesses_fl = list(itertools.chain.from_iterable(roster_wise_team_fitnesses))
        # Sort according to NSGA2
        sorted_teams = pg.sort_population_mo(points=roster_wise_team_fitnesses_fl) # NOTE: better teams first
        
        # Perform a Borda count
        for roster_idx in self.pop:
            roster_idx.reset_borda_points()
        for rank, team_idx in enumerate(sorted_teams):
            if rank < self.pop_size//2:
                print(roster_wise_team_fitnesses_fl[team_idx])
            # The roster is the row number in which this team lies
            roster_idx = team_idx // len(roster_wise_team_combinations[0])
            # Assign points to the roster
            self.pop[roster_idx].borda_points += len(sorted_teams) - rank # NOTE: more points for smaller the rank
        
        # Sort the population by borda_points in descending order
        sorted_pop = sorted(self.pop, key=lambda roster_ind: roster_ind.borda_points, reverse=True)
        # Copy the top half into the parent set
        parent_set = sorted_pop[:self.pop_size // 2]

        # Create offsprings, leaving 2 empty spots
        offspring_set = []
        while len(offspring_set) < (self.pop_size//2):
        # for _ in range(self.pop_size//2 - 2):
            idx1, idx2 = random.sample(range(len(parent_set)), 2) # As the parent set is sorted
            parent1 = parent_set[idx1] if idx1 < idx2 else parent_set[idx2] # Choose the lower index
            idx1, idx2 = random.sample(range(len(parent_set)), 2)
            parent2 = parent_set[idx1] if idx1 < idx2 else parent_set[idx2] # Choose the lower index
            # Crossover the parents and mutate the offsprings
            offspring_roster1, offspring_roster2 = self._crossover_policies(parent1.roster, parent2.roster)
            self._mutate_policy_in_place(offspring_roster1)
            self._mutate_policy_in_place(offspring_roster2)
            # Create new Roster_Inds with offspring policies
            offspring1 = Roster_Ind(roster=offspring_roster1)
            offspring2 = Roster_Ind(roster=offspring_roster2)
            
            offspring_set.append(offspring1)
            if len(offspring_set) < self.pop_size // 2:
                offspring_set.append(offspring2)

        # Set the population as parent_set + offspring_set
        self.pop = parent_set + offspring_set
        # print("--", len(self.pop), len(parent_set), len(offspring_set))


if __name__ == "__main__":
    r_buffs = [ReplayBuffer.ReplayBuffer("/home/raghav/Research/IJCAI25/MOMERL/config/MARMOTConfig.yaml") for _ in range(2)]
    evo = NSGAII(alg_config_filename="/home/raghav/Research/IJCAI25/MOMERL/config/MARMOTConfig.yaml", rover_config_filename="/home/raghav/Research/IJCAI25/MOMERL/config/MORoverEnvConfig.yaml", replay_buffers=r_buffs)
    for i in range(10000):
        print("Generation: ", i, "-----")
        evo.evolve()