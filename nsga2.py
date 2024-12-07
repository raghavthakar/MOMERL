import pygmo as pg
import multiheaded_actor as mha
import torch
from MORoverInterface import MORoverInterface
import random
import copy
import numpy as np
import yaml
import more_itertools
import ReplayBuffer as replay_buffer

class MHAWrapper():
    def __init__(self, mha, team_indices=None, fitnesses=None):
        self.mha = mha
        self.super_id = mha.id
        self.team_indices = team_indices # this is which heads are active
        self.fitnesses = fitnesses
        self.indices_from_fitness_lst = [] # this stores the indices associated with the fitness list that is sorted by pygmo
        

class NSGAII:
    def __init__(self, alg_config_filename, rover_config_filename, replay_buffers):
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
        """
        self.interface = MORoverInterface(rover_config_filename)
        self.config_filename = alg_config_filename
        self._read_config()

        # MERL hidden size is 100
        self.team_size = self.interface.get_team_size()
        assert self.num_heads >= self.team_size, "number of heads of MHA must be gte the number of agents on a team"
        assert self.popsize % 2 == 0, "population size should be even"

        self.state_size = self.interface.get_state_size()
        self.num_actions = self.interface.get_action_size()
        self.parent = [mha.MultiHeadActor(self.state_size, self.num_actions, self.hidden_size, self.num_heads, mha_id) for mha_id in range(self.popsize // 2)]
        self.offspring = None
        self.next_id = self.parent[-1].id + 1
        self.replay_buffers = replay_buffers

        self.all_team_combos = list(more_itertools.distinct_combinations(range(self.num_heads), self.team_size))
        assert len(self.all_team_combos) >= self.num_teams_formed_each_MHA, "There are fewer unique team combinations than the specified number of teams to form"
    
    def _read_config(self):
        """Read and load NSGAII configuration from the YAML file."""
        with open(self.config_filename, 'r') as config_file:
            self.config_data = yaml.safe_load(config_file)
            print('[NSGAII]: YAML config read.')

        self._load_config()
    
    def _load_config(self):
        """Load internal NSGAII configuration."""
        self.popsize = self.config_data["NSGAII"]["popsize"]
        self.noise_std = self.config_data["NSGAII"]["noise_std"]
        self.noise_mean = self.config_data["NSGAII"]["noise_mean"]
        self.hidden_size = self.config_data["MHA"]["hidden_size"]
        self.num_heads = self.config_data["Shared"]["roster_size"]
        self.num_teams_formed_each_MHA = self.config_data["NSGAII"]["num_teams_formed_each_MHA"]
        self.crossover_eta = 15
    
    def _give_mha_id(self, mha):
        """
        Assigns a multiheaded actor (roster) an id to track later on

        Parameters:
        - mha (MultiHeadActor): Neural network policy
        """
        mha.id = self.next_id
        self.next_id += 1
    
    def crossover_policies(self, x1, x2):
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

    def mutate_policy(self, policy):
        """
        Mutates a given policy by adding noise to weights according to the std dev + mean noise params

        Parameters:
        - policy (MultiHeadActor): Neural network policy
        """
        policy.mutate(self.noise_mean, self.noise_std)
    
    def make_new_pop(self, rosters, scores_dict):
        """
        Creates mutated offspring population from the parent. Deepcopy when using this as it mutates the input.

        Parameters:
        - rosters (list of MultiHeadActors)
        - scores_dict (mha_d:borda_score dictionary)

        Returns:
        - new_pop (list of MultiHeadActors): List of offspring MultiHeadActors after crossover and mutation
        """
        new_pop = []

        # Simply mutate parent set in the first generation
        if scores_dict is None:
            for mha in rosters:
                offspring = copy.deepcopy(mha)
                self.mutate_policy(offspring)
                self._give_mha_id(offspring)
                new_pop.append(offspring)

        # If roster scores are available
        else:
            while len(new_pop) < (self.popsize // 2):
                roster1, roster2 = random.sample(rosters, 2)
                parent1 = roster1 if scores_dict[roster1.id] > scores_dict[roster2.id] else roster2
                
                roster1, roster2 = random.sample(rosters, 2)
                parent2 = roster1 if scores_dict[roster1.id] > scores_dict[roster2.id] else roster2

                offspring1, offspring2 = self.crossover_policies(parent1, parent2)
                
                self.mutate_policy(offspring1)
                self._give_mha_id(offspring1)
                
                self.mutate_policy(offspring2)
                self._give_mha_id(offspring2)

                new_pop.extend([offspring1, offspring2])

        return new_pop
    
    def form_teams_for_mha(self, mha):
        """
        Creates lists of indices where each list represents a team that is formed from the roster and adds this list to a MHAWrapper

        Parameters:
        - mha (MultiHeadActor): A roster of agent policies

        Returns:
        - mhainfo (MHAWrapper): Data about the given roster along with the teams formed
        """
        #team_list = [np.random.choice(self.num_heads, size=self.team_size, replace=False).tolist() for _ in range(self.num_teams_formed_each_MHA)]
        indices_to_pick = [np.random.choice(len(self.all_team_combos), size=1, replace=False).tolist()[0] for _ in range(self.num_teams_formed_each_MHA)]

        team_list = [self.all_team_combos[i] for i in indices_to_pick]

        mhainfo = MHAWrapper(mha=mha, team_indices=team_list)

        return mhainfo
        # this returns [[indices for team 1], [indices for team 2]...[indices for team n]] -> all for 1 mha
    
    def find_best_rosters(self, front_crowd_sort, all_rosters):
        """
        Scores each roster (multiheaded actor) based on the teams that were formed from it

        Parameters:
        - front_crowd_sort (list of indices of fitnesses): An arg sorted list of fitnesses from all teams
        - all_rosters (list of MHAWrappers): A list containing all the rosters

        Returns:
        - dict (dict where k:v is mha id: score ): Sorted (by values) dictionary in reverse order
        """
        scores_dict = {mha.super_id: 0 for mha in all_rosters}
        # intializing all scores to 0

        count_value = len(front_crowd_sort) - 1 # -1 so the last team gets a score of 0
        print("count_value", count_value)

        for team_ind in front_crowd_sort:
            for ros in all_rosters:
                if(team_ind in ros.indices_from_fitness_lst):
                    scores_dict[ros.super_id] += count_value
                    break
            
            count_value -= 1
    
        return {k: v for k, v in sorted(scores_dict.items(), key=lambda item: item[1], reverse=True)} # sorting the values of scores_dict based on the scores (value of each entry)

    def get_roster_for_RL(self, all_fitnesses, all_rosters):
        print("allfits:", all_fitnesses)
        non_dom_lst = pg.non_dominated_front_2d(points=all_fitnesses).tolist()
        print("nondomlist:",non_dom_lst)
        team_picked = random.choice(non_dom_lst) # this is the index because non_dom_lst uses args
        print("team_picked:", team_picked)

        for ros in all_rosters:
            print("this ros.indices from lst", ros.super_id, "vals:", ros.indices_from_fitness_lst)
            if(team_picked in ros.indices_from_fitness_lst):
                print("mha is found in roster func, id:", ros.super_id)
                print("indices in fitness:",ros.indices_from_fitness_lst)
                print("team_indices", ros.team_indices)
                print("index to grab from team", team_picked - ros.indices_from_fitness_lst[0])
                return (ros.mha, ros.team_indices[team_picked - ros.indices_from_fitness_lst[0]]) # getting the index value of team_indices by subtracting the current value by the first value in the list

    def evolve_pop(self, print_fitness=False):
        """
        Completes one generation of NSGA2 (combine offspring + parent, sort, retain best, and create offspring)

        Returns:
        - champ_mha (MultiHeadActor): A roster from which a team when evaluated was on the Pareto Front
        - champ_team (list of ints): A list of the heads of the roster that are active (list of ints)
        """
        r_set = self.parent + (self.offspring or [])
        print("length of r_set is", len(r_set))
        # r_set has the population of mulitheaded actors from parent and offspring

        # now we need to form teams from r_set
        all_rosters = [self.form_teams_for_mha(roster) for roster in r_set] # all_rosters holds a list of MHAWrappers 

        all_fitnesses = self.evaluate_fitnesses(all_rosters) # indices_from_fitness_lst are also added to each MHA here
        # time to sort these fitnesses

        champ_mha, champ_team = self.get_roster_for_RL(all_fitnesses, all_rosters)
        print("MHA id selected:", champ_mha.id)
        print("Team picked", champ_team)

        scores_dict = None
        
        if(self.offspring is None):
            remaining_mhas = [ros.mha for ros in all_rosters]
        else:
            # print(all_fitnesses)
            
            front_crowd_sort = pg.sort_population_mo(points=all_fitnesses)

            scores_dict = self.find_best_rosters(front_crowd_sort, all_rosters)
            print(scores_dict)
            # print(front_crowd_sort)
            # only need to do this if you're not in the first generation of NSGA
            remaining_mhas = []

            for k, v in scores_dict.items():
                if(len(remaining_mhas) < self.popsize // 2):
                    for ros in all_rosters:
                        if(ros.super_id == k):
                            remaining_mhas.append(ros.mha)
                            print("Fitnesses for mha id:", ros.super_id, ros.fitnesses)
                            break
                    
                else:
                    break
            print()
            for ros in all_rosters:
                print("Second Fitnesses for mha id:", ros.super_id, ros.fitnesses)
        
        self.parent = remaining_mhas
        self.offspring = self.make_new_pop(copy.deepcopy(remaining_mhas), scores_dict)
        
        #return (copy.deepcopy(champ_mha), copy.deepcopy(champ_team))
        copied_champ_mha = copy.deepcopy(champ_mha)
        self._give_mha_id(copied_champ_mha)

        return (copied_champ_mha, copy.deepcopy(champ_team))

    def add_traj_to_rep_buff(self, traj, active_agents_indices):
        for agent_idx in active_agents_indices:
                for transition in traj[agent_idx]:
                    self.replay_buffers[agent_idx].add(transition)

    def evaluate_fitnesses(self, all_rosters):
        """
        Calls function to perform rollout then assigns fitness to each index

        Parameters:
        - r_set (list of MHAWrappers): Parent and offspring populations combined together

        Returns:
        - fitnesses (list of lists): List of fitness vectors, where each vector contains n values for n objectives
        """
        
        # all_rosters is a list of MHAWrappers
        all_fitnesses = []

        for roster in all_rosters:
            fitnesses = []
            counter = len(all_fitnesses)
            for team in roster.team_indices:
                roster.indices_from_fitness_lst.append(counter)
                #print("Added the index", counter)
                counter += 1

                traj, global_reward = self.interface.rollout(roster.mha, team)
                self.add_traj_to_rep_buff(traj, team)
                # adding to replay buffer

                g_list = [None] * len(global_reward)

                for key in global_reward:
                    g_list[key] = -global_reward[key]
                
                assert None not in g_list, "One of the objectives was not found in the global_reward dict"

                fitnesses.append(g_list)
            
            roster.fitnesses = fitnesses
            all_fitnesses.extend(fitnesses)

        # all_fitnesses has all the fitnesses as a 2d list for pygmo to sort
        
        return all_fitnesses


if __name__ == "__main__":
    r_buffs = [replay_buffer.ReplayBuffer("/Users/sidd/Desktop/ijcai25/marmot_combine/MOMERL/config/MARMOTConfig.yaml") for _ in range(2)]
    evo = NSGAII(alg_config_filename="/Users/sidd/Desktop/ijcai25/marmot_combine/MOMERL/config/MARMOTConfig.yaml", rover_config_filename="/Users/sidd/Desktop/ijcai25/marmot_combine/MOMERL/config/MORoverEnvConfig.yaml", replay_buffers=r_buffs)

    print_fits = True
    for i in range(5000):
        print("Generation:", i)
        evo.evolve_pop(print_fitness=True)
        print()
    
    for mha in evo.parent:
        print()
        print(evo.interface.rollout(mha, [0,1]))
    print("done")
