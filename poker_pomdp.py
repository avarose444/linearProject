import numpy as np
import random
import matplotlib.pyplot as plt
import pomdp_py
from pomdp_py import POMCP
from pomdp_py import Particles
from pomdp_py import Action

### necessary class definitions to make sure solver works ###

class PokerState(pomdp_py.State):
    def __init__(self, stage, hand_strength, pot_size):
        self.stage = stage
        self.hand_strength = hand_strength
        self.pot_size = pot_size 

    def __hash__(self):
        return hash((self.stage, self.hand_strength, self.pot_size))

    def __eq__(self, other):
        return (
            self.stage == other.stage and
            self.hand_strength == other.hand_strength and 
            self.pot_size == other.pot_size
        )

    def __str__(self):
        return f"PokerState(stage={self.stage}, hand_strength={self.hand_strength}, pot_size={self.pot_size})"

    def as_tuple(self):
        return (self.stage, self.hand_strength, self.pot_size)

class FoldAction(pomdp_py.Action):
    def __init__(self):
        super().__init__()

    def __str__(self):
        return "fold"

    def __eq__(self, other):
        return isinstance(other, FoldAction)

    def __hash__(self):
        return hash("fold")
    
class CheckCallAction(pomdp_py.Action):
    def __init__(self):
        super().__init__()

    def __str__(self):
        return "check_call"

    def __eq__(self, other):
        return isinstance(other, CheckCallAction)

    def __hash__(self):
        return hash("check_call")

class BetRaiseAction(pomdp_py.Action):
    def __init__(self):
        super().__init__()

    def __str__(self):
        return "bet_raise"

    def __eq__(self, other):
        return isinstance(other, BetRaiseAction)

    def __hash__(self):
        return hash("bet_raise")

### DEFINING POMDP COMPONENTS ###

#state space
stages = ["preflop", "flop", "turn", "river"]
hand_strengths = ["weak", "neutral", "strong"]
pot_sizes = ["small", "medium", "large"]

states = [
    PokerState(stage, hand, pot)
    for stage in stages
    for hand in hand_strengths
    for pot in pot_sizes
]

#action space
actions = {
    0: FoldAction(),
    1: CheckCallAction(),
    2: BetRaiseAction()
}

#observation space
opponent_actions = ["opp_fold", "opp_check", "opp_bet"]
board_strengths = ["neutral_board", "strong_board"]
observed_pot_sizes = ["small_pot", "medium_pot", "large_pot"]

observations = [
    (opp_action, board, pot)
    for opp_action in opponent_actions
    for board in board_strengths
    for pot in observed_pot_sizes
]

#state transitions
def transition_function(state, action):
    stage, hand_strength, pot_size = state

    stage_order = ["preflop", "flop", "turn", "river"]
    next_stage = stage_order[stage_order.index(stage)+1] if stage != "river" else "river"

    if isinstance(action, FoldAction):
        return{}
    elif isinstance(action, CheckCallAction):
        if hand_strength == "weak":
            return {
                (next_stage, "weak", pot_size): 0.7,
                (next_stage, "neutral", pot_size): 0.2,
                (next_stage, "strong", pot_size): 0.1,
            }
        elif hand_strength == "neutral":
            return {
                (next_stage, "weak", pot_size): 0.1,
                (next_stage, "neutral", pot_size): 0.6,
                (next_stage, "strong", pot_size): 0.3,
            }
        elif hand_strength == "strong":
            return {
                (next_stage, "weak", pot_size): 0.0,
                (next_stage, "neutral", pot_size): 0.2,
                (next_stage, "strong", pot_size): 0.8,
            }
    elif isinstance(action, BetRaiseAction):
        next_pot_size = {"small": "medium", "medium": "large", "large": "large"}[pot_size]
        if hand_strength == "weak":
            return {
                (next_stage, "weak", next_pot_size): 0.5,
                (next_stage, "neutral", next_pot_size): 0.3,
                (next_stage, "strong", next_pot_size): 0.2,
            }
        elif hand_strength == "neutral":
            return {
                (next_stage, "weak", next_pot_size): 0.1,
                (next_stage, "neutral", next_pot_size): 0.5,
                (next_stage, "strong", next_pot_size): 0.4,
            }
        elif hand_strength == "strong":
            return {
                (next_stage, "weak", next_pot_size): 0.0,
                (next_stage, "neutral", next_pot_size): 0.1,
                (next_stage, "strong", next_pot_size): 0.9,
            }
    else: 
        raise ValueError("Unknown action: {}".format(action))
    
transition_table = {}
for state in states:
    for action in actions.values():
        state_tuple = state.as_tuple()
        next_states = transition_function(state_tuple, action)
        transition_table[(state, action)] = {
            PokerState(*next_state): prob for next_state, prob in next_states.items()
        }

#rewards
def reward_function(state, action):
    stage, hand_strength, pot_size = state

    pot_value = {"small":10, "medium":50, "large":100}[pot_size]
    hand_value = {"weak":-10, "neutral":0, "strong":10}[hand_strength]

    if isinstance(action, FoldAction):
        return -pot_value * 0.2
    elif isinstance(action, CheckCallAction):
        if hand_strength == "weak":
            return -5
        elif hand_strength == "neutral":
            return 0
        elif hand_strength == "strong":
            return 10
    elif isinstance(action, BetRaiseAction):
        if hand_strength == "weak":
            return -20
        elif hand_strength == "neutral":
            return 10
        elif hand_strength == "strong":
            return pot_value * 0.5
    else:
        raise ValueError("Unknown action: {}".format(action))

def final_reward(winning, pot_size):
    pot_value = {"small":10, "medium":50, "large":100}[pot_size]

    if winning: 
        return pot_value
    else: 
        return -pot_value

reward_table = {}
for state in states:
    for action in actions.values():
        reward_table[(state, action)] = reward_function(state.as_tuple(), action)

#observations

#function returns probability distribution over all possible observations 
def observation_function(state, action, next_state):
    stage, hand_strength, pot_size = state.as_tuple()
    next_stage, next_hand_strength, next_pot_size = next_state.as_tuple()

    if isinstance(action, FoldAction):
        return {}
    elif isinstance(action, CheckCallAction):
        opp_action_prob = {
            "opp_fold": 0.2,
            "opp_check": 0.5,
            "opp_bet": 0.3,
        }
    elif isinstance(action, BetRaiseAction):
        opp_action_prob = {
            "opp_fold": 0.4, 
            "opp_check": 0.3,
            "opp_bet": 0.3,
        }
    else: 
        raise ValueError("Unknown action: {}".format(action))

    if next_hand_strength == "strong":
        board_state_prob = {
            "neutral_board": 0.4,
            "strong_board": 0.6,
        }
    elif next_hand_strength == "neutral":
        board_state_prob = {
            "neutral_board": 0.6,
            "strong_board": 0.4,
        }
    elif next_hand_strength == "weak":
        board_state_prob = {
            "neutral_board": 0.8,
            "strong_board": 0.2,
        }

    observed_pot_size = f"{next_pot_size}_pot"

    observations = {}
    for opp_action, opp_prob in opp_action_prob.items():
        for board_state, board_prob in board_state_prob.items():
            obs = (opp_action, board_state, observed_pot_size)
            observations[obs] = opp_prob * board_prob

    return observations

observation_table = {}
for state in states: 
    for action in actions.values():
        for next_state in states: 
            observation_table[(state, action, next_state)] = observation_function(state, action, next_state)

### DEFINING POMDP ENVIRONMENT/AGENT ###

class PokerEnvironment(pomdp_py.Environment):
    def __init__(self, states, transition_table, reward_table):
        self.states = states
        self.transition_table = transition_table
        self.reward_table = reward_table

        initial_state = random.choice(states)
        super().__init__(initial_state)

    def transition(self, state, action):
        return self.transition_table.get((state, action), {})

    def reward(self, state, action):
        return self.reward_table.get((state, action), 0)
    
    def state_transition(self, action):
        transition_probs = self.transition(self.state, action)
        if not transition_probs:
            return None
        
        next_state = random.choices(
            list(transition_probs.keys()),
            weights=list(transition_probs.values())
        )[0]
        
        self.apply_transition(next_state)

poker_env = PokerEnvironment(states, transition_table, reward_table)

### SOLVING THE POMDP ## 

class PokerTransitionModel(pomdp_py.TransitionModel):
    def probability(self, next_state, state, action):
        trans_probs = transition_function(state.as_tuple(), action)
        return trans_probs.get(next_state.as_tuple())

    def sample(self, state, action):
        trans_probs = transition_function(state.as_tuple())
        next_states = list(trans_probs.keys())
        probabilities = list(trans_probs.values())
        return random.choices(next_states, weights=probabilities)[0]

class PokerRewardModel(pomdp_py.RewardModel):
    def sample(self, state, action, next_state):
        return reward_function(state.as_tuple(), action)

class PokerObservationModel(pomdp_py.ObservationModel):
    def probability(self, observation, next_state, action):
        obs_probs = observation_function(state, action, next_state)
        return obs_probs.get(observation, 0)

    def sample(self, next_state, action):
        obs_probs = observation_function(state, action, next_state)
        observations = list(obs_probs.keys())
        probabilities = list(obs_probs.values())
        return random.choices(observations, weights=probabilities)[0]

class PokerPolicyModel(pomdp_py.framework.basics.PolicyModel):
    def __init__(self, actions):
        self._actions = actions

    def sample(self, state):
        return random.choice(list(self._actions.values()))

    def argmax(self, state):
        return "check_call"

    def get_all_actions(self, state=None, history=None):
        print("get_all_actions called")
        return self._actions.values()

class PokerAgent(pomdp_py.Agent):
    def __init__(self, belief, policy_model, transition_model, observation_model, reward_model):
        super().__init__(belief, policy_model, transition_model, observation_model, reward_model)

def create_poker_belief(states, num_particles=100):
    particles = [states[i % len(states)] for i in range(num_particles)]
    belief = pomdp_py.Particles(particles)
    return belief

belief = create_poker_belief(states)
policy_model = PokerPolicyModel(actions)
transition_model = PokerTransitionModel()
reward_model = PokerRewardModel()
observation_model = PokerObservationModel()
agent = PokerAgent(belief, policy_model, transition_model, observation_model, reward_model)
solver = POMCP()

# Simulate one decision-making step
print("calling solver plan")
action = solver.plan(agent)  # Decide action based on belief and model
# print("Action chosen:", action)

# # Environment responds
# poker_env.state_transition(action)
# new_observation = poker_env.observe(agent)
# reward = poker_env.reward(agent.state, action)

# print("New Observation:", new_observation)
# print("Reward received:", reward)

# # Update agent's belief
# agent.update(action, new_observation)