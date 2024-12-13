import numpy as np
import random
import matplotlib.pyplot as plt
import pomdp_py

### DEFINING POMDP COMPONENTS ###

#state space
stages = ["preflop", "flop", "turn", "river"]
hand_strengths = ["weak", "neutral", "strong"]
pot_sizes = ["small", "medium", "large"]

states = [
    (stage, hand, pot)
    for stage in stages
    for hand in hand_strengths
    for pot in pot_sizes
] + [("end", "none", "none")]

#action space
actions = {
    0: "fold",
    1: "check_call",
    2: "bet_raise"
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

    if stage == "end":
        return {("end", hand_strength, pot_size): 1.0}
    
    stage_order = ["preflop", "flop", "turn", "river"]
    next_stage = stage_order[stage_order.index(stage)+1] if stage != "river" else "river"

    if action == "fold":
        if hand_strength == "weak":
            return {("end", "none", "none"): 1.0}
    elif action == "check_call": 
        if hand_strength == "weak":
            return {
                (next_stage, "weak", pot_size): 0.6,
                (next_stage, "neutral", pot_size): 0.3,
                (next_stage, "strong", pot_size): 0.1,
            }
        elif hand_strength == "neutral":
            return {
                (next_stage, "weak", pot_size): 0.3,
                (next_stage, "neutral", pot_size): 0.5,
                (next_stage, "strong", pot_size): 0.2,
            }
        elif hand_strength == "strong":
            return {
                (next_stage, "weak", pot_size): 0.1,
                (next_stage, "neutral", pot_size): 0.3,
                (next_stage, "strong", pot_size): 0.6,
            }
    elif action == "bet_raise":
        next_pot_size = {"small": "medium", "medium": "large", "large": "large"}[pot_size]
        if hand_strength == "weak":
            return {
                (next_stage, "weak", next_pot_size): 0.7,
                (next_stage, "neutral", next_pot_size): 0.2,
                (next_stage, "strong", next_pot_size): 0.1,
            }
        elif hand_strength == "neutral":
            return {
                (next_stage, "weak", next_pot_size): 0.2,
                (next_stage, "neutral", next_pot_size): 0.5,
                (next_stage, "strong", next_pot_size): 0.3,
            }
        elif hand_strength == "strong":
            return {
                (next_stage, "weak", next_pot_size): 0.1,
                (next_stage, "neutral", next_pot_size): 0.2,
                (next_stage, "strong", next_pot_size): 0.7,
            }
    else: 
        raise ValueError("Unknown action: {}".format(action))
    
transition_table = {}
for state in states:
    for action in actions.values():
        next_states = transition_function(state, action)
        transition_table[(state, action)] = next_states
#rewards
def reward_function(state, action):
    stage, hand_strength, pot_size = state

    if stage == "end":
        return {}

    pot_value = {"small":10, "medium":50, "large":100}[pot_size]
    hand_value = {"weak":-10, "neutral":0, "strong":10}[hand_strength]

    if action == "fold":
        if hand_strength == "weak":
            return 0
        else:
            return -pot_value * 0.2
    elif action == "check_call":
        if hand_strength == "weak":
            return -10
        elif hand_strength == "neutral":
            return 0
        elif hand_strength == "strong":
            return 10
    elif action == "bet_raise":
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
        reward_table[(state, action)] = reward_function(state, action)

#observations

#function returns probability distribution over all possible observations 
def observation_function(state, action, next_state):
    stage, hand_strength, pot_size = state
    next_stage, next_hand_strength, next_pot_size = next_state

    if stage == "end":
        return {}
   
    if action == "fold":
        return {}
    elif action == "check_call":
        opp_action_prob = {
            "opp_fold": 0.2,
            "opp_check": 0.5,
            "opp_bet": 0.3,
        }
    elif action == "bet_raise":
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
    else:
        board_state_prob = {
            "no_board": 1.0,
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

class PokerState(pomdp_py.State):
    def __init__(self, stage, hand_strength, pot_size):
        self.stage = stage
        self.hand_strength = hand_strength
        self.pot_size = pot_size
    
    def __repr__(self):
        return f"({self.stage}, {self.hand_strength}, {self.pot_size})"


class PokerEnvironment(pomdp_py.Environment):
    def __init__(self, states, transition_table, reward_table):
        self.states = states
        self.transition_table = transition_table
        self.reward_table = reward_table

        initial_state = random.choice(states)
        self_state = PokerState(*initial_state)
        super().__init__(self_state)

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

class BeliefState:
    def __init__(self, states):
        self.beliefs = {state: 1/len(states) for state in states}

    def update(self, action, observation, transition_function, observation_function):
        new_beliefs = {}

        for state in self.beliefs:
            belief = self.beliefs[state]
            transition_probs = transition_function.get((state, action), {})
            observation_prob = observation_function.get((state, observation), 0)
            new_beliefs[state] = belief * sum(transition_probs.get(next_state, 0) * observation_prob for next_state in transition_probs)
        
        total = sum(new_beliefs.values())
        if total > 0:
            for state in new_beliefs:
                new_beliefs[state] /= total
        self.beliefs = new_beliefs

    def most_likely_state(self):
        return max(self.beliefs, key=self.beliefs.get)

poker_env = PokerEnvironment(states, transition_table, reward_table)

def value_iteration(belief_state, max_iterations=100, gamma=0.6, epsilon=0.01):
    V = {state: 0 for state in states}

    for iteration in range(max_iterations):
        delta = 0
        new_V = {}
        for belief in belief_state.beliefs:
            expected_values = []
            for action in actions.values():
                expected_value = 0
                for state, belief_prob in belief_state.beliefs.items():
                    transition_probs = transition_table.get((state, action), {})
                    if transition_probs:
                        for next_state, trans_prob in transition_probs.items():
                            obs_probs = observation_table.get((state, action, next_state), {})
                            for obs, obs_prob in obs_probs.items():
                                reward = reward_table.get((state, action), 0)
                                expected_value += (
                                    belief_prob * trans_prob * obs_prob * (reward + gamma + V.get(next_state, 0))
                                )
                expected_values.append(expected_value)
            
            best_value = max(expected_values)
            new_V[belief] = best_value
            delta = max(delta, abs(V[belief] - best_value))
        
        V.update(new_V)
        if delta < epsilon:
                print(f"Belief Value Iteration converged after {iteration + 1} iterations")
                break
        return V

def extract_policy(belief_state, V, gamma=0.6):
    policy = {}
    for belief in belief_state.beliefs:
        best_action = None
        max_value = float('-inf')

        for action in actions.values():
            expected_value = 0

            for state, belief_prob in belief_state.beliefs.items():
                transition_probs = transition_table.get((state, action), {})
                if transition_probs:
                    for next_state, trans_prob in transition_probs.items():
                        obs_probs = observation_table.get((state, action, next_state), {})
                        for obs, obs_prob in obs_probs.items():
                            reward = reward_table.get((state, action), 0)
                            expected_value += (
                                belief_prob * trans_prob * obs_prob * (reward + gamma * V.get(next_state, 0))
                            )
            
            if expected_value > max_value:
                max_value = expected_value
                best_action = action
            policy[belief] = best_action

        return policy

states = [
    ('preflop', 'weak', 'small'), ('preflop', 'weak', 'medium'), ('preflop', 'weak', 'large'),
    ('preflop', 'neutral', 'small'), ('preflop', 'neutral', 'medium'), ('preflop', 'neutral', 'large'),
    ('preflop', 'strong', 'small'), ('preflop', 'strong', 'medium'), ('preflop', 'strong', 'large'),
    ('flop', 'weak', 'small'), ('flop', 'weak', 'medium'), ('flop', 'weak', 'large'),
    ('flop', 'neutral', 'small'), ('flop', 'neutral', 'medium'), ('flop', 'neutral', 'large'),
    ('flop', 'strong', 'small'), ('flop', 'strong', 'medium'), ('flop', 'strong', 'large'),
    ('turn', 'weak', 'small'), ('turn', 'weak', 'medium'), ('turn', 'weak', 'large'),
    ('turn', 'neutral', 'small'), ('turn', 'neutral', 'medium'), ('turn', 'neutral', 'large'),
    ('turn', 'strong', 'small'), ('turn', 'strong', 'medium'), ('turn', 'strong', 'large'),
    ('river', 'weak', 'small'), ('river', 'weak', 'medium'), ('river', 'weak', 'large'),
    ('river', 'neutral', 'small'), ('river', 'neutral', 'medium'), ('river', 'neutral', 'large'),
    ('river', 'strong', 'small'), ('river', 'strong', 'medium'), ('river', 'strong', 'large'),
    ('end', 'none', 'none')
]

initial_belief_state = BeliefState(states)
belief_values = value_iteration(initial_belief_state)
optimal_policy = extract_policy(initial_belief_state, belief_values)
print(optimal_policy)

# for belief, action in optimal_policy.items():
#     print(f"Belief: {belief}, Action: {action}")