import numpy as np
import mdptoolbox
import matplotlib.pyplot as plt
import pomdp_py

#state space
stages = ["preflop", "flop", "turn", "river"]
hand_strengths = ["weak", "neutral", "strong"]
pot_sizes = ["small", "medium", "large"]

states = [
    (stage, hand, pot)
    for stage in stages
    for hand in hand_strengths
    for pot in pot_sizes
]

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

    stage_order = ["preflop", "flop", "turn", "river"]
    next_stage = stage_order[stage_order.index(stage)+1] if stage != "river" else "river"

    if action == "fold":
        return{}
    elif action == "check_call":
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
    elif action == "bet_raise":
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
        transition_table[(state, action)] = transition_function(state, action)

#rewards
def reward_function(state, action):
    stage, hand_strength, pot_size = state

    pot_value = {"small":10, "medium":50, "large":100}[pot_size]
    hand_value = {"weak":-10, "neutral":0, "strong":10}[hand_strength]

    if action == "fold":
        return -pot_value * 0.2
    elif action == "check_call":
        if hand_strength == "weak":
            return -5
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

# Example usage
print(reward_table[(("flop", "weak", "small"), "fold")])  # Accessing a specific reward