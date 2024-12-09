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

def transition_function(state, action):
    stage, hand_strength, pot_size = state

    stage_order = ["preflop", "flop", "turn", "river"]
    next_stage = stage_order[stage_order.index(stage)+1] if stage != "river" else "river"

    if action == "fold":
        return{}
    elif action == "check_call":
        return {
            (next_stage, hand_strength, pot_size): 0.8,
            (next_stage, "medium", pot_size): 0.2 if hand_strength=="weak" else 0.1, 
            (next_stage, "strong", pot_size): 0.1 if hand_strength!="strong" else 0.0,
        }
    elif action == "bet_raise":
        next_pot_size = {"small": "medium", "medium": "large", "large": "large"}[pot_size]
        return {
            (next_stage, hand_strength, next_pot_size): 0.6,
            (next_stage, "medium", next_pot_size): 0.3 if hand_strength == "weak" else 0.2,
            (next_stage, "strong", next_pot_size): 0.1,
        }
    else: 
        raise ValueError("Unknown action: {}".format(action))
    
transition_table = {}
for state in states:
    for action in actions.values():
        transition_table[(state, action)] = transition_function(state, action)