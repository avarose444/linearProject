import numpy as np
import mdptoolbox
import matplotlib.pyplot as plt
import pomdp_py

#defining state space
stages = ["preflop", "flop", "turn", "river"]
hand_strengths = ["weak", "medium", "strong"]
pot_sizes = ["small", "medium", "large"]

states = [
    (stage, hand, pot)
    for stage in stages
    for hand in hand_strengths
    for pot in pot_sizes
]

#define action space
actions = {
    0: "fold",
    1: "check_call",
    2: "bet_raise"
}

print("Actions: ", actions)
