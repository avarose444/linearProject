import numpy as np
import mdptoolbox
import matplotlib.pyplot as plt

# 1. Define the MDP Components

# Transition probability matrices (normalized)
P = [
    np.array([[0.8, 0.2, 0.0, 0.0],  # Action: Fold
              [0.6, 0.4, 0.0, 0.0],
              [0.0, 0.7, 0.3, 0.0],
              [0.0, 0.0, 0.5, 0.5]]),

    np.array([[0.7, 0.3, 0.0, 0.0],  # Action: Check
              [0.4, 0.6, 0.0, 0.0],
              [0.0, 0.6, 0.4, 0.0],
              [0.0, 0.0, 0.3, 0.7]]),

    np.array([[0.5, 0.5, 0.0, 0.0],  # Action: Bet
              [0.3, 0.7, 0.0, 0.0],
              [0.0, 0.4, 0.6, 0.0],
              [0.0, 0.0, 0.2, 0.8]])
]

# Normalize transition matrices to ensure valid probabilities
P = [matrix / matrix.sum(axis=1, keepdims=True) for matrix in P]

# Reward matrix
R = np.array([
    [-1, -1, 0, 0],  # Action: Fold
    [0, 0, 1, 1],    # Action: Check
    [2, 2, 3, 3]     # Action: Bet
])

# Transpose R to align with P
R = R.T

# Discount factor
discount = 0.9

# 2. Solve the MDP
try:
    vi = mdptoolbox.mdp.ValueIteration(P, R, discount)
    vi.run()
    # Print the results
    print("Optimal Policy:", vi.policy)
    print("Value Function:", vi.V)
except Exception as e:
    print(f"Error solving MDP: {e}")
    vi = None

# 3. Simulate the Game
def simulate_poker_game(policy, num_rounds=10):
    states = ["pre-flop", "flop", "turn", "river"]
    current_state = 0  # Start at pre-flop
    total_reward = 0

    for _ in range(num_rounds):
        action = policy[current_state]
        print(f"State: {states[current_state]}, Action Taken: {action}")
        
        # Simulate the transition
        try:
            next_state = np.random.choice(
                range(len(states)),
                p=P[action][current_state]
            )
        except Exception as e:
            print(f"Error during transition: {e}")
            break
        
        # Update total reward and move to the next state
        total_reward += R[current_state, action]
        current_state = next_state
    
    print("Total Reward:", total_reward)

if vi:
    simulate_poker_game(vi.policy)

# 4. Visualize Results
if vi:
    states = range(len(vi.V))
    plt.bar(states, vi.V)
    plt.xlabel("States")
    plt.ylabel("Value")
    plt.title("Value Function")
    plt.show()
