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
        return {}
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
            return 1
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

reward_table = {}
for state in states:
    for action in actions.values():
        reward_table[(state, action)] = reward_function(state, action)

#observations

#function returns probability distribution over all possible observations 
def observation_function(state, action, next_state):
    stage, hand_strength, pot_size = state
    next_stage, next_hand_strength, next_pot_size = next_state
   
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
    
    def __eq__(self, other):
        if not isinstance(other, PokerState):
            return False
        return (self.stage == other.stage and
                self.hand_strength == other.hand_strength and
                self.pot_size == other.pot_size)

    def __hash__(self):
        return hash((self.stage, self.hand_strength, self.pot_size))
    
    def update_state(self, new_state):
        self._state = new_state


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

def value_iteration(belief_state, max_iterations=100, gamma=0.8, epsilon=0.01):
    V = {belief: 0 for belief in belief_state.beliefs}

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
                                    belief_prob * trans_prob * obs_prob * (reward + gamma * V.get(next_state, 0))
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
    ('river', 'strong', 'small'), ('river', 'strong', 'medium'), ('river', 'strong', 'large')
]

initial_belief_state = BeliefState(states)
belief_values = value_iteration(initial_belief_state)
optimal_policy = extract_policy(initial_belief_state, belief_values)

for belief in optimal_policy:
    if belief == ('preflop', 'weak', 'small') or belief == ('preflop', 'weak', 'medium') or belief == ('preflop', 'weak', 'large'):
        optimal_policy[belief] = "fold"
    if belief == ('preflop', 'neutral', 'small') or belief == ('preflop', 'neutral', 'medium') or belief == ('preflop', 'neutral', 'large'):
        optimal_policy[belief] = "check_call"
    if belief == ('flop', 'weak', 'small') or belief == ('flop', 'weak', 'medium') or belief == ('flop', 'weak', 'large'):
        optimal_policy[belief] = "fold"
    if belief == ('flop', 'neutral', 'small') or belief == ('flop', 'neutral', 'medium') or belief == ('flop', 'neutral', 'large'):
        optimal_policy[belief] = "check_call"
    if belief == ('turn', 'weak', 'small') or belief == ('turn', 'weak', 'medium') or belief == ('turn', 'weak', 'large'):
        optimal_policy[belief] = "fold"
    if belief == ('turn', 'neutral', 'small') or belief == ('turn', 'neutral', 'medium') or belief == ('turn', 'neutral', 'large'):
        optimal_policy[belief] = "check_call"
    if belief == ('river', 'weak', 'small') or belief == ('river', 'weak', 'medium') or belief == ('river', 'weak', 'large'):
        optimal_policy[belief] = "fold"
    if belief == ('river', 'neutral', 'small') or belief == ('river', 'neutral', 'medium') or belief == ('river', 'neutral', 'large'):
        optimal_policy[belief] = "check_call"

for state, action in optimal_policy.items():
    reward = reward_function(belief, action)
    print(f"State: {state}, Action: {action}, Reward: {reward}")

import matplotlib.pyplot as plt

# Group rewards by stage
stages = ["preflop", "flop", "turn", "river"]
rewards_by_stage = {stage: [] for stage in stages}

for state, action in optimal_policy.items():
    stage, _, _ = state  # Extract the stage
    reward = reward_function(state, action)  # Get the reward
    rewards_by_stage[stage].append(reward)

# Calculate average reward per stage
avg_rewards = {stage: sum(rewards) / len(rewards) for stage, rewards in rewards_by_stage.items()}

# Bar chart
plt.figure(figsize=(8, 6))
plt.bar(avg_rewards.keys(), avg_rewards.values(), color='skyblue')
plt.title("Average Reward by Stage", fontsize=16)
plt.xlabel("Stage", fontsize=14)
plt.ylabel("Average Reward", fontsize=14)
plt.show()


def plot_belief_updates(initial_belief, actions_taken, observations_received, transition_function, observation_function):
    beliefs = [initial_belief.beliefs.copy()]
    current_belief = initial_belief

    for action, observation in zip(actions_taken, observations_received):
        current_belief.update(action, observation, transition_function, observation_function)
        beliefs.append(current_belief.beliefs.copy())

    # Plot belief evolution
    states = list(initial_belief.beliefs.keys())
    for state in states:
        belief_values = [b[state] for b in beliefs]
        plt.plot(range(len(beliefs)), belief_values, label=str(state))

    plt.xlabel('Step')
    plt.ylabel('Belief Probability')
    plt.title('Belief State Evolution')
    plt.legend()
    plt.show()

# Example usage:
initial_belief = BeliefState(states)
actions_taken = ["check_call", "bet_raise"]
observations_received = [("opp_check", "neutral_board", "small_pot"), ("opp_bet", "strong_board", "medium_pot")]
plot_belief_updates(initial_belief, actions_taken, observations_received, transition_table, observation_table)

mock_policy = {
    ('preflop', 'weak', 'small'): 'fold',
    ('preflop', 'weak', 'medium'): 'fold',
    ('preflop', 'neutral', 'small'): 'check_call',
    ('preflop', 'strong', 'large'): 'bet_raise',
    ('flop', 'neutral', 'medium'): 'check_call',
    ('turn', 'strong', 'large'): 'bet_raise',
    # Add more states as needed
}

# Map actions to numerical values for visualization
action_to_num = {'fold': 0, 'check_call': 1, 'bet_raise': 2}
states = list(mock_policy.keys())
actions = [action_to_num[mock_policy[state]] for state in states]

# Create heatmap-friendly data
stage_idx = {stage: i for i, stage in enumerate(stages)}
hand_idx = {hand: i for i, hand in enumerate(hand_strengths)}
pot_idx = {pot: i for i, pot in enumerate(pot_sizes)}

heatmap = np.full((len(stages), len(hand_strengths), len(pot_sizes)), -1)

for state, action in zip(states, actions):
    stage, hand, pot = state
    heatmap[stage_idx[stage], hand_idx[hand], pot_idx[pot]] = action

# Plot the heatmap for one stage as a 2D grid
plt.figure(figsize=(8, 6))
for stage in stages:
    stage_id = stage_idx[stage]
    plt.matshow(heatmap[stage_id], cmap="coolwarm", fignum=False)
    plt.colorbar(label="Action (0=Fold, 1=Check/Call, 2=Bet/Raise)")
    plt.title(f"Optimal Actions for {stage.capitalize()} Stage")
    plt.xlabel("Hand Strength")
    plt.ylabel("Pot Size")
    plt.xticks(range(len(hand_strengths)), hand_strengths)
    plt.yticks(range(len(pot_sizes)), pot_sizes)
    plt.show()





def simulate_poker_game(poker_env, initial_belief_state, policy, num_episodes=100):
    total_rewards = 0
    game_results = []

    for episode in range(num_episodes):
        poker_env = PokerEnvironment(states, transition_table, reward_table)
        belief_state = BeliefState(states)

        episode_reward = 0
        game_over = False

        print(f"--- Episode {episode + 1} ---")
        while not game_over:
            belief = belief_state.most_likely_state()
            action = policy.get(belief, "check_call")  
            
            print(f"Agent action: {action}")
            
            current_state = poker_env.state
            next_state = poker_env.state_transition(action)
            reward = poker_env.reward(current_state, action)

            episode_reward += reward

            if action == "fold" or (current_state.stage == "river" and next_state.stage == "river"):
                game_over = True

            if next_state:
                obs_probs = observation_function(current_state, action, next_state)
                observation = random.choices(list(obs_probs.keys()), weights=list(obs_probs.values()))[0]
                print(f"Observation: {observation}")

                belief_state.update(action, observation, transition_table, observation_table)

            if next_state:
                state.update(next_state)

        total_rewards += episode_reward
        game_results.append(episode_reward)
        print(f"Episode Reward: {episode_reward}\n")

    average_reward = total_rewards / num_episodes
    print(f"Average Reward over {num_episodes} episodes: {average_reward}")
    return game_results, average_reward


game_results, avg_reward = simulate_poker_game(poker_env, initial_belief_state, optimal_policy, num_episodes=100)
