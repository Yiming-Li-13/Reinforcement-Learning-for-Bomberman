# Reinforcement-Learning-for-Bomberman
Training RL agents to play Bomberman. Compares Double DQN with engineered features vs. Linear Q-Learning under a shared feature/reward design: dense reward shaping, BFS guidance, danger maps, curriculum training, and incremental checkpointing.


# Deep RL Agents for Bomberman

Course final project: training reinforcement learning agents to play the classic **Bomberman** game. The repository contains two agents developed with genuine teamwork and a shared feature/reward design:

| Agent | Method | Owner |
|-------|--------|-------|
| `dqn_agent` | **Double DQN** with 26-dim hand-crafted features | Yiming Li |
| `lq_agent` | **Linear Q-Learning** on engineered features | Ningyun Chen |

Both members jointly designed the feature set and reward scheme, trained, debugged, and evaluated **both** agents.

## Highlights

- **No hard-coded strategy** — action selection is 100% determined by the learned Q-values. The only filter is a *physical legality mask* (walls, crates, bombs, enemies block movement; flames and self-damaging bombs are deliberately *not* filtered — the network must learn to avoid them from rewards).
- **BFS guidance as a feature, not a rule** — a breadth-first search recommends a direction (escape / coin / crate), but the network is free to adopt or ignore it.
- **Curriculum training** — 4 phases from pure coin collection to 4-player melee, with a separate bomb-exploration schedule that is disabled until the agent masters navigation and evasion.
- **Incremental training** — model weights, optimizer state, and replay buffer are persisted every round, so training resumes seamlessly across phases/processes.
- **Dense reward shaping** — balanced opposing rewards (e.g., toward-coin +0.5 / away-coin -0.25), bomb-quality shaping (trapped bomb -12), and a *death-forfeit* rule that annuls all positive loot on the step the agent dies.

## Requirements

- Python 3.10+
- PyTorch
- NumPy

The course-provided `bomberman_rl` framework (with `main.py`, `events.py`, scenarios, and the rule-based agents) is expected in the parent directory of `agent_code`.

## Repository Structure

```
agent_code/
└── dqn_agent3/           # this agent
   ├── callbacks.py       # features, network, action selection
   ├── train.py           # training loop, reward shaping, Double DQN update
   ├── my_saved_model.pt  # trained Q-network weights
   └── train.log          # training log
```
```
lq-agent-code/
├── lq_agent-t1/           # Linear Q-Learning agent
│   ├── callbacks.py       # features, network, action selection
│   ├── train.py           # training loop, reward shaping
│   ├── lq-model-t1.pt     # trained Q-network weights
│   └── lq_agent-t1.log    # training log
├── lq_agent-t2/            
│   ├── callbacks.py      
│   ├── train.py           
│   ├── lq-model-t2.pt 
│   └── lq_agent-t2.log
├── lq_agent-t3/            
│   ├── callbacks.py       
│   ├── train.py          
│   ├── lq-model-t3.pt  
│   └── lq_agent-t3.log
├── lq_agent-final/          
│   ├── callbacks.py    
│   ├── train.py         
│   ├── lq-model-final.pt  
│   └── lq_agent.log   
└── eval_data/             # Evaluation Results (CSV) of DQN agent and Linear Q-Learning agent
```
## Training (DQN agent)

Training follows a 4-phase curriculum (run from the framework root, `--no-gui` for speed):

```bash
# Phase 1: coin collection on an empty board
python main.py play --train 1 --agents dqn_agent3 --scenario coin-heaven --no-gui --n-rounds 400

# Phase 2: crate destruction + evasion
python main.py play --train 1 --agents dqn_agent3 --scenario loot-crate --no-gui --n-rounds 600

# Phase 3: 1v1 combat against the rule-based agent
python main.py play --train 1 --agents dqn_agent3 rule_based_agent --scenario classic --no-gui --n-rounds 800

# Phase 4: full 4-player melee
python main.py play --train 1 --agents dqn_agent3 rule_based_agent rule_based_agent rule_based_agent --scenario classic --no-gui --n-rounds 1000
```

Each phase is an independent process; because all state is checkpointed to disk, later phases automatically continue from the previous ones. Only deleting the checkpoint files (`my_saved_model.pt`, `trainer_state.pt`, `replay_buffer.pt`) triggers a cold restart.

## Training (Linear Q-Learning agent)

Training follows a 4-phase curriculum:

```bash
# Phase 1: coin collection on an empty board
python main.py play --train 1 --agents lq_agent-t1 --scenario coin-heaven --no-gui --n-rounds 500

# Phase 2: crate destruction + evasion
python main.py play --train 1 --agents lq_agent-t2 --scenario coin-heaven --no-gui --n-rounds 1000

# Phase 3: combat against the peaceful_agent and coin_collector_agent
python main.py play --train 1 --agents lq_agent-t3 peaceful_agent coin_collector_agent --scenario classic --no-gui --n-rounds 1500 

# Phase 4: full 4-player melee
python main.py play --train 1 --agents lq_agent-final rule_based_agent rule_based_agent rule_based_agent --no-gui --n-rounds 2000
```

Each phase adds features based on the previous phase to improve training effectiveness.


### Key hyperparameters

| Parameter | Value |
|-----------|-------|
| Network | MLP 26 → 128 → 128 → 64 → 6 |
| Optimizer | Adam, lr = 5e-5 |
| Discount γ | 0.95 |
| Replay buffer | 100,000 transitions, batch size 128 |
| Target network | hard update every 1,000 gradient steps |
| Loss | Huber (Smooth L1), gradient clip 1.0 |
| Exploration | ε-greedy: 1.0 → 0.02 (τ = 25k steps); separate bomb ε enabled after 58k steps |
| Updates | 2 gradient steps per environment step |

## Playing / Evaluation

```bash
# Watch the agent in the GUI
python main.py play --agents dqn_agent3 rule_based_agent rule_based_agent rule_based_agent --scenario classic --n-rounds 3 --update-interval 0.08
python main.py play --agents lq_agent-final rule_based_agent rule_based_agent rule_based_agent --n-rounds 10

# Fast headless evaluation
python main.py play --agents dqn_agent3 rule_based_agent rule_based_agent rule_based_agent --scenario classic --n-rounds 100 --no-gui
python main.py play --agents lq_agent-final rule_based_agent rule_based_agent rule_based_agent --scenario classic --n-rounds 100 --no-gui
```

## Feature Engineering (DQN agent)

The raw `game_state` is compressed into a 26-dimensional vector covering four groups:

- **Danger perception** — in-danger flag, per-direction danger urgency from a bomb blast countdown map, per-direction passability, safe-neighbour ratio
- **Navigation** — BFS recommended direction (escape > coin > crate-adjacent), normalized offset to the nearest coin
- **Combat** — offset/distance to nearest enemy, adjacent crate density, bomb availability, bomb-escape feasibility (BFS ≤ 3 steps), enemy-in-blast flag
- **Global state** — step progress, visible coin ratio, surviving enemy ratio


## Feature Engineering (Linear Q-learning agent)

The raw `game_state` is compressed into a 36-dimensional vector covering five groups:

* **Danger & escape** — in-danger flag, safe escape direction
* **Navigation** — BFS directions towards coins, crates, and nearby enemies
* **Combat** — bomb availability, enemy-in-blast flag, adjacent enemy positions
* **Local map** — surrounding walls and adjacent crates
* **Movement** — previous movement direction and loop detection


## Authors

- **Yiming Li** — Double DQN agent (features, training pipeline, experiments)
- **Ningyun Chen** — Linear Q-Learning agent (feature design, training loop)

See `report.pdf` for the full description of methods, training procedure, and experimental results.
