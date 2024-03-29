# bc4rl
Bisimulation Critic for Reinforcement Learning

**Usage:**

```
python train.py <algo> <policy> <env> -d <device>
```

**Example:**

```
python train.py bsac BSACMlpPolicy lunarlander -d cuda:0
```