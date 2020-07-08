# 696-DS MSFT-UMass Human-Agent Cooperative Reinforcement Learning
Repository for 696DS Microsoft-III Project.

#### Dependencies
1. [Anaconda](https://www.anaconda.com/), Python 3

## Important Files
Shared Autonomy in these files uses a pre-defined Q-table.
#### Results plots
- nqcost_gridworld.py: nqcost algorithm implementation
- shared_autonomy.py: Shared Autonomy algorithm implementation (closest actions not defined)
- shared_modified.py: Modified Shared Autonomy algorithm implementation
- comparison_gridworld.py: compares the three algorithms and plots them (results in paper)

#### Baseline plots
- human_shared.py: using advanced gridworld with 8 directions, create the plots for the baselines. i.e. plots the returns as we increase the constraint. Uses it's own implementation of Shared Autonomy with closest actions defined.
