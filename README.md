## Learning Dynamics

- **URL**: https://github.com/Giovanni3A/LearningDynamics
---

### Abstract

Simluation module of different model dynamics applied to simple norm games. Each dynamics is defined by it's differential equation, derived from the model.

### Structure

    ├── README.md                                  
    ├── .gitignore                                 
    ├── requirements.txt                           <- List of necessary packages
    ├── examples                                   <- Examples of use, notebooks and generated data
    ├── LDynamics                                  <- Folder with main scripts
        ├── _dynamic_simulator.py                      <- Simulation method

### Results

As analysis results, we can compare how different methods develop strategies in games. The HawksDoves example uses the Hawks and Doves game (https://en.wikipedia.org/wiki/Chicken_(game)) and show how the methods converge to different states in different ways.

![replicator.png](/examples/HawksDoves/results/imgs/replicator.png)

![ppo.png](/examples/HawksDoves/results/imgs/ppo.png)


#### References

https://arxiv.org/pdf/1906.00190.pdf

https://www.ssc.wisc.edu/~whs/research/egt.pdf