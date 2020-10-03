# Student-Initiated Action Advising via Advice Novelty

## Requirements
**Dependencies**
- numpy
- opencv-python
- tensorflow=1.13.1  

**Environments**  
- [GridWorld](https://github.com/ercumentilhan/GridWorld) 
- [MinAtar](https://github.com/ercumentilhan/MinAtar/tree/original) 

## Execution

A training process can be started as follows with appropriate argument(s):
```
python action_advising/main.py --experiment-setup <experiment setup>
```

In order to employ an expert agent, the saved model identifier, seed, checkpoint state must be specified in \<ID\> (string), \<SEED\> (string), \<CHECKPOINT\> (integer) fields in constants.py as follows:
```python
EXPERT = {
    'GridWorld': ('NONE', 'NONE', 0),
    'MinAtar-asterix': (<ID>, <SEED>, <CHECKPOINT>),
    ...
}
```


The list of all hyperparameters can be found in `main.py`.

**Values to be passed as --env-name argument:**

- **GridWorld:** 'GridWorld'
- **MinAtar games:** 'MinAtar-asterix', 'MinAtar-breakout', 'MinAtar-freeway', 'MinAtar-seaquest', 'MinAtar-spaceinvaders'

**Format of _experiment-setup_:**  
This an integer argument with three digits **abc** (except for _no advising_ which takes 0) which defines the setup of the experiment in terms of action advising method and action advising budget to be used.

- **a:** Action advising method
  - **1:** Early advising
  - **2:** Uniformly random advising
  - **3:** Uncertainty-based advising
  - **4:** State novelty-based advising
  - **5:** Advice novelty-based advising
- **c:** Budget (the first columun is for GridWorld and the second column is for MinAtar; these are defined in `executor.py`)

| c        | GridWorld Budget           | MinAtar Budget  |
| --- |:-------:|:-------:|
| 0 | 100     | 1000    |
| 1 | 250     | 2500    |
| 2 | 500     | 5000    |
| 3 | 1000    | 10000    |
| 4 | 2500    | 25000    |
| 5 | 5000    | 50000    |
| 6 | 10000    | 100000    |
| 7 | 25000    | 250000    |
| 8 | 50000    | 500000    |
| 9 | 100000    | 1000000    |

**Experiment setups used in the study:** 
- **GridWorld**
  - **No Advising:** 0
  - **Early Advising:** 105, 108
  - **Uniformly Random Advising:** 205, 208
  - **Uncertainty-based Advising:** 305, 308 
  - **State Novelty-based Advising:** 405, 408 
  - **Advice Novelty-based Advising:** 505, 508 
  
- **MinAtar (for each game)**
  - **No Advising:** 0
  - **Early Advising:** 105, 107
  - **Uniformly Random Advising:** 205, 207
  - **Uncertainty-based Advising:** 305, 307
  - **State Novelty-based Advising:** 405, 407 
  - **Advice Novelty-based Advising:** 505, 507 
