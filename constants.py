# Universal
NONE = 0

# DQN
EGREEDY_DQN = 0
NOISYNETS_DQN = 1

# Environment observation
NONSPATIAL = 0
SPATIAL = 1

# Environment type
GRIDWORLD = 0
MINATAR = 1

DQN_TYPE_ABRV = {
    EGREEDY_DQN: 'EG',
    NOISYNETS_DQN: 'NN'
}

# Env. information dictionary:
# Abbreviation, Type, Obs. Type, States are countable, Codebase name,
# Difficulty ramping, Level, Initial difficulty, Timesteps limit

ENV_INFO = {
    'GridWorld': ('GW000', GRIDWORLD, SPATIAL, True, 'gridworld', False, 0, 0, 50),
    'MinAtar-asterix': ('MA00R', MINATAR, SPATIAL, False, 'asterix', False, 0, 0, 1000),
    'MinAtar-breakout': ('MA10R', MINATAR, SPATIAL, False, 'breakout', False, 0, 0, 1000),
    'MinAtar-freeway': ('MA20R', MINATAR, SPATIAL, False, 'freeway', False, 0, 0, 1000),
    'MinAtar-seaquest': ('MA30R', MINATAR, SPATIAL, False, 'seaquest', False, 0, 0, 1000),
    'MinAtar-spaceinvaders': ('MA40R', MINATAR, SPATIAL, False, 'space_invaders', False, 0, 0, 1000),
}

EXPERT = {
    'GridWorld': ('NONE', 'NONE', 0),
    'MinAtar-asterix': ('MA00R_EG_000_000_20200609-083555', '104', 3000000),
    'MinAtar-breakout': ('MA10R_EG_000_001_20200609-083555', '102', 3000000),
    'MinAtar-freeway': ('MA20R_NN_000_000_20200613-095023', '104', 2800000),
    'MinAtar-seaquest': ('MA30R_EG_000_001_20200609-190804', '100', 3000000),
    'MinAtar-spaceinvaders': ('MA40R_EG_000_000_20200613-223415', '100', 3000000),
}

# Precomputed for 5000 observations
RND_MEAN_COEFFS = {
    'GridWorld': [0.01234568, 0.01234568, 0.17283951],
    'MinAtar-asterix': [0.01, 0.01382899, 0.02005002, 0.00622099],
    'MinAtar-breakout': [0.02943401, 0.01, 0.01, 0.29559526],
    'MinAtar-freeway': [0.01, 0.08, 0.0, 0.0, 0.02197401, 0.02906104, 0.02896495],
    'MinAtar-seaquest': [9.9999998e-03, 9.9999998e-03, 8.8000212e-05, 2.7792018e-02, 2.0630020e-03, 1.3262015e-02,
                         5.8560017e-03, 5.4857921e-02, 1.8419992e-03, 1.2393015e-02],
    'MinAtar-spaceinvaders': [0.01, 0.23063892, 0.17967683, 0.05096298, 0.002311, 0.00545999],
}

# Precomputed for 5000 observations
RND_STD_COEFFS = {
    'GridWorld': [0.11042311, 0.11042311, 0.37810846],
    'MinAtar-asterix': [0.09949878, 0.09991241, 0.12762503, 0.05361485],
    'MinAtar-breakout': [0.16885684, 0.09949878, 0.09949878, 0.4562406 ],
    'MinAtar-freeway': [0.09949875, 0.27129334, 0.0, 0.0, 0.13479413, 0.16020493, 0.16085072],
    'MinAtar-seaquest': [0.09949879, 0.09949879, 0.00087559, 0.14389507, 0.01965935, 0.09730721, 0.05139352, 0.21288691,
                         0.01717723, 0.09478367],
    'MinAtar-spaceinvaders': [0.09949875, 0.42100674, 0.32598287, 0.09502289, 0.02252811, 0.05432627],
}