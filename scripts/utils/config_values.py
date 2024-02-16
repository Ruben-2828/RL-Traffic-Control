
# Frozenset is used to create immutable sets

# Possible metric values for plotting
Metric = frozenset({
    'system_total_stopped',
    'system_total_waiting_time',
    'system_mean_waiting_time',
    'system_mean_speed'
})

# Possible agent types for training models
AgentType = frozenset({
    'FIXED',
    'QL',
    'DQN',
    'SARSA',
    'SARSA_decay'
})

# Possible traffic types to train/test models with
TrafficType = frozenset({
    'low',
    'medium',
    'high'
})
