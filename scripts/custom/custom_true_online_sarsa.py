from linear_rl.true_online_sarsa import TrueOnlineSarsaLambda

class TrueOnlineSarsaLambdaDecay(TrueOnlineSarsaLambda):
    def __init__(self, state_space, action_space, basis='fourier', min_max_norm=False, alpha=0.0001, lamb=0.9,
                 gamma=0.99, epsilon=0.05, fourier_order=7, max_non_zero_fourier=2, decay=0.99, min_epsilon=0.01):
        super().__init__(state_space, action_space, basis, min_max_norm, alpha, lamb, gamma, epsilon, fourier_order,
                         max_non_zero_fourier)
        self.decay = decay
        self.min_epsilon = min_epsilon

    def get_action(self, features):
        action = super().get_action(features)
        self.epsilon = max(self.epsilon * self.decay, self.min_epsilon)
        return action
