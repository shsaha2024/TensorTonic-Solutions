import numpy as np
def value_iteration_step(values, transitions, rewards, gamma):
    """
    Perform one step of value iteration and return updated values.
    """
    # Write code here
    vals = rewards+ gamma*np.matmul(transitions,values)
    new_values = np.max(vals, axis=1)
    return new_values.tolist()