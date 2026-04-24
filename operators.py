# %%
# Import Library
import numpy as np
# %%
def ContinuousCrossover(x1, x2, gamma):
    
    alpha = np.random.uniform(
        -gamma,
        1 + gamma,
        size = (x1.shape[0], x1.shape[1])
    )
    
    y1 = alpha * x1 + (1 - alpha) * x2
    y2 = alpha * x2 + (1 - alpha) * x1
    
    return y1, y2
# %%
# Mutate the genes of offspring
# Adds a random mutation to portion of population in otder to explore new areas of search space 
def ContinuousMutation(x1, problem):
    x1 = x1 + (problem['VarMax'] - problem['VarMin']) * np.random.uniform(low = -1, high = 1, size = (x1.shape[0], x1.shape[1]))
    return x1