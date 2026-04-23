# %%
# Import library
import numpy as np


# %%
def initialization(problem):

    # Empty Solution Template
    empty_solution = problem['ProblemStructure']

    # Extract RMFS Problem
    CostFunction = problem['CostFunction']
    RepairFunction = problem['RepairFunction']
    VarMin = np.asarray(problem['VarMin'], dtype=np.float32)
    VarMax = np.asarray(problem['VarMax'], dtype=np.float32)
    nVar = problem['nVar']

    max_mode = problem.get('Max/Min', 1) == 1
    init_pop_size = int(problem.get('InitPopSize'))
    if init_pop_size <= 0:
        raise ValueError("InitPopSize must be positive.")

    rng = np.random.default_rng(problem.get('Seed', None))
    shape = (nVar[0], nVar[1])
    eps = 1e-8

    # Seed chaotic state once; then evolve it across candidates.
    chaotic_state = rng.uniform(low=0.0, high=1.0, size=shape).astype(np.float32, copy=False)
    chaotic_state = np.clip(chaotic_state, eps, 1.0).astype(np.float32, copy=False)

    # Create Initial Population
    initialpop = []
    for i in range(init_pop_size):
        candidate = empty_solution.copy()
        position = chaotic_state.copy()
        candidate['position'] = np.clip(position, VarMin, VarMax).astype(np.float32, copy=False)

        # Q and Y are derived from X inside RepairFunction
        candidate['quantity_num'] = np.zeros(shape, dtype=np.int32)
        candidate['compartment_num'] = np.zeros(shape, dtype=np.int32)

        # Repair and evaluate cost
        candidate = RepairFunction(candidate, problem)
        candidate = CostFunction(candidate, problem)
        initialpop.append(candidate)

    initialpop.sort(key=lambda x: x['cost'], reverse=max_mode)
    return initialpop