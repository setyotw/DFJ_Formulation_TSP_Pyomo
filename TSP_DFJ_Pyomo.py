## Problem: Traveling salesman problem (DFJ Formulation)
## Solver: Gurobi
## Language: Python (written in Pyomo)
## Written by: @setyotw (inspired from http://www.opl.ufc.br/post/tsp/)
## Date: September 1st, 2022

#%% import packages
import pyomo.environ as pyo
import numpy as np
from itertools import chain, combinations

#%%%%%%%%%%%%%%%%%%%%%%%%%%
#  DEVELOPMENT PARTS
#%% define powerset for building the DFJ subtour elimination constraints
# powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
def powerset(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

#%%
def TSP_DFJ_Formulation(n, costMatrix):
    #%
    # 1 | initialize sets and notations
    N0 = [i for i in range(0,n+1)]
    N = [i for i in range(1,n)]
    arc_IJ = [(i,j) for i in N0 for j in N0 if i!=j]
    c = {(i,j) : costMatrix[i-1][j-1] for (i,j) in arc_IJ}

    # 2 | initialize the model
    model = pyo.ConcreteModel()

    # 3 | initialize decision variables
    model.x = pyo.Var(arc_IJ, within=pyo.Binary)

    # 4 | define objective function
    model.objective = pyo.Objective(
        expr= sum(model.x[i,j]*c[i,j] for (i,j) in arc_IJ),
        sense=pyo.minimize)

    # 5 | define constraints
    model.constraints = pyo.ConstraintList()

    # a) each node is visited only once (restrict the inbound arc to 1)
    for j in N:
        model.constraints.add(sum([model.x[i,j] for i in N0 if i!=j]) == 1)

    # b) each node is visited only once (restrict the outbound arc to 1)
    for i in N:
        model.constraints.add(sum([model.x[i,j] for j in N0 if i!=j]) == 1)

    # c) subtour elimination constraints
    SubtourSet = list(powerset(N0))[1:-1]
    for S in SubtourSet:
        if len(S) >= 2:
            model.constraints.add(sum([model.x[i,j] for i in S for j in S if i!=j]) <= len(S)-1)

    # 6 | call the solver (we use Gurobi here, but you can use other solvers i.e. PuLP or CPLEX)
    model.pprint()
    solver = pyo.SolverFactory('gurobi')
    completeResults = solver.solve(model,tee = True)

    # 7 | extract the results
    solutionObjective = model.objective()
    tourRepo = []
    for i in model.x:
        if model.x[i].value > 0:
            tourRepo.append((i, model.x[i].value))
            print(str(model.x[i]), model.x[i].value)
    solutionGap = (completeResults.Problem._list[0]['Upper bound'] - completeResults.Problem._list[0]['Lower bound']) / completeResults.Problem._list[0]['Upper bound']
    runtimeCount = completeResults.Solver._list[0]['Time']

    return solutionObjective, solutionGap, tourRepo, runtimeCount, completeResults

#%%%%%%%%%%%%%%%%%%%%%%%%%%
#  IMPLEMENTATION PARTS
#%% input problem instance
# a simple TSP case with 1 depot and 10 customer nodes

# symmetric distance matrix [11 x 11]
costMatrix = np.array([
[0.000, 2.768, 7.525, 9.689, 28.045, 36.075, 25.754, 3.713, 2.701, 8.286, 7.944],
[2.768, 0.000, 9.164, 11.482, 27.779, 37.431, 25.488, 7.305, 1.928, 7.911, 10.758],
[7.525, 9.164, 0.000, 16.297, 33.406, 42.246, 31.115, 4.065, 8.062, 13.970, 1.594],
[9.689, 11.482, 16.297, 0.000, 19.053, 27.198, 16.762, 12.484, 8.810, 17.182, 20.464],
[28.045, 27.779, 33.406, 19.053, 0.000, 10.138, 5.596, 31.349, 26.286, 36.994, 38.820],
[36.075, 37.431, 42.246, 27.198, 10.138, 0.000, 10.634, 38.434, 34.759, 43.132, 46.850],
[25.754, 25.488, 31.115, 16.762, 5.596, 10.634, 0.000, 29.058, 23.995, 32.498, 36.529],
[3.713, 7.305, 4.065, 12.484, 31.349, 38.434, 29.058, 0.000, 6.005, 8.615, 5.651],
[2.701, 1.928, 8.062, 8.810, 26.286, 34.759, 23.995, 6.005, 0.000, 9.862, 9.656],
[8.286, 7.911, 13.970, 17.182, 36.994, 43.132, 32.498, 8.615, 9.862, 0.000, 15.313],
[7.944, 10.758, 1.594, 20.464, 38.820, 46.850, 36.529, 5.651, 9.656, 15.313, 0.000]])

# number of nodes on the graph, can be calculated as the horizontal/vertical length of the distance matrix
n = len(costMatrix[0:])

#%% implement the mathematical formulation
# solutionObjective --> best objective value found by the solver
# solutionGap --> solution gap, (UB-LB)/UB
# tourRepo --> return the active variables
# runtimeCount --> return the runtime in seconds
# completeResults --> return the complete results storage
solutionObjective, solutionGap, tourRepo, runtimeCount, completeResults = TSP_DFJ_Formulation(n, costMatrix)


# %%
