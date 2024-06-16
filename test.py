import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linprog
import time
from optimization import transportation_simplex_method, get_total_cost

def compare_solvers(costs, supply, demand):
    # Convert costs, supply, demand to numpy arrays
    costs = np.array(costs)
    supply = np.array(supply)
    demand = np.array(demand)
    
    # Number of sources and destinations
    num_sources, num_destinations = costs.shape
    
    # Solve using the Transportation Simplex Method
    start_time = time.time()
    simplex_solution = transportation_simplex_method(supply, demand, costs)
    simplex_total_cost = get_total_cost(costs, simplex_solution)
    simplex_time = time.time() - start_time
    
    # Solve using Scipy's linprog (interior point method)
    c = costs.flatten()
    A_eq = []
    
    for i in range(num_sources):
        row = np.zeros(num_sources * num_destinations)
        row[i * num_destinations:(i + 1) * num_destinations] = 1
        A_eq.append(row)
    
    for j in range(num_destinations):
        col = np.zeros(num_sources * num_destinations)
        col[j::num_destinations] = 1
        A_eq.append(col)
    
    A_eq = np.array(A_eq)
    b_eq = np.concatenate([supply, demand])
    bounds = [(0, None)] * (num_sources * num_destinations)
    
    start_time = time.time()
    result = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
    linprog_solution = result.x.reshape(num_sources, num_destinations)
    linprog_total_cost = get_total_cost(costs, linprog_solution)
    linprog_time = time.time() - start_time
    
    return simplex_solution, simplex_total_cost, simplex_time, linprog_solution, linprog_total_cost, linprog_time

def plot_results(simplex_solution, linprog_solution):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(simplex_solution, cmap='Blues')
    axes[0].set_title('Our Solution')
    for (i, j), val in np.ndenumerate(simplex_solution):
        axes[0].text(j, i, f'{val:.1f}', ha='center', va='center', color='red')
    
    axes[1].imshow(linprog_solution, cmap='Blues')
    axes[1].set_title('Linprog Solution')
    for (i, j), val in np.ndenumerate(linprog_solution):
        axes[1].text(j, i, f'{val:.1f}', ha='center', va='center', color='red')
    
    plt.tight_layout()
    plt.show()

def main():
    costs = [
        [8, 6, 10, 9],
        [9, 12, 13, 7],
        [14, 9, 16, 5]
    ]
    supply = [35, 50, 40]
    demand = [45, 20, 30, 30]

    simplex_solution, simplex_total_cost, simplex_time, linprog_solution, linprog_total_cost, linprog_time = compare_solvers(costs, supply, demand)
    
    print("Our Method Solution:")
    print(simplex_solution)
    print("Our Method Total Cost:", simplex_total_cost)
    print("Our Method Computation Time:", simplex_time)
    
    print("\nLinprog Method Solution:")
    print(linprog_solution)
    print("Linprog Method Total Cost:", linprog_total_cost)
    print("Linprog Method Computation Time:", linprog_time)

    plot_results(simplex_solution, linprog_solution)

if __name__ == "__main__":
    main()
