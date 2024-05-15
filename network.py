import networkx as nx
import numpy as np

def calculate_head_loss(flow_rate, diameter, length, roughness, fluid_density, fluid_viscosity):
    area = np.pi * (diameter / 2)**2
    velocity = flow_rate / area
    reynolds_number = (fluid_density * velocity * diameter) / fluid_viscosity
    if reynolds_number < 2300:
        friction_factor = 64 / reynolds_number
    else:
        friction_factor = (0.25 / (np.log10(roughness / (3.7 * diameter) + 5.74 / reynolds_number**0.9)))**2
    head_loss = (friction_factor * (length / diameter) * (velocity**2) / (2 * 9.81)) # g = 9.81 m/s^2
    return head_loss

def simulate_pipe_network(graph, fluid_density, fluid_viscosity, boundary_conditions, tolerance=1e-6, max_iterations=1000):
    flow_rates = {edge: 1.0 for edge in graph.edges()}  # Initial guess
    pressures = {node: None for node in graph.nodes()}  # Initial pressures unknown

    # Apply initial pressures from boundary conditions
    for node, value in boundary_conditions.items():
        pressures[node] = value

    converged = False
    iteration = 0

    while not converged and iteration < max_iterations:
        max_error = 0

        # Update pressures at nodes not in boundary conditions based on current flow rates
        for node in pressures:
            if pressures[node] is None:
                connected_edges = graph.edges(node, data=True)
                pressure_sum = 0
                count = 0
                for _, neighbor, data in connected_edges:
                    if pressures[neighbor] is not None:  # Neighbor has a defined pressure
                        flow = flow_rates[(node, neighbor) if (node, neighbor) in flow_rates else (neighbor, node)]
                        head_loss = calculate_head_loss(flow, data['diameter'], data['length'], data['roughness'], fluid_density, fluid_viscosity)
                        pressure_sum += pressures[neighbor] + head_loss if flow > 0 else pressures[neighbor] - head_loss
                        count += 1
                if count > 0:
                    pressures[node] = pressure_sum / count  # Average of neighboring pressures adjusted for head loss

        # Update flow rates and check for convergence
        for edge in graph.edges():
            start_node, end_node = edge
            diameter = graph.edges[edge]['diameter']
            length = graph.edges[edge]['length']
            roughness = graph.edges[edge]['roughness']
            
            head_loss = calculate_head_loss(flow_rates[edge], diameter, length, roughness, fluid_density, fluid_viscosity)
            pressure_difference = pressures[start_node] - pressures[end_node]
            new_flow_rate = (pressure_difference - head_loss) / length if length else 0

            error = abs(new_flow_rate - flow_rates[edge])
            max_error = max(max_error, error)

            flow_rates[edge] = new_flow_rate

        if max_error < tolerance:
            converged = True

        iteration += 1

    if not converged:
        raise RuntimeError("Failed to converge after {} iterations".format(max_iterations))

    return flow_rates, pressures

# Example usage
G = nx.Graph()
G.add_edge('A', 'B', diameter=0.1, length=100, roughness=0.01)
G.add_edge('B', 'C', diameter=0.1, length=100, roughness=0.01)
G.add_edge('C', 'D', diameter=0.1, length=100, roughness=0.01)
G.add_edge('A', 'D', diameter=0.1, length=100, roughness=0.01)

fluid_density = 1000  # kg/m^3
fluid_viscosity = 0.001  # Pa·s
boundary_conditions = {'A': 100, 'D': 0}  # Pressures at nodes A and D

flow_rates, pressures = simulate_pipe_network(G, fluid_density, fluid_viscosity, boundary_conditions)

print("Flow Rates:", flow_rates)
print("Pressures:", pressures)
