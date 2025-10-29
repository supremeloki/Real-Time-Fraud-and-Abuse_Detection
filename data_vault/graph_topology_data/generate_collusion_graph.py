import networkx as nx
import pandas as pd
import numpy as np
import uuid
from datetime import datetime, timedelta

def generate_base_graph(num_users=1000, num_drivers=500, num_rides=5000):
    G = nx.Graph()

    for i in range(num_users):
        G.add_node(f"user_{i}", node_type="user", attributes={"created_at": datetime.now() - timedelta(days=np.random.randint(1, 365))})

    for i in range(num_drivers):
        G.add_node(f"driver_{i}", node_type="driver", attributes={"rating": np.random.uniform(3.5, 5.0), "vehicle_type": np.random.choice(["sedan", "motorcycle"])})

    for _ in range(num_rides):
        user = f"user_{np.random.randint(0, num_users)}"
        driver = f"driver_{np.random.randint(0, num_drivers)}"
        ride_id = str(uuid.uuid4())[:8]
        ride_timestamp = datetime.now() - timedelta(minutes=np.random.randint(1, 10000))
        
        G.add_edge(user, driver, edge_type="completed_ride", ride_id=ride_id, timestamp=ride_timestamp,
                   fare=np.random.uniform(30000, 150000), distance=np.random.uniform(1, 20))
    
    return G

def inject_collusion_patterns(G, num_collusion_groups=5, group_size=3):
    collusion_nodes = []

    # Get list of driver and user nodes
    driver_nodes = [n for n, attrs in G.nodes(data=True) if attrs.get("node_type") == "driver"]
    user_nodes = [n for n, attrs in G.nodes(data=True) if attrs.get("node_type") == "user"]

    for i in range(num_collusion_groups):
        colluding_drivers = np.random.choice(driver_nodes, min(group_size, len(driver_nodes)), replace=False).tolist()
        colluding_users = np.random.choice(user_nodes, min(group_size // 2, len(user_nodes)), replace=False).tolist()

        for driver in colluding_drivers:
            G.nodes[driver]["attributes"]["collusion_suspect"] = True
            collusion_nodes.append(driver)
        for user in colluding_users:
            G.nodes[user]["attributes"]["collusion_suspect"] = True
            collusion_nodes.append(user)

        # Create suspicious interactions within the group
        for driver in colluding_drivers:
            for user in colluding_users:
                for _ in range(np.random.randint(5, 15)): # Multiple short rides
                    ride_id = str(uuid.uuid4())[:8]
                    ride_timestamp = datetime.now() - timedelta(minutes=np.random.randint(1, 100))
                    G.add_edge(user, driver, edge_type="collusion_ride", ride_id=ride_id, timestamp=ride_timestamp,
                               fare=np.random.uniform(15000, 25000), distance=np.random.uniform(0.5, 1.5), is_fraud=True)
            
            # Link colluding drivers with each other (e.g., proximity, frequent non-ride interactions)
            for other_driver in colluding_drivers:
                if driver != other_driver and not G.has_edge(driver, other_driver):
                    G.add_edge(driver, other_driver, edge_type="driver_network_interaction", interaction_type="proximity_meet",
                               timestamp=datetime.now() - timedelta(hours=np.random.randint(1, 24)))

    return G, collusion_nodes

if __name__ == "__main__":
    print("Generating base graph...")
    base_graph = generate_base_graph(num_users=5000, num_drivers=2000, num_rides=20000)
    print("Injecting collusion patterns...")
    fraud_graph, fraud_nodes = inject_collusion_patterns(base_graph, num_collusion_groups=10, group_size=5)

    print(f"Generated graph with {fraud_graph.number_of_nodes()} nodes and {fraud_graph.number_of_edges()} edges.")
    print(f"Injected {len(fraud_nodes)} potential fraudster nodes.")

    node_data = []
    for node, attrs in fraud_graph.nodes(data=True):
        node_data.append({
            "node_id": node,
            "node_type": attrs.get("node_type"),
            "is_collusion_suspect": attrs.get("attributes", {}).get("collusion_suspect", False)
        })
    pd.DataFrame(node_data).to_csv("graph_nodes.csv", index=False)

    edge_data = []
    for u, v, attrs in fraud_graph.edges(data=True):
        edge_data.append({
            "source": u,
            "target": v,
            "edge_type": attrs.get("edge_type"),
            "is_fraud_edge": attrs.get("is_fraud", False),
            "timestamp": attrs.get("timestamp")
        })
    pd.DataFrame(edge_data).to_csv("graph_edges.csv", index=False)
    
    print("Graph node and edge data saved to graph_nodes.csv and graph_edges.csv")