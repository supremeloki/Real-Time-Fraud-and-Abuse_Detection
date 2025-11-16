import networkx as nx
import pandas as pd
import numpy as np
import logging
import argparse
from pathlib import Path
from src.utils.common_helpers import load_config, setup_logging

logger = setup_logging(__name__)


class GraphFeatureExtractor:
    def __init__(self, config_path: Path, env: str):
        self.config = load_config(config_path, env)
        self.logger = setup_logging(
            "GraphFeatureExtractor", self.config["environment"]["log_level"]
        )
        self.graph_data_path = Path("./data_vault/graph_topology_data/")

    def load_graph_data(self) -> nx.Graph:
        self.logger.info("Loading graph topology data.")
        try:
            nodes_df = pd.read_csv(self.graph_data_path / "graph_nodes.csv")
            edges_df = pd.read_csv(self.graph_data_path / "graph_edges.csv")

            G = nx.Graph()
            for _, row in nodes_df.iterrows():
                G.add_node(
                    row["node_id"],
                    node_type=row["node_type"],
                    is_collusion_suspect=row["is_collusion_suspect"],
                )

            for _, row in edges_df.iterrows():
                G.add_edge(
                    row["source"],
                    row["target"],
                    edge_type=row["edge_type"],
                    is_fraud_edge=row["is_fraud_edge"],
                )

            self.logger.info(
                f"Loaded graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges."
            )
            return G
        except FileNotFoundError:
            self.logger.error(
                "Graph data files not found. Please run data_vault/graph_topology_data/generate_collusion_graph.py first."
            )
            return nx.Graph()
        except Exception as e:
            self.logger.error(f"Error loading graph data: {e}", exc_info=True)
            return nx.Graph()

    def compute_centrality_features(self, G: nx.Graph) -> pd.DataFrame:
        self.logger.info("Computing centrality features.")
        if G.number_of_nodes() == 0:
            return pd.DataFrame()

        try:
            degree_centrality = nx.degree_centrality(G)
            betweenness_centrality = nx.betweenness_centrality(
                G, k=min(1000, G.number_of_nodes())
            )  # Use approximation for large graphs
            closeness_centrality = nx.closeness_centrality(G)

            centrality_df = pd.DataFrame(
                [
                    {
                        "node_id": node,
                        "degree_centrality": degree_centrality.get(node, 0),
                        "betweenness_centrality": betweenness_centrality.get(node, 0),
                        "closeness_centrality": closeness_centrality.get(node, 0),
                    }
                    for node in G.nodes()
                ]
            )
            self.logger.info("Centrality features computed.")
            return centrality_df
