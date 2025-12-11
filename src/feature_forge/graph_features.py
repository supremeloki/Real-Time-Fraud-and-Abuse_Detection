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
        except Exception as e:
            self.logger.error(
                f"Error computing centrality features: {e}", exc_info=True
            )
            return pd.DataFrame()

    def compute_community_features(self, G: nx.Graph) -> pd.DataFrame:
        self.logger.info("Computing community detection features.")
        if G.number_of_nodes() == 0 or G.number_of_edges() == 0:
            return pd.DataFrame()

        try:
            # Using Louvain method for community detection
            import community as co

            partition = co.best_partition(G)

            community_df = pd.DataFrame(
                [
                    {"node_id": node, "community_id": community_id}
                    for node, community_id in partition.items()
                ]
            )
            self.logger.info("Community detection features computed.")
            return community_df
        except ImportError:
            self.logger.warning(
                "python-louvain not installed. Skipping community detection features. Install with 'pip install python-louvain'"
            )
            return pd.DataFrame()
        except Exception as e:
            self.logger.error(f"Error computing community features: {e}", exc_info=True)
            return pd.DataFrame()

    def combine_graph_features(self, G: nx.Graph) -> pd.DataFrame:
        self.logger.info("Combining all graph features.")

        centrality_df = self.compute_centrality_features(G)
        community_df = self.compute_community_features(G)

        # Merge with node types
        nodes_info = pd.DataFrame(
            [
                {
                    "node_id": node,
                    "node_type": G.nodes[node].get("node_type"),
                    "is_collusion_suspect": G.nodes[node].get(
                        "is_collusion_suspect", False
                    ),
                }
                for node in G.nodes()
            ]
        )

        combined_df = nodes_info.merge(centrality_df, on="node_id", how="left").fillna(
            0
        )
        combined_df = combined_df.merge(community_df, on="node_id", how="left").fillna(
            -1
        )  # -1 for nodes not in any community

        self.logger.info("Graph features combined.")
        return combined_df

    def store_graph_features(self, graph_features_df: pd.DataFrame):
        self.logger.info("Storing graph features.")
        graph_features_df.to_csv(
            self.graph_data_path / "processed_graph_features.csv", index=False
        )
        self.logger.info("Graph features saved to processed_graph_features.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Snapp Graph Feature Extractor")
    parser.add_argument(
        "--env", type=str, default="dev", help="Environment (dev or prod)"
    )
    args = parser.parse_args()

    current_dir = Path(__file__).parent
    project_root = current_dir.parent.parent
    config_directory = project_root / "config"

    extractor = GraphFeatureExtractor(config_directory, args.env)
    graph = extractor.load_graph_data()

    if graph.number_of_nodes() > 0:
        combined_features = extractor.combine_graph_features(graph)
        extractor.store_graph_features(combined_features)
    else:
        logger.warning("No graph data to process for graph features.")
