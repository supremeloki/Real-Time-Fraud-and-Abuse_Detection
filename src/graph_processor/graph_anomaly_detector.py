# src/graph_processor/graph_anomaly_detector.py

import networkx as nx
import logging
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple
from collections import defaultdict
from src.data_access.redis_cache_manager import (
    RedisCacheManager,
)  # Assuming RedisCacheManager exists
from src.utils.common_helpers import setup_logging

logger = setup_logging(__name__)


class GraphAnomalyDetector:
    """
    Detects structural and behavioral anomalies within the user-driver interaction graph
    using real-time updates and historical graph snapshots.
    """

    def __init__(
        self,
        redis_config: Dict[str, Any],
        window_hours: int = 24,
        centrality_deviation_threshold: float = 3.0,
        min_nodes_for_community: int = 5,
    ):

        self.redis_client = RedisCacheManager(redis_config)
        self.window_hours = window_hours
        self.centrality_deviation_threshold = centrality_deviation_threshold
        self.min_nodes_for_community = min_nodes_for_community

        # In-memory graph for real-time analysis within the window
        self.current_graph = (
            nx.MultiDiGraph()
        )  # Directed graph for user -> driver interactions, allows multiple edges
        self.node_activity_timestamps: Dict[str, datetime] = (
            {}
        )  # Last active time for nodes

        # Store historical centrality means/stds in Redis for anomaly detection
        self.historical_centrality_key = "graph_metrics:centrality_stats"
        self.historical_centrality: Dict[str, Dict[str, float]] = (
            self.redis_client.get_value(self.historical_centrality_key) or {}
        )

        logger.info(
            f"GraphAnomalyDetector initialized with {window_hours}-hour window."
        )

    def _clean_graph(self, current_time: datetime):
        """Removes old nodes and edges from the in-memory graph."""
        cutoff_time = current_time - timedelta(hours=self.window_hours)

        # Remove old edges
        edges_to_remove = []
        for u, v, data in self.current_graph.edges(data=True):
            if "timestamp" in data and data["timestamp"] < cutoff_time:
                edges_to_remove.append((u, v, data.get("key")))  # For MultiDiGraph

        for u, v, key in edges_to_remove:
            self.current_graph.remove_edge(u, v, key=key)

        # Remove isolated nodes that haven't been active recently
        nodes_to_remove = [
            node
            for node in self.current_graph.nodes()
            if self.current_graph.degree(node) == 0
            and self.node_activity_timestamps.get(node, datetime.min) < cutoff_time
        ]
        for node in nodes_to_remove:
            self.current_graph.remove_node(node)
            if node in self.node_activity_timestamps:
                del self.node_activity_timestamps[node]

        logger.debug(
            f"Graph cleaned. Nodes: {self.current_graph.number_of_nodes()}, Edges: {self.current_graph.number_of_edges()}"
        )

    def update_graph_state(self, event: Dict[str, Any]):
        """Updates the in-memory graph with a new event."""
        event_timestamp = datetime.fromisoformat(event["event_timestamp"])
        user_id = event.get("user_id")
        driver_id = event.get("driver_id")

        if not user_id or not driver_id:
            logger.warning(
                f"Event {event.get('event_id')} missing user_id or driver_id. Skipping graph update."
            )
            return

        # Add nodes if new, update activity timestamp
        self.current_graph.add_node(user_id, type="user")
        self.current_graph.add_node(driver_id, type="driver")
        self.node_activity_timestamps[user_id] = event_timestamp
        self.node_activity_timestamps[driver_id] = event_timestamp

        # Add edge (user -> driver) with timestamp
        self.current_graph.add_edge(
            user_id,
            driver_id,
            timestamp=event_timestamp,
            event_id=event.get("event_id"),
        )

        # Clean up old data in the graph
        self._clean_graph(event_timestamp)
        logger.debug(f"Graph updated with event {event.get('event_id')}.")

    def _calculate_metrics_for_graph(self) -> Dict[str, Dict[str, float]]:
        """Calculates real-time graph metrics (e.g., centrality) for the current graph state."""
        metrics = {"node_degree_centrality": {}, "node_betweenness_centrality": {}}
        if self.current_graph.number_of_nodes() > 1:
            try:
                degrees = self.current_graph.degree()
                for node, degree in degrees:
                    metrics["node_degree_centrality"][node] = degree

                # Betweenness is computationally expensive, only run for smaller graphs or sample
                if self.current_graph.number_of_nodes() < 500:  # Heuristic limit
                    betweenness = nx.betweenness_centrality(self.current_graph)
                    metrics["node_betweenness_centrality"] = betweenness
            except Exception as e:
                logger.error(
                    f"Error calculating graph centralities: {e}", exc_info=True
                )
        return metrics

    def _detect_centrality_anomalies(
        self, current_metrics: Dict[str, Dict[str, float]]
    ) -> Dict[str, Any]:
        """Compares current centrality metrics against historical stats to detect anomalies."""
        anomalies = {}
        for metric_type, node_metrics in current_metrics.items():
            for node, value in node_metrics.items():
                if node in self.historical_centrality:
                    hist_data = self.historical_centrality[node]
                    mean = hist_data.get(f"{metric_type}_mean")
                    std = hist_data.get(f"{metric_type}_std")

                    if mean is not None and std is not None and std > 0:
                        z_score = abs((value - mean) / std)
                        if z_score >= self.centrality_deviation_threshold:
                            anomalies[f"{node}_high_{metric_type}_anomaly"] = {
                                "current_value": value,
                                "historical_mean": mean,
                                "historical_std": std,
                                "z_score": z_score,
                            }
                            logger.warning(
                                f"Anomaly detected for {node} in {metric_type}: z-score={z_score:.2f}"
                            )
        return anomalies

    def _detect_community_anomalies(self) -> Dict[str, Any]:
        """Detects unusual community structures or changes."""
        community_anomalies = {}
        # Requires a non-directed graph for many community detection algorithms
        undirected_graph = self.current_graph.to_undirected()

        if undirected_graph.number_of_nodes() < self.min_nodes_for_community:
            return community_anomalies

        try:
            # Using Louvain method for community detection
            from networkx.algorithms import community

            communities_generator = community.label_propagation_communities(
                undirected_graph
            )
            # communities = tuple(sorted(c) for c in next(communities_generator)) # Get first partitioning

            # In a real system, you'd track community changes over time, e.g.,
            # - A node suddenly joining a very different community
            # - A new, very dense, small community forming rapidly (potential collusion)
            # For this demo, let's just detect if very small, dense communities exist.

            for comm in communities_generator:
                if (
                    len(comm) > 1 and len(comm) < self.min_nodes_for_community * 2
                ):  # Small community
                    subgraph = undirected_graph.subgraph(comm)
                    if subgraph.number_of_edges() > (
                        len(comm) * (len(comm) - 1) / 4
                    ):  # Relatively dense (more than 25% possible edges)
                        community_anomalies[f"dense_small_community_detected"] = list(
                            comm
                        )
                        logger.warning(f"Dense small community detected: {list(comm)}")
                        break  # Only report one for simplicity
        except Exception as e:
            logger.warning(f"Error detecting community anomalies: {e}", exc_info=True)
        return community_anomalies

    def analyze_graph_for_anomalies(self, event_timestamp: datetime) -> Dict[str, Any]:
        """
        Main function to analyze the current graph state for various anomalies.
        """
        self._clean_graph(event_timestamp)  # Ensure graph is up-to-date with window

        anomalies = {}

        # Calculate current graph metrics
        current_metrics = self._calculate_metrics_for_graph()

        # Detect centrality anomalies (requires historical data)
        anomalies.update(self._detect_centrality_anomalies(current_metrics))

        # Detect community anomalies
        anomalies.update(self._detect_community_anomalies())

        # Additional anomaly checks can be added here:
        # - High fan-out/fan-in rate for a node (newly active user/driver making many connections)
        # - Sudden change in the diameter or density of the entire graph

        return anomalies

    def update_historical_centrality_stats(
        self, new_metrics: Dict[str, Dict[str, float]]
    ):
        """
