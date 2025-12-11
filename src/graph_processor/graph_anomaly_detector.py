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
        Periodically updates the historical mean and std dev for centrality measures
        based on the current graph state. This would typically be a scheduled batch job.
        """
        # This function would be called by a separate batch process after a period
        # to update the `historical_centrality` in Redis.
        # For simplicity, we just update it in memory for demo.

        for metric_type, node_metrics in new_metrics.items():
            for node, value in node_metrics.items():
                if node not in self.historical_centrality:
                    self.historical_centrality[node] = {
                        f"{metric_type}_history": [value],
                        f"{metric_type}_mean": value,
                        f"{metric_type}_std": 0.0,
                    }
                else:
                    history = self.historical_centrality[node].get(
                        f"{metric_type}_history", []
                    )
                    history.append(value)
                    # Keep history bounded for calculation
                    if len(history) > 100:  # Max 100 points for stats
                        history = history[-100:]

                    self.historical_centrality[node][f"{metric_type}_history"] = history
                    self.historical_centrality[node][f"{metric_type}_mean"] = np.mean(
                        history
                    )
                    self.historical_centrality[node][f"{metric_type}_std"] = np.std(
                        history
                    )

        self.redis_client.set_value(
            self.historical_centrality_key, self.historical_centrality
        )
        logger.info("Historical centrality stats updated in Redis.")


if __name__ == "__main__":
    import json

    # Dummy Redis configuration for local testing
    redis_conf_demo = {
        "redis_host": "localhost",
        "redis_port": 6379,
        "redis_db": 5,  # Dedicated DB for this demo
        "default_ttl_seconds": 3600,
    }

    detector = GraphAnomalyDetector(
        redis_conf_demo, window_hours=1, centrality_deviation_threshold=2.0
    )

    try:
        detector.redis_client.check_connection()
    except Exception as e:
        print(
            f"Could not connect to Redis: {e}. Please ensure Redis is running on localhost:6379."
        )
        exit()

    current_time = datetime.now()

    # Simulate initial events to build up graph and historical stats
    print("--- Building Initial Graph and Historical Stats ---")
    for i in range(10):
        event = {
            "event_id": f"initial_e{i}",
            "event_timestamp": (
                current_time - timedelta(minutes=5 * (9 - i))
            ).isoformat(),
            "user_id": f"user_{i%3}",
            "driver_id": f"driver_{i%2}",
        }
        detector.update_graph_state(event)

    # Manually update historical stats (normally a background job)
    detector.update_historical_centrality_stats(detector._calculate_metrics_for_graph())
    print(
        f"Initial historical centrality stats: {json.dumps(detector.historical_centrality, indent=2)}"
    )

    # Scenario 1: Normal event, no anomalies
    print("\n--- Scenario 1: Normal Event ---")
    normal_event = {
        "event_id": "normal_e1",
        "event_timestamp": (current_time + timedelta(minutes=1)).isoformat(),
        "user_id": "user_0",
        "driver_id": "driver_1",
    }
    detector.update_graph_state(normal_event)
    anomalies_1 = detector.analyze_graph_for_anomalies(
        datetime.fromisoformat(normal_event["event_timestamp"])
    )
    print(f"Anomalies detected: {json.dumps(anomalies_1, indent=2)}")

    # Scenario 2: High degree centrality spike for a user
    print("\n--- Scenario 2: High Degree Centrality Spike ---")
    spike_user_id = "user_spike_A"
    for i in range(5):
        event = {
            "event_id": f"spike_e{i}",
            "event_timestamp": (current_time + timedelta(minutes=10 + i)).isoformat(),
            "user_id": spike_user_id,
            "driver_id": f"driver_new_{i}",
        }
        detector.update_graph_state(event)

    anomalies_2 = detector.analyze_graph_for_anomalies(
        datetime.fromisoformat(event["event_timestamp"])
    )
    print(
        f"Anomalies detected (User {spike_user_id} now in graph): {json.dumps(anomalies_2, indent=2)}"
    )

    # To see a Z-score anomaly, 'user_spike_A' needs a historical mean and a low std_dev first,
    # then a sudden spike. For this demo, it would be 'new_node_high_degree' behavior, not Z-score.
    # To demonstrate Z-score, we'd need to simulate historical data for 'user_spike_A' first.
    # For now, if it appears as a new node with high degree, it implicitly indicates an anomaly.

    # Scenario 3: Potential collusion (dense small community forming)
    print("\n--- Scenario 3: Potential Collusion (Dense Small Community) ---")
    collusion_time = current_time + timedelta(minutes=30)
    collusion_users = ["coll_u1", "coll_u2"]
    collusion_drivers = ["coll_d1", "coll_d2"]

    events_collusion = [
        {
            "event_id": "coll_e1",
            "event_timestamp": (collusion_time + timedelta(minutes=1)).isoformat(),
            "user_id": collusion_users[0],
            "driver_id": collusion_drivers[0],
        },
        {
            "event_id": "coll_e2",
            "event_timestamp": (collusion_time + timedelta(minutes=2)).isoformat(),
            "user_id": collusion_users[1],
            "driver_id": collusion_drivers[0],
        },
        {
            "event_id": "coll_e3",
            "event_timestamp": (collusion_time + timedelta(minutes=3)).isoformat(),
            "user_id": collusion_users[0],
            "driver_id": collusion_drivers[1],
        },
        {
            "event_id": "coll_e4",
            "event_timestamp": (collusion_time + timedelta(minutes=4)).isoformat(),
            "user_id": collusion_users[1],
            "driver_id": collusion_drivers[1],
        },
    ]
    for event in events_collusion:
        detector.update_graph_state(event)

    anomalies_3 = detector.analyze_graph_for_anomalies(
        datetime.fromisoformat(events_collusion[-1]["event_timestamp"])
    )
    print(
        f"Anomalies detected (Collusion attempt): {json.dumps(anomalies_3, indent=2)}"
    )

    # Clean up Redis
    detector.redis_client.delete_key(detector.historical_centrality_key)
    print(f"\nCleaned up Redis DB {redis_conf_demo['redis_db']}.")
