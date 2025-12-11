import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.nn import GraphSAGE, SAGEConv
from torch_geometric.data import Data
import networkx as nx
import pandas as pd
import numpy as np
import logging
import argparse
import mlflow
import mlflow.pytorch
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    precision_recall_curve,
    auc,
    accuracy_score,
    confusion_matrix,
)
from pathlib import Path
from src.utils.common_helpers import load_config, setup_logging

logger = setup_logging(__name__)


class GNNModel(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))
        self.dropout = dropout

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class GNNTrainer:
    def __init__(self, config_path: Path, env: str):
        self.config = load_config(config_path, env)
        self.logger = setup_logging(
            "GNNTrainer", self.config["environment"]["log_level"]
        )
        self.model_params = self.config["model_config"]["gnn_collusion_detector"]
        self.data_path = Path("./data_vault/")
        self.mlflow_tracking_uri = self.config["environment"]["mlflow_tracking_uri"]
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        mlflow.set_experiment(f"snapp_fraud_gnn_{env}")
        self.logger.info(f"MLflow tracking URI: {self.mlflow_tracking_uri}")

    def load_and_prepare_graph_data(self) -> Data:
        self.logger.info("Loading and preparing graph data for GNN training.")
        try:
            nodes_df = pd.read_csv(
                self.data_path / "graph_topology_data" / "graph_nodes.csv"
            )
            edges_df = pd.read_csv(
                self.data_path / "graph_topology_data" / "graph_edges.csv"
            )
            graph_features_df = pd.read_csv(
                self.data_path / "graph_topology_data" / "processed_graph_features.csv"
            )

            # Merge node types and features
            nodes_df = nodes_df.merge(
                graph_features_df, on="node_id", how="left"
            ).fillna(0)

            # Map node_ids to integer indices
            node_id_to_idx = {
                node_id: i for i, node_id in enumerate(nodes_df["node_id"])
            }
            nodes_df["node_idx"] = nodes_df["node_id"].map(node_id_to_idx)
            edges_df["source_idx"] = edges_df["source"].map(node_id_to_idx)
            edges_df["target_idx"] = edges_df["target"].map(node_id_to_idx)

            # Prepare node features (x)
            feature_cols = [
                col
                for col in nodes_df.columns
                if col
                not in ["node_id", "node_type", "is_collusion_suspect", "node_idx"]
            ]
            x = torch.tensor(nodes_df[feature_cols].values, dtype=torch.float)

            # Prepare edge_index
            edge_index = (
                torch.tensor(
                    edges_df[["source_idx", "target_idx"]].values, dtype=torch.long
                )
                .t()
                .contiguous()
            )

            # Prepare target labels (y)
            # We want to predict 'is_collusion_suspect' for users/drivers, which are nodes in the graph
            y = torch.tensor(nodes_df["is_collusion_suspect"].values, dtype=torch.long)

            # Create torch_geometric Data object
            data = Data(x=x, edge_index=edge_index, y=y)
            self.logger.info(f"Graph data prepared: {data}")
            return data

        except FileNotFoundError:
            self.logger.error(
                "Required graph data files not found. Run graph_topology_data/generate_collusion_graph.py and src/feature_forge/graph_features.py first."
            )
            return None
        except Exception as e:
            self.logger.error(
                f"Error loading or preparing graph data: {e}", exc_info=True
            )
            return None

    def train_model(self, data: Data):
        self.logger.info("Starting GNN model training.")
        if data is None or data.num_nodes == 0:
            self.logger.error("No valid graph data for GNN training.")
            return

        with mlflow.start_run():
            mlflow.log_params(self.model_params)

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            data = data.to(device)

            model = GNNModel(
                in_channels=data.x.shape[1],
                hidden_channels=self.model_params["hidden_channels"],
                out_channels=2,  # Binary classification for suspect/non-suspect
                num_layers=self.model_params["num_layers"],
                dropout=self.model_params["dropout"],
            ).to(device)

            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=self.model_params["learning_rate"],
                weight_decay=self.model_params["weight_decay"],
            )
            criterion = F.cross_entropy  # For binary classification (0 or 1)

            # Simple train/val split for nodes (can be more sophisticated for GNNs)
            train_mask_idx, val_mask_idx = train_test_split(
                np.arange(data.num_nodes),
                test_size=0.2,
                random_state=42,
                stratify=data.y.cpu().numpy(),  # Ensure fraud nodes are proportionally split
            )
            train_mask = torch.zeros(data.num_nodes, dtype=torch.bool, device=device)
            val_mask = torch.zeros(data.num_nodes, dtype=torch.bool, device=device)
            train_mask[train_mask_idx] = True
            val_mask[val_mask_idx] = True

            for epoch in range(1, self.model_params["epochs"] + 1):
                model.train()
                optimizer.zero_grad()
                out = model(data.x, data.edge_index)
                loss = criterion(out[train_mask], data.y[train_mask])
                loss.backward()
                optimizer.step()

                model.eval()
                val_out = model(data.x, data.edge_index)
                val_loss = criterion(val_out[val_mask], data.y[val_mask])

                # Metrics
                preds = val_out[val_mask].argmax(dim=1)
                labels = data.y[val_mask]

                accuracy = accuracy_score(labels.cpu(), preds.cpu())

                # AUC for binary classification
                probabilities = (
                    F.softmax(val_out[val_mask], dim=1)[:, 1].detach().cpu().numpy()
                )
                auc_score = roc_auc_score(labels.cpu(), probabilities)

                self.logger.info(
                    f"Epoch: {epoch:03d}, Train Loss: {loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {accuracy:.4f}, Val AUC: {auc_score:.4f}"
                )
                mlflow.log_metrics(
                    {
                        "train_loss": loss.item(),
                        "val_loss": val_loss.item(),
                        "val_accuracy": accuracy,
                        "val_auc_score": auc_score,
                    },
                    step=epoch,
                )

            self.logger.info("GNN model training completed.")
            mlflow.pytorch.log_model(
                model,
                "gnn_collusion_model",
                registered_model_name="GNNCollusionDetector",
            )
            self.logger.info("GNN model logged to MLflow.")
            return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train GNN Collusion Detection Model")
    parser.add_argument(
        "--env", type=str, default="dev", help="Environment (dev or prod)"
    )
    args = parser.parse_args()

    current_dir = Path(__file__).parent
    project_root = current_dir.parent.parent
    config_directory = project_root / "config"

    trainer = GNNTrainer(config_directory, args.env)
    graph_data = trainer.load_and_prepare_graph_data()

    if graph_data:
        trainer.train_model(graph_data)
    else:
        logger.error("No valid graph data loaded. Cannot train GNN model.")
