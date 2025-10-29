# 🗄️ Data Vault Directory

This directory serves as the **central data repository** for the Real-Time Fraud & Abuse Detection system, containing schemas, simulated data, and graph topologies. 🏦

## 🗂️ Structure Overview

- **📋 event_schemas/** - JSON schemas for ride events and data structures
- **🎭 fraud_pattern_simulator/** - Synthetic fraud scenario generation scripts
- **🕸️ graph_topology_data/** - Collusion graphs and topology creation scripts

## 📄 Key Files

- **📋 ride_event_schema.json** - Ride event data schema definition
- **🎭 generate_abuse_scenarios.py** - Synthetic fraud pattern generator
- **🕸️ generate_collusion_graph.py** - Collusion graph data creation

## 📦 Dependencies

- **🐍 Python 3.8+**
- **🔧 Libraries:** networkx, pandas, numpy (graph generation & data manipulation)

## 🚀 Usage

### 🎭 Generating Fraud Scenarios
```python
from fraud_pattern_simulator.generate_abuse_scenarios import FraudSimulator
simulator = FraudSimulator()
scenarios = simulator.generate_scenarios(num_scenarios=100)
```

### 🕸️ Creating Graph Topologies
```python
from graph_topology_data.generate_collusion_graph import GraphGenerator
generator = GraphGenerator()
graph_data = generator.generate_collusion_network(num_nodes=1000)
```

## 💾 Data Storage Notes

- **📦 Local S3 data:** Stored in `../local_s3_data/`
- **🏷️ Model registry:** Metadata in `../model_registry_metadata/`
- **🔐 Production:** Ensure proper S3 access controls