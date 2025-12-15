
# 🧬 Graph-Based Drug Repurposing Recommender

![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?logo=python&logoColor=white)
![Neo4j](https://img.shields.io/badge/Neo4j-Graph_DB-008CC1?logo=neo4j&logoColor=white)
![PyTorch Geometric](https://img.shields.io/badge/PyTorch_Geometric-GNN-EE4C2C?logo=pytorch&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?logo=streamlit&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

**A graph-based, supervised learning system for identifying novel drug–disease associations using a heterogeneous biomedical knowledge graph and Graph Neural Networks (GNN).**

This project demonstrates how raw biomedical data can be transformed into an interpretable, bias-corrected recommendation framework. It integrates large-scale data extraction, knowledge graph construction, and deep learning to predict potential new uses for existing drugs.

---

##  What is Drug Repurposing?

**Drug repurposing** (or drug repositioning) aims to discover new therapeutic uses for existing, approved drugs. It significantly reduces development costs and timelines compared to traditional drug discovery. 

In this project, we model biomedical knowledge as a **Heterogeneous Graph** consisting of **Drugs**, **Proteins**, and **Diseases**, and apply supervised link prediction using a **HeteroGraphSAGE**-based GNN to predict potential drug–disease treatment relationships.

---

## 🧠 Key Features

* **🕸️ Heterogeneous Knowledge Graph:** Models complex relationships between Drugs, Proteins, and Diseases with multi-relational edges (e.g., `TARGETS`, `TREATS`).
* **🏥 Biomedical NER:** Utilizes **SciSpaCy** (`en_ner_bc5cdr_md`) to extract disease entities from unstructured clinical text (indications, pharmacology) automatically.
* **🤖 Supervised GNN:** Learns latent representations of biomedical entities using **HeteroGraphSAGE** for link prediction.
* **⚖️ Bias Correction:** Implements a mechanism to adjust predictions and reduce disease popularity bias (countering data imbalance).
* **📊 Interactive Dashboard:** A **Streamlit** app for real-time exploration of recommendations and model diagnostics.

---

## 🏗️ System Architecture

The pipeline consists of five main stages, moving from raw XML data to an interactive frontend:

```mermaid
graph LR
    A[DrugBank XML] --> B(Data Extraction & NER)
    B --> C[(Neo4j Knowledge Graph)]
    C --> D[HeteroGraphSAGE Training]
    D --> E[Bias-Corrected Inference]
    E --> F[Streamlit Dashboard]
````

1.  **Data Extraction:** Parsing DrugBank XML & applying NER.
2.  **Graph Construction:** Ingesting nodes/edges into Neo4j.
3.  **Training:** Supervised GNN training with PyTorch Geometric.
4.  **Inference:** Generating probabilities for unconnected pairs.
5.  **Visualization:** Interactive analysis via Streamlit.
<img width="700" height="800" alt="image" src="https://github.com/user-attachments/assets/5fc67086-c352-4518-a38c-a35c2de56d01" />
<br></br>
<br></br>

 <img width="700" height="500" alt="image" src="https://github.com/user-attachments/assets/6e88bced-79ee-4f62-b2d3-9f0746e51bb3" />

Knowledge Graph Visualization
Below is a high-level visualization of the constructed heterogeneous biomedical knowledge graph, illustrating the complexity and interconnectedness of the data.

<img width="500" height="300" alt="image" src="https://github.com/user-attachments/assets/f147d090-5ae0-426f-8e42-559906e8c155" />


## ⚙️ Methodology

###  1.Data Extraction & Preprocessing (`xml_to_csv.py`)

  * Parses **DrugBank XML** using streaming XML parsing for scalability.
  * Extracts drug and protein entities from structured fields.
  * Applies **SciSpaCy NER** on unstructured text (indication, description, pharmacology) to identify diseases.
  * Normalizes and clusters disease names using **FuzzyWuzzy** matching.
  * **Output:** Clean CSV files for graph construction.

###  2.Knowledge Graph Construction (`build_neo4j_graph.py`)

  * Loads cleaned data into a **Neo4j** graph database.
  * Enforces uniqueness constraints on node identifiers.
  * Constructs a heterogeneous biomedical graph with specific relationships:
      * `(:Drug)-[:TARGETS]->(:Protein)`
      * `(:Drug)-[:TREATS]->(:Disease)`
Example Node and Relationships
The image below shows a detailed view of the drug Etanercept and its direct connections to proteins (like TNF) and diseases (like rheumatoid arthritis and psoriasis), demonstrating the graph's structure.
<img width="300" height="700" alt="image" src="https://github.com/user-attachments/assets/9bd2e593-cf5c-4d67-92ac-b38ac0a6ddde" />


###  3.GNN Training (`model_training.py`)

  * **Framework:** PyTorch Geometric.
  * **Model:** **HeteroGraphSAGE** (Graph Sample and Aggregate) to learn node embeddings via message passing.
  * **Task:** Binary classification to predict `TREATS` links.
  * **Validation:** Uses negative sampling, early stopping, and ROC-AUC based validation.

###  4.Recommendation & Bias Correction (`recommender.py`)

  * Generates link probabilities for all possible drug–disease pairs.
  * Applies a **post-processing bias correction** step to counter the data imbalance where well-known diseases have disproportionately more links.
  * **Output:** Ranked, interpretable Top-K recommendations.

-----

## 📂 Project Structure

```bash
Graph-Based-Drug-Repurposing-Recommender/
│
├── xml_to_csv.py              # XML parsing, NER, normalization, CSV generation
├── build_neo4j_graph.py       # Neo4j graph construction and ingestion
├── model_training.py          # Supervised GNN training (HeteroGraphSAGE)
├── recommender.py             # Inference and bias-corrected recommendations
├── streamlit_app.py           # Interactive visualization dashboard
│
├── data_cleaning/             # Intermediate preprocessing utilities
├── clean_output/              # Final recommendation outputs (CSV)
└── requirements.txt           # Project dependencies
```

-----

## 🚀 Getting Started

### Prerequisites

  * Python 3.8+
  * Neo4j Desktop or Sandbox (Running locally on `bolt://localhost:7687`)

### Installation

1.  **Clone the repository**

    ```bash
    git clone [https://github.com/yourusername/drug-repurposing-gnn.git](https://github.com/yourusername/drug-repurposing-gnn.git)
    cd drug-repurposing-gnn
    ```

2.  **Install dependencies**

    ```bash
    pip install -r requirements.txt
    ```

3.  **Install SciSpaCy Model**

    ```bash
    pip install [https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.1/en_ner_bc5cdr_md-0.5.1.tar.gz](https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.1/en_ner_bc5cdr_md-0.5.1.tar.gz)
    ```

4.  **Run the Pipeline**

    ```bash
    # 1. Extract Data
    python xml_to_csv.py

    # 2. Build Graph (Ensure Neo4j is running)
    python build_neo4j_graph.py

    # 3. Train Model
    python model_training.py

    # 4. Generate Recommendations
    python recommender.py
    ```

5.  **Launch the Dashboard**

    ```bash
    streamlit run streamlit_app.py
    ```

-----

## 📊 Outputs

The system generates two key CSV reports in the `clean_output/` folder:

1.  `top_20_global_adjusted.csv` – Top 20 global drug–disease predictions.
2.  `top_5_per_drug_adjusted.csv` – Top 5 recommendations per drug.

-----

## 👩‍💻 Author
**A.R.Keerthana** 


## 📜 License

This project is for academic and research purposes.


