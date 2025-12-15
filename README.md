<?xml version="1.0" encoding="UTF-8"?>
<Project name="Graph-Based Drug Repurposing Recommender" version="1.0">

    <Header>
        <Title>Graph-Based Drug Repurposing Recommender</Title>
        <Subtitle>A graph-based, supervised learning system for identifying novel drug–disease associations.</Subtitle>
        <Author>
            <Name>Keerthana</Name>
            <Role>Final Year Computer Science Engineering Student</Role>
            <Interests>Graph Learning, Biomedical AI, Knowledge Graphs</Interests>
        </Author>
        <Status>Academic &amp; Research Project</Status>
        <License>For academic and research purposes only.</License>
    </Header>

    <Overview>
        <Description>
            Drug repurposing aims to discover new therapeutic uses for existing drugs, reducing cost and development time compared to traditional drug discovery. 
            In this project, we model biomedical knowledge as a heterogeneous graph consisting of drugs, proteins, and diseases, and apply supervised link prediction using a HeteroGraphSAGE-based GNN to predict potential drug–disease treatment relationships.
        </Description>
        <CoreIntegration>
            <Item>Large-scale biomedical data extraction (DrugBank)</Item>
            <Item>Knowledge graph construction (Neo4j)</Item>
            <Item>Supervised GNN training (PyTorch Geometric)</Item>
            <Item>Bias-corrected recommendation generation</Item>
            <Item>Interactive visualization (Streamlit)</Item>
        </CoreIntegration>
    </Overview>

    <KeyFeatures>
        <Feature name="Heterogeneous Knowledge Graph">
            Models Drugs, Proteins, and Diseases with multi-relational edges (e.g., TARGETS, TREATS).
        </Feature>
        <Feature name="Biomedical NER">
            Uses SciSpaCy (en_ner_bc5cdr_md) to extract disease entities from unstructured text.
        </Feature>
        <Feature name="Supervised GNN for Link Prediction">
            Learns latent representations of biomedical entities using HeteroGraphSAGE.
        </Feature>
        <Feature name="Bias Correction Mechanism">
            Adjusts predictions to reduce disease popularity bias (countering data imbalance).
        </Feature>
        <Feature name="Interactive Dashboard">
            Enables real-time exploration of recommendations and model diagnostics via Streamlit.
        </Feature>
    </KeyFeatures>

    <SystemArchitecture>
        <Flow>
            <Step order="1" component="DrugBank XML">Raw Input Source</Step>
            <Step order="2" component="Data Extraction &amp; Cleaning" script="xml_to_csv.py">Parsing, NER, Normalization</Step>
            <Step order="3" component="Knowledge Graph Construction" script="build_neo4j_graph.py">Neo4j Ingestion</Step>
            <Step order="4" component="Supervised GNN Training" script="model_training.py">HeteroGraphSAGE Learning</Step>
            <Step order="5" component="Bias-Corrected Inference" script="recommender.py">Prediction Generation</Step>
            <Step order="6" component="Visualization &amp; Interaction" script="streamlit_app.py">User Dashboard</Step>
        </Flow>
    </SystemArchitecture>

    <DirectoryStructure>
        <Tree><![CDATA[
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
        ]]></Tree>
    </DirectoryStructure>

    <Methodology>
        <Phase number="1" title="Data Extraction &amp; Preprocessing">
            <Details>
                - Parses DrugBank XML using streaming XML parsing for scalability.
                - Extracts drug and protein entities from structured fields.
                - Applies SciSpaCy NER on unstructured text (indication, description, pharmacology) to identify diseases.
                - Normalizes and clusters disease names using FuzzyWuzzy.
                - Outputs clean CSV files for graph construction.
            </Details>
        </Phase>
        <Phase number="2" title="Knowledge Graph Construction">
            <Details>
                - Loads cleaned data into Neo4j.
                - Enforces uniqueness constraints on node identifiers.
                - Constructs a heterogeneous biomedical graph with [:TARGETS] and [:TREATS] relationships.
            </Details>
        </Phase>
        <Phase number="3" title="GNN Training (Supervised Link Prediction)">
            <Details>
                - Uses PyTorch Geometric for heterogeneous GNN modeling.
                - Learns node embeddings via HeteroGraphSAGE message passing.
                - Trains a binary classifier to predict drug–disease treatment links.
                - Employs negative sampling, early stopping, and ROC-AUC based validation.
            </Details>
        </Phase>
        <Phase number="4" title="Recommendation &amp; Bias Correction">
            <Details>
                - Generates predictions for all possible drug–disease pairs.
                - Applies disease-wise bias correction to counter data imbalance.
                - Produces ranked, interpretable Top-K recommendations.
            </Details>
        </Phase>
        <Phase number="5" title="Visualization">
            <Details>
                - Streamlit-based dashboard for querying predictions and viewing diagnostics.
            </Details>
        </Phase>
    </Methodology>

    <TechStack>
        <Language>Python</Language>
        <Database type="Graph">Neo4j</Database>
        <Framework type="GNN">PyTorch Geometric</Framework>
        <NLP>SciSpaCy</NLP>
        <Matching>FuzzyWuzzy</Matching>
        <Frontend>Streamlit</Frontend>
    </TechStack>

    <Outputs>
        <File name="top_20_global_adjusted.csv">Top 20 global drug–disease predictions.</File>
        <File name="top_5_per_drug_adjusted.csv">Top 5 recommendations per drug.</File>
    </Outputs>

    <UseCases>
        <Case>Drug repurposing research</Case>
        <Case>Biomedical knowledge graph analysis</Case>
        <Case>Graph Neural Network experimentation</Case>
        <Case>Healthcare AI and decision-support systems</Case>
    </UseCases>

    <FutureEnhancements>
        <Plan>Integration of clinical trial data</Plan>
        <Plan>Temporal graph modeling</Plan>
        <Plan>Explainable AI (XAI) for GNN predictions</Plan>
        <Plan>Deployment as a full-stack web application</Plan>
    </FutureEnhancements>

</Project>
