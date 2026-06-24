"""
Phase 1: Graph construction and LLM refinement (enhanced NetworkX version)

Note: PyTorch >= 2.6 is recommended. Install with: pip install torch>=2.6 --upgrade
Features:
1. Load the dataset and extract features.
2. Build the initial undirected KNN graph with NetworkX.
3. Compute seven topology features.
4. Perform optional LLM dual-task refinement on the NetworkX graph:
   - Task A: edge pruning (keep/drop)
   - Task B: node description generation
5. Convert the optimized graph to DGL.
6. Vectorize descriptions and fuse features.
7. Save the optimized graph.

Label display policy:
- Target nodes: show CWE labels for training nodes and "Unlabeled" for validation/test nodes.
- Neighbor nodes: use the same policy.

Advantages:
- NetworkX natively supports undirected edges and avoids duplicate bidirectional edge evaluation.
- Edge traversal and deletion are simpler.
- The final graph is converted to DGL once for efficient training.
"""

import os

import argparse
import pandas as pd
import numpy as np
import torch
import dgl
import json
import pickle
import networkx as nx
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from transformers import AutoTokenizer, AutoModel
import faiss
import torch.nn.functional as F
from tqdm import tqdm
from collections import Counter
from datetime import datetime
import logging
import random
import re

# ===================== Configuration =====================
class Config:
    # Dataset path
    DATA_PATH = os.getenv('SCULPT_DATA_PATH', 'data/bigvul_10GB_CWE.csv')
    
    # CodeT5+ model path or Hugging Face model name
    CODET5P_PATH = os.getenv('SCULPT_CODET5P_PATH', 'Salesforce/codet5p-110m-embedding')
    
    # Output directory
    OUTPUT_DIR = os.getenv('SCULPT_OUTPUT_DIR', 'optimized_graphs')
    
    # LLM configuration (DeepSeek API)
    USE_LLM_OPTIMIZATION = os.getenv('SCULPT_USE_LLM', '0').lower() in {'1', 'true', 'yes'}
    LLM_API_KEY = os.getenv('DEEPSEEK_API_KEY', '')
    LLM_MODEL = os.getenv('SCULPT_LLM_MODEL', 'deepseek-chat')
    LLM_BASE_URL = os.getenv('SCULPT_LLM_BASE_URL', 'https://api.deepseek.com/v1')
    
    # LLM refinement parameters
    LLM_FILTER_TOP_PERCENT = 0.05  # Refine the top X% high-degree nodes; set to 1.0 for the full graph.
    LLM_MIN_DEGREE = 10  # Minimum degree threshold
    LLM_MAX_NEIGHBORS_PER_CALL = 30  # Maximum neighbors evaluated in each LLM call
    
    # Conservative edge-pruning strategy
    LLM_ENABLE_EDGE_DROP = True  # Enable edge pruning. Set to False to generate descriptions only.
    LLM_MAX_DROPS_PER_NODE = 3  # Maximum number of dropped edges per node
    LLM_MIN_DROP_CONFIDENCE = 0.85  # Minimum confidence threshold for dropping an edge
    
    # Checkpoint/resume configuration
    CHECKPOINT_INTERVAL = 10  # Save one checkpoint every N processed nodes.
    CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, 'checkpoints')  # Checkpoint directory
    RESUME_FROM_CHECKPOINT = True  # Resume from an existing checkpoint if available
    
    # Feature fusion parameters
    DESC_EMBED_DIM = 256  # Projected description vector dimension, aligned with code vectors.
    
    # Graph construction parameters
    MIN_SAMPLES_PER_CLASS = 100
    
    # Random seed
    SEED = 42


current_time = datetime.now().strftime('%Y%m%d_%H%M%S')


def setup_logging():
    """Configure logging after command-line options have updated Config."""
    if not os.path.exists(Config.OUTPUT_DIR):
        os.makedirs(Config.OUTPUT_DIR)

    log_file = os.path.join(Config.OUTPUT_DIR, f'graph_construction_{current_time}.log')
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )


def configure_from_args(args=None):
    parser = argparse.ArgumentParser(description='SCULPT Phase 1: graph construction and optional LLM refinement')
    parser.add_argument('--data_path', type=str, default=Config.DATA_PATH,
                        help='CSV file with columns: func_before, CWE ID')
    parser.add_argument('--codet5p_path', type=str, default=Config.CODET5P_PATH,
                        help='Local path or Hugging Face model name for CodeT5+ embeddings')
    parser.add_argument('--output_dir', type=str, default=Config.OUTPUT_DIR,
                        help='Directory for graph artifacts and logs')
    parser.add_argument('--min_samples_per_class', type=int, default=Config.MIN_SAMPLES_PER_CLASS,
                        help='Minimum samples required for each CWE class')
    parser.add_argument('--seed', type=int, default=Config.SEED,
                        help='Random seed')
    parser.add_argument('--use_llm', action='store_true',
                        help='Enable LLM-based edge refinement and node descriptions')
    parser.add_argument('--llm_model', type=str, default=Config.LLM_MODEL,
                        help='LLM model name')
    parser.add_argument('--llm_base_url', type=str, default=Config.LLM_BASE_URL,
                        help='OpenAI-compatible LLM base URL')
    parser.add_argument('--llm_filter_top_percent', type=float, default=Config.LLM_FILTER_TOP_PERCENT,
                        help='Top percent of high-degree nodes to refine with the LLM')
    parser.add_argument('--llm_min_degree', type=int, default=Config.LLM_MIN_DEGREE,
                        help='Minimum node degree for LLM refinement')
    parser.add_argument('--llm_max_neighbors_per_call', type=int, default=Config.LLM_MAX_NEIGHBORS_PER_CALL,
                        help='Maximum neighbors evaluated per LLM call')
    parser.add_argument('--disable_edge_drop', action='store_true',
                        help='Generate LLM descriptions without dropping graph edges')
    parser.add_argument('--max_drops_per_node', type=int, default=Config.LLM_MAX_DROPS_PER_NODE,
                        help='Maximum dropped edges per LLM-refined node')
    parser.add_argument('--min_drop_confidence', type=float, default=Config.LLM_MIN_DROP_CONFIDENCE,
                        help='Minimum confidence for an LLM drop decision')
    parser.add_argument('--checkpoint_interval', type=int, default=Config.CHECKPOINT_INTERVAL,
                        help='Save LLM refinement checkpoint every N processed nodes')
    parser.add_argument('--no_resume', action='store_true',
                        help='Do not resume LLM refinement from an existing checkpoint')

    parsed = parser.parse_args(args)
    Config.DATA_PATH = parsed.data_path
    Config.CODET5P_PATH = parsed.codet5p_path
    Config.OUTPUT_DIR = parsed.output_dir
    Config.CHECKPOINT_DIR = os.path.join(Config.OUTPUT_DIR, 'checkpoints')
    Config.MIN_SAMPLES_PER_CLASS = parsed.min_samples_per_class
    Config.SEED = parsed.seed
    Config.USE_LLM_OPTIMIZATION = parsed.use_llm or Config.USE_LLM_OPTIMIZATION
    Config.LLM_MODEL = parsed.llm_model
    Config.LLM_BASE_URL = parsed.llm_base_url
    Config.LLM_FILTER_TOP_PERCENT = parsed.llm_filter_top_percent
    Config.LLM_MIN_DEGREE = parsed.llm_min_degree
    Config.LLM_MAX_NEIGHBORS_PER_CALL = parsed.llm_max_neighbors_per_call
    Config.LLM_ENABLE_EDGE_DROP = not parsed.disable_edge_drop
    Config.LLM_MAX_DROPS_PER_NODE = parsed.max_drops_per_node
    Config.LLM_MIN_DROP_CONFIDENCE = parsed.min_drop_confidence
    Config.CHECKPOINT_INTERVAL = parsed.checkpoint_interval
    Config.RESUME_FROM_CHECKPOINT = not parsed.no_resume
    Config.LLM_API_KEY = os.getenv('DEEPSEEK_API_KEY', '')
    set_seed(Config.SEED)
    return parsed

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(Config.SEED)


# ===================== Prompt Templates =====================
SYSTEM_PROMPT = """You are a security analysis assistant specializing in software vulnerability semantics. You are CONSERVATIVE in your decisions - when uncertain, prefer to KEEP connections."""

LLM_PROMPT_TEMPLATE = """You are a security analysis assistant. Given a vulnerable function node and its neighbors in a similarity-based graph, do two things:

(A) **Connectivity Refinement (BE VERY CONSERVATIVE)**:
- DEFAULT is KEEP. Only drop edges when you are HIGHLY CONFIDENT (>0.85) that there is NO mechanism-level relation.
- KEEP if: shared root cause, shared unsafe APIs, same CWE family, similar vulnerability pattern, or ANY reasonable connection.
- DROP ONLY if: completely different vulnerability mechanisms with NO overlap whatsoever.
- **IMPORTANT: Drop at most 3-5 edges per node. If all neighbors seem related, keep them all.**
- When in doubt, KEEP the edge.

(B) **Node Representation**: produce a description capturing vulnerability semantics.

Constraints: Be conservative. Prefer keeping edges over dropping. The graph structure is valuable.


# Target Node Information

Function ID: {node_id}
Vulnerability Type: {vulnerability_type}
Source Code:
```c
{source_code}
```
Node Properties in Graph:
- Degree Centrality: {degree_centrality:.4f}
- Closeness Centrality: {closeness_centrality:.4f}
- Betweenness Centrality: {betweenness_centrality:.4f}
- Clustering Coefficient: {clustering_coefficient:.4f}
- Square Clustering Coefficient: {square_clustering:.4f}
- Katz Centrality: {katz_centrality:.4f}
- Eigenvector Centrality: {eigenvector_centrality:.4f}


# Neighbors (Top-k by Similarity)

Based on semantic embeddings from a code pre-trained model, following are the top-k mutual neighbors currently connected to the target node.

{neighbors_section}


# Output Format

Output in JSON format. Remember: DEFAULT is KEEP. Only drop when highly confident (>0.85).
```json
{{
  "node_representation": {{
    "description": "High-level description capturing vulnerability semantics...",
    "rationale": "Explanation for ground truth (if labeled) or candidate type prediction (if unlabeled)..."
  }},
  "connectivity_refinement": [
    {{
      "neighbor_id": 123,
      "decision": "keep",
      "confidence": 0.8,
      "evidence": "Both functions share memory operation patterns..."
    }},
    {{
      "neighbor_id": 456,
      "decision": "keep",
      "confidence": 0.7,
      "evidence": "Similar code structure, keeping for potential relation..."
    }},
    {{
      "neighbor_id": 789,
      "decision": "drop",
      "confidence": 0.92,
      "evidence": "Completely unrelated: one is network I/O, other is pure math computation with no shared APIs or patterns..."
    }}
  ]
}}
```
"""

NEIGHBOR_TEMPLATE = """
Neighbor {index}:
Function ID: {neighbor_id}
Vulnerability Type: {vulnerability_type}
Similarity Score: {similarity_score:.4f}
Source Code:
```c
{neighbor_code}
```
"""


# ===================== Code Processor =====================
class CodeProcessor:
    """CodeT5p feature extractor for vectorizing code and descriptions.
    
    Note: this uses T5EncoderModel plus hierarchical feature aggregation.
    """
    
    def __init__(self):
        from transformers import T5EncoderModel
        self.tokenizer = AutoTokenizer.from_pretrained(Config.CODET5P_PATH, trust_remote_code=True)
        # Use T5EncoderModel to obtain hidden_states for hierarchical aggregation.
        self.model = T5EncoderModel.from_pretrained(Config.CODET5P_PATH).to(device)
        self.model.eval()

    def vectorize(self, texts, is_code=True):
        """
        Feature extraction with hierarchical feature aggregation.
        Args:
            texts: List of input texts.
            is_code: Whether the input is code (True) or natural language descriptions (False).
        """
        results = []
        batch_size = 16

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            if isinstance(batch, pd.Series):
                batch = batch.tolist()
            elif isinstance(batch, np.ndarray):
                batch = batch.tolist()

            inputs = self.tokenizer(
                batch,
                padding='max_length',
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(device)

            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)
            
            # Use hierarchical feature aggregation.
            hidden_states = outputs.hidden_states
            batch_features = self._hierarchical_feature_aggregation(hidden_states, inputs['attention_mask'])
            
            results.append(batch_features.cpu().detach())

        return torch.cat(results, dim=0).numpy()
    
    def _hierarchical_feature_aggregation(self, hidden_states, attention_mask):
        """Hierarchical feature aggregation over all Transformer layer outputs."""
        num_layers = len(hidden_states)
        
        # Learnable layer weights
        if not hasattr(self, 'layer_weights'):
            layer_weights_init = torch.log(torch.linspace(0.5, 2.0, num_layers)).to(device)
            self.layer_weights = layer_weights_init.detach().clone()

        normalized_weights = F.softmax(self.layer_weights, dim=0)

        # 1. Weighted CLS feature aggregation
        cls_features = []
        for i, (hidden, weight) in enumerate(zip(hidden_states, normalized_weights)):
            cls_feat = hidden[:, 0]
            weighted_cls = cls_feat * weight
            cls_features.append(weighted_cls)

        stacked_cls = torch.stack(cls_features, dim=1)
        cls_attention_weights = F.softmax(
            torch.matmul(stacked_cls, stacked_cls.transpose(-2, -1)).mean(dim=-1),
            dim=-1
        )
        final_cls = torch.sum(stacked_cls * cls_attention_weights.unsqueeze(-1), dim=1)

        # 2. Weighted global average pooling
        pooled_features = []
        attention_mask_expanded = attention_mask.unsqueeze(-1)

        for i, (hidden, weight) in enumerate(zip(hidden_states, normalized_weights)):
            masked_hidden = hidden * attention_mask_expanded
            pooled = torch.sum(masked_hidden, dim=1) / torch.sum(attention_mask_expanded, dim=1)
            weighted_pooled = pooled * weight
            pooled_features.append(weighted_pooled)

        stacked_pooled = torch.stack(pooled_features, dim=1)
        pooled_attention_weights = F.softmax(
            torch.matmul(stacked_pooled, stacked_pooled.transpose(-2, -1)).mean(dim=-1),
            dim=-1
        )
        final_pooled = torch.sum(stacked_pooled * pooled_attention_weights.unsqueeze(-1), dim=1)

        # 3. Multi-scale feature extraction from inter-layer differences
        layer_differences = []
        for i in range(1, num_layers):
            diff = hidden_states[i][:, 0] - hidden_states[i - 1][:, 0]
            layer_differences.append(diff)

        if layer_differences:
            diff_weights = F.softmax(torch.ones(len(layer_differences)).to(device), dim=0)
            weighted_diffs = torch.stack([diff * weight for diff, weight in zip(layer_differences, diff_weights)], dim=1)
            final_diff = torch.mean(weighted_diffs, dim=1)
        else:
            final_diff = torch.zeros_like(final_cls)

        # 4. Dynamic weighted fusion
        cls_weight = torch.sigmoid(torch.mean(final_cls, dim=-1, keepdim=True))
        pooled_weight = torch.sigmoid(torch.mean(final_pooled, dim=-1, keepdim=True))
        diff_weight = torch.sigmoid(torch.mean(final_diff, dim=-1, keepdim=True))

        total_weight = cls_weight + pooled_weight + diff_weight + 1e-8
        cls_weight = cls_weight / total_weight
        pooled_weight = pooled_weight / total_weight
        diff_weight = diff_weight / total_weight

        combined_features = (
            final_cls * cls_weight +
            final_pooled * pooled_weight +
            final_diff * diff_weight * 0.3
        )

        # L2 normalization
        combined_features = F.normalize(combined_features, p=2, dim=1)

        return combined_features
    
    def vectorize_descriptions(self, descriptions):
        """
        Vectorize LLM-generated descriptions.
        Args:
            descriptions: List of description texts.
        Returns:
            numpy array of shape [len(descriptions), embed_dim]
        """
        return self.vectorize(descriptions, is_code=False)


# ===================== Topology Feature Computation =====================
def compute_topology_features(nx_g):
    """
    Compute all seven topology features directly on the NetworkX graph.
    
    Args:
        nx_g: Undirected NetworkX graph.
    
    Returns:
        topo_features: numpy array [N, 7]
    """
    logging.info("Computing topology features...")
    
    n_nodes = nx_g.number_of_nodes()
    
    # 1. Degree centrality
    logging.info("  - Computing degree centrality...")
    degree_cent = np.array(list(nx.degree_centrality(nx_g).values()))
    
    # 2. Closeness centrality
    logging.info("  - Computing closeness centrality...")
    closeness_cent = np.array(list(nx.closeness_centrality(nx_g).values()))
    
    # 3. Betweenness centrality with sampling for acceleration
    logging.info("  - Computing betweenness centrality...")
    if n_nodes > 1000:
        betweenness_cent = np.array(list(
            nx.betweenness_centrality(nx_g, k=min(500, n_nodes)).values()
        ))
    else:
        betweenness_cent = np.array(list(nx.betweenness_centrality(nx_g).values()))
    
    # 4. Clustering coefficient
    logging.info("  - Computing clustering coefficient...")
    clustering_coef = np.array(list(nx.clustering(nx_g).values()))
    
    # 5. Square clustering coefficient
    logging.info("  - Computing square clustering coefficient...")
    try:
        square_clustering = np.array(list(nx.square_clustering(nx_g).values()))
    except:
        square_clustering = np.zeros(n_nodes)
    
    # 6. Katz centrality
    logging.info("  - Computing Katz centrality...")
    try:
        # Compute the largest eigenvalue to determine alpha.
        eigenvalues = nx.adjacency_spectrum(nx_g)
        max_eigenvalue = max(abs(eigenvalues))
        alpha = 0.9 / max_eigenvalue if max_eigenvalue > 0 else 0.01
        katz_cent = np.array(list(nx.katz_centrality(nx_g, alpha=alpha, max_iter=1000).values()))
    except:
        logging.warning("    Katz centrality failed; using zeros")
        katz_cent = np.zeros(n_nodes)
    
    # 7. Eigenvector centrality
    logging.info("  - Computing eigenvector centrality...")
    try:
        eigenvector_cent = np.array(list(nx.eigenvector_centrality(nx_g, max_iter=1000).values()))
    except:
        logging.warning("    Eigenvector centrality failed; using zeros")
        eigenvector_cent = np.zeros(n_nodes)
    
    # Combine features into a feature matrix.
    topo_features = np.stack([
        degree_cent, closeness_cent, betweenness_cent,
        clustering_coef, square_clustering, katz_cent, eigenvector_cent
    ], axis=1).astype(np.float32)
    
    logging.info(f"Topology feature computation completed, shape: {topo_features.shape}")
    return topo_features


# ===================== Initial Graph Construction (NetworkX) =====================
def build_initial_graph_networkx(vectors, labels, train_idx, val_idx, test_idx):
    """
    Build the initial KNN graph as an undirected NetworkX graph.
    
    Returns:
        nx_g: Undirected NetworkX graph.
        normalized: Standardized features.
        scaler: Fitted standard scaler.
        edge_similarities: Edge similarity dictionary, keyed by (u, v).
    """
    logging.info("Building initial KNN graph (undirected NetworkX graph)...")
    
    vectors = np.asarray(vectors)
    train_idx = np.asarray(train_idx, dtype=np.int64)
    val_idx = np.asarray(val_idx, dtype=np.int64)
    test_idx = np.asarray(test_idx, dtype=np.int64)
    n_samples, dim = vectors.shape

    # Standardize features using the training split only.
    scaler = StandardScaler()
    scaler.fit(vectors[train_idx])
    normalized = np.zeros_like(vectors, dtype=np.float32)
    normalized[train_idx] = scaler.transform(vectors[train_idx])
    normalized[val_idx] = scaler.transform(vectors[val_idx])
    normalized[test_idx] = scaler.transform(vectors[test_idx])

    # FAISS index
    feats32 = normalized.astype('float32')
    faiss.normalize_L2(feats32)

    index_main = faiss.IndexFlatIP(dim)
    if torch.cuda.is_available():
        try:
            res = faiss.StandardGpuResources()
            index_main = faiss.index_cpu_to_gpu(res, 0, index_main)
        except:
            pass
    index_main.add(feats32)

    # Dynamic k value
    base_k = min(max(20, int(10 * np.log10(n_samples))), 60)

    # Create an undirected NetworkX graph.
    nx_g = nx.Graph()
    nx_g.add_nodes_from(range(n_samples))
    
    edge_count = 0
    edge_similarities = {}

    for i in range(n_samples):
        local_k = base_k
        search_k = min(local_k * 2, n_samples - 1)
        scores, indices = index_main.search(feats32[i:i + 1], search_k + 1)

        scores = scores[0][1:]
        indices = indices[0][1:]

        if len(scores) > 0:
            sorted_scores = np.sort(scores)[::-1]
            threshold_idx = max(1, int(len(sorted_scores) * 0.7))
            threshold = sorted_scores[min(threshold_idx, len(sorted_scores) - 1)]

            valid_mask = scores >= max(threshold, 0.1)
            valid_indices = indices[valid_mask]
            valid_scores = scores[valid_mask]

            if len(valid_indices) < local_k // 3:
                top_k = min(local_k // 2, len(indices))
                valid_indices = indices[:top_k]
                valid_scores = scores[:top_k]

            # Use i < j to avoid duplicate edges; mutual-neighbor matching is not required.
            for j, s in zip(valid_indices, valid_scores):
                if j == i:
                    continue
                if i < j:  # Avoid duplicate edges.
                    weight = float(s * 0.8 + 0.2)
                    nx_g.add_edge(i, j, weight=weight, similarity=float(s))
                    edge_similarities[(min(i, j), max(i, j))] = float(s)
                    edge_count += 1

    logging.info(f"Total nodes: {n_samples}, undirected edges: {edge_count}")
    logging.info(f"Initial NetworkX graph built: {nx_g.number_of_nodes()} nodes, {nx_g.number_of_edges()} undirected edges")
    
    return nx_g, normalized, scaler, edge_similarities


def networkx_to_dgl(nx_g, node_features, labels, train_idx, val_idx, test_idx, add_self_loop=True):
    """
    Convert an undirected NetworkX graph to a DGL graph by materializing bidirectional edges.
    
    Args:
        nx_g: Undirected NetworkX graph.
        node_features: Node feature numpy array [N, D].
        labels: Node label numpy array [N].
        train_idx, val_idx, test_idx: Data split indices.
        add_self_loop: Whether to add self-loops.
    
    Returns:
        dgl_g: DGL graph.
    """
    logging.info("Converting NetworkX graph to DGL graph...")
    
    n_nodes = nx_g.number_of_nodes()
    
    # Collect all edges and generate bidirectional edges.
    src, dst, weights = [], [], []
    for u, v, data in nx_g.edges(data=True):
        weight = data.get('weight', 1.0)
        # Add bidirectional edges for DGL message passing.
        src.extend([u, v])
        dst.extend([v, u])
        weights.extend([weight, weight])
    
    # Add self-loops manually to avoid edata issues with dgl.add_self_loop.
    if add_self_loop:
        for i in range(n_nodes):
            src.append(i)
            dst.append(i)
            weights.append(1.0)  # Self-loop weight.
    
    # Create the DGL graph.
    dgl_g = dgl.graph((src, dst), num_nodes=n_nodes)
    dgl_g.edata['weight'] = torch.tensor(weights, dtype=torch.float32)
    
    # Node features and labels.
    dgl_g.ndata['feat'] = torch.from_numpy(node_features)
    dgl_g.ndata['label'] = torch.LongTensor(labels)
    
    # Split masks.
    train_mask = torch.zeros(n_nodes, dtype=torch.bool)
    train_mask[train_idx] = True
    val_mask = torch.zeros(n_nodes, dtype=torch.bool)
    val_mask[val_idx] = True
    test_mask = torch.zeros(n_nodes, dtype=torch.bool)
    test_mask[test_idx] = True
    
    dgl_g.ndata['train_mask'] = train_mask
    dgl_g.ndata['val_mask'] = val_mask
    dgl_g.ndata['test_mask'] = test_mask
    
    logging.info(f"DGL graph conversion completed: {dgl_g.num_nodes()} nodes, {dgl_g.num_edges()} edges including bidirectional edges and self-loops")
    
    return dgl_g


# ===================== LLM Dual-Task Evaluator =====================
class LLMNodeEvaluator:
    """LLM dual-task evaluator for edge pruning and description generation."""
    
    def __init__(self, api_key, model="gpt-4", base_url=None, max_neighbors=10):
        self.model = model
        self.max_neighbors = max_neighbors
        
        # Initialize the OpenAI-compatible client.
        try:
            from openai import OpenAI
            if base_url:
                self.client = OpenAI(api_key=api_key, base_url=base_url)
            else:
                self.client = OpenAI(api_key=api_key)
            self.available = True
        except ImportError:
            logging.warning("OpenAI library is not installed; skipping LLM refinement")
            self.available = False
        except Exception as e:
            logging.warning(f"Failed to initialize the LLM client: {e}")
            self.available = False
    
    def truncate_code(self, code, max_length):
        """Truncate source code."""
        if len(code) <= max_length:
            return code
        half = max_length // 2 - 20
        return code[:half] + "\n// ... code truncated ...\n" + code[-half:]
    
    def build_prompt(self, node_info, neighbors_info):
        """Build the LLM prompt."""
        
        # Build the neighbor information section.
        neighbors_section = ""
        for i, neighbor in enumerate(neighbors_info, 1):
            neighbors_section += NEIGHBOR_TEMPLATE.format(
                index=i,
                neighbor_id=neighbor['id'],
                vulnerability_type=neighbor['vulnerability_type'],
                similarity_score=neighbor['similarity'],
                neighbor_code=self.truncate_code(neighbor['code'], 500)
            )
        
        prompt = LLM_PROMPT_TEMPLATE.format(
            node_id=node_info['id'],
            vulnerability_type=node_info['vulnerability_type'],
            source_code=self.truncate_code(node_info['code'], 1000),
            degree_centrality=node_info['degree_centrality'],
            closeness_centrality=node_info['closeness_centrality'],
            betweenness_centrality=node_info['betweenness_centrality'],
            clustering_coefficient=node_info['clustering_coefficient'],
            square_clustering=node_info['square_clustering'],
            katz_centrality=node_info['katz_centrality'],
            eigenvector_centrality=node_info['eigenvector_centrality'],
            neighbors_section=neighbors_section
        )
        
        return prompt
    
    def evaluate_node(self, node_info, neighbors_info):
        """
        Evaluate a single node.
        
        Returns:
            dict: {
                'description': str,
                'rationale': str,
                'edge_decisions': [{'neighbor_id': int, 'decision': str, 'confidence': float, 'evidence': str}, ...]
            }
        """
        if not self.available:
            return None
        
        if len(neighbors_info) == 0:
            # If there are no neighbors, generate only the node description.
            return self._generate_description_only(node_info)
        
        # Process neighbors in batches if there are too many.
        all_edge_decisions = []
        description = None
        rationale = None
        
        for i in range(0, len(neighbors_info), self.max_neighbors):
            batch = neighbors_info[i:i + self.max_neighbors]
            result = self._call_llm(node_info, batch)
            
            if result:
                # Use the first batch's description as the final description.
                if description is None:
                    description = result.get('description', '')
                    rationale = result.get('rationale', '')
                
                all_edge_decisions.extend(result.get('edge_decisions', []))
        
        return {
            'description': description or '',
            'rationale': rationale or '',
            'edge_decisions': all_edge_decisions
        }
    
    def _generate_description_only(self, node_info):
        """Generate only a description when the node has no neighbors."""
        simple_prompt = f"""Analyze this vulnerable function and provide:
1. A high-level description capturing its vulnerability semantics
2. A rationale explaining the vulnerability type

Function ID: {node_info['id']}
Vulnerability Type: {node_info['vulnerability_type']}
Source Code:
```c
{self.truncate_code(node_info['code'], 1500)}
```

Output in JSON:
```json
{{
  "description": "...",
  "rationale": "..."
}}
```
"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": simple_prompt}
                ],
                temperature=0.1,
                max_tokens=1000
            )
            
            result = self._parse_response(response.choices[0].message.content)
            if result:
                return {
                    'description': result.get('description', ''),
                    'rationale': result.get('rationale', ''),
                    'edge_decisions': []
                }
            else:
                return {
                    'description': '',
                    'rationale': '',
                    'edge_decisions': []
                }
        except Exception as e:
            logging.warning(f"LLM description generation failed: {e}")
            return {
                'description': '',
                'rationale': '',
                'edge_decisions': []
            }
    
    def _call_llm(self, node_info, neighbors_info, max_retries=2):
        """Call the LLM with retries."""
        prompt = self.build_prompt(node_info, neighbors_info)
        
        for attempt in range(max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,
                    max_tokens=4000  # Increase token budget to support more neighbors.
                )
                
                content = response.choices[0].message.content
                
                # Check for abnormal finish_reason values.
                finish_reason = response.choices[0].finish_reason
                if finish_reason == 'length':
                    logging.warning("LLM output was truncated because max_tokens was insufficient")
                elif finish_reason != 'stop':
                    logging.warning(f"LLM did not finish normally: finish_reason={finish_reason}")
                
                result = self._parse_response(content)
                if result is not None:
                    return result
                
                # Retry after a parsing failure if attempts remain.
                if attempt < max_retries:
                    logging.info(f"Parsing failed; retrying ({attempt + 1}/{max_retries})...")
                    import time
                    time.sleep(1)  # Brief pause before retrying.
                    
            except Exception as e:
                logging.warning(f"LLM call failed (attempt {attempt + 1}): {e}")
                if attempt < max_retries:
                    import time
                    time.sleep(2)
        
        return None
    
    def _parse_response(self, response_text):
        """Parse the LLM response."""
        if not response_text or not response_text.strip():
            logging.warning("LLM returned an empty response")
            return None
        
        # Try multiple JSON extraction strategies.
        json_str = None
        
        # Strategy 1: ```json ... ```
        json_match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        
        # Strategy 2: ``` ... ``` without a language tag.
        if not json_str:
            json_match = re.search(r'```\s*(.*?)\s*```', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
        
        # Strategy 3: directly locate a { ... } object.
        if not json_str:
            json_match = re.search(r'\{[\s\S]*\}', response_text)
            if json_match:
                json_str = json_match.group(0)
        
        # Strategy 4: use the raw response text.
        if not json_str:
            json_str = response_text.strip()
        
        try:
            result = json.loads(json_str)
            
            # Extract node representation.
            node_repr = result.get('node_representation', {})
            description = node_repr.get('description', result.get('description', ''))
            rationale = node_repr.get('rationale', result.get('rationale', ''))
            
            # Extract edge decisions.
            edge_decisions = result.get('connectivity_refinement', [])
            
            return {
                'description': description,
                'rationale': rationale,
                'edge_decisions': edge_decisions
            }
        except json.JSONDecodeError as e:
            # Print the first 200 characters for debugging.
            preview = response_text[:200].replace('\n', '\\n') if response_text else "(empty)"
            logging.warning(f"JSON parsing failed: {e}, response preview: {preview}...")
            return None


# ===================== Checkpoint Management =====================
class CheckpointManager:
    """Checkpoint manager for resumable LLM refinement."""
    
    def __init__(self, checkpoint_dir):
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_file = os.path.join(checkpoint_dir, 'llm_checkpoint.pkl')
        
        # Create the checkpoint directory.
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
    
    def save(self, data):
        """Save a checkpoint."""
        temp_file = self.checkpoint_file + '.tmp'
        try:
            with open(temp_file, 'wb') as f:
                pickle.dump(data, f)
            # Atomic replacement.
            if os.path.exists(self.checkpoint_file):
                os.remove(self.checkpoint_file)
            os.rename(temp_file, self.checkpoint_file)
            return True
        except Exception as e:
            logging.error(f"Failed to save checkpoint: {e}")
            if os.path.exists(temp_file):
                os.remove(temp_file)
            return False
    
    def load(self):
        """Load a checkpoint."""
        if not os.path.exists(self.checkpoint_file):
            return None
        try:
            with open(self.checkpoint_file, 'rb') as f:
                data = pickle.load(f)
            logging.info(f"Successfully loaded checkpoint: {self.checkpoint_file}")
            return data
        except Exception as e:
            logging.error(f"Failed to load checkpoint: {e}")
            return None
    
    def exists(self):
        """Check whether the checkpoint exists."""
        return os.path.exists(self.checkpoint_file)
    
    def clear(self):
        """Clear the checkpoint."""
        if os.path.exists(self.checkpoint_file):
            os.remove(self.checkpoint_file)
            logging.info("Checkpoint cleared")


# ===================== LLM Graph Optimization (NetworkX, Resumable) =====================
def optimize_graph_with_llm(nx_g, code_texts, topo_features, labels, label_encoder, 
                            train_idx, val_idx, test_idx, edge_similarities, evaluator):
    """
    Optimize graph structure with an LLM through edge pruning and description generation.
    Operates on an undirected NetworkX graph, so edges are naturally unique.
    Supports resumable execution by saving checkpoints every N processed nodes.
    
    Args:
        nx_g: Undirected NetworkX graph.
        code_texts: List of source-code texts.
        topo_features: Topology features [N, 7].
        labels: Label array.
        label_encoder: Label encoder.
        train_idx, val_idx, test_idx: Data split indices.
        edge_similarities: Edge similarity dictionary.
        evaluator: LLM evaluator.
    
    Returns:
        nx_g_new: Optimized NetworkX graph.
        node_descriptions: {node_id: {'description': str, 'rationale': str}}
        evaluation_log: List of evaluation records.
    """
    if not evaluator.available:
        logging.info("LLM is unavailable; skipping graph refinement")
        return nx_g, {}, []
    
    logging.info("Starting LLM dual-task refinement (edge pruning + description generation)...")
    logging.info("Using an undirected NetworkX graph; edges are naturally unique")
    logging.info(f"Resumable execution: saving every {Config.CHECKPOINT_INTERVAL} nodes")
    
    # Initialize the checkpoint manager.
    ckpt_manager = CheckpointManager(Config.CHECKPOINT_DIR)
    
    # Create dataset masks.
    n_nodes = nx_g.number_of_nodes()
    train_mask = np.zeros(n_nodes, dtype=bool)
    train_mask[train_idx] = True
    val_mask = np.zeros(n_nodes, dtype=bool)
    val_mask[val_idx] = True
    test_mask = np.zeros(n_nodes, dtype=bool)
    test_mask[test_idx] = True
    
    # Compute degrees for all nodes in the undirected NetworkX graph.
    degrees = np.array([nx_g.degree(i) for i in range(n_nodes)])
    
    # Select high-degree nodes for refinement.
    if Config.LLM_FILTER_TOP_PERCENT >= 1.0:
        high_degree_mask = degrees >= Config.LLM_MIN_DEGREE
        logging.info("Mode: full-graph refinement")
    else:
        degree_threshold = np.percentile(degrees, 100 - Config.LLM_FILTER_TOP_PERCENT * 100)
        high_degree_mask = (degrees >= degree_threshold) & (degrees >= Config.LLM_MIN_DEGREE)
        logging.info(f"Mode: refining the top {Config.LLM_FILTER_TOP_PERCENT*100:.0f}% high-degree nodes")
    
    nodes_to_evaluate = np.where(high_degree_mask)[0].tolist()
    total_nodes = len(nodes_to_evaluate)
    
    # Count nodes from each split.
    train_count = np.sum(high_degree_mask & train_mask)
    val_count = np.sum(high_degree_mask & val_mask)
    test_count = np.sum(high_degree_mask & test_mask)
    
    logging.info(f"Total nodes: {n_nodes}")
    logging.info(f"Nodes to evaluate with LLM: {total_nodes}")
    logging.info(f"  - Train nodes: {train_count}")
    logging.info(f"  - Validation nodes: {val_count}")
    logging.info(f"  - Test nodes: {test_count}")
    
    # Count initial edges.
    initial_edges = nx_g.number_of_edges()
    logging.info(f"Initial undirected edges: {initial_edges}")
    
    # Try to resume from an existing checkpoint.
    evaluated_edges = set()
    edges_to_remove = []
    node_descriptions = {}
    evaluation_log = []
    processed_nodes = set()
    start_idx = 0
    
    if Config.RESUME_FROM_CHECKPOINT and ckpt_manager.exists():
        ckpt_data = ckpt_manager.load()
        if ckpt_data is not None:
            # Verify that the checkpoint matches the current task.
            if (ckpt_data.get('total_nodes') == total_nodes and 
                ckpt_data.get('n_nodes') == n_nodes):
                
                evaluated_edges = ckpt_data.get('evaluated_edges', set())
                edges_to_remove = ckpt_data.get('edges_to_remove', [])
                node_descriptions = ckpt_data.get('node_descriptions', {})
                evaluation_log = ckpt_data.get('evaluation_log', [])
                processed_nodes = ckpt_data.get('processed_nodes', set())
                
                logging.info(f"=" * 50)
                logging.info("Resuming from checkpoint:")
                logging.info(f"  - Processed nodes: {len(processed_nodes)}/{total_nodes}")
                logging.info(f"  - Evaluated edges: {len(evaluated_edges)}")
                logging.info(f"  - Generated descriptions: {len(node_descriptions)}")
                logging.info(f"  - Edges marked for removal: {len(edges_to_remove)}")
                logging.info(f"=" * 50)
            else:
                logging.warning("Checkpoint does not match the current task; restarting")
                ckpt_manager.clear()
    
    # Filter out already processed nodes.
    remaining_nodes = [n for n in nodes_to_evaluate if n not in processed_nodes]
    logging.info(f"Remaining nodes to process: {len(remaining_nodes)}")
    
    # Process nodes.
    processed_count = len(processed_nodes)
    
    for idx, node_id in enumerate(tqdm(remaining_nodes, desc="LLM node evaluation")):
        node_id = int(node_id)
        
        try:
            # Get this node's neighbors directly from the undirected NetworkX graph.
            neighbors = list(nx_g.neighbors(node_id))
            
            # Filter out already evaluated edges.
            unevaluated_neighbors = []
            for nid in neighbors:
                edge_key = frozenset({node_id, nid})
                if edge_key not in evaluated_edges:
                    unevaluated_neighbors.append(nid)
            
            # Mark these edges as evaluated.
            for nid in unevaluated_neighbors:
                evaluated_edges.add(frozenset({node_id, nid}))
            
            # Determine the vulnerability label shown for the target node.
            if train_mask[node_id]:
                target_vuln_type = label_encoder.classes_[labels[node_id]]
            else:
                target_vuln_type = "Unlabeled"
            
            # Prepare target node information.
            node_info = {
                'id': node_id,
                'code': code_texts[node_id],
                'vulnerability_type': target_vuln_type,
                'degree_centrality': float(topo_features[node_id, 0]),
                'closeness_centrality': float(topo_features[node_id, 1]),
                'betweenness_centrality': float(topo_features[node_id, 2]),
                'clustering_coefficient': float(topo_features[node_id, 3]),
                'square_clustering': float(topo_features[node_id, 4]),
                'katz_centrality': float(topo_features[node_id, 5]),
                'eigenvector_centrality': float(topo_features[node_id, 6]),
            }
            
            # Prepare neighbor information for unevaluated edges only.
            neighbors_info = []
            for nid in unevaluated_neighbors:
                edge_key = (min(node_id, nid), max(node_id, nid))
                sim = edge_similarities.get(edge_key, 0.5)
                
                if train_mask[nid]:
                    neighbor_vuln_type = label_encoder.classes_[labels[nid]]
                else:
                    neighbor_vuln_type = "Unlabeled"
                
                neighbors_info.append({
                    'id': nid,
                    'code': code_texts[nid],
                    'similarity': float(sim),
                    'vulnerability_type': neighbor_vuln_type
                })
            
            # Call the LLM evaluator.
            result = evaluator.evaluate_node(node_info, neighbors_info)
            
            if result:
                # Save node description.
                node_descriptions[node_id] = {
                    'description': result['description'],
                    'rationale': result['rationale']
                }
                
                # Process edge decisions; this can be disabled in the configuration.
                if Config.LLM_ENABLE_EDGE_DROP:
                    MAX_DROPS_PER_NODE = Config.LLM_MAX_DROPS_PER_NODE
                    MIN_DROP_CONFIDENCE = Config.LLM_MIN_DROP_CONFIDENCE
                    
                    # Select high-confidence drop decisions.
                    drop_candidates = []
                    for decision in result['edge_decisions']:
                        neighbor_id = decision.get('neighbor_id')
                        action = decision.get('decision', '').lower()
                        confidence = decision.get('confidence', 0)
                        
                        if action == 'drop' and neighbor_id is not None and confidence >= MIN_DROP_CONFIDENCE:
                            drop_candidates.append({
                                'neighbor_id': int(neighbor_id),
                                'confidence': confidence,
                                'evidence': decision.get('evidence', '')
                            })
                    
                    # Sort by confidence and keep at most MAX_DROPS_PER_NODE decisions.
                    drop_candidates.sort(key=lambda x: x['confidence'], reverse=True)
                    drops_to_apply = drop_candidates[:MAX_DROPS_PER_NODE]
                    
                    for drop in drops_to_apply:
                        edges_to_remove.append((node_id, drop['neighbor_id']))
                        evaluation_log.append({
                            'source': node_id,
                            'target': drop['neighbor_id'],
                            'decision': 'drop',
                            'confidence': drop['confidence'],
                            'evidence': drop['evidence']
                        })
                    
                    # Record the effect of the conservative pruning strategy.
                    if len(drop_candidates) > MAX_DROPS_PER_NODE:
                        logging.info(f"Node {node_id}: LLM suggested dropping {len(drop_candidates)} edges; applying {len(drops_to_apply)} under the conservative strategy")
                else:
                    # Edge pruning is disabled; log only.
                    logging.debug(f"Node {node_id}: edge pruning is disabled; skipping")
            
            # Mark the node as processed.
            processed_nodes.add(node_id)
            processed_count += 1
            
            # Save JSON after each node for real-time inspection.
            desc_path = os.path.join(Config.OUTPUT_DIR, 'node_descriptions_latest.json')
            desc_to_save = {str(k): v for k, v in node_descriptions.items()}
            with open(desc_path, 'w', encoding='utf-8') as f:
                json.dump(desc_to_save, f, indent=2, ensure_ascii=False)
            
            log_path = os.path.join(Config.OUTPUT_DIR, 'llm_evaluation_log_latest.json')
            with open(log_path, 'w', encoding='utf-8') as f:
                json.dump(evaluation_log, f, indent=2, ensure_ascii=False)
            
            # Save checkpoints periodically because pickle files can be large.
            if processed_count % Config.CHECKPOINT_INTERVAL == 0:
                ckpt_data = {
                    'evaluated_edges': evaluated_edges,
                    'edges_to_remove': edges_to_remove,
                    'node_descriptions': node_descriptions,
                    'evaluation_log': evaluation_log,
                    'processed_nodes': processed_nodes,
                    'total_nodes': total_nodes,
                    'n_nodes': n_nodes,
                    'timestamp': datetime.now().isoformat()
                }
                if ckpt_manager.save(ckpt_data):
                    logging.info(f"Checkpoint saved: {processed_count}/{total_nodes} nodes processed")
                    
        except Exception as e:
            logging.error(f"Error while processing node {node_id}: {e}")
            # Save current progress.
            ckpt_data = {
                'evaluated_edges': evaluated_edges,
                'edges_to_remove': edges_to_remove,
                'node_descriptions': node_descriptions,
                'evaluation_log': evaluation_log,
                'processed_nodes': processed_nodes,
                'total_nodes': total_nodes,
                'n_nodes': n_nodes,
                'timestamp': datetime.now().isoformat(),
                'error_node': node_id,
                'error_message': str(e)
            }
            ckpt_manager.save(ckpt_data)
            
            # Also save JSON files on error.
            desc_path = os.path.join(Config.OUTPUT_DIR, 'node_descriptions_latest.json')
            desc_to_save = {str(k): v for k, v in node_descriptions.items()}
            with open(desc_path, 'w', encoding='utf-8') as f:
                json.dump(desc_to_save, f, indent=2, ensure_ascii=False)
            
            logging.info("Checkpoint and JSON files saved after the error; execution can be resumed later")
            raise  # Re-raise the exception.
    
    # Processing is complete; clear the checkpoint.
    ckpt_manager.clear()
    
    logging.info("LLM evaluation completed:")
    logging.info(f"  - Evaluated edges: {len(evaluated_edges)}")
    logging.info(f"  - Nodes with generated descriptions: {len(node_descriptions)}")
    logging.info(f"  - Edges suggested for removal: {len(edges_to_remove)}")
    
    # Remove edges from the NetworkX graph.
    nx_g_new = nx_g.copy()
    removed_count = 0
    
    for u, v in edges_to_remove:
        if nx_g_new.has_edge(u, v):
            nx_g_new.remove_edge(u, v)
            removed_count += 1
    
    logging.info("Edge removal completed:")
    logging.info(f"  - Undirected edges before removal: {initial_edges}")
    logging.info(f"  - Undirected edges after removal: {nx_g_new.number_of_edges()}")
    logging.info(f"  - Actual removed edges: {removed_count}")
    
    return nx_g_new, node_descriptions, evaluation_log


# ===================== Description Vectorization and Feature Fusion =====================
def vectorize_and_fuse_features(g, node_descriptions, processor, n_nodes, code_embed_dim):
    """
    Vectorize LLM-generated descriptions and fuse them with code embeddings.
    
    Args:
        g: DGL graph.
        node_descriptions: {node_id: {'description': str, 'rationale': str}}
        processor: CodeProcessor instance.
        n_nodes: Total number of nodes.
        code_embed_dim: Code embedding dimension.
    
    Returns:
        enhanced_features: Fused features [N, code_embed_dim + desc_embed_dim].
    """
    logging.info("Starting description vectorization and feature fusion...")
    
    # Initialize description vectors; unprocessed nodes use zero vectors.
    desc_vectors = np.zeros((n_nodes, code_embed_dim), dtype=np.float32)
    
    # Collect nodes that have descriptions.
    nodes_with_desc = []
    descriptions = []
    
    for node_id, desc_info in node_descriptions.items():
        desc_text = desc_info.get('description', '')
        if desc_text:
            nodes_with_desc.append(node_id)
            # Combine description and rationale.
            combined_desc = desc_text
            rationale = desc_info.get('rationale', '')
            if rationale:
                combined_desc += " " + rationale
            descriptions.append(combined_desc)
    
    if len(descriptions) > 0:
        logging.info(f"Vectorizing {len(descriptions)} node descriptions...")
        
        # Batch vectorization.
        desc_embeddings = processor.vectorize_descriptions(descriptions)
        
        # Fill vectors into the corresponding node positions.
        for i, node_id in enumerate(nodes_with_desc):
            desc_vectors[node_id] = desc_embeddings[i]
    
    # Get original code features.
    code_features = g.ndata['feat'].numpy()
    
    # Project description vectors to the same dimension if needed.
    # CodeT5p outputs already have the same dimension, so no projection is needed.
    
    # Concatenate features.
    enhanced_features = np.concatenate([code_features, desc_vectors], axis=1)
    
    logging.info("Feature fusion completed:")
    logging.info(f"  - Code feature shape: {code_features.shape}")
    logging.info(f"  - Description feature shape: {desc_vectors.shape}")
    logging.info(f"  - Fused feature shape: {enhanced_features.shape}")
    
    return enhanced_features


# ===================== Main Pipeline =====================
def main():
    configure_from_args()
    setup_logging()

    logging.info("=" * 60)
    logging.info("Phase 1: Graph construction and LLM dual-task refinement (NetworkX version)")
    logging.info("=" * 60)
    
    # 1. Load data.
    logging.info("Loading data...")
    if not os.path.exists(Config.DATA_PATH):
        raise FileNotFoundError(
            f"Dataset not found: {Config.DATA_PATH}. "
            "Provide --data_path or set SCULPT_DATA_PATH."
        )
    df = pd.read_csv(Config.DATA_PATH)
    required_columns = {'func_before', 'CWE ID'}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing_columns)}")
    logging.info(f"Total samples: {len(df)}")
    
    # 2. Filter classes.
    cwe_counts = df['CWE ID'].value_counts()
    selected_cwe = cwe_counts[cwe_counts >= Config.MIN_SAMPLES_PER_CLASS].index
    filtered_df = df[df['CWE ID'].isin(selected_cwe)]
    
    selected_dfs = []
    for cwe in selected_cwe:
        cwe_df = filtered_df[filtered_df['CWE ID'] == cwe]
        selected_dfs.append(cwe_df)
    
    if not selected_dfs:
        raise ValueError(
            "No classes remain after filtering. "
            "Lower --min_samples_per_class or provide a larger dataset."
        )
    selected_df = pd.concat(selected_dfs).reset_index(drop=True)
    if selected_df.empty:
        raise ValueError(
            "No classes remain after filtering. "
            "Lower --min_samples_per_class or provide a larger dataset."
        )
    logging.info(f"Samples after filtering: {len(selected_df)}, number of classes: {len(selected_cwe)}")
    
    # 3. Encode labels.
    le = LabelEncoder()
    labels = le.fit_transform(selected_df['CWE ID'])
    logging.info(f"Classes: {le.classes_}")
    
    # 4. Split data before graph construction.
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=Config.SEED)
    train_idx, temp_idx = next(sss1.split(selected_df, labels))
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=Config.SEED)
    val_rel, test_rel = next(sss2.split(selected_df.iloc[temp_idx], labels[temp_idx]))
    val_idx, test_idx = temp_idx[val_rel], temp_idx[test_rel]
    
    logging.info(f"Data split: train {len(train_idx)}, validation {len(val_idx)}, test {len(test_idx)}")
    
    # 5. Extract features.
    logging.info("Extracting code features...")
    processor = CodeProcessor()
    code_texts = selected_df['func_before'].values
    
    vectors = []
    batch_size = 64
    for i in tqdm(range(0, len(selected_df), batch_size), desc="Feature extraction"):
        batch_texts = code_texts[i:i + batch_size]
        batch_vectors = processor.vectorize(pd.Series(batch_texts))
        vectors.append(batch_vectors)
    
    vectors = np.vstack(vectors)
    code_embed_dim = vectors.shape[1]
    logging.info(f"Code feature shape: {vectors.shape}")
    
    # 6. Build the initial NetworkX graph.
    nx_g, normalized_features, scaler, edge_similarities = build_initial_graph_networkx(
        vectors, labels, train_idx, val_idx, test_idx
    )
    
    # 7. Compute topology features directly on NetworkX.
    topo_features = compute_topology_features(nx_g)
    
    # 8. Optional LLM dual-task refinement on NetworkX.
    node_descriptions = {}
    evaluation_log = []
    
    if Config.USE_LLM_OPTIMIZATION:
        if not Config.LLM_API_KEY:
            raise ValueError(
                "LLM refinement is enabled but DEEPSEEK_API_KEY is not set. "
                "Export DEEPSEEK_API_KEY or run without --use_llm."
            )
        evaluator = LLMNodeEvaluator(
            api_key=Config.LLM_API_KEY,
            model=Config.LLM_MODEL,
            base_url=Config.LLM_BASE_URL,
            max_neighbors=Config.LLM_MAX_NEIGHBORS_PER_CALL
        )
        
        nx_g, node_descriptions, evaluation_log = optimize_graph_with_llm(
            nx_g, code_texts, topo_features, labels, le,
            train_idx, val_idx, test_idx, edge_similarities,
            evaluator
        )
    
    # 9. Convert to a DGL graph.
    logging.info("Converting optimized NetworkX graph to DGL graph...")
    g = networkx_to_dgl(
        nx_g, normalized_features, labels,
        train_idx, val_idx, test_idx,
        add_self_loop=True
    )
    
    # Add topology features to the DGL graph.
    g.ndata['topo_feat'] = torch.from_numpy(topo_features)
    
    # 10. Description vectorization and feature fusion.
    if len(node_descriptions) > 0:
        enhanced_features = vectorize_and_fuse_features(
            g, node_descriptions, processor, g.num_nodes(), code_embed_dim
        )
        g.ndata['enhanced_feat'] = torch.from_numpy(enhanced_features)
    else:
        # If no descriptions exist, enhanced features are raw features concatenated with zeros.
        zero_desc = np.zeros((g.num_nodes(), code_embed_dim), dtype=np.float32)
        enhanced_features = np.concatenate([g.ndata['feat'].numpy(), zero_desc], axis=1)
        g.ndata['enhanced_feat'] = torch.from_numpy(enhanced_features)
    
    # 11. Save results.
    logging.info("Saving the optimized graph...")
    
    save_data = {
        # Graph structure: DGL graph used for training.
        'graph': g,
        
        # Raw data.
        'labels': labels,
        'label_encoder': le,
        'scaler': scaler,
        'code_texts': code_texts,
        'vectors': vectors,
        'topo_features': topo_features,
        
        # Data splits.
        'train_idx': train_idx,
        'val_idx': val_idx,
        'test_idx': test_idx,
        
        # LLM-generated data.
        'node_descriptions': node_descriptions,
        
        # Configuration.
        'config': {
            'use_llm': Config.USE_LLM_OPTIMIZATION,
            'llm_model': Config.LLM_MODEL if Config.USE_LLM_OPTIMIZATION else None,
            'llm_filter_percent': Config.LLM_FILTER_TOP_PERCENT,
            'code_embed_dim': code_embed_dim,
            'enhanced_feat_dim': g.ndata['enhanced_feat'].shape[1],
            'seed': Config.SEED,
            'timestamp': current_time,
            'graph_lib': 'networkx->dgl'  # Graph construction backend.
        }
    }
    
    save_path = os.path.join(Config.OUTPUT_DIR, f'optimized_graph_{current_time}.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump(save_data, f)
    logging.info(f"Graph data saved to: {save_path}")
    
    # Save evaluation log.
    if evaluation_log:
        log_path = os.path.join(Config.OUTPUT_DIR, f'llm_evaluation_log_{current_time}.json')
        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump(evaluation_log, f, indent=2, ensure_ascii=False)
        logging.info(f"LLM evaluation log saved to: {log_path}")
    
    # Save node descriptions.
    if node_descriptions:
        desc_path = os.path.join(Config.OUTPUT_DIR, f'node_descriptions_{current_time}.json')
        # Convert keys to strings for JSON.
        desc_to_save = {str(k): v for k, v in node_descriptions.items()}
        with open(desc_path, 'w', encoding='utf-8') as f:
            json.dump(desc_to_save, f, indent=2, ensure_ascii=False)
        logging.info(f"Node descriptions saved to: {desc_path}")
    
    # Save the latest graph copy.
    latest_path = os.path.join(Config.OUTPUT_DIR, 'latest_graph.pkl')
    with open(save_path, 'rb') as f_in:
        with open(latest_path, 'wb') as f_out:
            f_out.write(f_in.read())
    logging.info(f"Latest graph copy created: {latest_path}")
    
    logging.info("=" * 60)
    logging.info("Phase 1 completed!")
    logging.info("Final graph statistics:")
    logging.info(f"  - Nodes: {g.num_nodes()}")
    logging.info(f"  - DGL edges including bidirectional edges and self-loops: {g.num_edges()}")
    logging.info(f"  - Raw feature dimension: {code_embed_dim}")
    logging.info(f"  - Enhanced feature dimension: {g.ndata['enhanced_feat'].shape[1]}")
    logging.info(f"  - Nodes with descriptions: {len(node_descriptions)}")
    logging.info("=" * 60)
    
    return save_path


if __name__ == "__main__":
    main()
