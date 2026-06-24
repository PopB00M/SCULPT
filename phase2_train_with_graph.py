"""
Phase 2: Train with the optimized graph
Features:
1. Load the optimized graph built by Phase 1, including enhanced features.
2. Train the enhanced GAT model.
3. Evaluate and save the final model.

Usage:
    python phase2_train_with_graph.py --graph_path optimized_graphs/latest_graph.pkl
    python phase2_train_with_graph.py --graph_path optimized_graphs/latest_graph.pkl --use_enhanced_feat
"""

import os
import argparse
import pickle
import pandas as pd
import numpy as np
import torch
import dgl
import json
from dgl.nn.pytorch import GATConv
from sklearn.metrics import classification_report, matthews_corrcoef
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import torch.nn.functional as F
from tqdm import tqdm
from collections import Counter
from datetime import datetime
import logging
import random


# ===================== Configuration =====================
class TrainConfig:
    # Default graph path
    DEFAULT_GRAPH_PATH = os.getenv('SCULPT_GRAPH_PATH', 'optimized_graphs/latest_graph.pkl')
    
    # Output directory
    OUTPUT_DIR = os.getenv('SCULPT_TRAINING_OUTPUT_DIR', 'training_results')
    
    # Model parameters
    HIDDEN_DIM = 1024
    DROPOUT = 0.4
    
    # Training parameters
    EPOCHS = 400
    PATIENCE = 20
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 0.005
    
    # Whether to use topology features
    USE_TOPO_FEATURES = True
    TOPO_FUSION_WEIGHT = 0.2  # Topology feature fusion weight
    
    # Whether to use enhanced features: code embeddings + LLM description embeddings
    USE_ENHANCED_FEAT = False  # Disabled by default to evaluate raw features + optimized graph structure first.
    
    # Random seed
    SEED = 42


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


set_seed(TrainConfig.SEED)


# ===================== Model Definitions =====================
class MultiScaleAttention(torch.nn.Module):
    """Multi-scale attention module."""
    
    def __init__(self, in_dim, num_scales=3):
        super(MultiScaleAttention, self).__init__()
        self.num_scales = num_scales
        self.in_dim = in_dim
        
        self.scale_attentions = torch.nn.ModuleList()
        for i in range(num_scales):
            base_heads = max(1, in_dim // (64 * (i + 1)))
            num_heads = base_heads
            while in_dim % num_heads != 0 and num_heads > 1:
                num_heads -= 1
            if num_heads == 0:
                num_heads = 1
            
            self.scale_attentions.append(
                torch.nn.MultiheadAttention(
                    embed_dim=in_dim,
                    num_heads=num_heads,
                    dropout=0.1,
                    batch_first=True
                )
            )
        
        self.scale_weights = torch.nn.Parameter(torch.ones(num_scales))
        self.fusion_layer = torch.nn.Linear(in_dim * num_scales, in_dim)
    
    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
            squeeze_output = True
        else:
            squeeze_output = False
        
        multi_scale_outputs = []
        for attention in self.scale_attentions:
            attn_output, _ = attention(x, x, x)
            multi_scale_outputs.append(attn_output)
        
        scale_weights_norm = F.softmax(self.scale_weights, dim=0)
        weighted_outputs = []
        for i, output in enumerate(multi_scale_outputs):
            weighted_outputs.append(output * scale_weights_norm[i])
        
        combined = torch.cat(weighted_outputs, dim=-1)
        fused_output = self.fusion_layer(combined)
        
        if squeeze_output:
            fused_output = fused_output.squeeze(1)
        
        return fused_output


class ResidualGate(torch.nn.Module):
    """Residual gating module."""
    
    def __init__(self, in_dim):
        super(ResidualGate, self).__init__()
        self.gate = torch.nn.Sequential(
            torch.nn.Linear(in_dim * 2, in_dim),
            torch.nn.Sigmoid()
        )
    
    def forward(self, x, residual):
        concat_features = torch.cat([x, residual], dim=-1)
        gate_weights = self.gate(concat_features)
        output = gate_weights * x + (1 - gate_weights) * residual
        return output


class EnhancedGATLayer(torch.nn.Module):
    """Enhanced GAT layer."""
    
    def __init__(self, in_dim, out_dim, num_heads=4, feat_drop=0.3, attn_drop=0.2, 
                 activation=F.elu, use_multi_scale=True):
        super(EnhancedGATLayer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.use_multi_scale = use_multi_scale
        
        self.gat = GATConv(
            in_feats=in_dim,
            out_feats=out_dim,
            num_heads=num_heads,
            feat_drop=feat_drop,
            attn_drop=attn_drop,
            residual=False,
            allow_zero_in_degree=True,
            activation=activation
        )
        
        gat_output_dim = out_dim * num_heads
        
        if self.use_multi_scale:
            self.multi_scale_attention = MultiScaleAttention(in_dim)
        
        self.residual_proj = torch.nn.Linear(in_dim, gat_output_dim)
        self.residual_gate = ResidualGate(gat_output_dim)
        self.norm = torch.nn.LayerNorm(gat_output_dim)
        self.drop = torch.nn.Dropout(0.15)
    
    def forward(self, g, h, edge_weights=None):
        if self.use_multi_scale:
            h_multi_scale = self.multi_scale_attention(h)
            h = h + h_multi_scale * 0.3
        
        if self.training and edge_weights is not None:
            edge_weights_mod = edge_weights + torch.randn_like(edge_weights) * 0.05
            edge_weights_mod = torch.clamp(edge_weights_mod, 0.1, 1.0)
            h_gat = self.gat(g, h, edge_weights_mod)
        else:
            h_gat = self.gat(g, h, edge_weights)
        
        h_gat = h_gat.reshape(h_gat.shape[0], -1)
        residual = self.residual_proj(h)
        gated_output = self.residual_gate(h_gat, residual)
        output = self.norm(gated_output)
        output = self.drop(output)
        
        return output


class DenseConnection(torch.nn.Module):
    """DenseNet-style cross-layer connections."""
    
    def __init__(self, layer_dims):
        super(DenseConnection, self).__init__()
        self.layer_dims = layer_dims
        self.projections = torch.nn.ModuleList()
        
        for i in range(1, len(layer_dims)):
            total_input_dim = sum(layer_dims[:i])
            self.projections.append(
                torch.nn.Sequential(
                    torch.nn.Linear(total_input_dim, layer_dims[i]),
                    torch.nn.LayerNorm(layer_dims[i]),
                    torch.nn.GELU()
                ) if total_input_dim != layer_dims[i] else torch.nn.Identity()
            )
    
    def forward(self, layer_outputs):
        enhanced_outputs = [layer_outputs[0]]
        
        for i in range(1, len(layer_outputs)):
            concatenated = torch.cat(enhanced_outputs, dim=-1)
            
            if i - 1 < len(self.projections):
                projected = self.projections[i - 1](concatenated)
                enhanced = layer_outputs[i] + projected
            else:
                enhanced = layer_outputs[i]
            
            enhanced_outputs.append(enhanced)
        
        return enhanced_outputs


class AdvancedFeatureTuner(torch.nn.Module):
    """Feature tuning module."""
    
    def __init__(self, feat_dim, hidden_dim=256, dropout=0.2):
        super(AdvancedFeatureTuner, self).__init__()
        
        self.feature_extractor = torch.nn.Sequential(
            torch.nn.Linear(feat_dim, hidden_dim),
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout)
        )
        
        self.feature_enhancer = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout * 0.5),
            torch.nn.Linear(hidden_dim, feat_dim)
        )
        
        self.gate = torch.nn.Sequential(
            torch.nn.Linear(feat_dim * 2, feat_dim),
            torch.nn.Sigmoid()
        )
        
        self.tune_ratio = torch.nn.Parameter(torch.FloatTensor([0.05]))
        self.enabled = False
        self.epochs_trained = 0
        self.phase = 0
        self.class_specific = torch.nn.Parameter(torch.zeros(1, feat_dim))
    
    def forward(self, features, original_features=None, labels=None, training_phase=0, train_mask=None):
        if not self.enabled:
            return features
        
        if original_features is None:
            original_features = features.detach()
        
        mid_features = self.feature_extractor(features)
        enhanced = self.feature_enhancer(mid_features)
        gate_input = torch.cat([features, enhanced], dim=1)
        gates = self.gate(gate_input)
        tuned_features = gates * enhanced + (1 - gates) * features
        
        if labels is not None and self.training and train_mask is not None:
            train_labels = labels[train_mask] if train_mask.any() else None
            if train_labels is not None:
                class_adjust = torch.zeros_like(tuned_features)
                class_adjust[train_mask] = self.class_specific.expand(train_mask.sum(), -1) * 0.01
                tuned_features = tuned_features + class_adjust
        
        tune_ratio_val = self.tune_ratio.item()
        actual_ratio = min(tune_ratio_val * (1 + 0.2 * training_phase), 0.6)
        final_features = (1 - actual_ratio) * original_features + actual_ratio * tuned_features
        
        return final_features
    
    def step_phase(self):
        self.epochs_trained += 1
        self.phase = self.epochs_trained // 8
        return self.phase


class TopoFeatureFusion(torch.nn.Module):
    """Topology feature fusion module."""
    
    def __init__(self, code_feat_dim, topo_feat_dim, fusion_weight=0.2):
        super(TopoFeatureFusion, self).__init__()
        
        self.fusion_weight = fusion_weight
        
        # Topology feature projection
        self.topo_proj = torch.nn.Sequential(
            torch.nn.Linear(topo_feat_dim, code_feat_dim // 4),
            torch.nn.LayerNorm(code_feat_dim // 4),
            torch.nn.GELU(),
            torch.nn.Linear(code_feat_dim // 4, code_feat_dim)
        )
        
        # Gated fusion
        self.gate = torch.nn.Sequential(
            torch.nn.Linear(code_feat_dim * 2, code_feat_dim),
            torch.nn.Sigmoid()
        )
    
    def forward(self, code_feat, topo_feat):
        # Project topology features
        topo_projected = self.topo_proj(topo_feat)
        
        # Gated fusion
        concat_feat = torch.cat([code_feat, topo_projected], dim=-1)
        gate = self.gate(concat_feat)
        
        # Weighted fusion
        fused = (1 - self.fusion_weight) * code_feat + self.fusion_weight * gate * topo_projected
        
        return fused


class EnhancedGATWithTopo(torch.nn.Module):
    """Enhanced GAT model with optional topology features."""
    
    def __init__(self, in_dim, hidden_dim, n_classes, topo_dim=5, dropout=0.5, use_topo=True):
        super(EnhancedGATWithTopo, self).__init__()
        
        self.use_topo = use_topo
        self.hidden_dim = hidden_dim
        self.n_classes = n_classes
        
        # Topology feature fusion
        if use_topo:
            self.topo_fusion = TopoFeatureFusion(in_dim, topo_dim, TrainConfig.TOPO_FUSION_WEIGHT)
        
        # Feature tuning
        self.feature_tuner = AdvancedFeatureTuner(
            in_dim,
            hidden_dim=hidden_dim // 2,
            dropout=dropout * 0.6
        )
        
        # Input transformation
        self.input_transform = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden_dim),
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout * 0.6)
        )
        
        # GAT layers
        self.gat_layers = torch.nn.ModuleList([
            EnhancedGATLayer(hidden_dim, hidden_dim // 4, num_heads=4, 
                           feat_drop=0.3, attn_drop=0.2, use_multi_scale=True),
            EnhancedGATLayer(hidden_dim, hidden_dim // 4, num_heads=4,
                           feat_drop=0.3, attn_drop=0.2, use_multi_scale=True),
            EnhancedGATLayer(hidden_dim, hidden_dim // 4, num_heads=4,
                           feat_drop=0.2, attn_drop=0.1, use_multi_scale=True)
        ])
        
        # DenseNet connections
        layer_dims = [hidden_dim, hidden_dim, hidden_dim, hidden_dim]
        self.dense_connections = DenseConnection(layer_dims)
        
        # Final fusion
        self.final_fusion = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim * 4, hidden_dim),
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout * 0.5)
        )
        
        # Classifier
        self.feature_extractor = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim // 2),
            torch.nn.LayerNorm(hidden_dim // 2),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout * 0.4)
        )
        
        self.classifier = torch.nn.Linear(hidden_dim // 2, n_classes)
        
        self.l2_reg = 0.0003
        self.tune_start_epoch = 5
        self.current_epoch = 0
    
    def forward(self, g, features, topo_features=None):
        # 1. Topology feature fusion
        if self.use_topo and topo_features is not None:
            features = self.topo_fusion(features, topo_features)
        
        # 2. Feature tuning
        if self.training and self.current_epoch >= self.tune_start_epoch:
            if not self.feature_tuner.enabled:
                self.feature_tuner.enabled = True
            
            training_phase = self.feature_tuner.phase
            original_features = g.ndata.get('orig_feat', features.detach())
            train_mask = g.ndata.get('train_mask', None)
            labels = g.ndata.get('label', None)
            
            features = self.feature_tuner(
                features, original_features, labels, training_phase, train_mask
            )
        
        # 3. Input transformation
        h0 = self.input_transform(features)
        
        if self.training:
            noise = torch.randn_like(h0) * 0.01
            h0 = h0 + noise
        
        # 4. GAT layers + DenseNet
        layer_outputs = [h0]
        for gat_layer in self.gat_layers:
            h_gat = gat_layer(g, layer_outputs[-1], g.edata.get('weight', None))
            layer_outputs.append(h_gat)
        
        enhanced_outputs = self.dense_connections(layer_outputs)
        
        # 5. Final fusion
        final_concat = torch.cat(enhanced_outputs, dim=-1)
        h_fused = self.final_fusion(final_concat)
        
        # 6. Classification
        feature_representation = self.feature_extractor(h_fused)
        logits = self.classifier(feature_representation)
        
        # 7. L2 regularization
        l2_loss = 0
        if self.training:
            for param in self.parameters():
                l2_loss += torch.norm(param, 2)
        
        return logits, self.l2_reg * l2_loss, feature_representation
    
    def update_epoch(self):
        self.current_epoch += 1
        if self.feature_tuner.enabled:
            self.feature_tuner.step_phase()
        return self.current_epoch


# ===================== Loss Functions =====================
class ContrastiveLoss(torch.nn.Module):
    def __init__(self, temperature=0.1):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
    
    def forward(self, features, labels):
        features = F.normalize(features, p=2, dim=1)
        batch_size = features.shape[0]
        
        similarity_matrix = torch.matmul(features, features.T) / self.temperature
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(features.device)
        
        logits_mask = torch.scatter(
            torch.ones_like(mask), 1,
            torch.arange(batch_size).view(-1, 1).to(features.device), 0
        )
        mask = mask * logits_mask
        
        exp_logits = torch.exp(similarity_matrix) * logits_mask
        log_prob = similarity_matrix - torch.log(exp_logits.sum(1, keepdim=True) + 1e-8)
        
        mask_sum = mask.sum(1)
        mask_sum = torch.clamp(mask_sum, min=1)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_sum
        
        return -mean_log_prob_pos.mean()


class AdaptiveFocalLoss(torch.nn.Module):
    def __init__(self, num_classes, gamma=2.0, smoothing=0.1, alpha=None, 
                 use_contrastive=True, contrastive_weight=0.2):
        super(AdaptiveFocalLoss, self).__init__()
        self.num_classes = num_classes
        self.gamma = gamma
        self.smoothing = smoothing
        self.use_contrastive = use_contrastive
        self.contrastive_weight = contrastive_weight
        
        if alpha is None:
            self.alpha = torch.ones(num_classes).to(device)
        else:
            self.alpha = torch.FloatTensor(alpha).to(device)
        
        if self.use_contrastive:
            self.contrastive_loss = ContrastiveLoss()
    
    def forward(self, pred, target, weights=None, epoch=0, features=None):
        eps = self.smoothing / self.num_classes
        target_one_hot = F.one_hot(target, self.num_classes).float().to(device)
        target_smooth = (1 - self.smoothing) * target_one_hot + eps
        
        log_softmax = F.log_softmax(pred, dim=1)
        softmax = torch.exp(log_softmax)
        pt = softmax.gather(1, target.unsqueeze(1)).squeeze(1)
        
        if epoch < 10:
            effective_gamma = self.gamma * 0.7
        else:
            effective_gamma = self.gamma * (1.1 - 0.2 * torch.tanh(2 * pt))
        
        focal_weight = (1 - pt) ** effective_gamma
        alpha_weight = self.alpha[target]
        
        if weights is not None:
            sample_weight = weights
        else:
            sample_weight = torch.ones_like(target).float().to(device)
        
        combined_weight = alpha_weight * focal_weight * sample_weight
        focal_loss = -combined_weight * torch.sum(target_smooth * log_softmax, dim=1)
        focal_loss = focal_loss.mean()
        
        contrastive_loss = 0
        if self.use_contrastive and features is not None:
            contrastive_loss = self.contrastive_loss(features, target)
        
        total_loss = focal_loss + self.contrastive_weight * contrastive_loss
        
        return total_loss


# ===================== Training Function =====================
def train_model(graph_path):
    # Configure logging
    if not os.path.exists(TrainConfig.OUTPUT_DIR):
        os.makedirs(TrainConfig.OUTPUT_DIR)

    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(TrainConfig.OUTPUT_DIR, f'training_{current_time}.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    logging.info("=" * 60)
    logging.info("Phase 2: Train with the optimized graph")
    logging.info("=" * 60)
    
    # 1. Load graph data
    logging.info(f"Loading graph data: {graph_path}")
    if not os.path.exists(graph_path):
        raise FileNotFoundError(
            f"Graph artifact not found: {graph_path}. "
            "Run phase1_graph_construction.py first or provide --graph_path."
        )
    with open(graph_path, 'rb') as f:
        data = pickle.load(f)
    
    g = data['graph'].to(device)
    labels = data['labels']
    le = data['label_encoder']
    topo_features = data['topo_features']
    
    logging.info(f"Graph statistics: {g.num_nodes()} nodes, {g.num_edges()} edges")
    logging.info(f"Classes: {le.classes_}")
    logging.info(f"Graph construction config: {data.get('config', {})}")
    
    # Choose enhanced features or raw features
    if TrainConfig.USE_ENHANCED_FEAT and 'enhanced_feat' in g.ndata:
        logging.info("Using enhanced features (code embeddings + LLM description embeddings)")
        g.ndata['feat'] = g.ndata['enhanced_feat'].clone()
        logging.info(f"Enhanced feature dimension: {g.ndata['feat'].shape[1]}")
        
        # Count nodes with LLM descriptions
        node_descriptions = data.get('node_descriptions', {})
        logging.info(f"Nodes with LLM descriptions: {len(node_descriptions)}")
    else:
        logging.info("Using raw features (code embeddings only)")
    
    # Ensure required node data exists
    if 'orig_feat' not in g.ndata:
        g.ndata['orig_feat'] = g.ndata['feat'].clone()
    
    # Topology features
    if 'topo_feat' not in g.ndata:
        g.ndata['topo_feat'] = torch.from_numpy(topo_features).to(device)
    else:
        g.ndata['topo_feat'] = g.ndata['topo_feat'].to(device)
    
    # 2. Compute class weights
    train_mask = g.ndata['train_mask'].cpu().numpy()
    train_labels = labels[train_mask]
    class_counts = Counter(train_labels)
    n_train = len(train_labels)
    
    class_weights = {
        c: n_train / (len(class_counts) * count * 0.8 + 0.2 * n_train / len(class_counts))
        for c, count in class_counts.items()
    }
    
    alpha = np.array([class_weights[i] for i in range(len(class_weights))])
    alpha = alpha / alpha.sum()
    
    # 3. Initialize model
    in_dim = g.ndata['feat'].shape[1]
    topo_dim = g.ndata['topo_feat'].shape[1]
    n_classes = len(le.classes_)
    
    model = EnhancedGATWithTopo(
        in_dim=in_dim,
        hidden_dim=TrainConfig.HIDDEN_DIM,
        n_classes=n_classes,
        topo_dim=topo_dim,
        dropout=TrainConfig.DROPOUT,
        use_topo=TrainConfig.USE_TOPO_FEATURES
    ).to(device)
    
    logging.info(f"Model initialized: in_dim={in_dim}, hidden_dim={TrainConfig.HIDDEN_DIM}, "
                 f"topo_dim={topo_dim}, n_classes={n_classes}")
    logging.info(f"Using topology features: {TrainConfig.USE_TOPO_FEATURES}")
    
    # 4. Loss function and optimizer
    criterion = AdaptiveFocalLoss(
        num_classes=n_classes,
        gamma=2.0,
        smoothing=0.1,
        alpha=alpha
    ).to(device)
    
    feature_params = list(model.feature_tuner.parameters())
    other_params = [p for p in model.parameters() if not any(p is fp for fp in feature_params)]
    
    optimizer = torch.optim.AdamW([
        {'params': feature_params, 'lr': 3e-5},
        {'params': other_params, 'lr': TrainConfig.LEARNING_RATE}
    ], weight_decay=TrainConfig.WEIGHT_DECAY)
    
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=8, T_mult=2, eta_min=1e-6)
    
    # 5. Training loop
    best_f1 = 0.0
    patience_counter = 0
    history = {'train_loss': [], 'train_acc': [], 'val_acc': [], 'val_f1': [], 'val_mcc': []}
    
    logging.info("Start training...")
    
    for epoch in range(TrainConfig.EPOCHS):
        model.update_epoch()
        model.train()
        optimizer.zero_grad()
        
        # Forward pass
        features = g.ndata['feat']
        topo_feat = g.ndata['topo_feat'] if TrainConfig.USE_TOPO_FEATURES else None
        
        logits, l2_loss, feat_repr = model(g, features, topo_feat)
        
        # Compute loss
        train_mask = g.ndata['train_mask']
        train_logits = logits[train_mask]
        train_labels = g.ndata['label'][train_mask]
        train_features = feat_repr[train_mask]
        
        loss = criterion(train_logits, train_labels, None, epoch, train_features)
        total_loss = loss + l2_loss
        
        # Backpropagation
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        if epoch > 0 and epoch % 8 == 0:
            scheduler.step()
        
        # Evaluation
        if epoch % 5 == 0 or epoch == TrainConfig.EPOCHS - 1:
            model.eval()
            with torch.no_grad():
                eval_logits, _, _ = model(g, g.ndata['feat'], 
                                         g.ndata['topo_feat'] if TrainConfig.USE_TOPO_FEATURES else None)
                pred = eval_logits.argmax(1)
                
                # Training accuracy
                train_acc = (pred[g.ndata['train_mask']] == g.ndata['label'][g.ndata['train_mask']]).float().mean().item()
                
                # Validation metrics
                val_mask = g.ndata['val_mask']
                y_true_val = g.ndata['label'][val_mask].cpu().numpy()
                y_pred_val = pred[val_mask].cpu().numpy()
                
                report_val = classification_report(y_true_val, y_pred_val, output_dict=True, zero_division=0)
                val_acc = report_val['accuracy']
                val_f1 = report_val['weighted avg']['f1-score']
                val_mcc = matthews_corrcoef(y_true_val, y_pred_val)
                
                history['train_loss'].append(loss.item())
                history['train_acc'].append(train_acc)
                history['val_acc'].append(val_acc)
                history['val_f1'].append(val_f1)
                history['val_mcc'].append(val_mcc)
                
                logging.info(f"Epoch {epoch}/{TrainConfig.EPOCHS} | Loss: {loss.item():.4f} | "
                            f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | "
                            f"Val F1: {val_f1:.4f} | Val MCC: {val_mcc:.4f}")
                
                # Early stopping
                if epoch > 5:
                    if val_f1 > best_f1 + 1e-5:
                        best_f1 = val_f1
                        patience_counter = 0
                        
                        # Save model
                        model_path = os.path.join(TrainConfig.OUTPUT_DIR, f'best_model_{current_time}.pth')
                        torch.save({
                            'model': model.state_dict(),
                            'epoch': epoch,
                            'val_f1': val_f1,
                            'label_encoder': le,
                            'config': {
                                'hidden_dim': TrainConfig.HIDDEN_DIM,
                                'use_topo': TrainConfig.USE_TOPO_FEATURES
                            }
                        }, model_path)
                        logging.info(f"Saved best model, Val F1: {val_f1:.4f}")
                    else:
                        patience_counter += 1
                
                if patience_counter >= TrainConfig.PATIENCE:
                    logging.info("Early stopping triggered; stop training")
                    break
    
    # 6. Final evaluation
    logging.info("\nFinal test set evaluation...")
    
    model_path = os.path.join(TrainConfig.OUTPUT_DIR, f'best_model_{current_time}.pth')
    if os.path.exists(model_path):
        # weights_only=False is required because the checkpoint includes non-tensor objects such as LabelEncoder.
        checkpoint = torch.load(model_path, weights_only=False)
        model.load_state_dict(checkpoint['model'])
        logging.info(f"Loaded best model (epoch {checkpoint.get('epoch', 'unknown')})")
    
    model.eval()
    with torch.no_grad():
        logits, _, _ = model(g, g.ndata['feat'],
                            g.ndata['topo_feat'] if TrainConfig.USE_TOPO_FEATURES else None)
        pred = logits.argmax(1)
        
        test_mask = g.ndata['test_mask']
        y_true = g.ndata['label'][test_mask].cpu().numpy()
        y_pred = pred[test_mask].cpu().numpy()
        
        final_mcc = matthews_corrcoef(y_true, y_pred)
        
        logging.info("\nFinal classification report:")
        logging.info(classification_report(y_true, y_pred, target_names=le.classes_, zero_division=0))
        logging.info(f"Final MCC: {final_mcc:.4f}")
        
        # Save results
        results = {
            'test_mcc': float(final_mcc),
            'classification_report': classification_report(y_true, y_pred, target_names=le.classes_, 
                                                          output_dict=True, zero_division=0),
            'history': history,
            'config': {
                'hidden_dim': TrainConfig.HIDDEN_DIM,
                'use_topo': TrainConfig.USE_TOPO_FEATURES,
                'graph_path': graph_path
            }
        }
        
        results_path = os.path.join(TrainConfig.OUTPUT_DIR, f'results_{current_time}.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        logging.info(f"Results saved to: {results_path}")
    
    logging.info("=" * 60)
    logging.info("Phase 2 completed!")
    logging.info("=" * 60)


# ===================== Main Function =====================
def main():
    parser = argparse.ArgumentParser(description='Phase 2: train with the optimized graph')
    parser.add_argument('--graph_path', type=str, default=TrainConfig.DEFAULT_GRAPH_PATH,
                        help='Path to the optimized graph artifact')
    parser.add_argument('--hidden_dim', type=int, default=TrainConfig.HIDDEN_DIM,
                        help='Hidden dimension')
    parser.add_argument('--epochs', type=int, default=TrainConfig.EPOCHS,
                        help='Number of training epochs')
    parser.add_argument('--patience', type=int, default=TrainConfig.PATIENCE,
                        help='Early-stopping patience')
    parser.add_argument('--learning_rate', type=float, default=TrainConfig.LEARNING_RATE,
                        help='Learning rate for the main model parameters')
    parser.add_argument('--weight_decay', type=float, default=TrainConfig.WEIGHT_DECAY,
                        help='AdamW weight decay')
    parser.add_argument('--dropout', type=float, default=TrainConfig.DROPOUT,
                        help='Dropout ratio')
    parser.add_argument('--output_dir', type=str, default=TrainConfig.OUTPUT_DIR,
                        help='Directory for training logs, checkpoints, and results')
    parser.add_argument('--seed', type=int, default=TrainConfig.SEED,
                        help='Random seed')
    parser.add_argument('--no_topo', action='store_true',
                        help='Disable topology features')
    parser.add_argument('--use_enhanced_feat', action='store_true',
                        help='Use enhanced features (code embeddings + LLM description embeddings)')
    parser.add_argument('--no_enhanced_feat', action='store_true',
                        help='Disable enhanced features and use raw code embeddings only')
    
    args = parser.parse_args()
    
    # Update configuration
    TrainConfig.HIDDEN_DIM = args.hidden_dim
    TrainConfig.EPOCHS = args.epochs
    TrainConfig.PATIENCE = args.patience
    TrainConfig.LEARNING_RATE = args.learning_rate
    TrainConfig.WEIGHT_DECAY = args.weight_decay
    TrainConfig.DROPOUT = args.dropout
    TrainConfig.OUTPUT_DIR = args.output_dir
    TrainConfig.SEED = args.seed
    TrainConfig.USE_TOPO_FEATURES = not args.no_topo
    set_seed(TrainConfig.SEED)
    
    # Handle enhanced-feature options
    if args.use_enhanced_feat:
        TrainConfig.USE_ENHANCED_FEAT = True
    elif args.no_enhanced_feat:
        TrainConfig.USE_ENHANCED_FEAT = False
    # Otherwise, keep the default value.
    
    train_model(args.graph_path)


if __name__ == "__main__":
    main()
