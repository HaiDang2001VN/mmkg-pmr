"""
RotatE model inheriting from VLKGEBase
"""

import torch
import torch.nn as nn
import math
from vlkge.models.vlkge import VLKGEBase


class RotatE(VLKGEBase):
    """
    RotatE: Knowledge Graph Embedding by Relational Rotation in Complex Space
    Uses rotation in complex space with relation as rotation angles
    
    Reference: Sun et al. "RotatE: Knowledge Graph Embedding by Relational Rotation in Complex Space" (ICLR 2019)
    """
    
    def __init__(self, num_entities, num_relations, embedding_dim,
                 visual_features=None, textual_features=None, relation_features=None,
                 visual_entity_to_index=None, textual_entity_to_index=None,
                 fusion_mode="average", 
                 use_structural=True, use_visual=False, use_textual=False,
                 freeze_visual=True, freeze_textual=True,
                 visual_proj=False, textual_proj=False,
                 shared_projection=None,
                 inductive=False,
                 modality_asymmetry=False,
                 normalize_before_fusion=False,
                 raw_margin=12.0,
                 device=torch.device('cpu')):
        """
        Args:
            ... (standard VLKGEBase args)
            relation_features: Optional pretrained relation features
            raw_margin: Margin value for scoring
        """
        super().__init__(
            num_entities=num_entities,
            num_relations=num_relations,
            embedding_dim=embedding_dim,
            visual_features=visual_features,
            textual_features=textual_features,
            visual_entity_to_index=visual_entity_to_index,
            textual_entity_to_index=textual_entity_to_index,
            fusion_mode=fusion_mode,
            use_structural=use_structural,
            use_visual=use_visual,
            use_textual=use_textual,
            freeze_visual=freeze_visual,
            freeze_textual=freeze_textual,
            visual_proj=visual_proj,
            textual_proj=textual_proj,
            shared_projection=shared_projection,
            inductive=inductive,
            modality_asymmetry=modality_asymmetry,
            normalize_before_fusion=normalize_before_fusion,
            device=device
        )
        
        self.raw_margin = nn.Parameter(torch.tensor(raw_margin), requires_grad=False)
        
        # Use final embedding_dim (which includes concat adjustment if needed)
        final_dim = self.embedding_dim
        
        # ==================== RotatE-Specific: Imaginary Embeddings ====================
        if self.inductive:
            # WikiArt: Complex imaginary construction
            self._setup_rotate_imaginary_wikiart(final_dim)
        else:
            # WN9-IMG: Simple imaginary embeddings
            self.entity_embeddings_imag = nn.Embedding(num_entities, final_dim)
        
        # ==================== RotatE-Specific: Relation Embeddings (as angles) ====================
        if relation_features is not None:
            self.relation_embeddings = nn.Embedding.from_pretrained(relation_features, freeze=False)
            self.relation_proj = nn.Linear(self.relation_embeddings.embedding_dim, final_dim)
        else:
            self.relation_embeddings = nn.Embedding(num_relations, final_dim)
            self.relation_proj = None
        
        # Initialize parameters
        self.reset_parameters()
    
    def _setup_rotate_imaginary_wikiart(self, dim):
        """Setup complex imaginary part for WikiArt"""
        # Projection from real to imaginary
        self.vis_to_imag = nn.Linear(dim, dim, bias=False)
        self.head_imag_gate = nn.Parameter(torch.tensor(0.0))
        
        # Textual-based imaginary embeddings
        if self.use_textual:
            num_textual = len(self.textual_entity_to_index)
            self.entity_embeddings_imag = nn.Embedding(num_textual, dim)
    
    def reset_parameters(self):
        """Initialize in exact order matching original implementation"""
        # 1. Structural embeddings (real part)
        if self.use_structural:
            nn.init.xavier_uniform_(self.entity_embeddings.weight)
        
        # 2. Imaginary embeddings
        if self.inductive:
            if self.use_textual and hasattr(self, 'entity_embeddings_imag'):
                nn.init.xavier_uniform_(self.entity_embeddings_imag.weight)
            if hasattr(self, 'vis_to_imag'):
                nn.init.zeros_(self.vis_to_imag.weight)
        else:
            nn.init.xavier_uniform_(self.entity_embeddings_imag.weight)
        
        # 3. Relation angles
        if self.relation_embeddings.weight.requires_grad:
            nn.init.uniform_(self.relation_embeddings.weight, 0, 2 * math.pi)
        
        # 4. Visual projection
        if self.use_visual and hasattr(self, 'visual_linear'):
            if isinstance(self.visual_linear, nn.Linear):
                nn.init.xavier_uniform_(self.visual_linear.weight)
        
        # 5. Textual projection
        if self.use_textual and hasattr(self, 'textual_linear'):
            if isinstance(self.textual_linear, nn.Linear):
                nn.init.xavier_uniform_(self.textual_linear.weight)
        
        # 6. Relation projection
        if self.relation_proj is not None:
            nn.init.xavier_uniform_(self.relation_proj.weight)
        
        # 7. Shared projection (if used)
        if len(self.shared_projection) > 0 and hasattr(self, 'unified_projection'):
            if isinstance(self.unified_projection, nn.Linear):
                nn.init.xavier_uniform_(self.unified_projection.weight)
    
    def get_imaginary_embedding_wn9img(self, entity_ids):
        """Get imaginary embeddings for WN9-IMG (simple)"""
        return self.entity_embeddings_imag(entity_ids)
    
    def get_imaginary_embedding_wikiart(self, entity_ids, real_emb):
        """Get imaginary embeddings for WikiArt (complex construction)"""
        imag_emb = torch.zeros_like(real_emb)
        
        # Prefer textual-based imaginary if available
        if self.use_textual and hasattr(self, 'entity_embeddings_imag'):
            textual_indices = self.textual_id_lookup[entity_ids]
            textual_mask = textual_indices != -1
            if textual_mask.any():
                imag_emb[textual_mask] = self.entity_embeddings_imag(textual_indices[textual_mask])
        
        # For entities without textual: project from real via learned transformation
        if self.use_visual:
            visual_indices = self.visual_id_lookup[entity_ids]
            visual_mask = visual_indices != -1
            
            if self.use_textual and hasattr(self, 'entity_embeddings_imag'):
                # Exclude entities that already have textual imaginary
                textual_indices = self.textual_id_lookup[entity_ids]
                textual_mask = textual_indices != -1
                visual_mask = visual_mask & (~textual_mask)
            
            if visual_mask.any():
                imag_emb[visual_mask] = torch.tanh(self.head_imag_gate) * self.vis_to_imag(real_emb[visual_mask])
        
        return imag_emb
   
    def get_relation_representations(self, relation_ids):
        """Get relation embeddings (real part)"""
        rel_emb = self.relation_embeddings(relation_ids)
        
        if self.relation_proj is not None:
            rel_emb = self.relation_proj(rel_emb)
        
        return rel_emb
        
    def forward(self, head, relation, tail):
        """
        RotatE forward pass: score = margin - ||h * r - t||
        "*" denotes element-wise (Hadamard) complex multiplication
        
        Args:
            head: Head entity IDs (batch_size,)
            relation: Relation IDs (batch_size,)
            tail: Tail entity IDs (batch_size,)
            
        Returns:
            scores: Triple scores (batch_size,), higher is better
        """
        # Get real parts (fused representations from VLKGEBase)
        head_real = self.get_entity_representations(head)
        tail_real = self.get_entity_representations(tail)
        
        # Get imaginary parts (RotatE-specific)
        if self.inductive:
            head_imag = self.get_imaginary_embedding_wikiart(head, head_real)
            tail_imag = self.get_imaginary_embedding_wikiart(tail, tail_real)
        else:
            head_imag = self.get_imaginary_embedding_wn9img(head)
            tail_imag = self.get_imaginary_embedding_wn9img(tail)
        
        # Relation angles
        relation_real = self.get_relation_representations(relation)
        
        rel_re = torch.cos(relation_real)
        rel_im = torch.sin(relation_real)
        
        # RotatE scoring
        re_score = (rel_re * head_real - rel_im * head_imag) - tail_real
        im_score = (rel_re * head_imag + rel_im * head_real) - tail_imag
        complex_score = torch.stack([re_score, im_score], dim=2)
        
        score = self.raw_margin - torch.linalg.vector_norm(complex_score, dim=(1, 2))
        
        return score