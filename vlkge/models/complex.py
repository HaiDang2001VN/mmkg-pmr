"""
ComplEx model inheriting from VLKGEBase
"""

import torch
import torch.nn as nn
from vlkge.models.vlkge import VLKGEBase


def triple_dot(x, y, z):
    """Compute Re(<x, y, z>) for ComplEx scoring"""
    return torch.sum(x * y * z, dim=-1)


class ComplEx(VLKGEBase):
    """
    ComplEx: Complex Embeddings for Simple Link Prediction
    Uses complex-valued embeddings with real and imaginary parts
    
    Reference: Trouillon et al. "Complex Embeddings for Simple Link Prediction" (ICML 2016)
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
                 device=torch.device('cpu')):
        """
        Args:
            ... (standard VLKGEBase args)
            relation_features: Optional pretrained relation features
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
        
        # Use final embedding_dim (which includes concat adjustment if needed)
        final_dim = self.embedding_dim
        
        # ==================== ComplEx-Specific: Entity Imaginary Embeddings ====================
        if self.inductive:
            # WikiArt: Complex imaginary construction
            self._setup_complex_imaginary_wikiart(final_dim)
        else:
            # WN9-IMG: Simple imaginary embeddings
            self.entity_embeddings_imag = nn.Embedding(num_entities, final_dim)
        
        # ==================== ComplEx-Specific: Relation Embeddings (real and imaginary) ====================
        if relation_features is not None:
            self.relation_embeddings = nn.Embedding.from_pretrained(relation_features, freeze=False)
            self.relation_proj = nn.Linear(self.relation_embeddings.embedding_dim, final_dim)
        else:
            self.relation_embeddings = nn.Embedding(num_relations, final_dim)
            self.relation_proj = None
        
        # Relation imaginary embeddings
        self.relation_embeddings_imag = nn.Embedding(num_relations, final_dim)
        
        # Initialize parameters
        self.reset_parameters()
    
    def _setup_complex_imaginary_wikiart(self, dim):
        """Setup complex imaginary part for WikiArt"""
        # Projection from real to imaginary
        self.vis_to_imag = nn.Linear(dim, dim, bias=False)
        self.head_imag_gate = nn.Parameter(torch.tensor(0.0))
        
        # Textual-based imaginary embeddings
        if self.use_textual:
            num_textual = len(self.textual_entity_to_index)
            self.entity_embeddings_imag = nn.Embedding(num_textual, dim)
    
    def reset_parameters(self):
        """Initialize ComplEx parameters with Xavier uniform"""
        # 1. Structural embeddings (real part)
        if self.use_structural:
            nn.init.xavier_uniform_(self.entity_embeddings.weight)
        
        # 2. Entity imaginary embeddings
        if self.inductive:
            # WikiArt setup
            if self.use_textual and hasattr(self, 'entity_embeddings_imag'):
                nn.init.xavier_uniform_(self.entity_embeddings_imag.weight)
            if hasattr(self, 'vis_to_imag'):
                nn.init.zeros_(self.vis_to_imag.weight)
        else:
            # WN9-IMG setup
            nn.init.xavier_uniform_(self.entity_embeddings_imag.weight)
        
        # 3. Relation embeddings (real and imaginary)
        nn.init.xavier_uniform_(self.relation_embeddings.weight)
        nn.init.xavier_uniform_(self.relation_embeddings_imag.weight)
        
        # 4. Relation projection if exists
        if self.relation_proj is not None:
            nn.init.xavier_uniform_(self.relation_proj.weight)
        
        # 5. Visual projection layers
        if self.use_visual and hasattr(self, 'visual_linear'):
            if isinstance(self.visual_linear, nn.Linear):
                nn.init.xavier_uniform_(self.visual_linear.weight)
        
        # 6. Textual projection layers
        if self.use_textual and hasattr(self, 'textual_linear'):
            if isinstance(self.textual_linear, nn.Linear):
                nn.init.xavier_uniform_(self.textual_linear.weight)
        
        # 7. Shared projection if exists
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
        ComplEx forward pass
        score = Re(<h, r, t*>) = Re(h) · Re(r) · Re(t) + Im(h) · Re(r) · Im(t)
                                + Re(h) · Im(r) · Im(t) - Im(h) · Im(r) · Re(t)
        
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
        
        # Get imaginary parts (ComplEx-specific)
        if self.inductive:
            head_imag = self.get_imaginary_embedding_wikiart(head, head_real)
            tail_imag = self.get_imaginary_embedding_wikiart(tail, tail_real)
        else:
            head_imag = self.get_imaginary_embedding_wn9img(head)
            tail_imag = self.get_imaginary_embedding_wn9img(tail)
        
        # Get relation embeddings (real and imaginary)
        relation_real = self.get_relation_representations(relation)
        relation_imag = self.relation_embeddings_imag(relation)
        
        # ComplEx scoring function
        score = (triple_dot(head_real, relation_real, tail_real) +
                triple_dot(head_imag, relation_real, tail_imag) +
                triple_dot(head_real, relation_imag, tail_imag) -
                triple_dot(head_imag, relation_imag, tail_real))
        
        return score