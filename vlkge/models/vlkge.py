"""
VLKGEBase: Vision-Language Knowledge Graph Embedding Base Class
Paper: "VL-KGE: Vision-Language Models Meet Knowledge Graph Embeddings"

Unified handling of multimodal features with flexible projection strategies.
Supports structural, visual, and textual modalities with various fusion modes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class VLKGEBase(nn.Module):
    """
    Base class for Vision-Language Knowledge Graph Embedding models.
    
    Features:
    - Multimodal embeddings (structural, visual, textual)
    - Sparse feature support via entity ID mapping
    - Optional residual adapters for visual/textual features
    - Unified projection layer sharing across modalities
    - Flexible fusion strategies (average, concat, weighted, addition)
    - Inductive learning support with entity masking
    - Modality asymmetry handling for datasets where entities have different modality combinations
    
    Subclasses must:
    - Define their own relation embeddings (model-specific)
    - Implement their own reset_parameters()
    - Implement their scoring function in forward()
    """
    
    def __init__(self, num_entities, num_relations, embedding_dim,
                 visual_features=None, textual_features=None,
                 visual_entity_to_index=None, textual_entity_to_index=None,
                 fusion_mode="average", 
                 use_structural=True, use_visual=False, use_textual=False,
                 freeze_visual=True, freeze_textual=True,
                 visual_proj=False, textual_proj=False,
                 vis_adapter_dim=64, txt_adapter_dim=64,
                 shared_projection=None,
                 inductive=False,
                 modality_asymmetry=False,
                 normalize_before_fusion=False,
                 device=torch.device('cpu')):
        """
        Args:
            num_entities: Total number of entities
            num_relations: Total number of relations (stored for subclasses)
            embedding_dim: Dimension of embeddings (also output dimension after fusion, 
                          unless fusion_mode="concat")
            visual_features: Pretrained visual features tensor (num_visual_entities, visual_dim)
            textual_features: Pretrained textual features tensor (num_textual_entities, textual_dim)
            visual_entity_to_index: Dict mapping entity_id -> index for sparse visual features
            textual_entity_to_index: Dict mapping entity_id -> index for sparse textual features
            fusion_mode: How to combine modalities ("average", "concat", "weighted", "addition")
            use_structural: Whether to use trainable structural embeddings
            use_visual: Whether to use visual features
            use_textual: Whether to use textual features
            freeze_visual: Freeze visual encoder weights
            freeze_textual: Freeze textual encoder weights
            visual_proj: Add residual adapter for visual features (default: False)
            textual_proj: Add residual adapter for textual features (default: False)
            shared_projection: List of modalities to share projection layer 
                              e.g., ["visual", "textual"] or ["structural", "visual", "textual"]
                              (default: None, each modality has its own projection)
            inductive: Enable inductive learning with entity masking (default: False)
            modality_asymmetry: Handle per-entity modality combinations (default: False)
                               Set to True for datasets like WikiArt where different entities 
                               have different modality combinations (e.g., artworks have visual, 
                               artists have textual). Set to False for datasets like WN9-IMG 
                               where all entities have the same modalities.
            normalize_before_fusion: L2 normalize embeddings before fusion
            device: Computation device
        """
        super().__init__()
        
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim
        self.base_embedding_dim = embedding_dim
        self.device = device
        
        # Fusion and normalization settings
        self.fusion_mode = fusion_mode
        self.normalize_before_fusion = normalize_before_fusion
        
        # Modality flags
        self.use_structural = use_structural
        self.use_visual = use_visual
        self.use_textual = use_textual
        self.freeze_visual = freeze_visual
        self.freeze_textual = freeze_textual
        
        # Projection settings
        self.visual_proj = visual_proj
        self.textual_proj = textual_proj
        self.shared_projection = shared_projection if shared_projection is not None else []
        
        # Inductive learning and modality asymmetry
        self.inductive = inductive
        self.modality_asymmetry = modality_asymmetry
        
        # ==================== Structural Embeddings ====================
        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim) if use_structural else None
        
        # ==================== Visual Features ====================
        if use_visual:
            if visual_features is None or visual_entity_to_index is None:
                raise ValueError("use_visual=True requires both visual_features and visual_entity_to_index")
            
            self.vis_adapter_dim = vis_adapter_dim
            self.visual_adapter = None
            self.visual_gate = None
            self.visual_ln = None
            
            # Entity ID -> index mapping
            self.visual_entity_to_index = visual_entity_to_index
            self.visual_id_lookup = torch.full((num_entities,), -1, dtype=torch.long)
            for ent_id, idx in self.visual_entity_to_index.items():
                self.visual_id_lookup[ent_id] = idx
            self.visual_id_lookup = self.visual_id_lookup.to(device)
            
            # Visual embeddings
            self.visual_embeddings = nn.Embedding.from_pretrained(visual_features, freeze=freeze_visual)
            
            # Automatic dimension projection (always when dims mismatch)
            v_in = self.visual_embeddings.embedding_dim
            self.visual_linear = (nn.Linear(v_in, embedding_dim, bias=False)
                                 if v_in != embedding_dim else nn.Identity())
            
            # Optional residual adapter
            if self.visual_proj:
                self.visual_adapter = nn.Sequential(
                    nn.Linear(self.embedding_dim, self.vis_adapter_dim, bias=False),
                    nn.ReLU(),
                    nn.Linear(self.vis_adapter_dim, self.embedding_dim, bias=False),
                )
                for m in self.visual_adapter:
                    if isinstance(m, nn.Linear):
                        nn.init.zeros_(m.weight)  # residual = no-op initially
                self.visual_gate = nn.Parameter(torch.tensor(0.0))
                self.visual_ln = nn.LayerNorm(self.embedding_dim, elementwise_affine=False)
        
        # ==================== Textual Features ====================
        if use_textual:
            if textual_features is None or textual_entity_to_index is None:
                raise ValueError("use_textual=True requires both textual_features and textual_entity_to_index")
            
            self.txt_adapter_dim = txt_adapter_dim
            self.textual_adapter = None
            self.textual_gate = None
            self.textual_ln = None
            
            # Entity ID -> index mapping
            self.textual_entity_to_index = textual_entity_to_index
            self.textual_id_lookup = torch.full((num_entities,), -1, dtype=torch.long)
            for ent_id, idx in self.textual_entity_to_index.items():
                self.textual_id_lookup[ent_id] = idx
            self.textual_id_lookup = self.textual_id_lookup.to(device)
            
            # Textual embeddings
            self.textual_embeddings = nn.Embedding.from_pretrained(textual_features, freeze=freeze_textual)
            
            # Automatic dimension projection (always when dims mismatch)
            t_in = self.textual_embeddings.embedding_dim
            self.textual_linear = (nn.Linear(t_in, embedding_dim, bias=False)
                                  if t_in != embedding_dim else nn.Identity())
            
            # Optional residual adapter
            if self.textual_proj:
                self.textual_adapter = nn.Sequential(
                    nn.Linear(self.embedding_dim, self.txt_adapter_dim, bias=False),
                    nn.ReLU(),
                    nn.Linear(self.txt_adapter_dim, self.embedding_dim, bias=False),
                )
                for m in self.textual_adapter:
                    if isinstance(m, nn.Linear):
                        nn.init.zeros_(m.weight)
                self.textual_gate = nn.Parameter(torch.tensor(0.0))
                self.textual_ln = nn.LayerNorm(self.embedding_dim, elementwise_affine=False)
        
        # ==================== Shared Projection Layer ====================
        if len(self.shared_projection) > 0:
            self._setup_shared_projection()
        
        # ==================== Fusion Configuration ====================
        # Adjust embedding_dim only for concat mode
        if fusion_mode == "concat":
            num_modalities = sum([use_structural, use_visual, use_textual])
            self.embedding_dim = embedding_dim * num_modalities
        
        # Weighted fusion parameters
        if fusion_mode == "weighted":
            num_modalities = sum([use_structural, use_visual, use_textual])
            self.fusion_weights = nn.Parameter(torch.ones(num_modalities))
        
        # ==================== Inductive Learning Support ====================
        if self.inductive:
            self.register_buffer("seen_in_train", torch.zeros(num_entities, dtype=torch.bool))
    
    def _setup_shared_projection(self):
        """
        Setup shared projection layer for specified modalities.
        Replaces individual linear/adapter projections with a shared one.
        """
        # Collect modality dimensions
        modality_dims = {}
        if "structural" in self.shared_projection and self.use_structural:
            modality_dims["structural"] = self.base_embedding_dim
        if "visual" in self.shared_projection and self.use_visual:
            modality_dims["visual"] = self.visual_embeddings.embedding_dim
        if "textual" in self.shared_projection and self.use_textual:
            modality_dims["textual"] = self.textual_embeddings.embedding_dim
        
        if len(modality_dims) == 0:
            raise ValueError("No valid modalities specified for shared projection")
        
        # Verify all have same input dimension (required for true sharing)
        unique_dims = set(modality_dims.values())
        if len(unique_dims) > 1:
            raise ValueError(
                f"Shared projection requires all modalities to have same input dimension. "
                f"Got: {modality_dims}"
            )
        
        input_dim = list(unique_dims)[0]
        
        # create a shared learnable projection across modalities
        self.unified_projection = nn.Linear(input_dim, self.base_embedding_dim, bias=False)
        
        # Point all specified modalities to use the shared projection
        if "visual" in self.shared_projection and self.use_visual:
            self.visual_linear = self.unified_projection
        
        if "textual" in self.shared_projection and self.use_textual:
            self.textual_linear = self.unified_projection
    
    def mark_seen_entities(self, seen_ids):
        """Mark entities as seen during training (for inductive evaluation)"""
        if self.inductive:
            self.seen_in_train[seen_ids.to(self.device)] = True
    
    def get_visual_embedding(self, entity_ids):
        """Get visual embeddings for entity IDs with optional residual adapter"""
        if not self.use_visual:
            return None
        
        base_dim = self.base_embedding_dim
        indices = self.visual_id_lookup[entity_ids]
        mask = indices != -1
        emb = torch.zeros(len(entity_ids), base_dim, device=self.device)
        
        if mask.any():
            v = self.visual_embeddings(indices[mask])
            
            if self.freeze_visual:
                v = v.detach()
            
            # Apply dimension projection
            if not isinstance(self.visual_linear, nn.Identity):
                v = self.visual_linear(v)
            
            # Apply optional residual adapter
            if self.visual_adapter is not None:
                v_adapted = self.visual_adapter(v)
                v = v + torch.tanh(self.visual_gate) * self.visual_ln(v_adapted)
            
            emb[mask] = v
        
        return emb

    def get_textual_embedding(self, entity_ids):
        """Get textual embeddings for entity IDs with optional residual adapter"""
        if not self.use_textual:
            return None
        
        base_dim = self.base_embedding_dim
        indices = self.textual_id_lookup[entity_ids]
        mask = indices != -1
        emb = torch.zeros(len(entity_ids), base_dim, device=self.device)
        
        if mask.any():
            t = self.textual_embeddings(indices[mask])
            
            if self.freeze_textual:
                t = t.detach()
            
            # Apply dimension projection
            if not isinstance(self.textual_linear, nn.Identity):
                t = self.textual_linear(t)
            
            # Apply optional residual adapter
            if self.textual_adapter is not None:
                t_adapted = self.textual_adapter(t)
                t = t + torch.tanh(self.textual_gate) * self.textual_ln(t_adapted)
            
            emb[mask] = t
        
        return emb
    
    def get_structural_embedding(self, entity_ids):
        """Get structural embeddings with masking for unseen entities"""
        if not self.use_structural:
            return None
        
        emb = self.entity_embeddings(entity_ids)
        
        # Mask unseen entities for inductive evaluation
        if self.inductive:
            emb = emb * self.seen_in_train[entity_ids].unsqueeze(1)
            
        # Apply shared projection if structural is included
        if "structural" in self.shared_projection:
            emb = self.unified_projection(emb)
            
        return emb
    
    def _fuse_embeddings_sparse(self, structural, visual, textual, entity_ids):
        """
        Per-entity fusion for sparse features (WikiArt-style datasets).
        Each entity may have different modalities available.
        
        This handles cases where:
        - Entity A has only visual
        - Entity B has only textual  
        - Entity C has both visual and textual
        
        All in the same batch, and we need correct per-entity averaging.
        """
        batch_size = entity_ids.shape[0]
        base_dim = self.base_embedding_dim
        
        # Determine which modalities each entity has (based on enabled modalities, not zero values)
        has_structural = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        has_visual = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        has_textual = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        
        if structural is not None and self.use_structural:
            # Structural is "available" for all entities when use_structural=True
            # Even if masked to zero for unseen entities
            has_structural = torch.ones(batch_size, dtype=torch.bool, device=self.device)
        
        if self.use_visual:
            visual_indices = self.visual_id_lookup[entity_ids]
            has_visual = (visual_indices != -1)
        
        if self.use_textual:
            textual_indices = self.textual_id_lookup[entity_ids]
            has_textual = (textual_indices != -1)
        
        # Count modalities per entity
        num_modalities_per_entity = (has_structural.long() + has_visual.long() + has_textual.long())
        
        # Handle different fusion modes
        if self.fusion_mode == "concat":
            # Concat: always pad to full dimension (already handled, just return)
            concat_parts = []
            
            if self.use_structural:
                if structural is not None:
                    concat_parts.append(structural)
                else:
                    concat_parts.append(torch.zeros(batch_size, base_dim, device=self.device))
            
            if self.use_visual:
                if visual is not None:
                    concat_parts.append(visual)
                else:
                    concat_parts.append(torch.zeros(batch_size, base_dim, device=self.device))
            
            if self.use_textual:
                if textual is not None:
                    concat_parts.append(textual)
                else:
                    concat_parts.append(torch.zeros(batch_size, base_dim, device=self.device))
            
            return torch.cat(concat_parts, dim=-1)
        
        elif self.fusion_mode == "average":
            # Average: sum all enabled modalities (including zeros), divide by count
            result = torch.zeros(batch_size, base_dim, device=self.device)
            
            if structural is not None and self.use_structural:
                result = result + structural  # Adds zeros for unseen, non-zero for seen
            if visual is not None and self.use_visual:
                result = result + visual
            if textual is not None and self.use_textual:
                result = result + textual
            
            # Divide by number of ENABLED modalities per entity (not non-zero modalities)
            num_modalities_per_entity = num_modalities_per_entity.clamp(min=1).float().unsqueeze(1)
            result = result / num_modalities_per_entity
            
            return result
        
        elif self.fusion_mode == "addition":
            # Addition: just sum
            result = torch.zeros(batch_size, base_dim, device=self.device)
            
            if structural is not None and self.use_structural:
                result = result + structural
            if visual is not None and self.use_visual:
                result = result + visual
            if textual is not None and self.use_textual:
                result = result + textual
            
            return result
        
        elif self.fusion_mode == "weighted":
            # Weighted: use all enabled modalities (including masked ones)
            result = torch.zeros(batch_size, base_dim, device=self.device)
            
            # Build modality list in order
            modality_masks = []
            modality_tensors = []
            
            if self.use_structural:
                modality_masks.append(has_structural)  # Now always True when use_structural=True
                modality_tensors.append(structural if structural is not None 
                                       else torch.zeros(batch_size, base_dim, device=self.device))
            if self.use_visual:
                modality_masks.append(has_visual)
                modality_tensors.append(visual if visual is not None 
                                       else torch.zeros(batch_size, base_dim, device=self.device))
            if self.use_textual:
                modality_masks.append(has_textual)
                modality_tensors.append(textual if textual is not None 
                                       else torch.zeros(batch_size, base_dim, device=self.device))
            
            # Stack masks: (num_modalities, batch_size)
            modality_masks_stacked = torch.stack(modality_masks, dim=0)
            
            # For each entity, compute weighted sum of available modalities
            for i in range(batch_size):
                available_mask = modality_masks_stacked[:, i]
                
                if available_mask.any():
                    # Get weights for available modalities and renormalize
                    available_weights = self.fusion_weights[available_mask]
                    weights = F.softmax(available_weights, dim=0)
                    
                    # Weighted sum
                    for j, (has_mod, tensor) in enumerate(zip(available_mask, modality_tensors)):
                        if has_mod:
                            weight_idx = available_mask[:j+1].sum() - 1
                            result[i] += weights[weight_idx] * tensor[i]
            
            return result
        
        else:
            raise ValueError(f"Unknown fusion mode: {self.fusion_mode}")
    
    def fuse_embeddings(self, structural, visual, textual):
        """
        Simple fusion for dense features (WN9-IMG style).
        Assumes all entities have the same modalities available.
        
        For sparse features, use _fuse_embeddings_sparse instead.
        """
        # Get batch size
        batch_size = None
        if structural is not None:
            batch_size = structural.shape[0]
        elif visual is not None:
            batch_size = visual.shape[0]
        elif textual is not None:
            batch_size = textual.shape[0]
        else:
            raise ValueError("No embeddings available for fusion")
        
        base_dim = self.base_embedding_dim
        
        # Collect available embeddings
        available = []
        if structural is not None:
            available.append(structural)
        if visual is not None:
            available.append(visual)
        if textual is not None:
            available.append(textual)
        
        if len(available) == 0:
            raise ValueError("No embeddings available for fusion")
        
        # Fusion logic
        if self.fusion_mode == "concat":
            concat_parts = []
            
            if self.use_structural:
                concat_parts.append(structural if structural is not None 
                                   else torch.zeros(batch_size, base_dim, device=self.device))
            if self.use_visual:
                concat_parts.append(visual if visual is not None 
                                   else torch.zeros(batch_size, base_dim, device=self.device))
            if self.use_textual:
                concat_parts.append(textual if textual is not None 
                                   else torch.zeros(batch_size, base_dim, device=self.device))
            
            return torch.cat(concat_parts, dim=-1)
        
        elif self.fusion_mode == "average":
            return sum(available) / len(available)
        
        elif self.fusion_mode == "addition":
            return sum(available)
        
        elif self.fusion_mode == "weighted":
            # Map available embeddings to indices
            modality_order = []
            if self.use_structural:
                modality_order.append('structural')
            if self.use_visual:
                modality_order.append('visual')
            if self.use_textual:
                modality_order.append('textual')
            
            available_names = []
            if structural is not None:
                available_names.append('structural')
            if visual is not None:
                available_names.append('visual')
            if textual is not None:
                available_names.append('textual')
            
            available_indices = [modality_order.index(name) for name in available_names]
            available_weights = self.fusion_weights[available_indices]
            weights = F.softmax(available_weights, dim=0)
            
            return sum(w * emb for w, emb in zip(weights, available))
        
        else:
            raise ValueError(f"Unknown fusion mode: {self.fusion_mode}")
            
    def get_entity_representations(self, entity_ids):
        """
        Get fused entity representations.
        Returns embeddings in self.embedding_dim dimensions.
        
        Handles both dense features (all entities have same modalities) and
        sparse features with modality asymmetry (entities have different modality combinations).
        """
        # Get individual modality embeddings
        structural = self.get_structural_embedding(entity_ids)
        visual = self.get_visual_embedding(entity_ids)
        textual = self.get_textual_embedding(entity_ids)
        
        # Optional normalization before fusion
        if self.normalize_before_fusion:
            if structural is not None:
                structural = F.normalize(structural, p=2, dim=-1)
            if visual is not None:
                visual = F.normalize(visual, p=2, dim=-1)
            if textual is not None:
                textual = F.normalize(textual, p=2, dim=-1)
        
        # Use appropriate fusion strategy
        if self.modality_asymmetry:
            # Per-entity fusion: each entity may have different modality combinations
            # (e.g., WikiArt: artworks have visual, artists have textual)
            fused = self._fuse_embeddings_sparse(structural, visual, textual, entity_ids)
        else:
            # Batch-level fusion: all entities have same modalities
            # (e.g., WN9-IMG: all entities have both visual and textual)
            fused = self.fuse_embeddings(structural, visual, textual)
        
        return fused
    
    def forward(self, head, relation, tail):
        """
        Forward pass - must be implemented by subclasses.
        
        Args:
            head: Head entity IDs
            relation: Relation IDs
            tail: Tail entity IDs
            
        Returns:
            Scores for the triples
        """
        raise NotImplementedError("Subclasses must implement forward()")