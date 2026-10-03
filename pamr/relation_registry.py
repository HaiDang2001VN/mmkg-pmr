"""Canonical relation and inverse-relation registry.

The registry deliberately keeps public dataset relation names separate from
internal inverse IDs.  Internal inverse relations are represented by integer
IDs and manifest metadata, not invented dataset strings.
"""

import hashlib
import json


class RelationRegistry:
    """Stable relation IDs plus explicit inverse semantics."""

    def __init__(self, raw_relations, inverse_pairs=None, symmetric_relations=None):
        raw_names = sorted({str(name) for name in raw_relations})
        if not raw_names:
            raise ValueError("Relation registry cannot be empty")

        inverse_pairs = inverse_pairs or {}
        symmetric_relations = set(symmetric_relations or [])

        if not isinstance(inverse_pairs, dict):
            raise TypeError("inverse_pairs must be a relation-name mapping")

        unknown_symmetric = symmetric_relations - set(raw_names)
        if unknown_symmetric:
            raise ValueError(
                "symmetric_relations contains unknown relations: "
                f"{sorted(unknown_symmetric)}"
            )

        raw_set = set(raw_names)
        pair_by_name = {}
        for left, right in inverse_pairs.items():
            if left not in raw_set or right not in raw_set:
                raise ValueError(
                    f"inverse pair ({left}, {right}) is not entirely present "
                    "in the raw relation vocabulary"
                )
            if left == right:
                raise ValueError(
                    f"Relation {left} cannot be both a directed pair and itself"
                )
            for source, target in ((left, right), (right, left)):
                existing = pair_by_name.get(source)
                if existing is not None and existing != target:
                    raise ValueError(
                        f"Relation {source} has conflicting inverse mappings: "
                        f"{existing} and {target}"
                    )
                pair_by_name[source] = target

        overlap = symmetric_relations & set(pair_by_name)
        if overlap:
            raise ValueError(
                "Relations cannot be both symmetric and directed inverse pairs: "
                f"{sorted(overlap)}"
            )

        self.raw_relation_names = tuple(raw_names)
        self.relation_to_id = {
            name: relation_id for relation_id, name in enumerate(raw_names)
        }
        self.id_to_relation = dict(enumerate(raw_names))
        self.inverse_relation_id = {}
        self.category = {}
        self.internal_inverse_of = {}

        for name in raw_names:
            relation_id = self.relation_to_id[name]
            if name in symmetric_relations:
                self.inverse_relation_id[relation_id] = relation_id
                self.category[relation_id] = "undirected"
            elif name in pair_by_name:
                inverse_name = pair_by_name[name]
                self.inverse_relation_id[relation_id] = self.relation_to_id[
                    inverse_name
                ]
                self.category[relation_id] = "directed_existing_inverse"

        next_id = len(raw_names)
        for name in raw_names:
            relation_id = self.relation_to_id[name]
            if relation_id in self.inverse_relation_id:
                continue
            inverse_id = next_id
            next_id += 1
            self.inverse_relation_id[relation_id] = inverse_id
            self.inverse_relation_id[inverse_id] = relation_id
            self.id_to_relation[inverse_id] = None
            self.internal_inverse_of[inverse_id] = relation_id
            self.category[relation_id] = "directed_internal_inverse"
            self.category[inverse_id] = "internal_inverse"

        # Existing inverse pairs need reciprocal entries even when the config
        # only specifies one direction.
        for source_name, target_name in pair_by_name.items():
            source_id = self.relation_to_id[source_name]
            target_id = self.relation_to_id[target_name]
            self.inverse_relation_id[source_id] = target_id
            self.inverse_relation_id[target_id] = source_id
            self.category[source_id] = "directed_existing_inverse"
            self.category[target_id] = "directed_existing_inverse"

        self.num_relations = next_id

        # Registry acceptance invariant: every raw relation has exactly one
        # inverse target, every inverse is reciprocal, and the internal
        # inverse allocation covers exactly the relations declared without a
        # public inverse pair.
        mapped_ids = set(self.inverse_relation_id.keys())
        if mapped_ids != set(range(self.num_relations)):
            raise RuntimeError(
                "Relation registry inverse map is incomplete: expected "
                f"one entry per relation ID, got {len(mapped_ids)} of "
                f"{self.num_relations}"
            )
        for relation_id, inverse_id in self.inverse_relation_id.items():
            back = self.inverse_relation_id.get(inverse_id)
            if back != relation_id:
                raise RuntimeError(
                    f"Non-reciprocal inverse mapping for relation "
                    f"{relation_id} -> {inverse_id} (back: {back})"
                )
        self.manifest = self._build_manifest()
        encoded = json.dumps(self.manifest, sort_keys=True, separators=(",", ":"))
        self.registry_hash = hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def _build_manifest(self):
        return {
            "raw_relation_names": list(self.raw_relation_names),
            "relation_to_id": dict(self.relation_to_id),
            "id_to_relation": {
                str(key): value for key, value in self.id_to_relation.items()
            },
            "inverse_relation_id": {
                str(key): value for key, value in self.inverse_relation_id.items()
            },
            "category": {str(key): value for key, value in self.category.items()},
            "internal_inverse_of": {
                str(key): value for key, value in self.internal_inverse_of.items()
            },
            "num_relations": self.num_relations,
        }

    def relation_id(self, relation_name):
        return self.relation_to_id.get(relation_name)

    def inverse_id(self, relation_id):
        try:
            return self.inverse_relation_id[int(relation_id)]
        except KeyError as exc:
            raise ValueError(f"No inverse registered for relation ID {relation_id}") from exc

    def public_relation_name(self, relation_id):
        return self.id_to_relation.get(int(relation_id))

    def is_public_relation_id(self, relation_id):
        return int(relation_id) < len(self.raw_relation_names)

    def to_dict(self):
        manifest = dict(self.manifest)
        manifest["registry_hash"] = self.registry_hash
        return manifest


def mapping_hash(mapping):
    """Return a stable hash for a name-to-ID mapping."""
    encoded = json.dumps(
        sorted((str(name), int(identifier)) for name, identifier in mapping.items()),
        separators=(",", ":"),
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()
