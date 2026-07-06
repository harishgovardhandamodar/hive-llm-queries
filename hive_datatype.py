"""
Hive Data Model — the core datatype for HiveMind knowledge graphs.

A Hive is a topic-specific directed multigraph with four node types:
- paper / graph_paper (research papers)
- concept (extracted keywords/ideas)
- graph_ref (cross-hive references)

Edges carry a typed relation from a controlled vocabulary of 11 valid types.

Serialization uses the NetworkX Node-Link JSON format.
This module is self-contained and has no dependency on HiveMind itself.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Iterator
from collections.abc import Callable


# ── Constants ──────────────────────────────────────────────────────────────

VALID_RELATIONS: frozenset[str] = frozenset({
    "cites", "introduces", "uses", "improves", "extends",
    "compares", "contrasts", "proposes", "related_to",
    "nests", "references",
})

LINEAGE_RELATIONS: frozenset[str] = frozenset({
    "extends", "improves", "overcomes", "fixes", "proposes", "cites",
})

D3_GROUP: dict[str, int] = {
    "paper": 0,
    "graph_paper": 0,
    "concept": 1,
    "graph_ref": 2,
}

MAX_LABEL_LEN = 60
MAX_AUTHORS_LEN = 120
MAX_ABSTRACT_LEN = 300
MAX_AFFILIATIONS_LEN = 200
MAX_DEFINITION_LEN = 200


# ── Enums ──────────────────────────────────────────────────────────────────

class NodeType(str, Enum):
    PAPER = "paper"
    GRAPH_PAPER = "graph_paper"
    CONCEPT = "concept"
    GRAPH_REF = "graph_ref"


class Relation(str, Enum):
    CITES = "cites"
    INTRODUCES = "introduces"
    USES = "uses"
    IMPROVES = "improves"
    EXTENDS = "extends"
    COMPARES = "compares"
    CONTRASTS = "contrasts"
    PROPOSES = "proposes"
    RELATED_TO = "related_to"
    NESTS = "nests"
    REFERENCES = "references"


# ── Validation helpers ─────────────────────────────────────────────────────

def normalize_name(name: str) -> str:
    return " ".join(name.split())


def validate_relation(relation: str) -> str:
    if relation not in VALID_RELATIONS:
        return "related_to"
    return relation


def default_relation() -> str:
    return "related_to"


# ── Data classes ───────────────────────────────────────────────────────────

@dataclass
class HiveStats:
    papers: int = 0
    graph_papers: int = 0
    concepts: int = 0
    graph_refs: int = 0
    relations: int = 0
    cross_edges: int = 0


@dataclass
class Node:
    id: str
    type: NodeType | str
    label: str = ""
    graph_id: str = ""

    # Paper / Graph Paper fields
    arxiv_id: str = ""
    authors: str = ""
    published: str = ""
    abstract: str = ""
    categories: list[str] = field(default_factory=list)
    affiliations: str = ""

    # Graph Paper only
    source_paper: str = ""
    expanded_at: str = ""

    # Concept fields
    definition: str = ""
    concept_type: str = "concept"

    # Graph Ref fields
    target_graph_id: str = ""

    # Layout
    layout_x: float | None = None
    layout_y: float | None = None

    def __post_init__(self) -> None:
        self.label = self.label[:MAX_LABEL_LEN]

    @property
    def d3_group(self) -> int:
        return D3_GROUP.get(self.type, 2)

    @property
    def is_paper(self) -> bool:
        return self.type in (NodeType.PAPER, NodeType.GRAPH_PAPER)

    @property
    def is_concept(self) -> bool:
        return self.type == NodeType.CONCEPT

    @property
    def is_graph_ref(self) -> bool:
        return self.type == NodeType.GRAPH_REF

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "id": self.id,
            "label": self.label[:MAX_LABEL_LEN],
            "type": self.type,
            "title": self.label,
            "group": self.d3_group,
        }
        if self.layout_x is not None and self.layout_y is not None:
            d["layout_x"] = self.layout_x
            d["layout_y"] = self.layout_y
        if self.is_paper:
            d["arxiv_id"] = self.arxiv_id
            d["authors"] = self.authors[:MAX_AUTHORS_LEN]
            d["abstract"] = self.abstract[:MAX_ABSTRACT_LEN]
            d["affiliations"] = self.affiliations[:MAX_AFFILIATIONS_LEN]
            if self.categories:
                d["categories"] = self.categories[:10]
            if self.type == NodeType.GRAPH_PAPER:
                d["source_paper"] = self.source_paper
        elif self.is_concept:
            d["definition"] = self.definition[:MAX_DEFINITION_LEN]
            if self.concept_type != "concept":
                d["concept_type"] = self.concept_type
        return d

    @classmethod
    def from_graph_data(cls, node_id: str, attrs: dict[str, Any]) -> Node:
        return cls(
            id=node_id,
            type=attrs.get("type", "unknown"),
            label=attrs.get("label", node_id),
            graph_id=attrs.get("graph_id", ""),
            arxiv_id=attrs.get("arxiv_id", ""),
            authors=attrs.get("authors", ""),
            published=attrs.get("published", ""),
            abstract=attrs.get("abstract", ""),
            categories=attrs.get("categories", []),
            affiliations=attrs.get("affiliations", ""),
            source_paper=attrs.get("source_paper", ""),
            expanded_at=attrs.get("expanded_at", ""),
            definition=attrs.get("definition", ""),
            concept_type=attrs.get("concept_type", "concept"),
            target_graph_id=attrs.get("target_graph_id", ""),
        )


@dataclass
class Edge:
    source: str
    target: str
    relation: str = "related_to"
    key: int = 0
    cross_graph: bool = False
    target_graph: str = ""

    def __post_init__(self) -> None:
        self.relation = validate_relation(self.relation)

    @property
    def is_lineage(self) -> bool:
        return self.relation in LINEAGE_RELATIONS

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "relation": self.relation,
            "cross_graph": self.cross_graph,
            "target_graph": self.target_graph,
        }

    @classmethod
    def from_graph_data(cls, u: str, v: str, data: dict[str, Any], key: int = 0) -> Edge:
        return cls(
            source=u,
            target=v,
            relation=data.get("relation", "related_to"),
            key=key,
            cross_graph=data.get("cross_graph", False),
            target_graph=data.get("target_graph", ""),
        )


@dataclass
class HiveGraph:
    """A single hive — a topic-specific knowledge graph."""

    id: str
    nodes: list[Node] = field(default_factory=list)
    edges: list[Edge] = field(default_factory=list)

    # ── Computed properties ──────────────────────────────────────────

    @property
    def stats(self) -> HiveStats:
        return HiveStats(
            papers=sum(1 for n in self.nodes if n.type == NodeType.PAPER),
            graph_papers=sum(1 for n in self.nodes if n.type == NodeType.GRAPH_PAPER),
            concepts=sum(1 for n in self.nodes if n.type == NodeType.CONCEPT),
            graph_refs=sum(1 for n in self.nodes if n.type == NodeType.GRAPH_REF),
            relations=len(self.edges),
            cross_edges=sum(1 for e in self.edges if e.cross_graph),
        )

    @property
    def paper_count(self) -> int:
        return sum(1 for n in self.nodes if n.is_paper)

    @property
    def concept_count(self) -> int:
        return sum(1 for n in self.nodes if n.is_concept)

    @property
    def lineage_edges(self) -> list[Edge]:
        return [e for e in self.edges if e.is_lineage]

    # ── Node lookup ──────────────────────────────────────────────────

    def get_node(self, node_id: str) -> Node | None:
        for n in self.nodes:
            if n.id == node_id:
                return n
        return None

    def get_nodes_by_type(self, node_type: NodeType | str) -> list[Node]:
        return [n for n in self.nodes if n.type == node_type]

    @property
    def papers(self) -> list[Node]:
        return [n for n in self.nodes if n.is_paper]

    @property
    def concepts(self) -> list[Node]:
        return [n for n in self.nodes if n.is_concept]

    @property
    def graph_refs(self) -> list[Node]:
        return [n for n in self.nodes if n.is_graph_ref]

    # ── Edge lookup ──────────────────────────────────────────────────

    def get_edges(self, node_id: str) -> list[Edge]:
        return [e for e in self.edges if e.source == node_id or e.target == node_id]

    def outgoing(self, node_id: str) -> list[Edge]:
        return [e for e in self.edges if e.source == node_id]

    def incoming(self, node_id: str) -> list[Edge]:
        return [e for e in self.edges if e.target == node_id]

    # ── Serialization ────────────────────────────────────────────────

    def to_node_link_dict(self) -> dict[str, Any]:
        """Serialize to NetworkX Node-Link JSON format."""
        return {
            "directed": True,
            "multigraph": True,
            "graph": {},
            "graph_id": self.id,
            "nodes": [n.to_dict() for n in self.nodes],
            "links": [
                {
                    "source": e.source,
                    "target": e.target,
                    "relation": e.relation,
                    "key": e.key,
                    "cross_graph": e.cross_graph,
                    "target_graph": e.target_graph,
                }
                for e in self.edges
            ],
        }

    @classmethod
    def from_node_link_dict(cls, data: dict[str, Any]) -> HiveGraph:
        """Deserialize from NetworkX Node-Link JSON format."""
        graph_id = data.get("graph_id", "")
        nodes = []
        for nd in data.get("nodes", []):
            node = Node(
                id=nd["id"],
                type=nd.get("type", "unknown"),
                label=nd.get("label", nd.get("title", nd["id"])),
                graph_id=nd.get("graph_id", graph_id),
                arxiv_id=nd.get("arxiv_id", ""),
                authors=nd.get("authors", ""),
                published=nd.get("published", ""),
                abstract=nd.get("abstract", ""),
                categories=nd.get("categories", []),
                affiliations=nd.get("affiliations", ""),
                source_paper=nd.get("source_paper", ""),
                expanded_at=nd.get("expanded_at", ""),
                definition=nd.get("definition", ""),
                concept_type=nd.get("concept_type", "concept"),
                target_graph_id=nd.get("target_graph_id", ""),
                layout_x=nd.get("layout_x"),
                layout_y=nd.get("layout_y"),
            )
            nodes.append(node)

        edges = []
        for ld in data.get("links", []):
            edge = Edge(
                source=ld["source"],
                target=ld["target"],
                relation=ld.get("relation", "related_to"),
                key=ld.get("key", 0),
                cross_graph=ld.get("cross_graph", False),
                target_graph=ld.get("target_graph", ""),
            )
            edges.append(edge)

        return cls(id=graph_id, nodes=nodes, edges=edges)

    # ── JSON file I/O ────────────────────────────────────────────────

    @classmethod
    def from_json_file(cls, path: str) -> HiveGraph:
        with open(path) as f:
            data = json.load(f)
        return cls.from_node_link_dict(data)

    def to_json_file(self, path: str, indent: int = 2) -> None:
        with open(path, "w") as f:
            json.dump(self.to_node_link_dict(), f, indent=indent)


# ── Meta-graph ─────────────────────────────────────────────────────────────

@dataclass
class MetaNode:
    id: str
    label: str = ""
    type: str = "knowledge_graph"
    visible: bool = True
    papers: int = 0
    graph_papers: int = 0
    concepts: int = 0
    relations: int = 0
    graph_refs: int = 0
    cross_edges: int = 0
    owner: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label or self.id,
            "type": self.type,
            "visible": self.visible,
            "papers": self.papers,
            "graph_papers": self.graph_papers,
            "concepts": self.concepts,
            "relations": self.relations,
            "graph_refs": self.graph_refs,
            "cross_edges": self.cross_edges,
            "owner": self.owner,
        }


@dataclass
class MetaEdge:
    source: str
    target: str
    relation: str = "references"
    label: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "relation": self.relation,
            "label": self.label or self.relation,
        }


@dataclass
class MetaGraph:
    nodes: list[MetaNode] = field(default_factory=list)
    edges: list[MetaEdge] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "nodes": [n.to_dict() for n in self.nodes],
            "edges": [e.to_dict() for e in self.edges],
        }


# ── Federation stats ───────────────────────────────────────────────────────

@dataclass
class FederationStats:
    graphs: int = 0
    papers: int = 0
    concepts: int = 0
    graph_refs: int = 0
    relations: int = 0
    cross_edges: int = 0
    meta_edges: int = 0


# ── Aggregation helpers ────────────────────────────────────────────────────

def aggregate_stats(graphs: list[HiveGraph]) -> FederationStats:
    total = FederationStats(graphs=len(graphs))
    for g in graphs:
        s = g.stats
        total.papers += s.papers
        total.concepts += s.concepts
        total.graph_refs += s.graph_refs
        total.relations += s.relations
        total.cross_edges += s.cross_edges
    return total


def build_meta_graph(graphs: list[HiveGraph]) -> MetaGraph:
    """Build a meta-graph from a list of hives using graph_ref edges."""
    mg = MetaGraph()
    hive_map = {g.id: g for g in graphs}
    for g in graphs:
        s = g.stats
        mg.nodes.append(MetaNode(
            id=g.id,
            papers=s.papers,
            graph_papers=s.graph_papers,
            concepts=s.concepts,
            relations=s.relations,
            graph_refs=s.graph_refs,
            cross_edges=s.cross_edges,
        ))
        for ref in g.graph_refs:
            if ref.target_graph_id in hive_map:
                mg.edges.append(MetaEdge(
                    source=g.id,
                    target=ref.target_graph_id,
                    relation="references",
                ))
    return mg
