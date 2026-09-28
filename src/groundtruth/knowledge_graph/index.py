from __future__ import annotations

from collections import deque
from collections.abc import Iterable

from groundtruth.knowledge_graph.schema import KnowledgeGraph


class KnowledgeGraphIndex:
    def __init__(self, graph: KnowledgeGraph):
        self.nodes = {node.id: node for node in graph.nodes}
        self.adjacency: dict[str, list[tuple[str, str]]] = {
            node_id: [] for node_id in self.nodes
        }
        for edge in graph.edges:
            if edge.source_id not in self.nodes or edge.target_id not in self.nodes:
                continue
            self.adjacency[edge.source_id].append((edge.target_id, edge.relation))
            self.adjacency[edge.target_id].append((edge.source_id, edge.relation))
        for neighbors in self.adjacency.values():
            neighbors.sort()

    def expand(
        self,
        seed_ids: Iterable[str],
        max_hops: int = 3,
        allowed_relations: set[str] | None = None,
    ) -> list[tuple[str, int]]:
        if max_hops < 0:
            raise ValueError("max_hops must be non-negative")
        distance = {}
        queue = deque()
        for seed in seed_ids:
            if seed in self.nodes and seed not in distance:
                distance[seed] = 0
                queue.append(seed)
        while queue:
            current = queue.popleft()
            current_distance = distance[current]
            if current_distance >= max_hops:
                continue
            for neighbor, relation in self.adjacency.get(current, []):
                if allowed_relations and relation not in allowed_relations:
                    continue
                if neighbor not in distance:
                    distance[neighbor] = current_distance + 1
                    queue.append(neighbor)
        return sorted(distance.items(), key=lambda item: (item[1], item[0]))
