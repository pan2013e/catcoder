from typing import Callable, List

from PrettyPrint import PrettyPrintTree

from . import StructureNode, TypeDef

__all__ = ["TreeNode"]

Forest = List['TreeNode']
PrettyPrint = PrettyPrintTree(lambda x: x.children, lambda x: x.payload,
                    orientation=PrettyPrintTree.Horizontal)
AcceptedKinds = ['function', 'impl']

StructureNode.__str__ = lambda self: f"{self.label}: {self.detail}"

class TreeNode:
    def __init__(self, payload: StructureNode):
        self.payload = payload
        self.parent = None
        self.children = []

    @staticmethod
    def from_flattened(nodes: List[StructureNode]) -> Forest:
        free_nodes = [TreeNode(node) for node in nodes]
        forest = []
        for i in range(len(free_nodes)):
            node = free_nodes[i]
            if node.payload.ppid is not None:
                node.parent = free_nodes[node.payload.ppid]
                node.parent.children.append(node)
        for node in free_nodes:
            if node.parent is None:
                forest.append(node)
        return TreeNode.prune_forest(forest, lambda x: x.payload.kind in AcceptedKinds, -1)
    
    @staticmethod
    def prune_forest(roots: Forest, rule: Callable[['TreeNode'], bool], max_levels: int) -> Forest:
        if max_levels == 0:
            return roots
        pruned_roots = [root for root in roots if rule(root)]
        for root in pruned_roots:
            root.children = TreeNode.prune_forest(root.children, rule, max_levels - 1)
        return pruned_roots
    
    @staticmethod
    def filter_by_type(roots: Forest, type: TypeDef) -> Forest:

        def impl_type_match(s1: str, s2: str) -> bool:
            s1 = s1.replace('\n', ' ')
            idx = s1.find('where')
            if idx != -1:
                s1 = s1[:idx].strip()
            idx = s1.find(s2)
            if idx == -1:
                return False
            if idx > 0:
                if s1[idx - 1] not in ['<', ' ', ',']:
                    return False
            if idx + len(s2) < len(s1):
                ch = s1[idx + len(s2)]
                return ch == '<' or ch == '>'
            return True

        return TreeNode.prune_forest(roots, lambda x: x.payload.kind == 'impl' and impl_type_match(x.payload.label, type.name), 1)
    
    def pp(self):
        PrettyPrint(self)

    def __str__(self):
        if self.payload.kind == 'impl':
            return f"{self.payload.label}"
        elif self.payload.kind == 'function':
            s = self.payload.detail
            idx = s.find('fn')
            s = s[:idx + 2] + ' ' + self.payload.label + s[idx + 2:]
            return s
        else:
            raise RuntimeError(f'Unexpected node kind {self.payload.kind}, which should be pruned')
