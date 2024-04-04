"""
Implementation of the folding algorithm based on:
Huber, K. T., Oxelman, B., Lott, M., & Moulton, V., "Reconstructing the evolutionary history of polyploids from multilabeled trees," Molecular Biology and Evolution, 2006.

This script specifically adapts the folding algorithm to allow for relaxed isomorphism checks and the use of gene tree clusters to choose the hybrid clade.
"""

import re
import treeswift
import collections
import networkx as nx
from multiset import FrozenMultiset, Multiset

from polyphest import utils
import PhyNetPy
from PhyNetPy.Graph import DAG
from PhyNetPy.Node import Node
from polyphest.multree_builder import Cluster, generate_multree
from typing import List, Counter, Dict


def is_near_isomorphic(
    network: DAG, clade1: Node, clade2: Node, threshold: float, normalize: bool = True
):
    cluster1 = utils.get_leaves_under_node(network, clade1)
    cluster2 = utils.get_leaves_under_node(network, clade2)

    if cluster1 != cluster2:
        return False

    dag1 = utils.clade_to_networkx(network, clade1)
    dag2 = utils.clade_to_networkx(network, clade2)
    distance = nx.graph_edit_distance(
        dag1, dag2, node_match=lambda u, v: u["species"] == v["species"]
    )
    if normalize:
        distance /= (dag1.number_of_nodes() + dag2.number_of_nodes())

    # print(f"{cluster1} vs. {cluster2}: {distance}")
    return distance <= threshold


def compute_isomorphism_code(network: DAG, node: Node):
    outdegree = network.out_degree(node)

    if outdegree == 0:
        return [node.get_name()]
    else:
        children_codes = []
        for child in network.get_children(node):
            code = compute_isomorphism_code(network, child)
            child.add_attribute("isomorphism_code", code)
            children_codes.append((len(code), code))

        # sort in lexicographical order
        children_codes.sort(key=lambda x: x[1])

        isomorphism_code = [str(len(children_codes))]
        for _, code in children_codes:
            isomorphism_code.extend(code)

        return isomorphism_code


def are_nodes_isomorphic(node: Node, other_node: Node):
    code = node.attribute_value_if_exists("isomorphism_code")
    other_code = other_node.attribute_value_if_exists("isomorphism_code")
    if not code and not other_code:
        return True

    if len(code) != len(other_code):
        return False

    for x, y in zip(code, other_code):
        if x != y:
            return False

    return True


def initialize(tree: DAG):
    utils.init_node_height(tree)
    compute_isomorphism_code(tree, tree.root()[0])


def find_isomorphic_trees(
    multree: DAG, node: Node, nodes_to_check, use_near_isomorphic, threshold
):
    isomorphic_trees = None
    indices_to_remove = []
    for i, other_node in enumerate(nodes_to_check):
        is_equivalent = (
            is_near_isomorphic(multree, node, other_node, threshold)
            if use_near_isomorphic
            else are_nodes_isomorphic(node, other_node)
        )

        if is_equivalent:
            if isomorphic_trees is None:
                isomorphic_trees = [node]
            isomorphic_trees.append(other_node)
            indices_to_remove.append(i)

    return isomorphic_trees, indices_to_remove


def compute_clusters(node: Node, net: DAG, clusters=None):
    if clusters is None:
        clusters = dict()

    # Base case: leaf
    if net.out_degree(node) == 0:
        cluster = FrozenMultiset([node.get_name()])
        clusters[node] = cluster
        return cluster

    # Recursive case: internal node
    cluster = Multiset()
    for child in net.get_children(node):
        child_cluster = compute_clusters(child, net, clusters)
        cluster = cluster.combine(child_cluster)

    clusters[node] = FrozenMultiset(cluster)
    return cluster


def compute_score_of_nontrivial_clusters(
    clade: Node, net: DAG, gt_clusters: Dict[FrozenMultiset, Counter[int]]
) -> float:
    clade_clusters = dict()
    compute_clusters(clade, net, clade_clusters)
    nontrivial_clusters = collections.Counter(
        {c for nd, c in clade_clusters.items() if len(c) > 1 and nd != clade}
    )
    score = 0
    for cluster, count in nontrivial_clusters.items():
        if cluster in gt_clusters:
            cur_score = 2 * count * gt_clusters[cluster][count * 2]
            if cur_score == 0:
                cur_score = count * gt_clusters[cluster][count]
            score += cur_score
    return score


def select_hybrid_node(
    isomorphic_trees: List[Node],
    net: DAG,
    use_clusters: bool = False,
    clusters: Dict[FrozenMultiset, Counter[int]] = None,
) -> Node:
    if use_clusters:
        return select_hybrid_node_based_on_clusters(isomorphic_trees, net, clusters)
    else:
        return select_hybrid_node_original(isomorphic_trees)


def select_hybrid_node_based_on_clusters(
    isomorphic_trees: List[Node], net: DAG, clusters: Dict[FrozenMultiset, Counter[int]]
) -> Node:
    target = isomorphic_trees[0]
    max_score = compute_score_of_nontrivial_clusters(target, net, clusters)
    for node in isomorphic_trees[1:]:
        score = compute_score_of_nontrivial_clusters(node, net, clusters)
        if score > max_score:
            target = node
            max_score = score

    return target


def select_hybrid_node_original(isomorphic_trees: List[Node]) -> Node:
    youngest_node = isomorphic_trees[0]
    for node in isomorphic_trees[1:]:
        if youngest_node.attribute_value_if_exists(
            "height"
        ) > node.attribute_value_if_exists("height"):
            youngest_node = node

    return youngest_node


def merge(node: Node, representative_node: Node, multree: DAG):
    donor = utils.insert_node_on_edge(node.get_parent(), node, multree)
    utils.merge_clades(representative_node.get_parent(), donor, multree)
    utils.remove_clade(multree, donor)  # Remove the donor clade from the tree


def merge_subtrees(
    isomorphic_trees: List[Node],
    multree: DAG,
    use_clusters: bool = False,
    clusters: Dict[FrozenMultiset, Counter[int]] = None,
):
    if not isomorphic_trees:
        return

    # Step 1: Choose a subtree to keep
    hybrid_node = select_hybrid_node(
        isomorphic_trees, multree, use_clusters, clusters
    )

    # Step 2: Merge subtrees
    utils.insert_node_on_edge(hybrid_node.get_parent(), hybrid_node, multree)
    for node in isomorphic_trees:
        if node == hybrid_node:
            continue
        merge(node, hybrid_node, multree)

    # Update the tree after merging
    utils.remove_binary_nodes(multree)
    utils.init_node_height(multree)


def convert_multree_to_network(
    multree: DAG,
    use_near_isomorphic: bool = False,
    use_clusters: bool = False,
    clusters: Counter[Cluster] = None,
    threshold=0.2,
) -> bool:
    """
    Converts a MUL-Tree to a network by merging isomorphic or near-isomorphic subtrees.

    Parameters:
    - multree: The MUL-Tree to convert.
    - use_near_isomorphic (bool): Whether to use near isomorphism for comparison (default False).
    - use_clusters (bool): Whether to use gene tree clusters to choose hybrid clade (default False).
    - clusters: A Counter of clusters obtained from gene trees (default None).
    - threshold (float): The threshold for near isomorphism comparison (default 0.2).

    Returns:
    - bool: True if any isomorphic subtrees were found and merged, False otherwise.
    """

    if not multree.root():
        raise ValueError("The MUL-Tree is empty.")

    initialize(multree)
    if use_near_isomorphic and use_clusters:
        gene_tree_clusters = collections.defaultdict(Counter)
        for cluster, count in clusters.items():
            gene_tree_clusters[cluster.cluster][cluster.n_copies] = count
    else:
        gene_tree_clusters = None
        use_clusters = False

    root_height = multree.root()[0].attribute_value_if_exists("height")
    height_list = [collections.deque() for _ in range(root_height + 1)]
    height_list[root_height].append(multree.root()[0])

    found_isomorphic = False
    for h in reversed(range(root_height + 1)):
        nodes_to_check = height_list[h]
        while nodes_to_check:
            node = nodes_to_check.popleft()
            isomorphic_trees, indexes_to_remove = find_isomorphic_trees(
                multree, node, nodes_to_check, use_near_isomorphic, threshold
            )

            if isomorphic_trees:
                found_isomorphic = True
                merge_subtrees(
                    isomorphic_trees, multree, use_clusters, gene_tree_clusters
                )

                for idx in sorted(indexes_to_remove, reverse=True):
                    del nodes_to_check[idx]

            # Update height list for children
            # This subtree may still have subtrees which are isomorphic to other subtrees
            for child in multree.get_children(node):
                height = child.attribute_value_if_exists("height")
                height_list[height].append(child)

    return found_isomorphic



