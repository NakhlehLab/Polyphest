"""
Implementation of the multi-labeled tree construction algorithm based on:
Lott, M., Spillner, A., Huber, K. T., Petri, A., Oxelman, B., & Moulton, V, "Inferring polyploid phylogenies from multiply-labeled gene trees," BMC evolutionary biology, 2009

This script adapts the MUL-tree construction algorithm and introduces an ILP formulation to solve the backbone cluster selection problem.
"""
import itertools
import math
import random
import collections
import re

import pulp
import treeswift
from multiset import FrozenMultiset, Multiset
from typing import List, Dict, Set, Counter, Tuple, Iterable

from polyphest import utils
import PhyNetPy
from PhyNetPy.Graph import DAG
from PhyNetPy.Node import Node


class Cluster:
    def __init__(self, cluster: FrozenMultiset, support: float = None, n_copies: int = 1):
        self.cluster = cluster
        self.n_copies = n_copies
        self.support = support

    def increment_support(self, support: float = 1):
        if self.support is None:
            self.support = support
        else:
            self.support += support

    def decrement_support(self, support: float = 1):
        if self.support is None:
            raise ValueError("Support is not set")
        self.support -= support
        if self.support < 0:
            self.support = 0

    def increment_multiplicity(self, n_copies: int = 1):
        self.n_copies += n_copies

    def decrement_multiplicity(self, n_copies: int = 1):
        self.n_copies -= n_copies

    def __repr__(self):
        return f"Cluster({self.n_copies} x {str(self.cluster)}: {self.support})"

    def __eq__(self, other):
        return self.cluster == other.cluster and self.n_copies == other.n_copies

    def __hash__(self):
        return hash((self.cluster, self.n_copies))

    def __len__(self):
        return len(self.cluster)


def identify_multree_root_cluster(trees: List[treeswift.Tree]) -> FrozenMultiset:
    # Step1: Work out in how many trees each number of copies of a taxa occur
    taxa_frequency_table = collections.defaultdict(lambda: collections.defaultdict(int))
    for tree in trees:
        taxa_freq = FrozenMultiset([lb for lb in tree.labels(internal=False)])
        for taxon, freq in taxa_freq.items():
            taxa_frequency_table[taxon][freq] += 1

    taxa = list(taxa_frequency_table.keys())
    for taxon in taxa:
        freqs = taxa_frequency_table[taxon]
        occurring_trees = sum(freqs.values())
        freqs[0] = len(trees) - occurring_trees

    # Step2: Use that number of copies which appears in the most trees to be in the agreed multiset
    consensus_multiset = collections.defaultdict(int)
    for taxon in taxa:
        possible_copies = list(taxa_frequency_table[taxon].keys())
        max_copies = max(possible_copies)

        for k in range(max_copies, -1, -1):
            n_supporting_trees = 0

            for possible_copy in possible_copies:
                if possible_copy >= k:
                    n_supporting_trees += taxa_frequency_table[taxon][possible_copy]

            if n_supporting_trees >= len(trees) / 2:
                consensus_multiset[taxon] = k
                break

    return FrozenMultiset(dict(consensus_multiset))


def compute_clusters_from_trees(weighted_trees: List[Tuple[treeswift.Tree, float]]):
    return [(compute_clusters(tree), weight) for tree, weight in weighted_trees]


def remove_redundant_clusters(tree_clusters: List[Tuple[Dict[treeswift.Node, FrozenMultiset], float]]):
    for clusters, weight in tree_clusters:
        nodes_to_remove = []

        for node, cluster in clusters.items():
            if not node.is_root() and cluster == clusters.get(node.get_parent(), None):
                nodes_to_remove.append(node)

        for node in nodes_to_remove:
            del clusters[node]


def reduce_clusters_on_leaves(tree_clusters: List[Tuple[Dict[treeswift.Node, FrozenMultiset], float]],
                              taxa: FrozenMultiset):
    """Reduce the clusters to only contain the taxa"""
    for clusters, weight in tree_clusters:
        for node, cluster in clusters.items():
            if not taxa.issuperset(cluster):
                clusters[node] = reduce_cluster(cluster, taxa)


def reduce_cluster(cluster: FrozenMultiset, expected_cluster: FrozenMultiset):
    updated_leaves = Multiset(cluster)
    for leaf, count in updated_leaves.items():
        if leaf in expected_cluster and count > expected_cluster[leaf]:
            updated_leaves[leaf] = expected_cluster[leaf]

    return FrozenMultiset(updated_leaves)


def compute_clusters(tree: treeswift.Tree):
    """Compute clusters for each node in the given tree"""
    treenode_clusters = dict()
    for node in tree.traverse_postorder():
        if node.is_leaf():
            treenode_clusters[node] = FrozenMultiset([node.label])
        else:
            for child in node.child_nodes():
                if node not in treenode_clusters:
                    treenode_clusters[node] = FrozenMultiset()
                treenode_clusters[node] = treenode_clusters[node].combine(
                    treenode_clusters[child]
                )
    return treenode_clusters


def extract_single_occurrence_taxa(root_cluster):
    return [leaf for leaf, count in root_cluster.items() if count == 1]


def compute_core_and_ambiguous_clusters(
        tree_clusters: List[Tuple[Dict[treeswift.Node, FrozenMultiset], float]],
        taxa: FrozenMultiset,
):
    """Computes core and ambiguous clusters based on clusters computed from a collection of trees."""
    singleton_leaves = FrozenMultiset(extract_single_occurrence_taxa(taxa))
    core_clusters, ambiguous_clusters = collections.Counter(), collections.Counter()

    for clusters, weight in tree_clusters:  # each tree
        tree_ambiguous_cluster_counter = collections.Counter()
        tree_cluster_counter = collections.Counter(clusters.values())

        for cluster, count in tree_cluster_counter.items():
            if len(cluster) == 1 or len(cluster) == len(taxa):  # ignore trivial clusters
                continue
            if cluster.intersection(singleton_leaves):  # core cluster
                new_cluster = Cluster(cluster=cluster, n_copies=1)
                core_clusters[new_cluster] += weight
            else:
                tree_ambiguous_cluster_counter[cluster] += count

        # update global count of ambiguous clusters
        for cur_acluster, a_count in tree_ambiguous_cluster_counter.items():
            new_cluster = Cluster(cluster=cur_acluster, n_copies=a_count)
            ambiguous_clusters[new_cluster] += weight

    return core_clusters, ambiguous_clusters


def filter_clusters(clusters: Counter[Cluster], filter_strategy: 'FilteringStrategy'):
    """Filter clusters based on the given strategy."""
    if filter_strategy is None:
        return collections.Counter(clusters)

    if not clusters:
        return collections.Counter()

    return filter_strategy.filter_clusters(clusters)


def is_disjoint(cluster1: Cluster, cluster2: Cluster, full_cluster: FrozenMultiset) -> bool:
    """
    We cannot use traditional cluster1.isdisjoint(cluster2) because we are dealing with multrees.
    e.g., given a tree (((x,y)I0,z)I1,(x,y,z)I2)I3; and cluster {y,z}
    {y,z} is not compatible with cluster under node I0, i.e., {x,y}.
    Still, {y,z} can be inserted into the clade under node I2.
    We also need to take into account the number of copies of the clusters.
    e.g., given a tree with taxa {a,b,x,x,y,y,z,z} and cluster 2x{y,z}, it is not compatible with cluster 1x{b,y}
    """
    return full_cluster.difference(cluster1.cluster * cluster1.n_copies).issuperset(
        cluster2.cluster * cluster2.n_copies)


def are_clusters_compatible(cluster1: Cluster, cluster2: Cluster, taxa: FrozenMultiset) -> bool:
    """
    Preliminary check if two clusters are compatible. This does not guarantee compatibility.
    This would fail to account for the case e.g., (z,z,y,y,((a,x,(b,x)I1)I2,o)I3)I0; {x,y,y,z,z}
    """
    return (
            is_disjoint(cluster1, cluster2, taxa)
            or cluster1.cluster.issuperset(cluster2.cluster)
            or cluster2.cluster.issuperset(cluster1.cluster)
    )


def select_backbone_clusters(core_clusters: Counter[Cluster], ambiguous_clusters: Counter[Cluster],
                             taxa: FrozenMultiset):
    clusters = list((core_clusters + ambiguous_clusters).items())
    # clusters = sorted(clusters, key=lambda item: (item[0].cluster, item[0].n_copies * item[1] * math.pow(len(item[0]), 0.65 )), reverse=True)
    grouped_clusters = collections.defaultdict(list)
    for index, cluster in enumerate(clusters):
        grouped_clusters[cluster[0].cluster].append(index)

    clusters_to_delete = [cluster for cluster, indices in grouped_clusters.items() if len(indices) < 2]

    for cluster in clusters_to_delete:
        del grouped_clusters[cluster]

    problem = pulp.LpProblem("BackboneClusterSelection", pulp.LpMaximize)
    num_clusters = len(clusters)

    x = pulp.LpVariable.dicts("x", range(num_clusters), lowBound=0, upBound=1, cat="Binary")
    problem += pulp.lpSum(
        # frequency * n_copies * size^0.65
        [x[i] * clusters[i][1] * clusters[i][0].n_copies * math.pow(len(clusters[i][0]), 0.65) for i in
         range(num_clusters)]
    )

    # compatibility constraints
    for i in range(num_clusters):
        for j in range(num_clusters):
            if not are_clusters_compatible(clusters[i][0], clusters[j][0], taxa):
                problem += x[i] + x[j] <= 1

    # select at most one cluster per group
    for cl, indices in grouped_clusters.items():
        problem += pulp.lpSum([x[i] for i in indices]) <= 1

    problem.solve(pulp.PULP_CBC_CMD(msg=False))
    if not pulp.LpStatus[problem.status] == "Optimal":
        print("Failed to solve the backbone cluster selection problem.")
        return None

    return [clusters[i] for i in range(num_clusters) if x[i].value() == 1]


def find_insertable_position(
        tree: DAG,
        cluster: FrozenMultiset,
        n_copies: int,
        node_heirs,
        node_to_cluster,
        label_to_nodes,
):
    """
    Find all positions where the given cluster with multiplicity can be inserted into the given tree.
    """
    label_to_nodes = {leaf: label_to_nodes[leaf] for leaf in cluster}

    # All combinations of leaf nodes for each label with multiplicity
    label_to_combo = {
        label: list(itertools.combinations(label_to_nodes[label], count))
        for label, count in cluster.items()
    }
    all_combinations = itertools.product(*label_to_combo.values())
    all_combinations = [
        {nd for subtuple in tup for nd in subtuple} for tup in all_combinations
    ]
    multiplicity = n_copies
    insertable_positions = set()
    for leaf_nodes in all_combinations:
        lca = tree.mrca(leaf_nodes)
        if node_to_cluster[lca] == cluster:
            multiplicity -= 1
            continue

        moved_children = set()
        size = 0
        for child in tree.get_children(lca):
            child_cluster = node_to_cluster[child]
            if child_cluster.issubset(cluster):
                child_leaf_descendants = node_heirs[child]
                if all([nd in leaf_nodes for nd in child_leaf_descendants]):
                    moved_children.add(child)
                    size += len(child_leaf_descendants)
            else:
                continue

            if moved_children and size == len(cluster):
                formed_cluster = Multiset()
                for ch in moved_children:
                    formed_cluster = formed_cluster.combine(node_to_cluster[ch])
                if FrozenMultiset(formed_cluster) == cluster:
                    insertable_positions.add(tuple(moved_children))
                break

    if len(insertable_positions) >= multiplicity:  # should use multiplicity here rather than cluster.n_copies
        return insertable_positions

    return None

def select_random_disjoint_collections(list_of_collections: List[Iterable], k: int):
    selected_collections = []
    available_collections = list_of_collections[:]

    while len(selected_collections) < k and available_collections:
        random.shuffle(available_collections)

        for candidate_collection in available_collections:
            if all(not any(node in selected_collection for node in candidate_collection) for selected_collection in selected_collections):
                selected_collections.append(candidate_collection)
                available_collections.remove(candidate_collection)
                break

        return selected_collections


def insert_cluster(
        net: DAG,
        cluster_to_insert: FrozenMultiset,
        multiplicity: int,
        node_heirs: Dict[Node, Set[Node]],
        node_to_cluster: Dict[Node, FrozenMultiset],
        label_to_nodes: Dict[str, Set[Node]]
):
    candidate_positions = find_insertable_position(net, cluster_to_insert, multiplicity, node_heirs, node_to_cluster,
                                                   label_to_nodes)

    if not candidate_positions:
        print(f"Cannot insert {cluster_to_insert} into {net.newick()}.")
        return

    # moved_children_list = random.choices(list(candidate_positions), k=multiplicity) # this might select the same moved_child multiple times

    moved_children_list = select_random_disjoint_collections(list(candidate_positions), multiplicity)
    for moved_children in moved_children_list:
        #print(f"Before insertion: {net.newick()}")
        lca = net.mrca(moved_children)
        new_child = Node()
        net.add_uid_node(new_child)
        net.add_edges([lca, new_child])
        node_to_cluster[new_child] = cluster_to_insert
        node_heirs[new_child] = {heir for child in moved_children for heir in node_heirs[child]}

        for child in moved_children:
            net.remove_edge([lca, child])
            net.add_edges([new_child, child])

        utils.remove_binary_nodes(net)
        net.remove_excess_branch()
        utils.remove_floaters(net)
        #print(f"\tAfter move {[nd.get_name() for nd in moved_children]}: {net.newick()}")


def generate_tree_from_clusters(clusters: List[Tuple[Cluster, float]], taxa: FrozenMultiset):
    """Generate a tree from the given clusters."""

    # Step 1: Create a star tree with the taxa
    net = DAG()
    root = Node()
    net.add_uid_node(root)

    root_cluster = taxa
    node_to_cluster = {root: root_cluster}

    label_to_nodes = collections.defaultdict(set)
    for taxon, copy_number in taxa.items():
        for k in range(copy_number):
            leaf = Node(name=taxon)
            label_to_nodes[taxon].add(leaf)
            c = FrozenMultiset(taxon)
            node_to_cluster[leaf] = c
            net.add_edges([root, leaf])
            leaf.set_parent([root])
            net.add_nodes([leaf])

    if not clusters:
        return net

    # Step 2: Insert the clusters into the tree
    clusters = sorted(clusters, key=lambda x: x[1] * x[0].n_copies * math.pow(len(x[0]), 0.65), reverse=True)
    non_trivial_clusters = [[x[0].cluster, x[0].n_copies, x[1]] for x in clusters if 1 < len(x[0]) < len(taxa)]
    refine_tree_with_clusters(net, non_trivial_clusters, label_to_nodes, taxa)
    leaves_cluster = FrozenMultiset([l.get_name() for l in net.get_leaves()])
    if leaves_cluster != taxa:
        raise ValueError(f"Backbone leaves {leaves_cluster} do not match the taxa {taxa}.\n{net.newick()}")
    return net


def refine_tree_with_clusters(tree: DAG, sorted_clusters: List[Tuple[FrozenMultiset, int, float]],
                              label_to_nodes: Dict[str, Set[Node]], taxa: FrozenMultiset):
    candidate_cluster_index_map = dict()
    for i, (cluster, n_copies, support) in enumerate(sorted_clusters):
        if not cluster.issubset(taxa):
            raise ValueError(f"{str(cluster)} is not a subset of the taxa {str(taxa)}.")

        candidate_cluster_index_map[cluster] = i

    node_heirs = tree.leaf_descendants_all()
    node_to_cluster = {
        node: FrozenMultiset([l.get_name() for l in heirs])
        for node, heirs in node_heirs.items()
    }

    cluster_to_nodes = collections.defaultdict(set)
    for nd, cl in node_to_cluster.items():
        cluster_to_nodes[cl].add(nd)
        if cl in candidate_cluster_index_map:
            idx = candidate_cluster_index_map[cl]
            sorted_clusters[idx][0][1] -= 1

    for cluster, n_copies, support in sorted_clusters:
        if n_copies <= 0:
            continue

        insert_cluster(tree, cluster, n_copies, node_heirs, node_to_cluster, label_to_nodes)


def resolve_node(
        tree: DAG,
        multifurcating_node: Node,
        core_clusters: Counter[Cluster],
        ambiguous_clusters: Counter[Cluster],
        node_to_cluster: Dict[Node, FrozenMultiset],
        cluster_to_nodes: Dict[FrozenMultiset, Set[Node]]
):
    parent = multifurcating_node
    children = [c for c in tree.get_children(multifurcating_node)]

    clusters = core_clusters + ambiguous_clusters
    grouped_clusters = collections.defaultdict(list)
    for cluster, support in clusters.most_common():
        grouped_clusters[cluster.cluster].append((cluster.n_copies, support))

    while len(children) > 2:
        combinations = itertools.combinations(children, 2)
        combs = list(combinations)  # after this line, combination is empty!
        max_score, max_pair = 0, []

        for child1, child2 in combs:
            merged_cluster = node_to_cluster[child1].combine(node_to_cluster[child2])
            if merged_cluster in grouped_clusters:
                for n_copies, score in grouped_clusters[merged_cluster]:
                    if merged_cluster not in cluster_to_nodes and score > max_score:
                        # TODO: if currently n_copies > 1, we should check if we can accommodate
                        # n copies of the cluster in the current tree since the score is associated with n_copies
                        max_score = score
                        max_pair = [child1, child2]

                    elif (
                            merged_cluster in cluster_to_nodes
                            and len(cluster_to_nodes[merged_cluster]) < n_copies
                            and score > max_score
                    ):
                        max_score = score
                        max_pair = [child1, child2]

        if not max_pair:  # randomly resolve
            max_pair = random.choice(combs)

        merged_cluster = node_to_cluster[max_pair[0]].combine(node_to_cluster[max_pair[1]])
        new_child = Node()
        tree.add_uid_node(new_child)
        tree.add_edges([parent, new_child])
        node_to_cluster[new_child] = merged_cluster
        cluster_to_nodes[merged_cluster].add(new_child)
        children.append(new_child)

        for child in max_pair:
            tree.remove_edge([parent, child])
            tree.add_edges([new_child, child])

        children.remove(max_pair[0])
        children.remove(max_pair[1])

        utils.remove_binary_nodes(tree)
        tree.remove_excess_branch()
        utils.remove_floaters(tree)


def resolve_nodes(tree: DAG, unresolved_nodes: List[Node], core_clusters, ambiguous_clusters):
    node_heirs = tree.leaf_descendants_all()
    node_to_cluster = {
        node: FrozenMultiset([l.get_name() for l in heirs])
        for node, heirs in node_heirs.items()
    }

    cluster_to_nodes = collections.defaultdict(set)
    for k, v in node_to_cluster.items():
        cluster_to_nodes[v].add(k)

    for node in unresolved_nodes:
        resolve_node(
            tree,
            node,
            core_clusters,
            ambiguous_clusters,
            node_to_cluster,
            cluster_to_nodes,
        )

    if utils.find_unresolved_node(tree):
        raise Exception("Failed to resolve all nodes.")


def generate_multree(gene_trees: List[Tuple[treeswift.Tree, float]], filter_strategy: 'FilteringStrategy',
                     root_cluster: FrozenMultiset = None):
    """
    Generate a multi-labeled tree from a list of gene trees.
    :param gene_trees: A list of gene trees, each with its frequency.
    :param filter_strategy: The filtering strategy to use for selecting clusters.
    :param root_cluster: The taxa of the multi-labeled tree.
    :return: A multi-labeled tree.
    """
    if root_cluster is None:
        root_cluster = identify_multree_root_cluster([tree for tree, _ in gene_trees])

    tree_clusters = compute_clusters_from_trees(gene_trees)
    reduce_clusters_on_leaves(tree_clusters, root_cluster)
    remove_redundant_clusters(tree_clusters)
    core_clusters, ambiguous_clusters = compute_core_and_ambiguous_clusters(tree_clusters, root_cluster)
    filtered_core_clusters = filter_clusters(core_clusters, filter_strategy)
    filtered_ambiguous_clusters = filter_clusters(ambiguous_clusters, filter_strategy)
    selected_backbone_clusters = select_backbone_clusters(filtered_core_clusters, filtered_ambiguous_clusters,
                                                          root_cluster)

    backbone_tree = generate_tree_from_clusters(selected_backbone_clusters, root_cluster)
    backbone_newick = backbone_tree.newick()
    unresolved_nodes = utils.find_unresolved_node(backbone_tree)
    if not unresolved_nodes:
        print(f"Final Mul-Tree: {backbone_tree.newick()}\n")
        return backbone_tree, filtered_ambiguous_clusters

    # Step 3: Resolve the backbone tree
    resolve_nodes(backbone_tree, unresolved_nodes, core_clusters, ambiguous_clusters)
    leaves_cluster = FrozenMultiset([l.get_name() for l in backbone_tree.get_leaves()])
    if leaves_cluster != root_cluster:
        raise ValueError(
            f"Mul-Tree leaves {str(leaves_cluster)} do not match the taxa {str(root_cluster)}.\n{backbone_tree.newick()}")
    #print(f"Backbone tree: {backbone_newick}")
    print(f"Final Mul-Tree: {backbone_tree.newick()}\n")
    return backbone_tree, filtered_ambiguous_clusters


