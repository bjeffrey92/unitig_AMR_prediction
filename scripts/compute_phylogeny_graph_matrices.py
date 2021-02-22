import pandas as pd
import networkx as nx
from Bio import Phylo


def prune_tree(metadata, tree, outcome):
    pruned_ids = metadata.loc[pd.isna(metadata[outcome])].id
    for sample in pruned_ids:
        try:
            tree.prune(sample)
        except ValueError: #ignore reference genome
            pass
    return tree


def parse_tree_matrices(tree):
    G = Phylo.to_networkx(tree)
    
    tree_terminals = tree.get_terminals() #to name nodes in graph
    mapping = {i:i.name for i in tree_terminals}
    G = nx.relabel_nodes(G, mapping)



if __name__ == '__main__':
    metadata_file = 'data/filtered_metadata.csv'
    metadata = pd.read_csv(metadata_file)

    tree_file = 'gwas/pruned_final_tree.nwk'
    tree = Phylo.read(tree_file, 'newick')

    outcomes = [i for i in metadata.columns if i.startswith('log2_')]
    for outcome in outcomes:
        outcome_tree = prune_tree(metadata, tree, outcome)
        outcome_tree.root_at_midpoint()
