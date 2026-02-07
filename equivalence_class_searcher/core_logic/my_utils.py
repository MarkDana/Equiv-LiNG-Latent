#!/usr/bin/python3
# -*- coding: utf-8 -*-

from itertools import product, combinations, permutations, chain
import numpy as np
import networkx as nx
from scipy.optimize import linear_sum_assignment
import pydot

def mat_to_config(mat):
    return int("".join(map(str, mat.flatten())), 2) if mat.size > 0 else 0

def config_to_mat(config, ydim, xdim):
    if ydim == 0 or xdim == 0: return np.zeros((ydim, xdim), dtype=np.uint8)
    bits = f"{config:0{ydim * xdim}b}"
    return np.array(list(map(int, bits)), dtype=np.uint8).reshape((ydim, xdim))

def digraph_adjmat_to_config(adjmat):
    # regardless of the diagonals, since we dont allow self loops and dont allow them.
    # just return the n*n-1 off diagonal entries' config.
    return mat_to_config(adjmat[~np.eye(adjmat.shape[0], dtype=bool)])

def digraph_config_to_adjmat(config, n, fill_diag_val=1):
    mat = np.zeros((n, n), dtype=np.uint8)
    mat[~np.eye(n, dtype=bool)] = config_to_mat(config, 1, n*(n - 1))[0]
    np.fill_diagonal(mat, fill_diag_val)
    return mat

def digraph_adjmat_to_edges_frozenset(adjmat, rename=None):
    # adjmat: with or without unit diagonals; doesnt matter. col to row.
    adjmat_zero_diag = np.copy(adjmat)
    np.fill_diagonal(adjmat_zero_diag, 0)
    rows, cols = np.nonzero(adjmat_zero_diag.T)
    base_edges = frozenset(zip(map(int, rows), map(int, cols)))
    return base_edges if rename is None else frozenset((rename.get(u, u), rename.get(v, v)) for u, v in base_edges)

def digraph_edges_iterable_to_adjmat(edges, n, fill_diag_val=1):
    # edges: iterable of (u, v) tuples; u -> v.
    mat = np.zeros((n, n), dtype=np.uint8)
    edges = np.array(list(edges))
    if edges.size > 0: mat[edges[:, 1], edges[:, 0]] = 1
    np.fill_diagonal(mat, fill_diag_val)
    return mat

def digraph_edges_iterable_to_readable(edges, rename=None):
    # edges: iterable of (u, v) tuples; u -> v.
    if rename is None: rename = {}
    return '(' + ', '.join(sorted(f'{rename.get(u, u)}->{rename.get(v, v)}' for u, v in edges)) + ')'

def is_DAG(adjmat):
    # adjmat: with or without unit diagonals; assumes no self loops (so just fill diagonals with 0).
    adjmat_zero_diag = np.copy(adjmat)
    np.fill_diagonal(adjmat_zero_diag, 0)
    nxG = nx.from_numpy_array(adjmat_zero_diag.T, create_using=nx.DiGraph)
    return nx.is_directed_acyclic_graph(nxG)

def get_all_row_perms_leading_to_unit_diagonals(mat):
    # mat: a square binary matrix.
    # return: a list of perms, each satisfying that mat[p] has all unit diagonals.
    n = mat.shape[0]
    results = []
    used_rows = [False] * n
    current_perm = [None] * n
    def backtrack(col):
        if col == n:
            results.append(current_perm[:])
            return
        for row in range(n):
            if not used_rows[row] and mat[row, col] == 1:
                used_rows[row] = True
                current_perm[col] = row
                backtrack(col + 1)
                used_rows[row] = False
    backtrack(0)
    return results

def get_new_graph_config_after_switch_edge(graph_config, n_nodes, va, vb):
    bit_index = vb * (n_nodes - 1) + (va - int(va > vb))
    return graph_config ^ (1 << (n_nodes * (n_nodes - 1) - 1 - bit_index))

def get_perm_rank_by_scipy(binary_matrix):
    # We want to maximize 1s on the diagonal => minimize cost of not having 1
    # currently, the last line (sum of row_ind, col_ind) is the bottleneck (even much slower than linear_sum_assignment)
    # on my machine, to run on n_nodes=9, n_latents=2, avg_degree 0 to 2,
    # per hit time:
    #     [~750 ns] cost_matrix = (1 - binary_matrix)
    #     [~1000 ns; not used] cost_matrix = (1 - binary_matrix).astype(np.uint8)
    #     [~900 ns] row_ind, col_ind = linear_sum_assignment(cost_matrix)
    #     [~1900 ns] return binary_matrix[row_ind, col_ind].sum()
    #     [~1900 ns; not used; for loop even faster..] return sum(binary_matrix[i, j] for i, j in zip(row_ind, col_ind))
    #     [~5500 ns; deprecated; so slow..] return np.sum(binary_matrix[row_ind, col_ind])
    cost_matrix = (1 - binary_matrix)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    return binary_matrix[row_ind, col_ind].sum()

def get_bases_of_a_mat(mat):
    rk = get_perm_rank_by_scipy(mat)
    return {frozenset(Z) for Z in combinations(range(mat.shape[0]), rk) if get_perm_rank_by_scipy(mat[list(Z)]) == rk}

def get_cocircuits_of_a_mat(mat):
    rk = get_perm_rank_by_scipy(mat)
    n = mat.shape[0]
    cocircuits = set()
    for k in range(1, n - rk + 2):  # minimal such sets
        for subset in combinations(range(n), k):
            if any(smaller < set(subset) for smaller in cocircuits): continue
            if get_perm_rank_by_scipy(mat[list(set(range(n)) - set(subset))]) < rk:
                cocircuits.add(frozenset(subset))
    return cocircuits


def get_new_mat_config_after_switch_edge(mat_config, xdim, ydim, vx, vy):
    bit_index = vx * ydim + vy
    return mat_config ^ (1 << (xdim * ydim - 1 - bit_index))


def plot_digraph(nodes, diedges, vlabels, vcolors, vshapes, vsizes, svpath, edge_styles=None):
    if edge_styles is None: edge_styles = {}
    graph = pydot.Dot(graph_type='digraph', strict=True)
    graph.set('rankdir', 'TB')  # Top-to-bottom layout
    graph.set('splines', 'true')  # Curved edges
    graph.set('overlap', 'false')  # Avoid overlapping nodes
    graph.set('layout', 'dot')  # Use dot layout engine
    graph.set_prog('dot')  # Specify engine for rendering
    for n in nodes:
        node_kwargs = {
            "label": vlabels[n],
            "style": "filled",
            "fillcolor": vcolors[n],
            "shape": vshapes[n],
            "fontsize": "16",
            "width": str(vsizes[n]),
            "height": str(vsizes[n]),
            "margin": "0.0",
            "fixedsize": "true",
        }
        graph.add_node(pydot.Node(str(n), **node_kwargs))

    for src, tgt in diedges:
        graph.add_edge(pydot.Edge(str(src), str(tgt), style=edge_styles.get((src, tgt), 'solid')))

    graph.write_svg(svpath)






