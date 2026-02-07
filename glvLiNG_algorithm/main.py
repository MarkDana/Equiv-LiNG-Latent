#!/usr/bin/python3
# -*- coding: utf-8 -*-
import random
import numpy as np
from scipy.optimize import linear_sum_assignment
from itertools import combinations
from utils_matroid import (
    reconstructTransversalMatroidProbs, getBipartiteGraph, translate_sage_bipartite_graph_to_mat,
    powerset, compute_circuits, harmonicAvg, avg, product, complementaryProduct
)
random.seed(42)
np.random.seed(42)


def glvLiNG(A_tilde, hyperparams):
    '''
    Given an estimated mixing matrix A_tilde from OICA, return a maximal digraph (as adjacency matrix).
    :param A_tilde: np.ndarray, shape (xdim, xdim+ldim), where xdim is the number of observables, ldim is the number of latents.
           we assume that A_tilde is column scaled and permuted from the true mixing matrix A.
           the row indices correspond to observables.
    :param hyperparams: to control for hyperparameters for determining full ranks.

    :return: Q, np.ndarray, shape (ldim+xdim, ldim+xdim), adjacency matrix of the maximal digraph.
           the first ldim indices are latents, the last xdim indices are observables.
           Q[Vi, Vj] == 1 if and only if Vj -> Vi is in the digraph.
           this digraph corresponds to the maximal presentation of the equivalence subclass (Theorem 4).
           if you want to further get edge types in this presentation, or to traverse from this digraph to the whole equivalence class,
           you can run the corresponding parts in the equivalence_class_searcher/ .
    '''
    xdim, vdim = A_tilde.shape
    ldim = vdim - xdim
    Q = np.zeros((vdim, vdim), dtype=np.uint8)

    # Preparation: get the basis probability scores for each xsubset, again L, and L+{xi}'s. (didnt do any pruning here; though doable)
    def full_rank_score(A):
        sigma_min = np.linalg.svd(A, compute_uv=False)[-1]
        return 1 / (1 + np.exp(-hyperparams["alpha"] * (sigma_min - hyperparams["epsilon"])))
    scoredict = dict()
    for Xi in [-1] + list(range(ldim, vdim)):
        scoredict[Xi] = dict()
        Lxi = list(range(ldim)) if Xi == -1 else list(range(ldim)) + [Xi]
        V_minus_Lxi = [i - ldim for i in range(vdim) if i not in Lxi]
        for S in combinations(range(vdim), len(Lxi)):
            V_minus_S = [i for i in range(vdim) if i not in S]
            scoredict[Xi][frozenset(S)] = full_rank_score(A_tilde[V_minus_Lxi][:, V_minus_S])

    # Phase 1: determine L->V edges
    if ldim > 0:
        basis_prob_scores = scoredict[-1]
        max_score = max(basis_prob_scores.values())
        max_key = [max_key for max_key, score in basis_prob_scores.items() if np.isclose(score, max_score)][0]
        basis_prob_scores[max_key] = 1.0  # ensure at least one basis is above cutoff.
        if ldim >= 2:
            M_LtoV = reconstructTransversalMatroidProbs(V=frozenset(range(vdim)), basisProbs=basis_prob_scores, r=ldim, config=hyperparams)
            Lmat_recovered = translate_sage_bipartite_graph_to_mat(getBipartiteGraph(M_LtoV), vdim)
            Q[:, :Lmat_recovered.shape[1]] = Lmat_recovered
        else:
            nonzero_entries = [i for i in range(vdim) if basis_prob_scores[frozenset([i])] >= hyperparams['basis-cutoff']]
            Q[nonzero_entries, 0] = 1
    if ldim == 0: ML_indep_sets = {frozenset([])}
    elif ldim == 1: ML_indep_sets = {frozenset([])}.union({frozenset([i]) for i in range(vdim) if Q[i,0] == 1})
    else: ML_indep_sets = set(frozenset(s) for s in M_LtoV.independent_sets())

    # Phase 2: determine X->V edges
    for xi in range(ldim, vdim):
        basis_prob_scores = scoredict[xi]
        MLx_bases = {S for S, score in basis_prob_scores.items() if score >= hyperparams['basis-cutoff']}
        MLx_indep_sets = set().union(*[{frozenset(s) for s in powerset(b)} for b in MLx_bases])
        renaming_basis_scores_sorted = sorted([(S, score) for S, score in basis_prob_scores.items() if S not in MLx_bases], key=lambda x: -x[1])
        while not (ML_indep_sets <= MLx_indep_sets):
            # ensure that after augmenting one column, those already independent sets are still independent.
            # this is a necessary condition; but still cannot guarantee that MLx can be augmented from ML in a legit way.
            S_to_add, _ = renaming_basis_scores_sorted.pop(0)
            MLx_bases.add(S_to_add)
            MLx_indep_sets.update({frozenset(s) for s in powerset(S_to_add)})
        MLx_circuits = compute_circuits(range(vdim), MLx_indep_sets)
        D_lemma10 = [i for i in range(vdim) if {frozenset(c - {i}) for c in MLx_circuits} & ML_indep_sets == set()]
        Q[D_lemma10, xi] = 1

    row_ind, col_ind = linear_sum_assignment(1 - Q)
    Q = Q[row_ind[np.argsort(col_ind)]]
    np.fill_diagonal(Q, 0)
    return Q



if __name__ == '__main__':

    # To show the usage of the algorithm, we run it on a synthetic example (G1 in Fig 3 in the paper).
    # In practice, you should first get the OICA estimated A_tilde from your data.
    #   In our experiments, we used the matlab OICA implementation in https://github.com/gilgarmish/oica (Podosinnikova et al., 2019)
    nodenames = ['L1', 'L2', 'X1', 'X2', 'X3']
    edges = [('L1', 'X1'), ('L1', 'X2'), ('L2', 'X2'), ('L2', 'X3'), ('X2', 'X3')]
    nodenum = len(nodenames)
    B_adjacency = np.zeros((nodenum, nodenum))
    for src, tgt in edges:
        B_adjacency[nodenames.index(tgt), nodenames.index(src)] = random.uniform(0.5, 2.5) * random.choice([-1, 1])
    A_mixing_full = np.linalg.inv(np.eye(nodenum) - B_adjacency)
    A_tilde = A_mixing_full[2:]  # we only observe X
    column_scaling_factors = np.random.uniform(0.5, 2.5, size=nodenum) * np.random.choice([-1, 1], size=nodenum)
    A_tilde = (A_tilde * column_scaling_factors)[:, np.random.permutation(nodenum)]  # do random column scaling and permutation to simulate the output of OICA.


    # With the OICA estimated A_tilde as input, we can run the algorithm as follows.
    # Due to the insufficiency of OICA in practice, the following hyperparameters may need to be tuned carefully.
    # Thw hyperparameters below are what we used in our experiments; you may start from it, but note that we didn't do much tuning.
    hyperparameters = {
        'alpha': 20.0,                    # related to sensitivity to SVD decomposition for rank computation; can choose freely
        'epsilon': 1e-4,                  # related to sensitivity to SVD decomposition for rank computation; can choose freely
        'basis-cutoff': 0.75,             # can choose from any value between 0 and 1
        'circuit-combinator': avg,        # can choose from [harmonicAvg, avg, min, max, product, complementaryProduct]
        'circuit-combinator-2': avg,      # can choose from [harmonicAvg, avg, min, max, product, complementaryProduct]
        'circuit-combinator-3': avg,      # can choose from [harmonicAvg, avg, min, max, product, complementaryProduct]
        'circuit-coloop-score': 1.0,      # suggest to fix at 1.0
        'closure-combinator': avg,        # can choose from [harmonicAvg, avg, min, max, product, complementaryProduct]
        'closure-cutoff': 0.75,           # can choose from any value between 0 and 1
        'closure-combinator-2': avg,      # can choose from [harmonicAvg, avg, min, max, product, complementaryProduct]
        'cyclic-flats-combinator': avg,   # can choose from [harmonicAvg, avg, min, max, product, complementaryProduct]
        'empty-cyclic-flat-score': 1.0    # suggest to fix at 1.0
     }
    Q_est = glvLiNG(A_tilde, hyperparameters)


    # We may find that in this simulation oracle example, indeed the correct maximal equivalent digraph G2 is recovered (up to L-renaming).
    # If you want to further get edge types in this presentation, or to traverse from this digraph to the whole equivalence class,
    #       you can run the corresponding parts in the equivalence_class_searcher/ .
    print(f'The estimated maximal digraph in equivalence subclass (with edge +/- but without cycle reversals) is:')
    for i in range(nodenum):
        for j in range(nodenum):
            if Q_est[j, i] == 1:
                print(f'  {nodenames[i]} -> {nodenames[j]}')



