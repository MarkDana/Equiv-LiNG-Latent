from itertools import product, combinations, permutations
import time, random
import numpy as np
import networkx as nx
import warnings
from matplotlib import MatplotlibDeprecationWarning
warnings.filterwarnings("ignore", category=MatplotlibDeprecationWarning)

from core_logic.my_utils import (
    get_all_row_perms_leading_to_unit_diagonals, \
    digraph_adjmat_to_config, digraph_config_to_adjmat, digraph_edges_iterable_to_adjmat, \
    get_perm_rank_by_scipy, mat_to_config, config_to_mat, get_new_mat_config_after_switch_edge
)

class EarlyStopException(Exception): pass
def check_early_stop(early_stop_at_size, early_stop_at_time, size_now, time_now):
    if early_stop_at_size is not None and size_now > early_stop_at_size: raise EarlyStopException
    if early_stop_at_time is not None and time_now > early_stop_at_time: raise EarlyStopException


def check_and_get_an_equivalent_irreducible_model(L_nodeset, X_nodeset, edgeset):
    '''
    :param L_nodeset: set of latent nodes
    :param X_nodeset: set of observed nodes, disjoint from L_nodeset
    :param edgeset: set of direct edges (i,j)
    :return:
        is_irreducible: bool, whether the input model is already irreducible
        L_nodeset_reduced: subset of L_nodeset
        new_edgeset: set of direct edges (i,j) among L_nodeset_reduced | X_nodeset, which is now irreducible and equiv. to the input model
    '''
    G = nx.DiGraph()
    G.add_nodes_from(L_nodeset | X_nodeset)
    G.add_edges_from(edgeset)
    latents_that_are_not_ancestors_to_any_X = {l for l in L_nodeset if len(set(nx.descendants(G, l)) & X_nodeset) == 0}
    G.remove_nodes_from(latents_that_are_not_ancestors_to_any_X)
    L_nodeset_reduced = L_nodeset - latents_that_are_not_ancestors_to_any_X
    L_children = {l: set(G.successors(l)) for l in L_nodeset_reduced}
    L_parents = {l: set(G.predecessors(l)) for l in L_nodeset_reduced}

    maximal_redundant_Lsets = set()
    for r in range(len(L_nodeset_reduced), 0, -1):  # sizes from n down to 1
        for lset in combinations(L_nodeset_reduced, r):
            lset = frozenset(lset)
            if not any(lset < maxrl for maxrl in maximal_redundant_Lsets):
                if len(set().union(*[L_children[l] for l in lset]) - lset) < 2:
                    maximal_redundant_Lsets.add(lset)
    # assert len(set().union(*maximal_redundant_Lsets)) == sum(len(lset) for lset in maximal_redundant_Lsets)  # mutually disjoint
    redundant_Lsingles = set().union(*maximal_redundant_Lsets)

    n_edges_before_operation = len(G.edges())
    for lset in maximal_redundant_Lsets:
        parents_outside = set().union(*[L_parents[l] for l in lset]) - set(lset)
        children_outside = set().union(*[L_children[l] for l in lset]) - set(lset)
        # assert len(parents_outside & redundant_Lsingles) == 0 and len(children_outside & redundant_Lsingles) == 0 # wont affect other redundant Ls
        # assert len(children_outside) == 1  # only have one child outside (if 0, already removed); and this lset | {c} is a proportional clique
        new_edges_to_add = list((product(parents_outside - children_outside, children_outside)))  # may intersect with existing edges, but that's ok
        G.remove_nodes_from(lset)
        G.add_edges_from(new_edges_to_add)
    # if len(maximal_redundant_Lsets) > 0: assert len(G.edges()) < n_edges_before_operation # not just removed nodes; edgenum also reduced.

    L_nodeset_reduced -= redundant_Lsingles
    is_irreducible = len(L_nodeset_reduced) == len(L_nodeset)
    return is_irreducible, L_nodeset_reduced, set(G.edges())


def get_union_and_intersection_and_traversal_of_colaugs(Q, ldim, max_class_size=None, max_time_seconds=None):
    '''
    ### Return:
        colaug_unions: a dict mapping each xi to a tuple.
        colaug_intersections: a dict mapping each xi to a tuple.
        colaug_traversals: a dict mapping each xi to a set of tuples.
        finished_flag: boolean.

    ### Given a matrix Q whose columns are partitioned into L and X, for each xi,
        we are interested in all column vectors that can be put in place of Q[:, xi], such that matroid(Q[:, L + {xi}]) remains unchanged. Denote this set as colaug(xi).

    ### About the union of colaug:
        1. There exists a unique maximum (in terms of the number of `1' entries) element in colaug(xi).
        2. Every element in colaug(xi) is a subset of this maximum element.
        3. This maximum element can be found by starting from any element, and find all places where current 0 can be flipped to 1 (in one step).
        4. This maximum element is characterized by Lemma 10 in the paper.

    ### About the intersection of colaug:
        Let minimum(colaug(xi)) denote the set with minimum number of `1' entries in colaug(xi).
        Let minimal(colaug(xi)) denote the set of all minimal elements in colaug(xi) (w.r.t. set inclusion).
        1. minimum(colaug(xi)) == minimal(colaug(xi)) == minimum_different_cocircuits (Corollary 2 in the paper). Note this set may contain multiple elements.
        2. The intersection of this minimum(colaug(xi)) is the intersection of the entire colaug(xi).
        3. This intersection can be found by starting from **the union element**, and find all places where current 1 can be flipped to 0 (in one step).
           Note: unlike the union case, this time we cannot start from any element. For example,
           Q = [[1, 0],  L and X are 1st, 2nd columns respectively.  colaug(2nd column) = [ [1,0]^T, [0,1]^T, [1,1]^T ], so intersection is just [0,0]^T.
                [1, 1]],                                             However, if starting from [0,1]^T, that `1' entry cannot be flipped, and we get wrong intersection [0,1]^T.

    ### My local speed test:
        odim: 2, ldim: 0, time used per run: 0.00001 s, avg num prod(colaugs): 1.00
        odim: 3, ldim: 0, time used per run: 0.00003 s, avg num prod(colaugs): 1.00
        odim: 3, ldim: 1, time used per run: 0.00003 s, avg num prod(colaugs): 5.77
        odim: 4, ldim: 0, time used per run: 0.00004 s, avg num prod(colaugs): 1.00
        odim: 4, ldim: 1, time used per run: 0.00006 s, avg num prod(colaugs): 26.95
        odim: 4, ldim: 2, time used per run: 0.00006 s, avg num prod(colaugs): 44.95
        odim: 5, ldim: 0, time used per run: 0.00007 s, avg num prod(colaugs): 1.00
        odim: 5, ldim: 1, time used per run: 0.00009 s, avg num prod(colaugs): 163.75
        odim: 5, ldim: 2, time used per run: 0.00013 s, avg num prod(colaugs): 972.21
        odim: 5, ldim: 3, time used per run: 0.00014 s, avg num prod(colaugs): 282.37
        odim: 6, ldim: 0, time used per run: 0.00010 s, avg num prod(colaugs): 1.00
        odim: 6, ldim: 1, time used per run: 0.00014 s, avg num prod(colaugs): 1165.16
        odim: 6, ldim: 2, time used per run: 0.00024 s, avg num prod(colaugs): 41508.15
        odim: 6, ldim: 3, time used per run: 0.00033 s, avg num prod(colaugs): 22551.73
        odim: 6, ldim: 4, time used per run: 0.00035 s, avg num prod(colaugs): 1527.70
        odim: 7, ldim: 0, time used per run: 0.00014 s, avg num prod(colaugs): 1.00
        odim: 7, ldim: 1, time used per run: 0.00020 s, avg num prod(colaugs): 14406.64
        odim: 7, ldim: 2, time used per run: 0.00038 s, avg num prod(colaugs): 2402197.98
        odim: 7, ldim: 3, time used per run: 0.00071 s, avg num prod(colaugs): 4186826.25
        odim: 7, ldim: 4, time used per run: 0.00092 s, avg num prod(colaugs): 384724.75
        odim: 7, ldim: 5, time used per run: 0.00085 s, avg num prod(colaugs): 7231.32
        odim: 8, ldim: 0, time used per run: 0.00018 s, avg num prod(colaugs): 1.00
        odim: 8, ldim: 1, time used per run: 0.00027 s, avg num prod(colaugs): 84249.05
        odim: 8, ldim: 2, time used per run: 0.00062 s, avg num prod(colaugs): 270907830.28
        odim: 8, ldim: 3, time used per run: 0.00136 s, avg num prod(colaugs): 1234535681.16
        odim: 8, ldim: 4, time used per run: 0.00255 s, avg num prod(colaugs): 235646592.50
        odim: 8, ldim: 5, time used per run: 0.00325 s, avg num prod(colaugs): 4966720.80
        odim: 8, ldim: 6, time used per run: 0.00291 s, avg num prod(colaugs): 36097.43
        odim: 9, ldim: 0, time used per run: 0.00025 s, avg num prod(colaugs): 1.00
        odim: 9, ldim: 1, time used per run: 0.00037 s, avg num prod(colaugs): 1564676.86
        odim: 9, ldim: 2, time used per run: 0.00102 s, avg num prod(colaugs): 30661294020.18
        odim: 9, ldim: 3, time used per run: 0.00295 s, avg num prod(colaugs): 848672781045.58
        odim: 9, ldim: 4, time used per run: 0.00663 s, avg num prod(colaugs): 275814528934.46
        odim: 9, ldim: 5, time used per run: 0.01103 s, avg num prod(colaugs): 8425001165.68
        odim: 9, ldim: 6, time used per run: 0.01177 s, avg num prod(colaugs): 49049834.35
        odim: 9, ldim: 7, time used per run: 0.01001 s, avg num prod(colaugs): 153941.28
        odim: 10, ldim: 0, time used per run: 0.00030 s, avg num prod(colaugs): 1.00
        odim: 10, ldim: 1, time used per run: 0.00046 s, avg num prod(colaugs): 25014680.72
        odim: 10, ldim: 2, time used per run: 0.00146 s, avg num prod(colaugs): 6909271127724.65
        odim: 10, ldim: 3, time used per run: 0.00561 s, avg num prod(colaugs): 786627031218084.25
        odim: 10, ldim: 4, time used per run: 0.01862 s, avg num prod(colaugs): 851509596861513.62
        odim: 10, ldim: 5, time used per run: 0.03637 s, avg num prod(colaugs): 35903114239498.42
        odim: 10, ldim: 6, time used per run: 0.04889 s, avg num prod(colaugs): 221235040414.21
        odim: 10, ldim: 7, time used per run: 0.05219 s, avg num prod(colaugs): 522432054.73
        odim: 10, ldim: 8, time used per run: 0.03969 s, avg num prod(colaugs): 662760.33
    '''
    xdim, ydim = Q.shape
    QL = Q[:, :ldim]

    rank_cache = dict()

    def rank_L(Z):
        key = sum(1 << i for i in Z)
        if key not in rank_cache: rank_cache[key] = get_perm_rank_by_scipy(QL[Z])
        return rank_cache[key]

    colaug_unions, colaug_intersections = dict(), dict()
    for va in range(ldim, ydim):
        current_va_children = Q[:, va].nonzero()[0].tolist()
        current_va_non_children = [i for i in range(xdim) if i not in current_va_children]
        rank_with_vb = rank_L(current_va_non_children)
        can_adds = [vb for vb in current_va_non_children if rank_L([i for i in current_va_non_children if i != vb]) < rank_with_vb]
        va_union = sorted(current_va_children + can_adds)
        va_non_union = [i for i in range(xdim) if i not in va_union]
        rank_without_vb = rank_L(va_non_union)
        can_removes = [vb for vb in va_union if rank_L(va_non_union + [vb]) > rank_without_vb]
        va_intersection = [i for i in va_union if i not in can_removes]
        colaug_unions[va], colaug_intersections[va] = tuple(va_union), tuple(va_intersection)

    early_stop_at_time = None if max_time_seconds is None else max_time_seconds + time.time()
    finished_flag = True
    colaug_traversals = dict()

    for va in range(ldim, ydim):
        if not finished_flag:
            colaug_traversals[va] = {colaug_unions[va]}
            continue
        rough_remaining_size = None if max_class_size is None else int(max_class_size / np.prod([len(colaug_traversals[xj]) for xj in range(ldim, va)]))
        known_equivs = set()
        queue = {colaug_unions[va]}
        try:
            while queue:
                currS = queue.pop()
                known_equivs.add(currS)
                removal_candidates = [i for i in currS if i not in colaug_intersections[va]]
                if not removal_candidates: continue
                non_currS = [i for i in range(xdim) if i not in currS]
                rank_without_vb = rank_L(non_currS)
                for vb in removal_candidates:  # find possible removals
                    newS = tuple(i for i in currS if i != vb)
                    if newS in known_equivs or newS in queue: continue  # since we already cached rank calls, we ignore those cannot points' cache.
                    if rank_L(non_currS + [vb]) > rank_without_vb: queue.add(newS)
                check_early_stop(rough_remaining_size, early_stop_at_time, len(known_equivs) + len(queue), time.time())
        except EarlyStopException: finished_flag = False
        if not finished_flag: known_equivs |= queue
        colaug_traversals[va] = known_equivs

    return colaug_unions, colaug_intersections, colaug_traversals, finished_flag


def get_union_and_intersection_and_traversal_of_equiv_matL(Q, max_class_size=None, max_time_seconds=None):
    '''
    ### Return:
        Qunion: a numpy array of shape as Q.
        Qintersection: a numpy array of shape as Q.
        Qtraversals: a set of integer configs of equiv mats.
        finished_flag: boolean.

    ### Given a matrix Q, we are interested in all the equivalent same-shaped matrices that induce the same row transversal matroid as Q.
    Let E(Q) denote this equivalence class of matrices;
    Let E+-(Q) denote the set of matrices reachable from Q by series of admission edge addition/deletions.
    We have:
        1. E+-(Q) ⊆ E(Q). For example,
           Q = [[1, 0],        E(Q) = { [[1, 0],     [[0, 1],          but E+-(Q) contains just this Q.
                [1, 0]]                  [1, 0]]  ,   [0, 1]]  }
        2. colperm(E+-(Q)) == E(Q), where colperm() means post-applying column permutations to elements obtained in E+-(Q).
           Though it's worth noting that E+-(Q) itself may also contain column-permutation-isomorphic matrices. For example,
           Q = [[1, 1],        E+-(Q) contains all 7 matrices, including both [[1, 0],   and   [[0, 1],
                [1, 1]]                                                        [0, 1]]          [1, 0]].

    ### Before introducing the union/intersection results, we also introduce another way to help traverse E+-(Q).
    Let Ecolaug(Q) denote the matrices by taking the Cartesian product of each Q's column augmentation against others.
    We have:
        1. colperm(E+-(Q)) ⊆ colperm(Ecolaug(Q)), where the additionals may be incorrect. For example,
           Q = [[1, 1],        colaug(1st column) = {[1,1]^T, [1,0]^T, [0,1]^T},   but after taking product, the resulting [[1, 1],  is incorrect.
                [1, 1]]        colaug(2nd column) = {[1,1]^T, [1,0]^T, [0,1]^T},                                            [0, 0]]

    ### Now, we are interested in the union and intersection of E+-(Q).

    About union, we have:
        1. There exists *a unique* mat in E+-(Q) that has the maximum number of edges (`1' entries). Denote it by Qmax.
        2. All other mats in E+-(Q) are proper subsets of Qmax.
        3. Note that in each column augmentation colaug(yi), there also exists a unique maximum/maximal member, and
           this Qmax is exactly formed by these maximum/maximal column vectors.

    About intersection, we have:
        1. E+-(Q)'s     minimal mats (no proper subsets of it can also be within E+-(Q))
            **equals**  minimum mats (in terms of the number of edges).
        2. The intersection of these mats is exactly the intersection of the whole E+-(Q).

        Then, how to link this intersection with the (faster) colaug of each column? Cannot direcly do so (as in union case). For example,
           Q = [[1, 0],        colaug(1st column) = {[1,1]^T, [1,0]^T}  ==intersection=> [1,0]^T   ==>  [[1, 0],
                [0, 1]]        colaug(2nd column) = {[1,1]^T, [0,1]^T}  ==intersection=> [0,1]^T         [0, 1]].
           But in fact, the intersection of 7 mats in E+-(Q) is emtpy, i.e., [[0, 0],
                                                                              [0, 0]].
           This is because the dependencies among columns are (not) considered when do column-wise augmentations.

        To obtain the full possible columns, we have to start with the Qmax obtained earlier:
        3. This intersection is the intersection of each column augmentation from Qmax.

    ### My local speed test:
        odim: 2, ldim: 0, time used per run: 0.00000 s, avg size E+-: 1.00
        odim: 3, ldim: 0, time used per run: 0.00000 s, avg size E+-: 1.00
        odim: 3, ldim: 1, time used per run: 0.00000 s, avg size E+-: 1.00
        odim: 4, ldim: 0, time used per run: 0.00000 s, avg size E+-: 1.00
        odim: 4, ldim: 1, time used per run: 0.00000 s, avg size E+-: 1.00
        odim: 4, ldim: 2, time used per run: 0.00020 s, avg size E+-: 9.25
        odim: 5, ldim: 0, time used per run: 0.00000 s, avg size E+-: 1.00
        odim: 5, ldim: 1, time used per run: 0.00000 s, avg size E+-: 1.00
        odim: 5, ldim: 2, time used per run: 0.00028 s, avg size E+-: 11.52
        odim: 5, ldim: 3, time used per run: 0.02879 s, avg size E+-: 1268.95
        odim: 6, ldim: 0, time used per run: 0.00000 s, avg size E+-: 1.00
        odim: 6, ldim: 1, time used per run: 0.00000 s, avg size E+-: 1.00
        odim: 6, ldim: 2, time used per run: 0.00034 s, avg size E+-: 12.52
        odim: 6, ldim: 3, time used per run: 0.07192 s, avg size E+-: 2737.51
    '''
    xdim, ydim = Q.shape
    if ydim < 2: return Q.copy(), Q.copy(), {mat_to_config(Q)}, True

    rank_cache = dict()  # TODO: be aware of possible of memory explosion here.

    def get_rank(mat):
        # assert mat.shape[1] == ydim - 1
        key = mat_to_config(mat)  # row and column permutations also wont change rank; but caching them is more complex.
        if key not in rank_cache: rank_cache[key] = get_perm_rank_by_scipy(mat)
        return rank_cache[key]

    Qunion = Q.copy()
    for va in range(ydim):
        Y_minus_va = [i for i in range(ydim) if i != va]
        current_va_children = Qunion[:, va].nonzero()[0].tolist()
        current_va_non_children = [i for i in range(xdim) if i not in current_va_children]
        rank_with_vb = get_rank(Qunion[current_va_non_children][:, Y_minus_va])
        can_adds = [vb for vb in current_va_non_children if get_rank(Qunion[[i for i in current_va_non_children if i != vb]][:, Y_minus_va]) < rank_with_vb]
        Qunion[can_adds, va] = 1

    Qintersection = Qunion.copy()
    for va in range(ydim):
        Y_minus_va = [i for i in range(ydim) if i != va]
        current_va_children = Qunion[:, va].nonzero()[0].tolist()  # have to remove from Qunion, not anything smaller.
        current_va_non_children = [i for i in range(xdim) if i not in current_va_children]
        rank_without_vb = get_rank(Qunion[current_va_non_children][:, Y_minus_va])
        can_removes = [vb for vb in current_va_children if get_rank(Qunion[current_va_non_children + [vb]][:, Y_minus_va]) > rank_without_vb]
        Qintersection[can_removes, va] = 0

    early_stop_at_time = None if max_time_seconds is None else max_time_seconds + time.time()
    finished_flag = True
    known_equivs = set()
    known_nonequivs = set()
    queue = {mat_to_config(Qunion)}
    try:
        while queue:
            Qconfig = queue.pop()
            known_equivs.add(Qconfig)
            Q = config_to_mat(Qconfig, xdim, ydim)
            for va in range(ydim):
                current_va_children = Q[:, va].nonzero()[0].tolist()
                removal_candidates = [i for i in current_va_children if Qintersection[i, va] == 0]
                if not removal_candidates: continue
                Y_minus_va = [i for i in range(ydim) if i != va]
                current_va_non_children = [i for i in range(xdim) if i not in current_va_children]
                rank_without_vb = get_rank(Q[current_va_non_children][:, Y_minus_va])
                for vb in removal_candidates:
                    new_config = get_new_mat_config_after_switch_edge(Qconfig, xdim, ydim, vb, va)
                    if new_config in known_equivs or new_config in queue or new_config in known_nonequivs: continue
                    if get_rank(Q[current_va_non_children + [vb]][:, Y_minus_va]) > rank_without_vb: queue.add(new_config)
                    else: known_nonequivs.add(new_config)
            check_early_stop(max_class_size, early_stop_at_time, len(known_equivs) + len(queue), time.time())
    except EarlyStopException: finished_flag = False

    if not finished_flag: known_equivs |= queue
    return Qunion, Qintersection, known_equivs, finished_flag


def traverse_dist_equiv_class_from_an_irreducible_graph_config(
    n_nodes,
    n_latents,
    graph_config,
    max_class_size_for_column_augmentation=None,
    max_time_seconds_for_column_augmentation=None,
    max_class_size_for_L_bipartite_traverse=None,
    max_time_seconds_for_L_bipartite_traverse=None,
    max_class_size_for_LX_combining=None,
    max_time_seconds_for_LX_combining=None,
    max_class_size_for_L_rename_deduplicate=None,
    max_time_seconds_for_L_rename_deduplicate=None,
):
    '''
    Given a graph config, traverse the whole distribution equivalence class it belongs to, by doing all possible transformations.
    :param n_nodes: int, total number of nodes
    :param n_latents: int, the first n_latents nodes are latent, the rest observed
    :param graph_config: int,
        an encoding of the graph's adjacency matrix. see my_utils for details.
        at input, this graph should already be irreducible.  for website speed, no further check is done within this function.

    # the following params are controlling early stop of the search, for website speed.
    # for local search, we may set all to None, i.e., to get the whole class, no matter how large it is.

    :return:
        a dict with the following keys
        'visited_graph_configs_after_search': set of ints,
            the set of graph configs visited after the search phase (before L-rename deduplication).
            suppose there is no early stop, then this set, after L-rename expansion, should cover the whole distribution equivalence class.
        'is_search_finished': bool, whether the search phase is finished,
            i.e., whether 'visited_graph_configs_after_search' is trustly to cover the whole class.
        'deduplicated_graph_configs_after_L_rename': set of ints,
            a subset of 'visited_graph_configs_after_search', after deduplicating L-isomorphic ones.
            suppose there is no early stop, then this set is all the non-L-isomorphic graphs in the distribution equivalence class.
        'whole_graph_configs_expanded_by_L_renaming': set of ints,
            a superset of 'visited_graph_configs_after_search', after expanding each graph by all possible L-renamings.
            suppose there is no early stop, then this set is exactly the whole distribution equivalence class.
        'is_L_rename_deduplicate_finished': bool, whether the L-rename deduplication phase is finished,
            i.e., whether items in 'deduplicated_graph_configs_after_L_rename' are trustly non-L-isomorphic.
    '''

    ##################### 1. preparations #####################
    Xlist = list(range(n_latents, n_nodes))
    gmat = digraph_config_to_adjmat(graph_config, n_nodes, fill_diag_val=1)
    finished_search_flag = True

    ##################### 2. independently decompose X columns' augmentation #####################
    colaug_unions, colaug_intersections, colaug_traversals, this_finished = get_union_and_intersection_and_traversal_of_colaugs(
        gmat, n_latents, max_class_size_for_column_augmentation, max_time_seconds_for_column_augmentation)
    finished_search_flag &= this_finished

    ##################### 3. traverse bipartite graphs to satify the L part #####################
    matL_union, matL_intersection, known_equivs_L_bipar, this_finished = get_union_and_intersection_and_traversal_of_equiv_matL(
        gmat[:, :n_latents], max_class_size_for_L_bipartite_traverse, max_time_seconds_for_L_bipartite_traverse)
    finished_search_flag &= this_finished

    ##################### 4. get the CPDAG-like presentation (edges that can and must appear) #####################
    gmat_cpdag_like_representation = np.zeros_like(gmat)
    gmat_cpdag_like_representation[:, :n_latents] = matL_intersection + matL_union
    for xi in Xlist:
        gmat_cpdag_like_representation[list(colaug_unions[xi]), xi] = 1
        gmat_cpdag_like_representation[list(colaug_intersections[xi]), xi] = 2

    ##################### 5. combine the two parts of L columns and X columns #####################
    early_stop_at_time = None if max_time_seconds_for_LX_combining is None else max_time_seconds_for_LX_combining + time.time()
    known_equiv_configs = {graph_config}
    already_parsed_row_keys = set()
    try:
        for matL_config in known_equivs_L_bipar:
            matL = config_to_mat(matL_config, n_nodes, n_latents)
            for xcolaugs in product(*[colaug_traversals[xi] for xi in Xlist]):
                matX = np.zeros((n_nodes, len(Xlist)), dtype=np.uint8)
                for i, colaug in enumerate(xcolaugs): matX[list(colaug), i] = 1
                mat = np.hstack([matL, matX])
                mat_rows_key = tuple(sorted([sum(1 << v for v in mat[i].nonzero()[0]) for i in range(n_nodes)]))
                if mat_rows_key in already_parsed_row_keys: continue
                for perm in get_all_row_perms_leading_to_unit_diagonals(mat):
                    known_equiv_configs.add(digraph_adjmat_to_config(mat[list(perm)]))
                    check_early_stop(max_class_size_for_LX_combining, early_stop_at_time, len(known_equiv_configs), time.time())
                already_parsed_row_keys.add(mat_rows_key)
    except EarlyStopException: finished_search_flag = False

    ##################### 6. post processing: L-rename deduplication #####################
    finished_L_rename_deduplicate_flag = True
    duplicated_graph_configs = set()
    tic = time.time()
    if n_latents >= 2:
        early_stop_at_time= None if max_time_seconds_for_L_rename_deduplicate is None else max_time_seconds_for_L_rename_deduplicate + time.time()
        expanded_graph_configs_by_L_renaming = set()
        try:
            for gconfig in known_equiv_configs:
                if gconfig in expanded_graph_configs_by_L_renaming:
                    duplicated_graph_configs.add(gconfig)
                    continue
                gmat = digraph_config_to_adjmat(gconfig, n_nodes, fill_diag_val=1)
                for permL in permutations(range(n_latents)):
                    permV = list(permL) + list(range(n_latents, n_nodes))
                    gmat_perm = gmat[np.ix_(permV, permV)]
                    newconfig = digraph_adjmat_to_config(gmat_perm)
                    expanded_graph_configs_by_L_renaming.add(newconfig)
                    check_early_stop(max_class_size_for_L_rename_deduplicate, early_stop_at_time, len(expanded_graph_configs_by_L_renaming), time.time())
        except EarlyStopException: finished_L_rename_deduplicate_flag = False
    else:
        expanded_graph_configs_by_L_renaming = known_equiv_configs
    time_L_rename_deduplicate = time.time() - tic

    return {
        'visited_graph_configs_after_search': known_equiv_configs,
        'is_search_finished': finished_search_flag,
        'deduplicated_graph_configs_after_L_rename': known_equiv_configs - duplicated_graph_configs,
        'whole_graph_configs_expanded_by_L_renaming': expanded_graph_configs_by_L_renaming,
        'is_L_rename_deduplicate_finished': finished_L_rename_deduplicate_flag,
        'time_L_rename_deduplicate': time_L_rename_deduplicate,
        'cpdag_like_representation': gmat_cpdag_like_representation.astype(np.uint8)
    }


