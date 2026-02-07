#!/usr/bin/env sage --python3 -i

# The algorithm follows directly from Ingleton, Piff (1973),
# Section 3: Mason's alpha-criterion
#

import sage
from sage import all as _S
from itertools import chain, combinations
import sage.matroids
from sage.matroids.matroid import Matroid
import numpy as np



def allFlats(M):
    """ (M)

        return a chain-iterator of all flats of M, with ascending rank """
    return chain.from_iterable(M.flats(r) for r in range(0, M.rank() + 1))


def getAlphaSystem(M):
    """
        (M)
        calculate the alpha-transversal system of M"""
    alpha = {}
    for x in allFlats(M):
        nlt = len(x) - M.rank(x)
        for y in alpha:
            if y.issubset(x):
                nlt -= alpha[y]
        if nlt > 0:
            alpha[x] = nlt
        elif nlt < 0:
            raise Exception(f"Matroid is not a strict gammoid!\n\nalpha({x})={nlt}")
    # delkeys = [x for x in alpha if alpha[x] == 0]
    # for k in delkeys:
    #    del alpha[k]
    return alpha


def getAlphaTransversalSystem(M):
    """
        (M)
        calculate the alpha-transversal system of M as bipartite graph;
        """
    alpha = getAlphaSystem(M)
    matroid_edges = [('e', e) for e in M.groundset()]
    alpha_copies = [('alpha', i, F) for F, m in alpha.items() for i in range(m)]
    partition = [matroid_edges, alpha_copies]
    data = {m: [a for a in alpha_copies if m[1] in a[2]] for m in matroid_edges}
    return _S.BipartiteGraph(data, partition)


def getAlphaTransversal(M):
    """
        (M)
        returns a transversal of the alpha system of M
        """
    G = getAlphaTransversalSystem(M)
    return [(m[1], a[2]) for (m, a, l) in G.matching()]


def getStrictGammoidDigraph(M):
    """
        (M) strict gammoid
        returns a digraph and a set of terminals representing the strict gammoid M
    """
    t = getAlphaTransversal(M)
    D = _S.DiGraph()
    D.add_vertices(M.groundset())
    T = set(M.groundset())
    for u, vs in t:
        T.discard(u)
        for v in vs:
            if (u == v): continue
            D.add_edge(u, v)
    return D, T


def getBipartiteGraph(M):
    """
        (M) transversal matroid

        returns a bipartite graph that represents the transversal matroid

        (Note that loops of M will not turn up in the graph!)
    """
    D, T = getStrictGammoidDigraph(M.dual())
    # T = D.sinks()
    matroid_elts = [(m,) for m in M.groundset()]
    prime_elts = [(m, "'") for m in M.groundset() if not m in T]
    data = {(m, "'"): [(m,)] + [(v,) for (u, v, l) in D.outgoing_edges(m)]
            for m in M.groundset() if not m in T}
    rhs = frozenset()
    for key in data:
        rhs = rhs.union(data[key])

    matroid_elts_b = [x for x in matroid_elts if x in rhs]
    partition = [matroid_elts_b, prime_elts]

    return _S.BipartiteGraph(data, partition)


def pullOutFaceMapFromBipartiteGraph(G):
    """
    (G) something returned from getBipartiteGraph
    returns the faceMap for the transversal matroid
    """
    E = [m[0] for m in G.vertices() if len(m) == 1]
    faces = {}
    nbr = 0
    for f in [m for m in G.vertices() if len(m) == 2]:
        faces[f] = nbr
        nbr += 1
    faceMap = {}
    for e in E:
        faceMap[e] = [faces[x] for y in G.edges((e,)) for x in y if not x is None and len(x) == 2]
    return faceMap


class TransversalMatroid(Matroid):
    def __init__(self, faceMap):
        """ creates a transversal matroid that uses bipartite graphs as a backend
        faceMap __ a map that maps each matroid element to a set of (abstract) vertices
                   defining the face it is on (in general position)

        ex: n = TransversalMatroid({'d': frozenset({1}), 'a': frozenset({0}), 'c': frozenset({0})})
        """
        faceMap = dict(faceMap)

        E = set(faceMap)

        fm = {}
        Vs = set()

        from collections.abc import Iterable
        for e in E:
            f = faceMap[e]
            if isinstance(f, Iterable):
                fm[e] = frozenset(f)
            else:
                fm[e] = frozenset({f})
            Vs = Vs.union(fm[e])

        V = tuple(sorted(Vs))
        for e in E:
            fm[e] = frozenset(map(V.index, fm[e]))

        partition = [[('v', i) for i in range(len(V))], [('e', e) for e in E]]
        data = dict(map(lambda y: (('e', y[0]), [('v', z) for z in y[1]]), fm.items()))

        G = sage.graphs.bipartite_graph.BipartiteGraph(data, partition)
        self.GE = partition[1]
        self.GV = partition[0]
        self.G = G
        self.facemap = fm
        self.E = frozenset(E)
        self.V = V

    def groundset(self):
        return self.E

    def _rank(self, X):
        Xs = set([('e', x) for x in X])
        GX = self.G.subgraph(Xs.union(self.GV))
        return len(GX.matching())

    def _repr_(self):
        return f"TransversalMatroid({self.facemap})"


def reconstructTransversalMatroidAlpha(M):
    """
    (M) transversal matroid

    returns a transversal matroid that is identical

    """
    G = getBipartiteGraph(M)
    faceMap = pullOutFaceMapFromBipartiteGraph(G)
    # correct the loops
    for m in M.groundset():
        if not m in faceMap:
            faceMap[m] = frozenset()
    return TransversalMatroid(faceMap)


### BETA STATISTICS CODE
#
#


def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


def cyclicFlats(M):
    """ (M)
        returns an iterator of all cyclic flats of M, i.e. flats where dropping any elements keeps the rank,
        i.e. M restricted to the flat has no coloops.
    """

    def isCyclic(F, M=M):
        r0 = M.rank(F)
        F = frozenset(F)
        for e in F:
            if M.rank(F.difference([e])) < r0:
                return False
        return True

    return filter(isCyclic, allFlats(M))


def cyclicFlats2(M):
    by_rank = {}
    by_rank[0] = frozenset([M.closure(frozenset())])
    all_found = by_rank[0]
    for c in M.circuits():
        cl_c = M.closure(c)
        rk = len(c) - 1
        if frozenset(cl_c) in all_found:
            continue
        by_rank[rk] = by_rank.get(rk, frozenset()).union([frozenset(cl_c)])
        now_found = frozenset([frozenset(cl_c)])
        for x in all_found:
            cl_x = M.closure(x.union(cl_c))
            rk = M.rank(cl_x)
            by_rank[rk] = by_rank.get(rk, frozenset()).union([frozenset(cl_x)])
            now_found = now_found.union([frozenset(cl_x)])
        all_found = all_found.union(now_found)

    cyclic_flats = []
    for r in range(M.rank() + 1):
        for c in by_rank.get(r, frozenset()):
            cyclic_flats.append(c)
    return cyclic_flats


def getBetaSystem(M):
    """ (M)
        calculate the beta-transversal system of M
    """
    rM = M.rank()
    beta = {}
    for F in reversed(list(cyclicFlats(M))):
        F = frozenset(F)
        rF = M.rank(F)
        sumB = 0
        for Z in beta:
            if F.issubset(Z):
                sumB += beta[Z]
        betaF = rM - rF - sumB
        if betaF > 0:
            beta[F] = betaF
    return beta


def getBetaSystem2(M):
    """ (M)
        calculate the beta-transversal system of M
    """
    rM = M.rank()
    beta = {}
    for F in reversed(list(cyclicFlats2(M))):
        F = frozenset(F)
        rF = M.rank(F)
        sumB = 0
        for Z in beta:
            if F.issubset(Z):
                sumB += beta[Z]
        betaF = rM - rF - sumB
        if betaF > 0:
            beta[F] = betaF
    return beta


def reconstructTransversalMatroidBeta(M):
    """
    (M) transversal matroid

    returns a transversal matroid that is identical

    """
    beta = getBetaSystem(M)
    E = frozenset(M.groundset())
    transversal_family = []
    for A in beta:
        Ac = E.difference(A)
        for i in range(beta[A]):
            transversal_family.append(Ac)
    faceMap = {}

    for e in E:
        faceMap[e] = frozenset((i for i in range(len(transversal_family)) if e in transversal_family[i]))

    return TransversalMatroid(faceMap)


def reconstructTransversalMatroidBeta2(M):
    """
    (M) transversal matroid

    returns a transversal matroid that is identical

    """
    beta = getBetaSystem2(M)
    E = frozenset(M.groundset())
    transversal_family = []
    for A in beta:
        Ac = E.difference(A)
        for i in range(beta[A]):
            transversal_family.append(Ac)
    faceMap = {}

    for e in E:
        faceMap[e] = frozenset((i for i in range(len(transversal_family)) if e in transversal_family[i]))

    return TransversalMatroid(faceMap)


#### PROBABILISTIC VERSION
#
#
#

def basisProbs(M, noiseFactor=0.0):
    probs = {}
    for c in combinations(M.groundset(), M.rank()):
        if M.is_independent(c):
            probs[frozenset(c)] = 1.0 - _S.random() * noiseFactor
        else:
            probs[frozenset(c)] = _S.random() * noiseFactor
    return probs


#
# Strategies for combining values
#

def harmonicAvg(a):
    p = 1.0
    for x in a:
        p *= x
    return p ** (1.0 / len(a))


def avg(a):
    return sum(a) / len(a)


def product(a):
    p = 1.0
    for x in a:
        p *= x
    return p


def complementaryProduct(a):
    p = 1.0
    for x in a:
        p *= 1.0 - x
    return 1.0 - p




def circuitDiscovery(V, basisProbs, r, config):
    circuits = {}
    cutOffBasis = config['basis-cutoff']
    basisCombination = config['circuit-combinator']
    circuitElementsCombination = config['circuit-combinator-2']
    circuitCombination = config['circuit-combinator-3']
    coloopScore = config['circuit-coloop-score']
    for X in combinations(V, r):
        X = frozenset(X)
        p = basisProbs[X]
        if p < cutOffBasis:
            continue
        for v in V:
            if v in X:
                continue
            circuit_elements = {}
            for x in X:
                X1 = X.difference([x]).union([v])
                p1 = basisProbs[X1]
                if p1 < cutOffBasis:
                    continue
                circuit_elements[x] = basisCombination([p, p1])
            cx = frozenset([v]).union(circuit_elements)
            if len(circuit_elements) == 0:
                score = coloopScore
            else:
                score = circuitElementsCombination(circuit_elements.values())
            vec = circuits.get(cx, [])
            vec.append(score)
            circuits[cx] = vec

    for key in circuits:
        circuits[key] = circuitCombination(circuits[key])

    return circuits


def circuitClosures(circuits, config):
    sorted_circuits = [x for x in circuits]
    sorted_circuits.sort(key=lambda c: -circuits[c])
    combinator = config['closure-combinator']
    cutoff = config['closure-cutoff']
    final_combinator = config['closure-combinator-2']
    circuit_closures = {}
    cc_ranks = {}
    for c0 in sorted_circuits:
        closure = frozenset(c0)
        rank = len(c0) - 1
        scores = [circuits[c0]]
        added_stuff = True
        while added_stuff:  # we might be missing a circuit after detection, but this may be fixed by going via two or more other circuits
            added_stuff = False
            for c1 in sorted_circuits:
                if c1.issubset(closure):
                    continue
                if len(closure.intersection(c1)) >= len(c1) - 1:
                    if combinator(scores + [circuits[c1]]) >= cutoff:
                        scores.append(circuits[c1])
                        closure = closure.union(c1)
                        added_stuff = True

        cl_scores = circuit_closures.get(closure, []) + [combinator(scores)]
        circuit_closures[closure] = cl_scores
        cc_ranks[closure] = rank
    for x in circuit_closures:
        circuit_closures[x] = final_combinator(circuit_closures[x])

    return circuit_closures, cc_ranks


def probCyclicFlats(ccs, cc_ranks, circuits, config):
    sorted_circuits = [x for x in circuits]
    sorted_circuits.sort(key=lambda c: -circuits[c])
    cl_combinator = config['closure-combinator']
    cl_cutoff = config['closure-cutoff']

    cyclic_flats = {}
    add_emptyset = True
    for x in cc_ranks:
        if cc_ranks[x] == 0:
            add_emptyset = False
            break
    if add_emptyset:
        cyclic_flats[frozenset()] = config['empty-cyclic-flat-score']

    found_flats = frozenset(cyclic_flats.keys())
    combinator = config['cyclic-flats-combinator']

    for c in ccs.keys():
        cyclic_flats[c] = ccs[c]
        additional_flats = [c]
        for f in found_flats:
            cvf = c.union(f)
            added_stuff = True
            scores = [circuits[x] for x in circuits if x.issubset(cvf)]
            while added_stuff:
                added_stuff = False
                for c1 in sorted_circuits:
                    if c1.issubset(cvf):
                        continue
                    if len(cvf.intersection(c1)) >= len(c1) - 1:
                        if cl_combinator(scores + [circuits[c1]]) >= cl_cutoff:
                            scores.append(circuits[c1])
                            cvf = cvf.union(c1)
                            added_stuff = True

            if not cvf in cyclic_flats:
                scores = [ccs[x] for x in ccs if x.issubset(cvf)]
                cyclic_flats[cvf] = combinator(scores)
                additional_flats.append(cvf)
        found_flats = found_flats.union(additional_flats)
    return cyclic_flats


def probFaceMap(basisProbs, V, zcal, r, config):
    cutoff = config['basis-cutoff']
    good_bases = [x for x in basisProbs if basisProbs[x] >= cutoff]
    zranks = {}
    for z in zcal:
        zranks[z] = max([len(z.intersection(b)) for b in good_bases])
    sorted_z = sorted(zranks, key=lambda t: -zranks[t])
    beta = {}
    beta_scoring = {}
    for F in sorted_z:
        F = frozenset(F)
        rF = zranks[F]
        sumB = 0
        for Z in beta:
            if F.issubset(Z):
                sumB += beta[Z]
        betaF = r - rF - sumB
        if betaF > 0:
            beta[F] = betaF
            beta_scoring[F] = zcal[F]
    transversal_system = []
    sorted_beta = sorted(beta, key=lambda t: -beta_scoring[t])
    for A in sorted_beta:
        m = min(r - len(transversal_system), beta[A])
        Ac = V.difference(A)
        for i in range(m):
            transversal_system.append(Ac)

    faceMap = {}
    for e in V:
        faceMap[e] = frozenset((i for i in range(len(transversal_system)) if e in transversal_system[i]))

    return faceMap


def reconstructTransversalMatroidProbs(V, basisProbs, r, M=None, config=None):
    hasM = False if M is None else True
    Cs = circuitDiscovery(V, basisProbs, r, config)
    if hasM:
        C0 = frozenset(M.circuits())
        C1 = frozenset(Cs.keys())
        print("Circuits reconstructed?", C0 == C1)
        for c in C0.difference(C1):
            print("  missing: ", c)
        for c in C1.difference(C0):
            print("  extra: ", c)

    CCs, CCranks = circuitClosures(Cs, config)
    if hasM:
        CC0 = frozenset([M.closure(c) for c in M.circuits()])
        CC1 = frozenset(CCs.keys())
        print("Circuit closures reconstructed?", CC0 == CC1)
        for c in CC0.difference(CC1):
            print("  missing: ", c)
        for c in CC1.difference(CC0):
            print("  extra: ", c)

    Zcal = probCyclicFlats(CCs, CCranks, Cs, config)
    if hasM:
        Z0 = frozenset(cyclicFlats2(M))
        Z1 = frozenset(Zcal.keys())
        print("Cyclic flats reconstructed?", Z0 == Z1)
        for c in Z0.difference(Z1):
            print("  missing: ", c)
        for c in Z1.difference(Z0):
            print("  extra: ", c)

    faceMap = probFaceMap(basisProbs, V, Zcal, r, config)
    return TransversalMatroid(faceMap)


def translate_sage_bipartite_graph_to_mat(BipartiteG, rowlen):
    '''
    e.g., Bipartite graph on 7 vertices
        [(0, "'"), (1, "'"), (3, "'"), (0,), (3,), (1,), (2,)]
        [((0,), (0, "'"), None), ((0, "'"), (3,), None), ((0,), (1, "'"), None), ((1, "'"), (3,), None), ((1,), (1, "'"), None), ((1, "'"), (2,), None), ((0,), (3, "'"), None), ((3,), (3, "'"), None)]
    '''
    vertices, edges = BipartiteG.vertices(), BipartiteG.edges()
    col_node_names = sorted([v[0] for v in vertices if len(v) == 2])
    row_node_names = sorted([v[0] for v in vertices if len(v) == 1])
    col_node_name_to_id = {v: i for i, v in enumerate(col_node_names)}
    assert set(row_node_names) <= set(range(rowlen))
    mat = np.zeros((rowlen, len(col_node_names)), dtype=np.uint8)
    for n1, n2, _ in edges:
        if len(n1) == 1: rown, coln = n1, n2
        else: rown, coln = n2, n1
        mat[rown[0], col_node_name_to_id[coln[0]]] = 1
    return mat


def compute_circuits(V, indep_sets):
    r = max(len(s) for s in indep_sets)  # rank
    circuits = set()
    for C_tuple in powerset(V):
        if len(C_tuple) > r + 1: break
        C = frozenset(C_tuple)
        if C not in indep_sets:
            if all(C-{x} in indep_sets for x in C):
                circuits.add(C)
    return circuits