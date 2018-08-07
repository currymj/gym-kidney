import networkx as nx
from gym_kidney import _solver
from gym.utils import seeding


def _default_model():
    M = 128
    K = 1024
    K = 580
    P = 0.05
    P_A = 0.05
    LEN = 3*K
    MODEL = models.HomogeneousModel(M, K, P, P_A, LEN)
    return MODEL

default_rng, seed = seeding.np_random(1)

def gen_random_graph(model, rng, n_steps=500):
    G = nx.DiGraph()
    for i in range(n_steps):
        G = model.arrive(G,rng)
    return G

def relabel(G):
    n_dd, n_ndd = 0, 0
    d_dd, d_ndd = {}, {}

    for u in G.nodes():
        if G.node[u]["ndd"]:
            d_ndd[u] = n_ndd
            n_ndd += 1
        else:
            d_dd[u] = n_dd
            n_dd += 1

    return n_dd, n_ndd, d_dd, d_ndd

def nx_to_ks(G):
    n_dd, n_ndd, d_dd, d_ndd = relabel(G)

    dd = _solver.Digraph(n_dd)
    for u, v, d in G.edges(data = True):
        if not G.node[u]["ndd"]:
            dd.add_edge(
                d["weight"] if ("weight" in d) else 1.0,
                dd.vs[d_dd[u]],
                dd.vs[d_dd[v]])

    ndds = [_solver.kidney_ndds.Ndd() for _ in range(n_ndd)]
    for u, v, d in G.edges(data = True):
        if G.node[u]["ndd"]:
            edge = _solver.kidney_ndds.NddEdge(
                dd.vs[d_dd[v]],
                d["weight"] if ("weight" in d) else 1.0)
            ndds[d_ndd[u]].add_edge(edge)

    return dd, ndds

def solve_graph(G, cycle_cap=3, chain_cap=3):
    dd, ndd = nx_to_ks(G)
    cfg = _solver.kidney_ip.OptConfig(
            dd,
            ndd,
            cycle_cap,
            chain_cap)
    soln  = _solver.solve_kep(cfg, "picef")
    rew_cycles = sum(map(lambda x: len(x), soln.cycles))
    rew_chains = sum(map(lambda x: len(x.vtx_indices), soln.chains))
    reward = rew_cycles + rew_chains
    
    return reward

def make_graph_score_pair(rng=default_rng):
    gr = gen_random_graph(MODEL, rng)
    score = solve_graph(gr)
    return (gr, score)
