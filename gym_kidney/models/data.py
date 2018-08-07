from gym_kidney import models

import numpy as np
import networkx as nx
import csv

BLOODS = ["A", "B", "AB", "O", "-"]

#
# DataModel evolves the graph by simulating on real exchange
# data.
# - m : Nat, expected vertices per period
# - k : Nat, ticks per period
# - data : String, path to CSV containing data
# - details : String, path to CSV containing vertex attributes
# - len : Nat, ticks per episode
#
class DataModel(models.Model):

    def __init__(self, m, k, data, details, len):
        self.m = m
        self.k = k
        self.data = data
        self.details = details
        self.len = len
        self._load_data()

        self.params = {
                "m": m,
                "k": k,
                "data": data,
                "details": details,
                "len": len
        }

        self.stats = {
                "arrived": 0,
                "departed": 0
        }

        for blood in BLOODS:
            self.stats["%s_patient_arrived" % blood] = 0
            self.stats["%s_donor_arrived" % blood] = 0
            self.stats["%s_patient_departed" % blood] = 0
            self.stats["%s_donor_departed" % blood] = 0

    def arrive(self, G, rng):
        R = self._ref
        n1 = G.order()
        n2 = rng.poisson(self.m / self.k)
        new = range(n1, n1 + n2)

        # label map
        r_to_g = self._inv(nx.get_node_attributes(G, "r_id"))

        for u in new:
            # add vertex
            r_id = rng.randint(0, R.order())
            attr_u = R.node[r_id]
            attr_u["r_id"] = r_id
            G.add_node(u, attr_u)
            self.stats["%s_patient_arrived" % attr_u["bp"]] += 1
            self.stats["%s_donor_arrived" % attr_u["bd"]] += 1

            # add to label map
            if r_id in r_to_g:
                r_to_g[r_id] += [u]
            else:
                r_to_g[r_id] = [u]

            # edges
            for vs in list(map(r_to_g.get, R.successors(r_id))):
                if vs == None: continue
                for v in vs:
                    G.add_edge(u, v)

            for vs in list(map(r_to_g.get, R.predecessors(r_id))):
                if vs == None: continue
                for v in vs:
                    G.add_edge(v, u)

        self.stats["arrived"] += n2
        return G

    def depart(self, G, rng):
        n1 = G.order()
        n2 = rng.binomial(n1, 1.0 / self.k)

        if G.order() <= n2:
            old = G.nodes()
        else:
            old = rng.choice(G.nodes(), n2, replace = False)
            old = old.tolist()

        for v in old:
            self.stats["%s_patient_departed" % G.node[v]["bp"]] += 1
            self.stats["%s_donor_departed" % G.node[v]["bd"]] += 1

        G.remove_nodes_from(old)
        self.stats["departed"] += n2
        return nx.convert_node_labels_to_integers(G)

    def done(self, tick):
        return tick >= self.len

    def _load_data(self):
        # adjacency matrix
        adj = np.loadtxt(self.data, delimiter = ",")
        self._ref = nx.DiGraph()
        self._ref = nx.from_numpy_matrix(adj, create_using = self._ref)

        # vertex attributes
        with open(self.details, mode = "r") as handle:
            read = csv.reader(handle)
            for row in read:
                u = self._ref.node[int(row[0])]
                u["ndd"] = row[1] == "1"
                u["bp"] = row[2]
                u["bd"] = row[3]

    def _inv(self, d):
        inv_map = {}
        for k, v in d.items():
            inv_map[v] = inv_map.get(v, [])
            inv_map[v].append(k)
        return inv_map
