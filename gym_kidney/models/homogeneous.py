from model import Model
import networkx as nx

#
# Homogeneous Erdős–Rényi model
#

class HomogeneousModel(Model):

	def __init__(self, m, k, p, p_a, len):
		self.m = m
		self.k = k
		self.p = p
		self.p_a = p_a
		self.len = len

		self.params = {
			"m": m,
			"k": k,
			"p": p,
			"p_a": p_a,
			"len": len
		}

		self.stats = {
			"arrived": 0,
			"departed": 0
		}

	def arrive(self, G, rng):
		n1 = G.order()
		n2 = rng.poisson(self.m / self.k)
		new = range(n1, n1 + n2)

		for u in new:
			ndd_u = rng.rand() < self.p_a
			G.add_node(u, ndd = ndd_u)

			for v in G.nodes():
				ndd_v = G.node[v]["ndd"]
				if u == v: continue
				if rng.rand() < p and not ndd_v:
					G.add_edge(u, v)
				if rng.rand() < p and not ndd_u:
					G.add_edge(v, u)

		self.stats["arrived"] += n2
		return G

	def depart(self, G, rng):
		n1 = G.order()
		n2 = rng.binomial(n1, 1.0 / self.k)
		old = rng.choice(G.nodes(), n2, replace = False).tolist()

		G.remove_nodes_from(old)
		self.stats["departed"] += n2
		return nx.convert_node_labels_to_integers(G)

	def done(self, tick):
		return tick >= self.len
