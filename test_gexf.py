import networkx as nx
import matplotlib.pyplot as plt

G = nx.read_gexf('GEXF/lol.gexf')
nx.draw_networkx(G, with_labels = True)
plt.savefig('GEXF/lol.png')