# visualization_task
Single-file algorithm for graph visualization with an ability to tweak the way certain parameter influences the output.

The algorithm is based on [this](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0098679) article.

The usage is simple. If you want to use it with ```networkx``` structures you can simply call it like that.

```python3
import networkx as nx
import matplotlib.pyplot as plt
from visualizer.visualizer import ForceAtlas2

G = nx.karate_club_graph()
forceatlas2 = ForceAtlas2(
                          # Outbound attraction distribution
                          distribute_outbound_attraction=False,
                          # Edge weight influence
                          edge_weight_influence=1.0,
                          # Jitter tolerance
                          jitter_tolerance=1.0,
                          # Scaling ratio between nodes
                          scaling_ratio=1.0,
                          # Strong gravity mode (more compact image)
                          strong_gravity_mode=False,
                          # Gravity pull
                          gravity=1.0)
positions = forceatlas2.forceatlas2_layout(G, pos=None, iterations=2500)
nx.draw_networkx(G, positions, cmap=plt.get_cmap('jet'), node_size=40, with_labels=False)
plt.show()
```

You can also do the same with adjacecy matrix, presented as 2-dimesional ```numpy.array``` with same length and witdth, as currenly the implementation supports only such arrays.
