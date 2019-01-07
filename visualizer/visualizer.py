import random
import numpy
import scipy
from math import sqrt



class Node:
    def __init__(self):
        self.mass = 0.0
        self.old_dx = 0.0
        self.old_dy = 0.0
        self.dx = 0.0
        self.dy = 0.0
        self.x = 0.0
        self.y = 0.0


class Edge:
    def __init__(self):
        self.node1 = -1
        self.node2 = -1
        self.weight = 0.0


# Repulsion function. 'n1' and 'n2' are nodes.
def lin_repulsion(n1, n2, coefficient=0):
    xDist = n1.x - n2.x
    yDist = n1.y - n2.y
    distance2 = xDist * xDist + yDist * yDist  # Distance squared

    if distance2 > 0:
        factor = coefficient * n1.mass * n2.mass / distance2
        n1.dx += xDist * factor
        n1.dy += yDist * factor
        n2.dx -= xDist * factor
        n2.dy -= yDist * factor


# Gravity repulsion function.
def lin_gravity(n, g):
    xDist = n.x
    yDist = n.y
    distance = sqrt(xDist * xDist + yDist * yDist)

    if distance > 0:
        factor = n.mass * g / distance
        n.dx -= xDist * factor
        n.dy -= yDist * factor


# Strong gravity force function. `n` should be a node, and `g` should be a constant by which to increase the force.
def strong_gravity(n, g, coefficient=0):
    xDist = n.x
    yDist = n.y

    if xDist != 0 and yDist != 0:
        factor = coefficient * n.mass * g
        n.dx -= xDist * factor
        n.dy -= yDist * factor


# Attraction function.  `n1` and `n2` should be nodes.
# Will directly ajust positions
def lin_attraction(n1, n2, e, distributed_attraction, coefficient=0):
    xDist = n1.x - n2.x
    yDist = n1.y - n2.y
    if not distributed_attraction:
        factor = -coefficient * e
    else:
        factor = -coefficient * e / n1.mass
    n1.dx += xDist * factor
    n1.dy += yDist * factor
    n2.dx -= xDist * factor
    n2.dy -= yDist * factor


def apply_repulsion(nodes, coefficient):
    i = 0
    for n1 in nodes:
        j = i
        for n2 in nodes:
            if j == 0:
                break
            lin_repulsion(n1, n2, coefficient)
            j -= 1
        i += 1


def apply_gravity(nodes, gravity, use_strong_gravity=False):
    if not use_strong_gravity:
        for n in nodes:
            lin_gravity(n, gravity)
    else:
        for n in nodes:
            strong_gravity(n, gravity)


def apply_attraction(nodes, edges, distributedAttraction, coefficient, edgeWeightInfluence):
    # Optimization, since usually edgeWeightInfluence is 0 or 1, and pow is slow
    if edgeWeightInfluence == 0:
        for edge in edges:
            lin_attraction(nodes[edge.node1], nodes[edge.node2], 1, distributedAttraction, coefficient)
    elif edgeWeightInfluence == 1:
        for edge in edges:
            lin_attraction(nodes[edge.node1], nodes[edge.node2], edge.weight, distributedAttraction, coefficient)
    else:
        for edge in edges:
            lin_attraction(nodes[edge.node1], nodes[edge.node2], pow(edge.weight, edgeWeightInfluence),
                           distributedAttraction, coefficient)


# Adjust speed and apply forces step
def adjust_speed_and_apply_forces(nodes, speed, speed_efficiency, jitter_tolerance):
    # Auto adjust speed.
    swing_amount = 0.0  # How much irregular movement
    total_effective_traction = 0.0  # How much useful movement
    for n in nodes:
        swinging = sqrt((n.old_dx - n.dx) * (n.old_dx - n.dx) + (n.old_dy - n.dy) * (n.old_dy - n.dy))
        swing_amount += n.mass * swinging
        total_effective_traction += .5 * n.mass * sqrt(
            (n.old_dx + n.dx) * (n.old_dx + n.dx) + (n.old_dy + n.dy) * (n.old_dy + n.dy))

    # Optimize jitter tolerance.
    estimated_optimal_jit_tolerance = .05 * sqrt(len(nodes))
    min_jit_tolerance = sqrt(estimated_optimal_jit_tolerance)
    max_jit_tolerance = 10
    jt = jitter_tolerance * max(min_jit_tolerance,
                                min(max_jit_tolerance, estimated_optimal_jit_tolerance * total_effective_traction / (
                                   len(nodes) * len(nodes))))

    min_speed_efficiency = 0.05

    # Protect against erratic behavior
    if swing_amount / total_effective_traction > 2.0:
        if speed_efficiency > min_speed_efficiency:
            speed_efficiency *= .5
        jt = max(jt, jitter_tolerance)

    target_speed = jt * speed_efficiency * total_effective_traction / swing_amount

    if swing_amount > jt * total_effective_traction:
        if speed_efficiency > min_speed_efficiency:
            speed_efficiency *= .7
    elif speed < 1000:
        speed_efficiency *= 1.3

    # Speed shoudn't rise too much too quickly
    max_rise = .5
    speed = speed + min(target_speed - speed, max_rise * speed)

    # Apply forces.
    for n in nodes:
        swinging = n.mass * sqrt((n.old_dx - n.dx) * (n.old_dx - n.dx) + (n.old_dy - n.dy) * (n.old_dy - n.dy))
        factor = speed / (1.0 + sqrt(speed * swinging))
        n.x = n.x + (n.dx * factor)
        n.y = n.y + (n.dy * factor)

    values = {}
    values['speed'] = speed
    values['speed_efficiency'] = speed_efficiency

    return values


class ForceAtlas2:
    def __init__(self,
                 # Размазывание по краям
                 distribute_outbound_attraction=False,
                 # Влияние веса ребра
                 edge_weight_influence=1.0,
                 # Степень свободы
                 jitter_tolerance=1.0,
                 # Степень оттлакивания
                 scaling_ratio=2.0,
                 # Режим сильной гравитации
                 strong_gravity_mode=False,
                 # Гравитация (к центру)
                 gravity=1.0):
        self.distribute_outbound_attraction = distribute_outbound_attraction
        self.edge_weight_influence = edge_weight_influence
        self.jitter_tolerance = jitter_tolerance
        self.scaling_ratio = scaling_ratio
        self.strong_gravity_mode = strong_gravity_mode
        self.gravity = gravity

    def init(self,
             G,  # 2D numpy ndarray or scipy sparse matrix format
             pos=None  # Array of initial positions
             ):
        is_sparse = False
        if isinstance(G, numpy.ndarray):
            # Check correctness
            assert G.shape == (G.shape[0], G.shape[0]), "G is not 2D square"
            assert numpy.all(G.T == G), "G is not symmetric"
            assert isinstance(pos, numpy.ndarray) or (pos is None), "Invalid node positions"
        elif scipy.sparse.issparse(G):
            # Check correctness for scipy
            assert G.shape == (G.shape[0], G.shape[0]), "G is not 2D square"
            assert isinstance(pos, numpy.ndarray) or (pos is None), "Invalid node positions"
            G = G.tolil()
            is_sparse = True
        else:
            assert False, "G is of unsupported type"

        # Put nodes into a data structure we use
        nodes = []
        for i in range(0, G.shape[0]):
            n = Node()
            if is_sparse:
                n.mass = 1 + len(G.rows[i])
            else:
                n.mass = 1 + numpy.count_nonzero(G[i])
            n.old_dx = 0
            n.old_dy = 0
            n.dx = 0
            n.dy = 0
            if pos is None:
                n.x = random.random()
                n.y = random.random()
            else:
                n.x = pos[i][0]
                n.y = pos[i][1]
            nodes.append(n)

        # Put edges into a data structure we use
        edges = []
        es = numpy.asarray(G.nonzero()).T
        for e in es:  # Iterate through edges
            # No duplicates
            if e[1] <= e[0]:
                continue
            edge = Edge()
            edge.node1 = e[0]  # The index of the first node
            edge.node2 = e[1]  # The index of the second node
            edge.weight = G[tuple(e)]
            edges.append(edge)

        return nodes, edges

    # This function returns a NetworkX layout.
    def forceatlas2_layout(self, G, pos=None, iterations=100):
        import networkx
        assert isinstance(G, networkx.classes.graph.Graph), "Not a networkx graph"
        assert isinstance(pos, dict) or (pos is None), "pos must be specified as a dictionary, as in networkx"
        M = networkx.to_scipy_sparse_matrix(G, dtype='f', format='lil')
        if pos is None:
            l = self.forceatlas2_not_networkx(M, pos=None, iterations=iterations)
        else:
            poslist = numpy.asarray([pos[i] for i in G.nodes()])
            l = self.forceatlas2_not_networkx(M, pos=poslist, iterations=iterations)
        return dict(zip(G.nodes(), l))

    # This should be used if not expecting networkx
    def forceatlas2_not_networkx(self,
                                 G,  # 2D numpy ndarray or scipy sparse matrix format
                    pos=None,  # Array of initial positions
                    iterations=100  # Number of times to iterate the main loop
                                 ):

        # speed and speed_efficiency stand for a scaling factor of dx and dy
        speed = 1.0
        speed_efficiency = 1.0
        nodes, edges = self.init(G, pos)
        outbound_att_compensation = 1.0
        if self.distribute_outbound_attraction:
            outbound_att_compensation = numpy.mean([n.mass for n in nodes])

        niters = range(iterations)
        for i in niters:
            for n in nodes:
                n.old_dx = n.dx
                n.old_dy = n.dy
                n.dx = 0
                n.dy = 0

            apply_repulsion(nodes, self.scaling_ratio)

            apply_gravity(nodes, self.gravity, use_strong_gravity=self.strong_gravity_mode)

            apply_attraction(nodes, edges, self.distribute_outbound_attraction, outbound_att_compensation,
                             self.edge_weight_influence)

            values = adjust_speed_and_apply_forces(nodes, speed, speed_efficiency, self.jitter_tolerance)
            speed = values['speed']
            speed_efficiency = values['speed_efficiency']

        return [(n.x, n.y) for n in nodes]

