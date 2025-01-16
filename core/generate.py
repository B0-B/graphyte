from core.graph import *
from numpy.random import uniform, poisson, choice

# ---- Generation ----
def random_graph (nodes: int, edge_density: float=0.2, directed: bool|None=None) -> PathGraph:

    # if directed is not defined will be sampled randomly
    directed = directed if directed != None else choice(2)
    
    G = PathGraph('Random_Graph_01', directed)

    # add nodes
    for _ in range(nodes):
        G.add_node()
    
    # add random edges
    for id in range(1, nodes+1):
        for nn_id in G.node_space:
            if id == nn_id or uniform() >= edge_density:
                continue
            if not G.is_linked(id, nn_id):
                G.add_edge(id, nn_id)
    
    return G

def generate_complete_graph ():
    pass

def generate_maze (solution_length: int) -> PathGraph:

    p_branching = 0.25

    G = PathGraph('Maze')  

    scale = int(solution_length)
    
    # create the main path
    for _ in range(scale):
        G.add_node()
    solution = list(G.node_space)
    for i in range(len(solution)-1):
        G.add_edge(solution[i], solution[i+1])
    
    # create random dead-ends
    for id in range(scale):

        # pick a random node in current graph
        id = np.random.choice(list(G.node_space))

        # can have at most 3 branches
        for _ in range(3):
            # sample random branchings
            if uniform() < p_branching:
                # create a path branching away
                n = int(poisson((scale/3,)))
                seq = [G.add_node() for _ in range(n)]
                seq.insert(0, G.node(id))
                G.connect_sequence(*seq)
    
    return G

def generate_network (nodes: int, mean_degree: float) -> Network:

    # Generate random connections using a poisson distribution.
    np.random.poisson()
    pass