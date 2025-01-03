from pyvis.network import Network
from core import GraphLike
import networkx as nx



def plot (graph: "GraphLike", 
          node_shape: str='circle', 
          node_color: str='#97c2fc',  
          highlight_color: str='#00ff55', 
          highlight_set: set[int]|None=None,
          **options):

    '''
    [Parameter]

    node_shape :    The shape defines what the node looks like. There are
                    two types of nodes. One type has the label inside of
                    it and the other type has the label underneath it. The
                    types with the label inside of it are: ellipse, circle,
                    database, box, text. The ones with the label outside of
                    it are: image, circularImage, diamond, dot, star,
                    triangle, triangleDown, square and icon.
    '''

    g = Network()

    # Add nodes
    for id in graph.node_space:
        node = graph.lookup_node[id]
        g.add_node(id, 
                   label=id, 
                   shape=node_shape, 
                   color=highlight_color if highlight_set and id in highlight_set else node_color, 
                   size=node.size * 10,
                   **options)

    # Add all edges
    for edge_tuple in graph.edge_space:
        g.add_edge(*edge_tuple, arrowStrikethrough=graph.directed)
    
    try:
        g.prep_notebook()
        g.show(f'graphyte_graph_{graph.name.lower()}_plot.html', notebook=True)
    except:
        g.show(f'graphyte_graph_{graph.name.lower()}_plot.html', notebook=False)

# nx_graph = nx.cycle_graph(10)
# nx_graph.nodes[1]['title'] = 'Number 1'
# nx_graph.nodes[1]['group'] = 1
# nx_graph.nodes[3]['title'] = 'I belong to a different group!'
# nx_graph.nodes[3]['group'] = 10
# nx_graph.add_node(20, size=20, title='couple', group=2)
# nx_graph.add_node(21, size=15, title='couple', group=2)
# nx_graph.add_edge(20, 21, weight=5)
# nx_graph.add_node(25, size=25, label='lonely', title='lonely node', group=3)
# nt = Network('500px', '500px')
# # populates the nodes and edges data structures
# nt.from_nx(nx_graph)
# nt.show('nx.html', notebook=False)