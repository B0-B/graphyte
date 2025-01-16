import numpy as np
from typing import Callable, Iterable
from core.utils import generate_random_hash


# ---- Global Graph Functions ----
def extract_node_id (node: "Node|int") -> int:

    '''
    Extracts the id from a node identifier which is either a Node object or int.

    [Return]

    node id as integer.
    '''

    # determine the node ids
    if inherits_from(node, Node):
        return node.id
    elif type(node) is int:
        return node
    else:
        raise TypeError(f'The provided node_a must be of type Node or int object, not {type(node)}.')

def inherits_from (object_or_class: any, class_type: type) -> bool:

    '''
    Checks if a provided object or class (type-like) inherits from another class or type.
    e.g. PathGraph, and the instance G = PathGraph(...) both inherit from GraphLike and BaseGraph.

    [Parameter]

    object_or_class :       the class or type to probe

    class_type :            provided super class of the potential probe class 

    [Return]

    Boolean condition 
    '''

    if type(object_or_class) == type:
        return issubclass(object_or_class, class_type)
    else:
        return isinstance(object_or_class, class_type)


# ---- Graphs ----
class Node:

    def __init__(self, parent_graph: "GraphLike", id: int|None=None, size: float=1.0, **properties):

        '''
        properties :        Optional non-constrained parameters.
                            Useful for distinguishing or classifying nodes.
                            Ex.: Node(.., color='red')
        '''
        
        self.graph = parent_graph # hold corresponding graph pointer
        self.marked = False   # set marker
        self.visits: int = 0 # count visits
        self.size: float = size # optional size parameter - useful for visualization

        # store additional properties
        if properties:
            self.__dict__.update(properties)

        # Node parameter
        self.assign_new_id(id) # -> assigns self.id
        
        # holds the pointers to adjacent "connected" nodes
        self.adjacent_nodes: set[int] = set()

        # Register node in graph nodes space and lookup table.
        self.graph.node_space.add(self.id)
        self.graph.lookup_node[self.id] = self
    
    def assign_new_id (self, id: int|None=None) -> int:

        '''
        Will process and assign a new id for this node.
        Latest provided id (not None) id will be used as floor with ascending count,
        except if a lower id is provided.
        '''

        # Assign new identifier.
        if type(id) is int:

            if id in self.graph.node_space:
                raise IndexError(f'The node with id "{id}" exists already in graph "{self.graph.name}".')
            self.id = id
            # shift the Node id counter to new floor but 
            # only if the freely assigned id is the greatest
            if id > self.graph.max_node_id:
                self.graph.max_node_id = id

        else:
            
            # just increment the node counter
            self.graph.max_node_id += 1

            # auto sample new id and assign
            self.id = self.graph.max_node_id

    def degree (self) -> int:

        '''
        Returns the degree of the node, i.e. the number of adjacent other nodes.
        '''

        return len(self.adjacent_nodes)

    def delete (self) -> None:

        # First disconnect from all other nodes by removing all corr. edges.
        # Second remove all pointers from adjacent nodes in their resp. adjacent nodes.
        iter_obj = self.graph.lookup_edge.copy()

        for tpl, edge in iter_obj.items():

            if self.id in tpl:
                
                adj_id = tpl[0] if tpl[0] != self.id else tpl[1]
                adj_node = self.graph.lookup_node[adj_id]

                # Delete the edge.
                edge.delete()

                # Mutually remove self id from the adjacent nodes's adjacent list.
                adj_node.adjacent_nodes.remove(self.id)

        # delete from lookup
        self.graph.lookup_node.pop(self.id)

    def link (self, node: "Node|int", weight: float|None=None) -> "Edge":
        
        '''
        Connects the node to another provided node.
        The connected node pointer will be added to adjacent nodes.

        [Parameter]

        node        a Node object or node identifier

        [Return]

        Returns the newly created Edge object.
        '''

        # determine the node id
        if type(node) is Node:
            node_id = node.id
        elif type(node) is int:
            node_id = node
        else:
            raise TypeError(f'the provided node must be a Node or int object, not {type(node)}.')

        # try to create edge - this will check for validity e.g. when the edge exists already
        _edge = Edge(self.graph, self.id, node_id, weight)

        # Add to adjacent nodes set.
        self.adjacent_nodes.add(node_id)
        # Connections are mutual if non-directed: if node_a holds node_b in adjacent nodes,
        # then node_b holds node_a in it's respective adjacent nodes.
        if not self.graph.directed:
            self.graph.lookup_node[node_id].adjacent_nodes.add(self.id)

        return _edge

    def mark (self) -> None:

        '''
        Toggles the boolean marker property. 
        The value is accessible via Node.marked instance variable.
        '''

        self.marked = not self.marked

    def unlink (self, node: "Node|int") -> None:

        '''
        Will delete all edges (self.id, other_ids) if directed.
        If the node is in a bi- or non-directed graph, it will also delete all
        tuples of the form (other_ids)
        '''

        # determine the node id
        if type(node) is Node:
            node_id = node.id
        elif type(node) is int:
            node_id = node
        else:
            raise TypeError(f'the provided node must be a Node or int object, not {type(node)}.')

        tpl = (self.id, node_id)
        pmt = (node_id, self.id)

        # adj_id = tpl[0] if tpl[0] != self.id else tpl[1]
        adj_node = self.graph.lookup_node[node_id]

        # remove adjacent node from adjacent node list
        self.adjacent_nodes.remove(node_id)

        # Delete the edge.
        if self.graph.directed:
            self.graph.lookup_edge[tpl].delete()
        else:
            if tpl in self.graph.lookup_edge:
                self.graph.lookup_edge[tpl].delete()
            if pmt in self.graph.lookup_edge:
                self.graph.lookup_edge[pmt].delete()

            # Mutually remove self id from the adjacent node's adjacent list.
            adj_node.adjacent_nodes.remove(self.id)

    def unlink_all (self) -> None:

        '''
        Unlinks all adjacent nodes.
        '''

        for id in self.adjacent_nodes.copy():

            self.unlink(id)

class ActiveNode ( Node ):

    '''
    Active nodes can perform tasks.
    '''

    def __init__(self,  
                 parent_graph: "GraphLike", 
                 id: int|None=None,
                 address: str|None=None, 
                 size: float=1.0, 
                 label: str|None=None, 
                 **properties):

        super().__init__(parent_graph, id, size=size, **properties)

        self.forward_task: Callable|None = None
        self.forward_task_kwargs: dict

        self.address: str|None = address       # if defined must be unique
        self.label: str|None = label           # does not have to be unique

    def forward (self, _input: "ForwardObject") -> "ForwardObject":

        '''
        Main forwarding function which performs the forward task.
        If no forward task is defined this function can be considered an identical map.
        '''

        if not self.forward_task:

            # Extract the input to process
            _input = _input.input if type(_input) is ForwardObject else _input

            # Determine the next target
            if not self.adjacent_nodes:
                target = None
            else: 
                # will take random neighbor if no forward task is defined
                target = np.random.choice(list(self.adjacent_nodes))

            return ForwardObject(_input, _input, self.id, target)
        
        return self.forward_task(self, _input, **self.forward_task_kwargs)

    def set_forward_task (self, callback: Callable) -> None:

        '''
        Sets the forward task callback. 
        The callback function should take 2 leading positional arguments:

            - node object
            - input of any type.
        
        and other optional keyword arguments which should be used during inference.  

        [Example]

        >>> def callback (received: ForwardObject, color_kwarg: str='red') -> ForwardObject:
        >>>     # extract information from received ForwardObject
        >>>     graph = received.graph
        >>>     current_id = received.target
        >>>     current_node = graph.node(current_id)
        >>>     data_to_process = received.input
        >>>     # access graph for instance
        >>>     if graph.directed:
        >>>         color_kwarg = 'blue'
        >>>     # extract information from the current node and add it to export data
        >>>     export_data = self.degree() 
        >>>     # route to the first neighbor otherwise terminate
        >>>     if node.adjacent_nodes:
        >>>         target = list(node.adjacent_nodes)[0]
        >>>     else:
        >>>         target = None
        >>>     # return a new ForwardObject
        >>>     return ForwardObject(graph, input=data_to_process, output=export_data, origin=current_id, target=target, color_kwarg=color_kwarg)
        '''

        self.forward_task = callback

class ForwardObject:

    '''
    Standard Object which is forwarded between ActiveNode pairs.

    [Parameter]

    input :         the input which the Node task should process

    output :        the task output which was derived from input

    origin :        the origin node id which took the input and produced the output.

    targets :        the set of target id of the next node to forward to.
                    if target is None, will terminate.

    kwargs :        Exported kwargs to give the next fallback.
    '''

    def __init__(self, 
                 graph: "PathGraph", 
                 input: any, 
                 output: any, 
                 origin: int|None, 
                 targets: set[int], 
                 **kwargs):
        
        self.graph = graph
        self.input = input
        self.output = output
        self.origin = origin
        self.targets = targets
        self.kwargs = kwargs

class Edge:

    '''
    [This class should not be instantiated manually!]
    '''

    def __init__(self, parent_graph: "GraphLike",  node_a: Node|int, node_b: Node|int, weight: float|None=None, length: float=1.0):
        
        self.graph = parent_graph
        self.marked = False
        self.visited = False

        self.directed = self.graph.directed
        self.weight = weight
        self.length = length

        # Extract IDs
        if type(node_a) is Node:
            node_a = node_a.id
        if type(node_b) is Node:
            node_b = node_b.id

        # Denote tuple for node identifiers
        self.tuple = (node_a, node_b)
        self.permuted = (node_b, node_a)

        # Check if the tuple is allowed in edge space by taking the directedity into account.
        if self.directed and self.tuple in self.graph.edge_space:
            raise LookupError(f'The edge "{self.tuple}" exists already in graph "{parent_graph.name}".')
        elif not self.directed:
            if self.tuple in self.graph.edge_space:
                raise LookupError(f'The edge "{self.tuple}" exists already in graph "{parent_graph.name}".')
            elif self.permuted in self.graph.edge_space:
                raise LookupError(f'The permuted edge "{self.permuted}" exists already and is bi-directed in graph "{parent_graph.name}". To enable the co-existence of both tuples i.e. ({self.tuple}) and ({self.permuted}), the graph needs to be directed.')
            
        # append to corr. edge space
        self.graph.edge_space.add(self.tuple)

        # override edge lookup
        self.graph.lookup_edge[self.tuple] = self
    
    def delete (self) -> None:

        '''
        Deletes the edge and removes it from all graph sets.
        '''
        
        # remove from edge space
        if self.tuple in self.graph.edge_space:
            self.graph.edge_space.remove(self.tuple)

        # repeat with permuted tuple if the graph is not directed
        if not self.graph.directed and self.permuted in self.graph.edge_space:
            self.graph.edge_space.remove(self.permuted)
        
        # pop the entry from edge lookup
        if self.tuple in self.graph.lookup_edge:
            self.graph.lookup_edge.pop(self.tuple)
        
        # repeat with permuted tuple if the graph is not directed
        if not self.graph.directed and self.permuted in self.graph.lookup_edge:
            self.graph.lookup_edge.pop(self.permuted)

class Path:

    '''
    [Base Path]

    This object just inherits the sequences of node identifiers without checking for links.
    Therefore the object should not be called manually but used when obtained from other objects.
    '''

    def __init__(self, parent_graph: "GraphLike", *nodes):
        
        '''
        A path is a sequence of nodes which are connected by edges.
        '''

        if len(nodes) < 2:
            raise ValueError('A path needs at least two nodes.')

        self.graph = parent_graph
        self.sequence = [n.id if type(n) is Node else n for n in nodes]
    
    def __call__(self):

        return self.sequence
    
    def connects (self, node_a: "Node|int", node_b: "Node|int") -> bool:

        '''
        If the underlying graph is directed, it will check if there is a path
        starting from node a to node b i.e. the sequence is treated in ascending order.
        Otherwise, in undirected graphs, it will check if the two nodes are connected,
        disregarding directions i.e. the sequence has no direction.

        [Return]
        Returns truth statement as boolean.
        '''

        id_a = extract_node_id(node_a)
        id_b = extract_node_id(node_b)

        # check if the nodes are included at all
        if id_a not in self.sequence or id_b not in self.sequence:
            return False
        
        # Directed case.
        if self.graph.directed:
            # check if id_a comes before id_b
            if self.sequence.index(id_a) < self.sequence.index(id_b):
                return True
            # otherwise no connection from a to b is possible
            return False
        
        # In undirected case we already proved that both ids are included
        # and thus are connected.
        return True

    def edge (self, node_a: "Node|int", node_b: "Node|int") -> Edge:

        '''
        Forwarder for BaseGraph.edge.
        Retrieves the corresponding edge which connects node_a and node_b,

            either directedly a --> b
            or non directedly a --- b.

        [Parameter]

        node_a :    Origin node

        node_b :    Target node to which to link the origin node

        [Return]

        Returns the sourced Edge object.
        '''

        return self.graph.edge(node_a, node_b)
    
    def includes_cycles (self) -> bool:

        '''
        Checks if cylces are in the path by counting duplicates in the sequence.
        '''

        for id in self.sequence:
            if self.sequence.count(id) > 1:
                return True
        return False

    def length (self) -> int:

        '''
        Returns the total number of all edges.
        '''

        return len(self.sequence)-1
        
    def show (self) -> None:

        '''
        Prints the path with all nodes in console.
        '''

        path_str = str(self.sequence[0])
        for id in self.sequence[1:]:
            path_str += f' -> {id}'

        print(path_str)

    def edge_length (self) -> float:

        '''
        Returns the weighted length of the path by considering the edge.length between node pairs. 
        '''

        length = 0
        for i in range(len(self.sequence)-1):
            edge = self.edge(self.sequence[i], self.sequence[i+1])
            length += edge.length
        return length

class GraphLike:

    '''
    Foundational definition of a graph.
    '''

    def __init__(self, name: str='Graph_01', directed: bool=False):

        self.name = name
        self.directed = directed
        self.max_node_id = 0

        # Every graph has a corresponding node space and edge space
        self.node_space: set[int] = set()
        self.edge_space: set[tuple[int]] = set() # set of all edge tuples

        # Lookup tables with pointers to objects
        self.lookup_node: dict[int, "Node"] = dict()        # lookup map from id to Node object
        self.lookup_edge: dict[tuple[int], Edge] = dict()   # lookup from tuple to Edge object

class BaseGraph ( GraphLike ):

    '''
    Involves basic graph methods like node and edge management, representation etc.
    '''

    def __init__(self, name: str = 'Graph_01', directed: bool = False):

        super().__init__(name, directed)

    def __mul__(self, other: "BaseGraph"): 

        '''
        Defines the multiplication context for tensor product 
        between two arbitrary graphs a and b as

            graph_a = BaseGraph(...)
            graph_b = BaseGraph(...)
            # graph tensor product
            graph_c = graph_a * graph_b

        The resulting graph will inherit the directed argument from graph_a.
        '''

        if not isinstance(other, BaseGraph): 
            raise TypeError(f"Unsupported operand type(s) for *: 'BaseGraph' and '{type(other).__name__}'")

        # create resulting matrix from the tensor product
        result_matrix = self.adjacency_matrix() @ other.adjacency_matrix() 

        # create new raw graph and add the topology from the resulting matrix
        result_graph = BaseGraph(f'{self.name} x {other.name}', self.directed)
        
        # add all nodes first
        for i in range(result_matrix.shape[0]):
            result_graph.add_node(i)
        
        # secondly, add all edges according to the new result matrix
        for i in range(result_matrix.shape[0]):
            for j in range(result_matrix.shape[0]):
                weight = result_matrix[i][j]
                if not result_graph.directed and (j, i) in result_graph.edge_space:
                    continue
                if weight != 0:
                    result_graph.add_edge(i, j, weight)

        return result_graph

    def add_edge (self, node_a: "Node|int", node_b: "Node|int", weight: float|None=None) -> Edge:

        '''
        Adds a new edge from node a to node b.

        [Parameter]

        node_a :    Origin node

        node_b :    Target node to which to link the origin node

        weight :    Optional float value for weight
        '''

        id_a = extract_node_id(node_a)
        node_a = self.node(id_a)

        return node_a.link(node_b, weight)

    def add_node (self, id: int|None=None, active: bool=False) -> Node:

        '''
        Adds a new node to the graph.

        [Parameter]

        id :    If int is provided it will be prioritized, otherwise will auto assign id.
        '''

        if active:
            return ActiveNode(self, id)
        return Node(self, id)
    
    def adjacency_list (self) -> dict[int, set[int]]:

        '''
        Returns an adjecency list for all nodes in the graph.
        '''

        adj_dict = dict()

        for id, node in self.lookup_node.items():

            adj_dict[id] = node.adjacent_nodes
        
        return adj_dict
    
    def adjacency_matrix (self, respect_edge_length: bool=False) -> np.ndarray:

        '''
        Returns a square adjacency matrix.
        The diagonal elements indicate the connection of each node to itself which is 0.
        All non-diagonal elements are 1 if the connection exists (row node id -> column node id)
        The matrix dimension spans with the node ids in ascending order in both dimensions.
        Hence if the node list is [2,4,7,8,16] this will map the row/col index to node id i.e. from [0,..,4] -> [2,4,7,8,16].
        The row id is the leading or origin node from where the direction starts whereas 
        the column is the target node. If the graph is non-directed both values are the same
        and the matrix becomes symmetric.
        '''

        node_list = list(self.node_space)
        node_list.sort() # sort in ascending order
        size = self.node_count()
        mat = []
        for i in range(size):
            row = []
            for j in range(size):
                id_i, id_j = node_list[i], node_list[j]
                tpl = (id_i, id_j)
                pmt = (id_j, id_i)
                val = 0
                if not self.directed and (tpl in self.edge_space or pmt in self.edge_space) or tpl in self.edge_space:
                    if respect_edge_length:
                        edge = self.edge(*tpl) if tpl in self.edge_space else self.edge(*pmt)
                        val = edge.length
                    else:
                        val = 1
                row.append(val)
            mat.append(row)

        return np.array(mat)
    
    def connect_sequence (self, *nodes: "Node|int") -> None:

        '''
        Connects nodes in a sequence with edges.
        '''

        for i in range(len(nodes)-1):
            id = extract_node_id(nodes[i])
            id_adj = extract_node_id(nodes[i+1])
            if (id, id_adj) not in self.edge_space:
                self.add_edge(id, id_adj)

    def distance (self, node_a: "Node|int", node_b: "Node|int") -> int|None:

        '''
        Returns the shortest distance between two connected nodes
        i.e. the number of edges that need to be crossed. If not connected
        will return None. Works in directed and undirected graphs.
        The distance is obtained by squaring the adjacency matrix.
        '''

        id_a = extract_node_id(node_a)
        id_b = extract_node_id(node_b)

        # derive index positions in the matrix
        node_list = list(self.node_space)
        ind_a = node_list.index(id_a)
        ind_b = node_list.index(id_b)

        adj_matrix = self.adjacency_matrix()

        # determine the longest path
        # the distance matrix is obtained from the 
        # adjacency_matrix squared.
        mat = np.identity(adj_matrix.shape[0])
        for n in range(1, len(self.edge_space)+1):
            mat = mat @ adj_matrix
            if mat[ind_a][ind_b] > 0:
                return n

        return None

    def node (self, node_id: int) -> Node:

        '''
        Retrieves the node object for provided node_id.
        '''

        if not node_id in self.lookup_node:
            raise ValueError(f'Node "{node_id}" not found.')

        return self.lookup_node[node_id]

    def node_count (self) -> int:

        '''
        Returns the number of nodes which are associated to the graph as an integer.
        '''

        return len(self.node_space)

    def node_entropy (self, node: "Node|int", base: float=np.e) -> float:

        '''
        Returns the Random Walker Entropy of a single node.
        Will consider the edge weights as likelihoods 
        if all adjacent node edges have weights.
        Assumes that all weights are normalized and complete.
        '''

        node = node if type(node) is Node else self.lookup_node[node]

        # Compute Entropy
        if not self.edges_all_weighted(node):
            # If not all edges are weighted will assume uniform probability distribution.
            # In uniform case, use the direct solution of Random Walker entropy.
            return np.emath.logn(base, node.degree())
        else:
            ent = 0
            # Accumulate the entropy across all adjacent node weights.
            for adj_id in node.adjacent_nodes:
                # n = self.lookup_node[adj_id]
                # get the corr. edge
                tpl = (node.id, adj_id)
                pmt = (adj_id, node.id)
                if self.directed and tpl in self.edge_space:
                    edge = self.lookup_edge[tpl]
                elif not self.directed and (tpl in self.edge_space or pmt in self.edge_space):
                    try:
                        edge = self.lookup_edge[tpl]
                    except:
                        edge = self.lookup_edge[pmt]
                else:
                    raise ValueError()
                # Add contribution to entropy
                val = edge.weight * np.emath.logn(base, edge.weight)
                ent -= val
            
            return ent
    
    def edge (self, node_a: "Node|int", node_b: "Node|int") -> Edge:

        '''
        Retrieves the corresponding edge which connects node_a and node_b,

            either directedly a --> b
            or non directedly a --- b.

        [Parameter]

        node_a :    Origin node

        node_b :    Target node to which to link the origin node

        [Return]

        Returns the sourced Edge object.
        '''

        id_a = extract_node_id(node_a)
        id_b = extract_node_id(node_b)

        # Check if the connection exists already.
        if not self.is_linked(id_a, id_b):
            raise LookupError(f'No edge exists which connects node "{id_a}" and "{id_b}".')
        
        if self.directed:
            pass
        elif (id_b, id_a) in self.lookup_edge:
            return self.lookup_edge[(id_b, id_a)]
        return self.lookup_edge[(id_a, id_b)]

    def edge_count (self) -> int:

        '''
        Returns the number of edges associated to the graph as an integer.
        '''

        return len(self.edge_space)
    
    def edges_all_weighted (self, node: "Node|int") -> bool:

        '''
        Checks if all edges of a node are weighted i.e. weight is not None.

        [Return]

        Returns truth statement as boolean.
        '''

        # get the node
        node = node if type(node) is Node else self.lookup_node[node]

        # check if the probabiities are weighted, 
        # which will only be true if all nodes have weights.
        # Otherwise will assume uniform distribution.
        for n in node.adjacent_nodes:
            
            # get the corr. edge
            tpl = (node.id, n)
            if self.directed and tpl in self.edge_space:
                edge = self.lookup_edge[tpl]
            elif not self.directed and tpl in self.edge_space:
                edge = self.lookup_edge[tpl]
            elif not self.directed and (n, node.id) in self.edge_space:
                edge = self.lookup_edge[(n, node.id)]
            else:
                raise ValueError()
            
            if edge.weight is None:

                return False

        return True
    
    def entropy (self, base: float=np.e) -> float:
        
        '''
        Returns the Random Walker Entropy of the whole graph.
        Accumulates all single node entropies using BaseGraph.entropy_node.
        '''
        
        ent = 0

        for node in self.lookup_node.values():

            ent += self.node_entropy(node, base)
        
        return ent

    def is_linked (self, node_a: "Node|int", node_b: "Node|int") -> bool:

        '''
        Determines if origin node a is connected to target node b.
        If non-directed, both nodes and edges are symmetric.

        [Parameter]

        node_a :    Origin node

        node_b :    Target node to which to link the origin node
        '''

        # determine the node ids
        if type(node_a) is Node:
            node_a = node_a.id
        elif type(node_a) is int:
            pass
        else:
            raise TypeError(f'The provided node_a must be a Node or int object, not {type(node_a)}.')
        if type(node_b) is Node:
            node_b = node_b.id
        elif type(node_b) is int:
            pass
        else:
            raise TypeError(f'The provided node_b must be a Node or int object, not {type(node_b)}.')
        
        if self.directed:
            return (node_a, node_b) in self.edge_space
        else:
            return (node_a, node_b) in self.edge_space or (node_b, node_a) in self.edge_space

    def remove_edge (self, edge: "Edge|tuple[int]") -> None:

        '''
        Secure method to remove an edge from graph.
        '''

        if type(edge) is tuple:
            edge = self.edge(*edge)
        edge.delete()

    def remove_node (self, node: "Node|int") -> None:

        '''
        Secure method to remove node from graph.
        '''

        id = extract_node_id(node)
        self.node(id).delete()

    def reset_node_labels (self) -> None:
        
        '''
        Resets all labels of all nodes like node.marker and node.visits.
        '''

        for node in self.lookup_node.values():
            
            node.marked = False
            node.visits = 0

    def spectrum (self) -> np.ndarray:
        
        '''
        Returns the spectrum of an undirected graph.
        Directed graphs hold no spectrum as their adjacency matrix is non-symmetric
        and therefore does not have real eigenvalues.
        The spectrum is the sorted list of the adjacency matrix eigenvalues.
        Starting with ev_1 >= ev_2 >= ev_3 ...

        [Return]

        1-D Array including the eigenvalues in descending order, i.e. the first element
        holds the largest eigenvalue.
        '''
        
        if self.directed:
            raise ValueError('Directed graphs have no spectrum as their adjacency matrix is non-symmetric.')

        adj_mat = self.adjacency_matrix()

        eigenvalues, _ = np.linalg.eig(adj_mat)
        
        return eigenvalues

    def to_mermaid (self, chart_type: str='flowchart', direction: str='LR') -> str:

        '''
        Generates a mermaid-compliant flow chart code for the Graph.

        [Return]

        Mermaid flowchart code as string.
        '''

        mermaid_wrapper = '''```mermaid\n{} {}\n{}```'''
        mermaid_code = ''
        
        # Generate nodes
        for node in self.node_space:
            mermaid_code += f'\tid_{node}(({node}))\n'

        # create edges
        for tpl in self.edge_space:
            if self.directed:
                mermaid_code += f'\tid_{tpl[0]} --> id_{tpl[1]}\n'
            else:
                mermaid_code += f'\tid_{tpl[0]} --- id_{tpl[1]}\n'

        return mermaid_wrapper.format(chart_type, direction.upper(), mermaid_code)

    def total_edge_weight (self) -> int:

        '''
        Returns the sum of weights of all edges in the graph.
        If no weight is defined for an edge it will be overriden with 0. 
        '''

        weight = 0
        for _, edge in self.lookup_edge.items():
            weight += 0 if edge.weight == None else edge.weight
        return weight

class PathGraph ( BaseGraph ):
    
    '''
    Involve graph search, paths, and connectivity algorithms.
    '''

    def __init__ (self, name: str='Graph_01', directed: bool=False):

        super().__init__(name, directed=directed)

        # cache object for counting purposes
        self.cache: set = set()

    def ats (self, node_a: "Node|int", node_b: "Node|int") -> Path:

        '''
        ### Adjacency Tree Search "ATS"

        Utilizes the adjacency matrix to determine neighbors at specific depth via mat_mul, 
        once determined the shortest path is traced back to origin node "a" which is always 
        the shortest path by edge count.

        [Background]

        This algorithm directly determines the final depth and will search only up to final depth.
        Hence, the approach is considered more efficient for just finding the shortest path than "branch and prune",
        which needs to determine all paths and thus also paths of lengths > final depth.
        However, while matrix multiplication can be efficient, it also introduces overhead, 
        especially for large matrices. Tensor managing can is memory heavy and should therefore 
        be considered for large graphs with large expected paths where memory management does not outweigh the benefits.

        Note: this algorithm does not consider edge weights, instead it determines the shortes path by the least number of intermediate edges.
        '''

        start_id = extract_node_id(node_a)
        target_id = extract_node_id(node_b)
        
        # determine the max_depth which will act as a radius or tree depth
        # which in essence is the distance between origin and targe nodes
        if not self.connected(node_a, node_b):
            raise ValueError(f'There exists no path between the nodes {start_id} and {target_id}, as they are not connected.')
        
        # compute the current adjacency matrix
        adj_mat = self.adjacency_matrix()
        mat = adj_mat.copy()
        
        # cache
        sequence: list = []
        tensors = []
        id_list = list(self.node_space) # maps the matrix index to node id

        # convert start and target node ids to matrix indices
        start_ind = id_list.index(start_id) # convert to matrix position (row)
        target_ind = id_list.index(target_id) # convert to matrix position (col)

        # The upper bound can be set to the longest possible path that connects all N nodes.
        # This path if it is a cycle would have N edges.
        max_depth = len(self.node_space)
        for _ in range(1, max_depth+1):
            tensors.append(mat)
            # stop if the target id is found
            if mat[start_ind][target_ind] > 0:
                sequence.append(target_ind)
                break
            # step forward to next depth - multiply current tensor with adjacency matrix
            mat = mat @ adj_mat


        # trace back from the target to origin
        for depth in range(len(tensors)-1, -1, -1):
            
            if depth == 0:
                sequence.append(start_ind)
                break
            
            # select the depth-1 adjacency matrix, this matrix yields which nodes are depth-1 neighbors
            tensor = tensors[depth-1]

            # denote all non-zeros indices in the start/origin row.
            # all candidates, surely lead to the start node.
            candidates = np.where(tensor[start_ind] > 0)[0]
            
            # if there is only one candidate it inevitably MUST be the path to origin
            if len(candidates) == 1:
                sequence.append(candidates[0])
                continue
            
            # otherwise check which candidate(s) is a next neighbor to the last node in sequence
            for ind in candidates:
                
                # check if the first order adjacency matrix yields an edge 
                # between previous node and current candidate
                if adj_mat[ind][sequence[-1]] > 0:
                    sequence.append(ind)
                    break
                    
        # convert sequence to ids
        sequence = [id_list[ind] for ind in sequence]
        sequence.reverse()

        # convert to path and return
        return Path(self, *sequence)

    def bps (self, node_a: "Node|int", node_b: "Node|int", verbose: bool=False) -> set[Path]|None:

        '''
        ### Branch and Prune Search "BPS"

        Pruning search algorithm for finding all paths between two connected nodes, a and b.
        If the aim is to find the shortest path this implementation performs better for shorter paths. 

        [Parameter]

        node_a :    Origin node

        node_b :    Target node

        [Return]

        Returns a set of all unique paths between a and b.

        --- 
        
        Algorithm in Pseudo Code

        0.  Initialize an array for paths:int->Path, branching:int->id, and a count map:id->count.
            Set a pointer variable to equal the start id.
        
        while loop
            
            1.  If unknown (i.e. not in the current sequence) add the pointer to the sequence list  
            2.  if pointer equals the target node id:
                    - add the current sequence to paths
                    - try to go to last id in branching array : pointer <- branching[-1]
                      if there is none, stop the search
                    - slice the sequence from right up to this id, but keep the id
                    - restart loop
            3.  Count number of possible exits at the pointer, 
                i.e. adjacent nodes which are not in the current sequence.
            4.  Decide on the number of exits whether:
                the # of exits equals 0, then revert:
                    - the pointer is a dead-end
                    -> try to go to last id in branching array : pointer <- branching[-1]
                    -> if there is none, stop the search:
                            break loop
                    -> slice the sequence from right up to this id, but keep the id
                    -> restart loop
                else if count equals 1:
                    - the pointer is an intermediate node
                    -> continue forward to the only neighbor: pointer <- neighbor
                      which is not in sequence yet 
                    -> restart loop
                else if the pointer id is not in branching list:
                    - the pointer is a new branching for this sequence 
                    -> push the pointer into branching
                    -> initialize a count for this branching i.e. count[pointer] -> 0
            5.  The pointer is a branching, check if the branch provides yet undiscovered exits, 
                otherwise break:
                -> if count[pointer] >= length(pointer.adjacents):
                        break loop
                -> Otherwise, select with pointer the next undiscovered exit 
                -> restart loop
            6.  Branch pruning: 
                - all exits were searched at current branch.
                -> discard the last known branching: branching.pop()
            7.  Revert:
                -> try to go to last id in branching array : pointer <- branching[-1]
                -> if there is none, stop the search:
                        break loop
                -> slice the sequence from right up to this id, but keep the id

        8. Return the paths list.            
        '''

        # clean the state        
        self.reset_node_labels()
        self.cache.clear()

        # extract node information
        root_id = extract_node_id(node_a)
        target_id = extract_node_id(node_b)

        # iteration variables
        pointer_id: int = root_id
        pointer_node: Node = self.node(pointer_id)
        path_set: list[list[int]] = list()
        sequence: list[int] = list()
        branching: list[int] = list()
        index_count: dict[int, int] = dict()

        if verbose:
            step = 0

        while True:

            # if this pointer is not being observed in the sequence yet, add it
            if pointer_id not in sequence:
                sequence.append(pointer_id)

            if verbose:
                if step > 1000:
                    break
                print(f'step {step}) Current Sequence:', sequence)
                step += 1

            # Check if the target is found
            if pointer_id == target_id:
                print('found sequence:', sequence) if verbose else None
                # denote the current sequence
                path_set.append(sequence)
                # Cut back sequence to previous branch to continue the search for other paths.
                # Except if there are no branches left, then stop the search.
                if not branching:
                    break
                # also check if the current pointer/target is a branching then pop it
                if branching[-1] == target_id:
                    branching.pop()
                # finally go back to last known branching
                pointer_id = branching[-1]
                pointer_node = self.node(pointer_id)
                cut_index = sequence.index(pointer_id) + 1
                sequence = sequence[:cut_index]
                continue
            
            # Count the number of exits at each node
            exits = pointer_node.degree() # marked and unmarked exits
            
            # Exclude cylcing exits i.e. which lead to a node which is already in the sequence
            if not self.directed:
                for n in pointer_node.adjacent_nodes:
                    if n in sequence:
                        exits -= 1
                        break

            # Categorize the current pointer based on the number of exits:
            if exits == 0:
                # No exits means the pointer is a dead-end.
                if verbose:
                    old = pointer_id
                # Check if there are branches left to return to
                # otherwise we are done here.
                if not branching:
                    break
                # -> return to last known branching
                pointer_id = branching[-1]
                pointer_node = self.node(pointer_id)
                # Branch pruning:
                # finally cut the sequence to last known branch
                cut_index = sequence.index(pointer_id) + 1
                sequence = sequence[:cut_index]
                print(f'\t{old} is a dead-end, return back to {pointer_id}') if verbose else 0
                continue
            elif exits == 1:
                # the node is just a intermediate node "which lies on a single edge"
                # step forward by sampling a new adjacent node
                # pointer_id = next(iter(pointer_node.adjacent_nodes)) # best way to extract the pointer from the set without copying the set
                for sample_id in pointer_node.adjacent_nodes:
                    if sample_id != pointer_id:
                        pointer_id = sample_id
                        break
                pointer_node = self.node(pointer_id)
                print(f'\tcontinue to {pointer_id}') if verbose else 0
                continue
            elif pointer_id not in branching:
                # exits > 1 -> branching detected
                branching.append(pointer_id) 
                index_count[pointer_id] = 0
                print(f'\t{pointer_id} is a new branch') if verbose else 0
            
            # At this point the current pointer is a branching
            # check if the branch provides unknown exits
            # i.e none in the sequence
            ajacents = list(pointer_node.adjacent_nodes)
            exit_id = None
            while index_count[pointer_id] < len(ajacents):
                neighbor = ajacents[index_count[pointer_id]]
                # directly increment the counter for this branch
                index_count[pointer_id] = index_count[pointer_id] + 1
                if neighbor not in sequence:
                    exit_id = neighbor
                    break
            if exit_id:
                # take new exit node
                exit_node = self.node(exit_id)
                pointer_id = exit_id
                pointer_node = exit_node
                print(f'\tgo to {pointer_id} ...') if verbose else 0
                continue

            # If no exits are found i.e. all found branches led to a dead-end,
            # the branching node itself becomes a dead-end.
            # -> Prune the branch i.e. remove from branching list
            print(f'\tAll exits searched at {pointer_id}, prune branch ...') if verbose else 0
            branching.pop()
            
            # If there are no branches left at this point the search is finished.
            if not branching:
                break

            # Otherwise revert to (second) last known branch 
            print(f'\tgo back to branch {branching[-1]} ...') if verbose else 0
            pointer_id = branching[-1]
            pointer_node = self.node(pointer_id)
            # -> cut the sequence to branching node
            cut_index = sequence.index(pointer_id) + 1
            sequence = sequence[:cut_index]

        # At this point no branchings are left anymore.
        # Check the found paths to select the shortest.
        if len(path_set) == 0:
            print(f'\tThere exists no path from node "{root_id}" to "{target_id}" in graph "{self.name}".') if verbose else 0
            return None
        else:
            return set([Path(self, *p) for p in path_set])
        
    def connected (self, node_a: "Node|int", node_b: "Node|int") -> bool:
        
        '''
        Checks if there is a connection from node a to node b.
        Works for directed and undirected case.

        [Parameter]

        node_a :    Origin node

        node_b :    Target node to which to link the origin node
        '''

        search_id = extract_node_id(node_b)
        return search_id in self.idfs(node_a, stop=search_id)

    def find_all_paths (self, node_a: "Node|int", node_b: "Node|int", verbose: bool=False) -> set[Path]|None:

        '''
        Finds all paths between two connected nodes, a and b.
        If the aim is to find the shortest path this implementation performs better for shorter paths.
        This is an alias for PathGraph.BPS algorithm.   

        [Return]

        Returns a set of PathGraph.Path objects which connect a to b or vice versa in undirected graphs.         
        '''

        return self.bps(node_a, node_b, verbose)

    def idfs (self, node: "Node|int", stop: "Node|int|None"=None) -> set[int]:
        
        '''
        Iterative Preorder Depth-first Search Algorithm (iDFS).
        Computes time scales better than worst case order O(|node space| + |edge space|)
        '''

        id = extract_node_id(node)
        node = self.node(id)

        # reset former node labels
        self.reset_node_labels()
        self.cache.clear()

        # label the first node as discovered by marking it
        node.mark()
        self.cache.add(id)

        stack = list(node.adjacent_nodes)
        next_node: Node = None

        if stop:
            stop_id = extract_node_id(stop)

        # Work down the stack.
        while stack:
            
            # Move to next node from stack
            next_id = stack[-1]
            next_node = self.node(next_id)

            if not next_node.marked:
                
                next_node.mark()
                self.cache.add(next_node.id)

                # break here if stop id was found
                if stop != None and stop_id == next_id:
                    break

                # find unmarked adjacents of the next_node
                unmarked_adjacents: list[int] = []
                for u_id in next_node.adjacent_nodes:
                    u_node = self.node(u_id)
                    if not u_node.marked:
                        unmarked_adjacents.append(u_id)

                # extend stack with unmarked adjacents
                stack = unmarked_adjacents + stack

                # finally remove the next node as it was marked
                stack.pop()
            
            else:
                
                # Directly remove marked node.id from stack.
                stack.pop()

        return self.cache

    def is_connected (self) -> bool:

        '''
        Checks if the graph is connected i.e. every node is connected to any other node.
        The algorithm compares the reachable nodes from a random pick to the whole node_space size.

        [Return]

        Boolean truth statement.
        '''

        # need to disable directed var for this
        save_directed = self.directed
        self.directed = False

        # the problem is symmetric -> pick any node
        pick = self.node(list(self.node_space)[0])

        # determine the reachable set from this node
        reachable_nodes = self.reachable_nodes( pick )

        # reset directed variable
        self.directed = save_directed

        # if the set of reachable nodes and the node space set (excluding the origin) are equal in size 
        # the graph is considered connected.
        return len(reachable_nodes) == len(self.node_space) - 1

    def rdfs (self, node: "Node|int", depth: int=1) -> set[int]:
        
        '''
        Recursive Depth-first Search (rDFS)

        Will explore and mark unvisited nodes recursively.
        The explored nodes are collected in cache for access.

        [Parameter]
        
        node :      root node as node object or node id

        depth :     recursion parameter (not meant for usage)


        [Return]

        Returns a set of explored node ids.
        '''
        
        # In the first run remember to reset all node labels.
        if depth == 1:
            self.cache.clear()
            self.reset_node_labels()
            # extract the node for this run
            id = extract_node_id(node) # convert the first time to id
        else:
            id = node
        
        # Mark the current node at pointer.
        node = self.node(id)
        node.mark()
        self.cache.add(id)

        # Iterate over adjacent_nodes,
        # note: adjacent_nodes respect directed and undirected paths
        # as the node is only adjacent if reachable by the defined direction.
        for nid in node.adjacent_nodes:
            neighbor = self.node(nid)
            if not neighbor.marked:
                self.rdfs(nid, depth+1)
        
        if depth == 1:
            return self.cache

    def reachable_nodes (self, node: "Node|int", algorithm: str='idfs') -> set[int]:
        
        '''
        DFS alias for discovering all reachable nodes from provided root node.

        [Parameter]

        node :      root node from where to start searching

        algorithm : underlying search algorithm (idfs, rdfs)

        [Return]

        Returns a set of reachable node ids.
        '''

        if algorithm == 'idfs':
            return self.idfs(node)
        elif algorithm == 'rdfs':
            return self.rdfs(node)

    def run (self, start_node: "Node|int", skip_non_active=False, verbose: bool=False) -> None:

        '''
        Runs the directed path from node to node with internal node decision_making.
        '''

        if not self.directed:
            ValueError('The graph needs to be directed to run it.')

        start_id = extract_node_id(start_node)

        print(f'Run "{self.name}" from node {start_id}:')

        pointer = start_id

        while True:

            node = self.node(pointer)

            if type(node) is not ActiveNode and not skip_non_active:
                raise TypeError('All nodes need to be an ActiveNode, otherwise use skip_non_active=True argument to skip non-active nodes.')

    def shortest_length (self, node_a: "Node|int", node_b: "Node|int") -> Path:

        '''
        Determines the path of shortest length by edge count.
        This is an alias for ATS.
        '''

        return self.ats(node_a, node_b)

    def shortest_path (self, node_a: "Node|int", node_b: "Node|int", respect_edge_length:bool=False, verbose: bool=False) -> Path|None:
        
        '''
        Allows to find the shortest path by cummulated path.length. If the length is not respected every 
        weight will be set to 1 and the algorithm becomes a direct alternative to shortest_length. 
        This is an alias for BPS or PathGraph.find_all_paths with minimization.
        

        [Parameter]

        node_a :    Origin node

        node_b :    Target node to which to link the origin node

        respect_edge_weight : if enabled, will account edge weights as length between nodes, 
                              otherwise will take 1 i.e. single count.

        '''

        # First find all paths.
        paths = self.find_all_paths(node_a, node_b, verbose)

        if not paths:
            return None
        
        # Minimize the set and return minimum.
        min, shortest = 0, None
        for p in paths:
            length = 0
            if respect_edge_length:
                length = p.edge_length()
            else:
                length = p.length()
            if length < min or shortest is None:
                shortest = p
                min = length

        return shortest
 
class ComponentGraph ( PathGraph ):

    '''
    Involve ConnectedComponent management and relations.
    '''

    def __init__(self, name: str='Graph_01', directed: bool=False):

        super().__init__(name, directed)

        # these variables will be adjusted automatically when 
        # ConnectedComponent is initialized with this graph as argument.
        self.component_id_counter = 0
        self.components: dict[int, ConnectedComponent] = dict()
    
    def split_into_components (self) -> list["ConnectedComponent"]:

        '''
        Splits the ComponentGraph into multiple ConnectedComponents.
        Returns a list of all ConnectedComponent objects for this Graph.
        '''

        # flush all current components
        self.components.clear()

        output_list = [] # aggregate component objects
        search_set = self.node_space.copy()

        while search_set:
            # select next pointer in search set
            pointer = list(search_set)[0]
            # determine the set all reachable nodes
            reach_set = self.reachable_nodes(pointer)
            reach_set.add(pointer)
            # denote the connected component
            component_set = reach_set.copy() # copy from current reach_set
            output_list.append(ConnectedComponent(self, component_set))
            # take the disjoint of the current search set and the component set
            search_set.difference_update(component_set)

        return output_list

class ConnectedComponent ( PathGraph ):

    '''
    A connected component follows all rules as PathGraph in which all nodes are connected, 
    but with a pointer reference to a parent component graph.
    '''

    def __init__(self, parent_graph: ComponentGraph, node_set: set[int]|list[int]):

        self.parent_graph = parent_graph
        self.component_id = parent_graph.component_id_counter

        # Init ComponentGraph.
        super().__init__(name=f'{parent_graph.name}_component_{self.component_id}', directed=parent_graph.directed)
        
        # Accept te node set as new node space.
        self.node_space = node_set if type(node_set) is set else set(node_set)

        # Copy only valid edges.
        self.edge_space = set()
        for tpl in parent_graph.edge_space:
            if tpl[0] in self.node_space or tpl[1] in self.node_space:
                self.edge_space.add(tpl)

        # store in parent object
        self.parent_graph.components[self.component_id] = self
        self.parent_graph.component_id_counter += 1 # increment
    
    def delete (self) -> None:

        '''
        Deletes the current component from graph object.
        '''

        self.parent_graph.components.pop(self.component_id)
        
class Network ( PathGraph ):

    '''
    A network is generally always a connected graph. 
    Separate nodes or disconnected components 
    are simply not part of the network and are hence not of interest.
    The main focus lies in finding routes, forwarding information 
    and or executing tasks at nodes. 

    Here Network is interpreted as 
    - a PathGraph which allows to route.
    - and contains only active nodes to execute tasks or forward.
    - every node has a degree of >=1.
    - nodes can disconnect with an impact on the network dynamic, 
      then the network participants have to re-route
    - every node has a unique address which is bijectively mapable to it's id
      similar to a domain name space e.g. '0.0.0.0' or '4 Privet Drive, Little Whinging, Surrey'
    - every node has a display label e.g. 'Google DNS' or 'Harry Potter'
      the label does not have to be unique, as multiple nodes with different addresses can be under the same label.
    '''

    def __init__(self, name: str='Network_01', directed: bool=False):

        super().__init__(name, directed)

        self.address_space: set[str] = set()
        # maps address to id
        self.address_lookup: dict[str, int] = dict()

    def add_node (self, id: int|None=None, address: str|None=None, label: str|None=None) -> ActiveNode:

        '''
        Adds a new node to the network.

        [Parameter]

        id :    If int is provided it will be prioritized, otherwise will auto assign id.
        '''

        node = ActiveNode(self, id)

        # generate address and store in the node
        if address != None and address in self.address_space:
            raise ValueError(f'Provided address must be unique but "{address}" exists already.')
        if address != None:
            node.address = address
        else:
            node.address = self.generate_address()

        # add label to node
        node.label = label

        # denote in address lookup
        self.address_space.add(node.address)
        self.address_lookup[node.address] = node.id

        return node
    
    def generate_address (self) -> str:
        
        '''
        Generates a unique address.
        '''

        address = generate_random_hash()
        while address in self.address_space:
            address = generate_random_hash()
        
        return address

# ---- Graph Operations ----
def canon (graph: GraphLike) -> GraphLike:

    id_list = list(graph.node_space)

    # create a degree sorted map D: id -> k, where k is the degree.
    degree_map: dict = dict()
    for id in id_list:
        degree_map[id] = graph.lookup_node[id].degree()
    sorted_degree_map = {k: v for k, v in sorted(degree_map.items(), key=lambda item: item[1])}
    
    # re-label the ids
    new_id = 1
    for id, deg in sorted_degree_map.items():
        # exchange the edges
        for i in range(len(graph.edge_space)):
            tpl = graph.edge_space[i]
            if tpl[0] == id:
                tpl[0] = new_id
            elif tpl[1] == id:
                tpl[1] = new_id
            graph.edge_space[i] = 1



# ---- Trees ----
class BaseTree ( PathGraph ):

    '''
    The BaseTree class implements a basic tree logic.
    Trees are graphs with a hirarchy which obeys the parent and children relation.

    Trees are by definition
     - connected and acyclic
     - contain directed path logic
     - Removing any edge disconnects the graph
     - Adding only an edge automatically creates a cycle

    '''

    def __init__ (self, name: str='Tree_01', directed: bool=True, depth: int|None=None, degree: int|None=None) -> None:

        super().__init__(name, directed)

        # initialize root node with id 0
        self.root = super().add_node(0, active=True)

        if depth:
            self.generate(self.root, depth, degree if degree else 2)

    def add_node (self, parent: ActiveNode|int, id: "any|None"=None, active: bool=True) -> ActiveNode:

        '''
        Modified add_node method which links the created nodes to parent node.
        This method automatically creates an edge from parent to child.
        '''

        parent_id = extract_node_id(parent)
        if not parent_id in self.node_space:
            raise ValueError(f'No parent node "{parent_id}" found.')
        # use inherited method
        node = super().add_node(id, active)
        # denote parent
        node.parent_id = parent_id 
        # add directed edge
        self.add_edge(parent_id, node.id)

        return node
    
    def children (self, node: ActiveNode|int) -> set[int]:

        '''
        Alias for adjacent nodes.
        '''

        return node.adjacent_nodes
    
    def generate (self, parent: int, depth: int, degree: int) -> None:
        
        '''
        Generates a tree of fixed depth and constant degree (across all nodes) recursively.
        '''

        if depth == 0:
            return
        
        for _ in range(degree):
            child = self.add_node(parent)
            self.generate(child, depth-1, degree)

    def depth_set (self, depth: int) -> set[int]:
        
        '''
        Returns all nodes at a specified depth.
        '''

        if depth == 0:
            return {0}

        adj_mat = self.adjacency_matrix()
        depth_adj_mat = adj_mat.copy()

        # compute the depth-order adjacency matrix
        # i.e. depth-many tensor products
        for _ in range(depth-1):
            depth_adj_mat = depth_adj_mat @ adj_mat

        # get indices which are ids
        index_array = []
        for i in range(len(depth_adj_mat[0])):
            if depth_adj_mat[0][i] > 0:
                index_array.append(i)

        return set(index_array)

class OrderTree ( BaseTree ):

    '''
    An ordered tree implementation.
    Child order matters and is additionally tracked in lists.
    '''

    def __init__ (self, name: str='Ordered_Tree_01', directed: bool=True, depth: int|None=None, degree: int|None=None):

        super().__init__(name, directed, depth, degree)

        self.child_list_map: dict[int, list[int]] = dict()
        # initialize root node in child map
        self.child_list_map[self.root.id] = list()
    
    def add_node (self, parent: ActiveNode|int, id = None, active = True):

        '''
        Extended method from super class BaseTree. 
        Same as BaseTree.add_node but adds the created node to parent child list in child map.
        '''
        
        node = super().add_node(parent, id, active)
        parent_id = extract_node_id(parent)

        # add parent in child map if not exists
        if not parent_id in self.child_list_map:
            self.child_list_map[parent_id] = list()

        # add ordered parent-child relation in child map
        if node.id not in self.child_list_map[parent_id]:
            self.child_list_map[parent_id].append(node.id)    

    def children(self, node) -> list[int]:

        '''
        Overrided children method -> differs from BaseTree super class.
        Children in ordered context return a list of child ids.
        '''

        return self.child_list_map[extract_node_id(node)]

    def permute (self, parent: ActiveNode|int, child_a: ActiveNode|int, child_b: ActiveNode|int) -> None:

        '''
        Exchanges the order of child a and b in parent child map.
        '''

        id_a = extract_node_id(child_a)
        id_b = extract_node_id(child_b)

        ind_a = self.child_list_map[parent].index(id_a)
        ind_b = self.child_list_map[parent].index(id_b)

        self.child_list_map[parent][ind_a] = id_b
        self.child_list_map[parent][ind_b] = id_a

    def remove_node (self, node: "Node|int") -> None:

        '''
        Extended method from super class BaseTree. 
        Same as BaseTree.remove_node but removes the node from parent child list in child map.
        '''

        super().remove_node(node)
        
        id = extract_node_id(node)

        # remove from child map
        if id in self.child_list_map:
            self.child_list_map.pop(id)
        
        # removing the connection from the parent list
        parent_id = self.node(id).parent_id
        if id in self.child_list_map[parent_id]:
            self.child_list_map[parent_id].remove(id)