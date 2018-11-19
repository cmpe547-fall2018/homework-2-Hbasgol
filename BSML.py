import numpy as np
import scipy as sc
from scipy.special import gammaln
from scipy.special import digamma

from itertools import combinations

import pygraphviz as pgv
from IPython.display import Image
from IPython.display import display

## This functions are adopted from Ali Taylan Cemgil's github page. 
## https://github.com/atcemgil/notes/blob/master/BayesianNetworks.ipynb
## Utility Functions

def normalize(A, axis=None):
    """Normalize a probability table along a specified axis"""
    Z = np.sum(A, axis=axis,keepdims=True)
    idx = np.where(Z == 0)
    Z[idx] = 1
    return A/Z

def find(cond):
    """
        finds indices where the given condition is satisfied.
    """
    return list(np.where(cond)[0])

## Random structure and parameter generators
    
def random_alphabet(N=20, first_letter='A'):
    """Generates unique strings to be used as index_names"""
    if N<27:
        alphabet = [chr(i+ord(first_letter)) for i in range(N)]
    else:
        alphabet = ['X'+str(i) for i in range(N)]    
    return alphabet

def random_parents(alphabet, max_indeg=3):
    """Random DAG generation"""
    N = len(alphabet)
    print(alphabet)
    indeg = lambda: np.random.choice(range(1,max_indeg+1))
    parents = {a:[b for b in np.random.choice(alphabet[0:(1 if i==0 else i)], replace=False, size=min(indeg(),i))] for i,a in enumerate(alphabet)}
    return parents

def random_cardinalities(alphabet, cardinality_choices=[2,3,4,5]):
    """Random cardinalities"""
    return [np.random.choice(cardinality_choices) for a in alphabet]
    
def states_from_cardinalities(alphabet, cardinalities):
    """Generate generic labels for each state"""
    return {a:[a+"_state_"+str(u) for u in range(cardinalities[i])] for i,a in enumerate(alphabet)}
    
def cardinalities_from_states(alphabet, states):
    """Count each cardinality according to the order implied by the alphabet list"""
    return [len(states[a]) for a in alphabet]    
    
def random_observations(cardinalities, visibles):
    """
    Samples a tensor of the shape of visibles. This function does not sample 
    from the joint distribution implied by the graph and the probability tables
    """
    return np.random.choice(range(10), size=clique_shape(cardinalities, visibles))
 
def random_dirichlet_cp_table(gamma, cardinalities, n, pa_n):
    '''
        gamma : Dirichlet shape parameter
        cardinalities : List of number of states of each variable
        n, pa_n : Output a table of form p(n | pa_n ), n is an index, pa_n is the list of parents of n 
    '''
    N = len(cardinalities)
    cl_shape = clique_shape(cardinalities, [n]+pa_n)
    U = clique_prior_marginal(cardinalities, cl_shape)
    return normalize(np.random.gamma(shape=gamma*U, size=cl_shape), axis=n)

def random_cp_tables(index_names, cardinalities, parents, gamma):
    """
    Samples a set of conditional probability tables consistent with the factorization
    implied by the graph.
    """
    N = len(index_names)
    theta = [[]]*N
    for n,a in enumerate(index_names):
        theta[n] = random_dirichlet_cp_table(gamma, cardinalities, n, index_names_to_num(index_names, parents[a]))
        #print(a, parents[a])
        #print(theta[n].shape)
        #print('--')
    return theta

def random_model(N=10, max_indeg=4):
    """
    Generates a random Bayesian Network
    """
    index_names = random_alphabet(N)
    parents = random_parents(index_names)
    cardinalities = random_cardinalities(index_names)
    states = states_from_cardinalities(index_names, cardinalities)
    
    return index_names, parents, cardinalities, states

## Graph Utilities and Visualizations
    
def clique_shape(cardinalities, family):
    N = len(cardinalities)
    size = [1]*N
    for i in family:
        size[i] = cardinalities[i] 
    return size

def clique_prior_marginal(cardinalities, shape):
    U = 1
    for a1,a2 in zip(shape, cardinalities):
        U = U*a2/a1
    return U
    
def index_names_to_num(index_names, names):
    name2idx = {name: i for i,name in enumerate(index_names)}
    return [name2idx[nm] for nm in names] 

def show_dag_image(index_names, parents, imstr='_BJN_tempfile.png'):
    name2idx = {name: i for i,name in enumerate(index_names)}
    A = pgv.AGraph(directed=True)
    for i_n in index_names:
        A.add_node(name2idx[i_n], label=i_n)
        for j_n in parents[i_n]:
            A.add_edge(name2idx[j_n], name2idx[i_n])
    A.layout(prog='dot')
    A.draw(imstr)
    display(Image(imstr))
    return 

def show_ug_image(UG, imstr='_BJN_tempfile.png'):
    A = pgv.AGraph(directed=False)

    for i_n in range(UG.shape[0]):
        A.add_node(i_n, label=i_n)
        for j_n in find(UG[i_n,:]):
            if j_n>i_n:
                A.add_edge(j_n, i_n)

    A.layout(prog='dot')
    A.draw(imstr)
    display(Image(imstr))
    return

def make_cp_tables(index_names, cardinalities, cp_tables):
    N = len(index_names)
    theta = [[]]*N

    for c in cp_tables:
        if not isinstance(c, tuple):
            nums = index_names_to_num(index_names, (c,))
        else:
            nums = index_names_to_num(index_names, c)
        #print(nums)
        n = nums[0]
        idx = list(reversed(nums))
        theta[n] = np.einsum(np.array(cp_tables[c]), idx, sorted(idx)).reshape(clique_shape(cardinalities,idx))
    
    return theta


    
def make_adjacency_matrix(index_names, parents):
        nVertex = len(index_names)
        name2idx = {name: i for i,name in enumerate(index_names)}

        ## Build Graph data structures
        # Adjacency matrix
        adj = np.zeros((nVertex, nVertex), dtype=int)
        for i_name in parents.keys():
            i = name2idx[i_name]
            for m_name in parents[i_name]:
                j = name2idx[m_name]
                adj[i, j] = 1

        return adj

def make_families(index_names, parents):
    nVertex = len(index_names)
    adj = make_adjacency_matrix(index_names, parents)
    # Possibly check topological ordering
    # toposort(adj)
    
    # Family, Parents and Children
    fa = [[]]*nVertex
    #pa = [[]]*nVertex
    #ch = [[]]*nVertex
    for n in range(nVertex):
        p = find(adj[n,:])
        #pa[n] = p
        fa[n] = [n]+p
        #c = find(adj[:,n])
        #ch[n] = c
    
    return fa

def permute_table(index_names, cardinalities, visible_names, X):
    '''
    Given a network with index_names and cardinalities, reshape a table X with 
    the given order as in visible_names so that it fits the storage convention of BNJNB.
    '''
    
    nums = index_names_to_num(index_names, visible_names)
    osize = [cardinalities[n] for n in nums]
    idx = list(nums)
    shape = clique_shape(cardinalities,idx)
    return np.einsum(X, idx, sorted(idx)).reshape(shape)


def make_cliques(families, cardinalities, visibles=None, show_graph=False):
    '''
        Builds the set of cliques of a triangulated graph.
    '''
    N = len(families)
        
    if visibles:
        C = families+[visibles]
    else:
        C = families

    # Moral Graph
    MG = np.zeros((N, N)) 

    for F in C:
        for edge in combinations(F,2):
            MG[edge[0], edge[1]] = 1  
            MG[edge[1], edge[0]] = 1  

#    if show_graph:
#        show_ug_image(MG,imstr='MG.png')


    elim = []
    Clique = []
    visited = [False]*N

    # Find an elimination sequence
    # Based on greedy search 
    # Criteria, select the minimum induced clique size
    for j in range(N):

        min_clique_size = np.inf
        min_idx = -1
        for i in range(N):
            if not visited[i]:
                neigh = find(MG[i,:])
                nm = np.prod(clique_shape(cardinalities, neigh+[i]))

                if min_clique_size > nm:
                    min_idx = i
                    min_clique_size = nm

        neigh = find(MG[min_idx,:])
        temp = set(neigh+[min_idx])

        is_subset = False
        for CC in Clique:
            if temp.issubset(CC):
                is_subset=True
        if not is_subset:
            Clique.append(temp)

        # Remove the node from the moral graph
        for edge in combinations(neigh,2):
            MG[edge[0], edge[1]] = 1
            MG[edge[1], edge[0]] = 1

        MG[min_idx,:] = 0
        MG[:, min_idx] = 0
        elim.append(min_idx)
        visited[min_idx] = True
#        if show_graph:
#            show_ug_image(MG,imstr='MG'+str(j)+'.png')

    return Clique, elim   

def topological_order(index_names, parents):
    """
    returns a topological ordering of the graph 
    """
    adj = make_adjacency_matrix(index_names, parents)
    nVertex = len(index_names)
    indeg = np.sum(adj, axis = 1)
    zero_in = find(indeg==0)
    topo_order = []
    while zero_in:
        n = zero_in.pop(0)
        topo_order.append(n)
        for j in find(adj[:,n]):
            indeg[j] -= 1
            if indeg[j] == 0:
                zero_in.append(j)
                
    if len(topo_order)<nVertex:
        return []
    else:
        return topo_order
    
    
## Spanning tree and graph traversal
        
def mst(E, N):
    """
    Generate a Spanning Tree of a graph with N nodes by Kruskal's algorithm, 
    given preordered edge set E with each edge as (weight, v1, v2)
    
    For a minimum spanning tree, use
    E.sort()
    mst(E, N)
    
    For a maximum spanning tree, use
    E.sort(reverse=True)
    mst(E, N)
    """
    
    parent = list(range(N))
    spanning_tree = {i:[] for i in range(N)}

    def find_v(vertex):
        v = vertex
        while parent[v] != v:
            v = parent[v]
        return v

    def union(v1, v2):
        root1 = find_v(v1)
        root2 = find_v(v2)
        if root1 != root2:
            parent[root2] = root1
    
    for edge in E:
        weight, v1, v2 = edge
        p1, p2 = find_v(v1), find_v(v2)
        if p1 != p2:
            union(p1, p2)
            spanning_tree[v1].append(v2)
            spanning_tree[v2].append(v1)
            
    return spanning_tree

def bfs(adj_list, root):
    """
        Breadth-first search starting from the root
        
        adj_list : A list of lists where adj_list[n] denotes the set of nodes that can be reached from node n
        
        Returns a BFS order, and a BFS tree as an array parent[i] 
        The root node has parent[rootnode] = -1
    """
    N = len(adj_list)
    visited = [False]*N
    parent = [-1]*N
    
    queue = [root]
    order = []
    while queue:
        v = queue.pop(0)
        if not visited[v]:
            visited[v] = True
            for w in adj_list[v]:
                if not visited[w]:
                    parent[w] = v
                    queue.append(w) 
            order.append(v)
            

    return order, parent

def is_leaf(i, parent):
    return  not (i in parent)

def is_root(i, parent):
    return parent[i] == -1

def make_list_receive_from(parent):
    lst = [[] for i in range(len(parent)) ]
    for i,p in enumerate(parent):
        if p!= -1:
            lst[p].append(i)
    
    return lst

## Samplers
    
def sample_indices(index_names, parents, cardinalities, theta, num_of_samples=1):
    '''
    Sample directly the indices given a Bayesian network
    '''
    N = len(index_names)
    order = topological_order(index_names, parents)
    X = []

    for count in range(num_of_samples):
        x = [[]]*N
        for n in order:
            varname = index_names[n] 

            idx = index_names_to_num(index_names,parents[varname])

            j = [0]*N
            for i in idx:
                j[i] = x[i]

            I_n = cardinalities[n]
            j[n] = tuple(range(I_n))
            #print(j)
            #print(theta[n][j])
            x[n] = np.random.choice(I_n, p=theta[n][j].flatten())
            #print(x)        
        X.append(x)        
    return X


def sample_states(var_names, states, index_names, parents, theta, num_of_samples=1):
    """
    Returns a dict with keys as state_name tuples and values as counts.
    This function generates each sample separately, so if 
    num_of_samples is large, consider using sample_counts
    
    """
    N = len(index_names)
    order = topological_order(index_names, parents)
    
    X = dict()
    nums = index_names_to_num(index_names,var_names)
    cardinalities = cardinalities_from_states(index_names, states)
    
    shape = clique_shape(cardinalities, nums)
    
    for count in range(num_of_samples):
        x = [[]]*N
        for n in order:
            varname = index_names[n] 

            idx = index_names_to_num(index_names,parents[varname])

            j = [0]*N
            for i in idx:
                j[i] = x[i]

            I_n = cardinalities[n]
            j[n] = tuple(range(I_n))
            #print(j)
            #print(theta[n][j])
            x[n] = np.random.choice(I_n, p=theta[n][j].flatten())
            #print(x)    
    
        key = tuple((states[index_names[n]][x[n]] for n in nums))
        X[key] = X.get(key, 0) + 1
    
    return X



def counts_to_table(var_names, ev_counts, index_names, states):
    """
    Given observed variables names as var_names and
    observations as key-value pairs {state_configuration: count} 
    create a table of counts.
    
    A state configuration is a tuple (state_name_0, ..., state_name_{K-1})
    where K is the lenght of var_names, and state_name_k is a state
    from states[var_names[k]]
    """
    var_nums = list(index_names_to_num(index_names, var_names))
    cardinalities = cardinalities_from_states(index_names, states)
    shape = clique_shape(cardinalities, var_nums)
    C = np.zeros(shape=shape)
    N = len(index_names)

    for rec in ev_counts.keys():
        conf = [0]*N
        for key, val in zip(var_names, rec):
            s = states[key].index(val)
            n = index_names_to_num(index_names, [key])[0]

            conf[n] = s

        #print(conf)
        # Final value is the count that the pair is observed
        C[tuple(conf)] += ev_counts[rec]
        
    return C 

def table_to_counts(T, var_names, index_names, states, clamped=[], threshold = 0):
    """
    Convert a table on index_names clamped on setdiff(index_names, var_names)
    to a dict of counts. Keys are state configurations. 
    """
    var_nums = list(index_names_to_num(index_names, var_names))
    M = len(index_names)
    
    ev_count = {}
    for u in zip(*(T>threshold).nonzero()):
        if not clamped:
            key = tuple((states[v][u[var_nums[i]]] for i,v in enumerate(var_names)))
        else:
            key = tuple((states[v][u[var_nums[i]]] if clamped[var_nums[i]] is None else states[v][clamped[var_nums[i]]] for i,v in enumerate(var_names)))
        ev_count[key] = T[u]
        
    return ev_count

## the inference Engine
    
def clamped_pot(X, ev_states):
    """
        Returns a subslice of a table. Used for clamping conditional probability tables 
        to a given set of evidence. 
        
        X: table
        ev_states: list of clamped states ev_states[i]==e (use None if not clamped)
    """
    
    #                  var is clamped,              var not clamped
    #                  ev_states[i]==e           ev_states[i]==None
    # var is member    idx[i] = e                   idx[i] = slice(0, X.shape[i])
    # var not member   idx[i] = None                idx[i] = slice(0, X.shape[i])
    
    card = list(X.shape)
    N = len(card)    
    idx = [[]]*N

    for i,e in enumerate(ev_states):
        if e is None and X.shape[i]>1: # the variable is unclamped or it is not a member of the potential
            idx[i] = slice(0, X.shape[i])
        else:
            if X.shape[i]==1:
                idx[i] = 0
            else:
                idx[i] = e
                card[i] = 1
    return X[tuple(idx)].reshape(card)


def multiply(theta, idx): 
    """Multiply a subset of a given list of potentials"""
    par = [f(n) for n in idx for f in (lambda n: theta[n], lambda n: range(len(theta)))]+[range(len(theta))]    
    return np.einsum(*par)

def condition_and_multiply(theta, idx, ev_states): 
    """Multiply a subset of a given list of potentials"""
    par = [f(n) for n in idx for f in (lambda n: clamped_pot(theta[n], ev_states), lambda n: range(len(theta)))]+[range(len(theta))]    
    return np.einsum(*par)

def marginalize(Cp, idx, cardinalities):
    return np.einsum(Cp, range(len(cardinalities)), [int(s) for s in sorted(idx)]).reshape(clique_shape(cardinalities,idx))


class Engine():
    def __init__(self, index_names, parents, states, theta, visible_names=[]):
        
        self.states = states
        cardinalities = [len(states[a]) for a in index_names]
        
        families = make_families(index_names, parents)        
        self.cardinalities = cardinalities
        self.index_names = index_names
        
        visibles = index_names_to_num(index_names, visible_names)
        self.Clique, self.elim = make_cliques(families, cardinalities, visibles)

        
        # Assign each conditional Probability table to one of the Clique potentials
        # Clique2Pot is the assignments
        self.Pot = families
        self.Clique2Pot = np.zeros((len(self.Clique), len(self.Pot)))
        selected = [False]*len(self.Pot)
        for i,c in enumerate(self.Clique):
            for j,p in enumerate(self.Pot):
                if not selected[j]:
                    self.Clique2Pot[i,j] = set(p).issubset(c)
                    if self.Clique2Pot[i,j]: 
                        selected[j] = True

        # Find the root clique
        # In our case it will be the one where all the visibles are a subset of
        self.RootClique = -1
        for i,c in enumerate(self.Clique):
            if set(visibles).issubset(c):
                self.RootClique = i
                break

        # Build the junction graph and compute a spanning tree
        junction_graph_edges = []
        for i,p in enumerate(self.Clique):
            for j,q in enumerate(self.Clique):
                ln = len(p.intersection(q))
                if i<j and ln>0:
                    junction_graph_edges.append((ln,i,j)) 
        junction_graph_edges.sort(reverse=True)
        self.mst = mst(junction_graph_edges, len(self.Clique))
        self.order, self.parent = bfs(self.mst, self.RootClique)
        self.receive_from = make_list_receive_from(self.parent)

        self.visibles = visibles
        
        # Setup the data structures for the Junction tree algorithm
        self.SeparatorPot = dict()
        self.CliquePot = dict()
        self.theta = theta
        self.cardinalities_clamped = []
    
    def propagate_observation(self, observed_configuration={}):
        
        ev_names = list(observed_configuration.keys())
        observed_states = [self.states[nm].index(observed_configuration[nm]) for nm in ev_names]
        
        nums = index_names_to_num(self.index_names, ev_names)
        #cardinalities_clamped = self.cardinalities.copy()
        cardinalities_clamped = [1 if i in nums else c for i,c in enumerate(self.cardinalities)]
        ev_states = [None]*len(self.cardinalities)
        for i,e in zip(nums, observed_states):
            ev_states[i] = e
        
        # Collect stage
        for c in reversed(self.order):
            self.CliquePot[c] = np.ones(clique_shape(cardinalities_clamped, self.Clique[c]))
            for p in self.receive_from[c]:
                self.CliquePot[c] *= self.SeparatorPot[(p,c)]

            # Prepare Clique Potentials 
            # Find probability tables that need to be multiplied into 
            # the Clique potential 
            idx = find(self.Clique2Pot[c, :])
            if idx:
                #print(idx)
                #print(ev_states)
                self.CliquePot[c] *= condition_and_multiply(self.theta, idx, ev_states) 

            # Set the separator potential
            if not is_root(c, self.parent):
                idx = self.Clique[self.parent[c]].intersection(self.Clique[c])
                self.SeparatorPot[(c,self.parent[c])] = marginalize(self.CliquePot[c], idx, cardinalities_clamped)

        # Distribution Stage
        for c in self.order[1:]:
            idx = self.Clique[self.parent[c]].intersection(self.Clique[c])
            self.CliquePot[c] *= marginalize(self.CliquePot[self.parent[c]], idx, cardinalities_clamped)/self.SeparatorPot[(c,self.parent[c])]        
        
        self.cardinalities_clamped = cardinalities_clamped
        self.values_clamped = ev_states
        
    def propagate_table(self, X=None):
        # Reset 
        self.values_clamped = [None]*len(self.cardinalities)
        # Collect stage
        for c in reversed(self.order):
            self.CliquePot[c] = np.ones(clique_shape(self.cardinalities, self.Clique[c]))
            for p in self.receive_from[c]:
                self.CliquePot[c] *= self.SeparatorPot[(p,c)]

            # Prepare Clique Potentials 
            # Find probability tables that need to be multiplied into 
            # the Clique potential 
            idx = find(self.Clique2Pot[c, :])
            if idx:
                self.CliquePot[c] *= multiply(self.theta, idx) 

            # Set the separator potential
            if not is_root(c, self.parent):
                idx = self.Clique[self.parent[c]].intersection(self.Clique[c])
                self.SeparatorPot[(c,self.parent[c])] = marginalize(self.CliquePot[c], idx, self.cardinalities)

        if X is not None:    
            SepX = marginalize(self.CliquePot[self.RootClique], self.visibles, self.cardinalities)
            # Note: Take care of zero divide
            self.CliquePot[self.RootClique] *= X/SepX

        # Distribution Stage
        for c in self.order[1:]:
            idx = self.Clique[self.parent[c]].intersection(self.Clique[c])
            self.CliquePot[c] *= marginalize(self.CliquePot[self.parent[c]], idx, self.cardinalities)/self.SeparatorPot[(c,self.parent[c])]
        
#    def propagate(self, ev_names=[],ev_counts=None):
#               
#        if ev_names:
#            X = evidence_to_table(ev_names, ev_counts, self.index_names, self.cardinalities, self.states)
#        else:
#            X = None

    def compute_ESS(self, X=[]):
        """Compute Expected Sufficient Statistics for each probability table"""
        E_S = dict()
        self.propagate_table(X)
        for c in self.order:
            for n in find(self.Clique2Pot[c, :]):
                E_S[n] = marginalize(self.CliquePot[c], self.Pot[n], self.cardinalities)
        return E_S
    
    def compute_marginal(self, var_names, normalization=False):
        """
        Compute a marginal table on variables in var_names 
        if the variables are the subset of a clique, otherwise returns None.
        var_names can be forced to be a subset of a clique by specifying
        Engine(..., visible_names=var_names) 
        """

        var_indices = index_names_to_num(self.index_names, var_names)
        idx = set(var_indices)
        j = None
        for c in self.order:
            if idx.issubset(self.Clique[c]):
                j = c
                break
    
        if j is not None:
            if self.cardinalities_clamped:
                if normalization:
                    return normalize(marginalize(self.CliquePot[j], var_indices, self.cardinalities_clamped))               
                else:
                    return marginalize(self.CliquePot[j], var_indices, self.cardinalities_clamped)                
            else:
                if normalization:
                    return normalize(marginalize(self.CliquePot[j], var_indices, self.cardinalities))                    
                else:
                    return marginalize(self.CliquePot[j], var_indices, self.cardinalities)
        else:
            print('Desired marginal is not a subset of any clique')
            return None
        
    def singleton_marginals(self, var_names, normalization=False):
        """ For each variable in var_names compute its marginal """
        L = {}
        var_indices = index_names_to_num(self.index_names, var_names)
        for j, v in enumerate(var_names):
            marg = self.compute_marginal([v])
            if normalization:
                marg = normalize(marg)
            if self.values_clamped[var_indices[j]] is None:
                L[v] = {self.states[v][i]: p for i,p in enumerate(marg.flatten())}
            else:
                L[v] = {self.states[v][self.values_clamped[var_indices[j]]]: p for i,p in enumerate(marg.flatten())}
            
        return L
        
            
    def sample_table(self, var_names, num_of_samples=1):
        #self.propagate_observation({})
        P = self.compute_marginal(var_names)
        if P is not None:
            return np.random.multinomial(num_of_samples, P.flatten()).reshape(P.shape)
        else:
            return None
        
    def marginal_table(self, marg_names, normalization=False):
        clamped = self.values_clamped
        return table_to_counts(self.compute_marginal(marg_names, normalization), marg_names, self.index_names, self.states, clamped)
