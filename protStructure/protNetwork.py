import numpy as np
import itertools
import operator
import csv
import random
import networkx as nx
from scipy.spatial.distance import pdist, squareform
from node2vec import Node2Vec
import sys
from matplotlib import pyplot as plt

# ATOM_FORMAT defines fixed-width ATOM field
# {fname: (start, end, format)}
# start/end is 1-based and inclusive

ATOM_FORMAT = {'ATOM': (1, 6, str),
               'No': (7, 11, int),
               'Name': (13, 16, str),
               'Alternate Location': (17, 17, str),
               'Residue Name': (18, 20, str),
               'Chain': (22, 22, str),
               'Residue No': (23, 26, int),
               'Insertion': (27, 27, str),
               'x': (31, 38, float),
               'y': (39, 46, float),
               'z': (47, 54, float),
               'Occupancy': (55, 60, float),
               'B Factor': (61, 66, float),
               'Element': (77, 78, str),
               'Charge': (79, 80, str)}

def parse_atom(atom_line):
    return {field_name:field_format(atom_line[start-1:end].strip()) for
                field_name, (start, end, field_format) in ATOM_FORMAT.items()}


def read_pdb(fname):
    # properties of atom in the final output
    atom_properties = ['Name', 'No', 'x', 'y', 'z']

    # read atoms to a list
    with open(fname) as pdb_file:
        atoms = []
        for line in pdb_file:
            if not line.startswith('ATOM'):
                continue
            atoms.append(parse_atom(line))

    # group chains
    chains = []
    for chain, chain_group in itertools.groupby(atoms, operator.itemgetter('Chain')):
        # group residues
        residues = []
        for (residue_no, residue_name), residue_group in itertools.groupby(chain_group,
                                                                           operator.itemgetter('Residue No',
                                                                                               'Residue Name')):
            residues.append({'No': residue_no,
                             'Name': residue_name,
                             'Atoms': [{key: atom[key] for key in atom_properties}
                                       for atom in residue_group]})
        chains.append({'Name': chain,
                       'Residues': residues})
    return chains

def coarse_chain(pdb, chain_id=0):
    try:
        chain = pdb[chain_id]
    except TypeError:
        for chain in pdb:
            if chain['Name'] == chain_id:
                break
    coarse = []
    for residue in chain['Residues']:
        sidechain_no = 4 if residue['Name'] != 'GLY' and (len(residue["Atoms"]) >5) else 1
        #if len(residue["Atoms"])<4 and residue['Name'] != 'GLY':
        print( len(residue["Atoms"]))
        print(sidechain_no)
        coarse.append(
            {'Backbone': (residue['Atoms'][1]['x'],
                     residue['Atoms'][1]['y'],
                     residue['Atoms'][1]['z']), # Calpha
             'Sidechain': (residue['Atoms'][sidechain_no]['x'],
                     residue['Atoms'][sidechain_no]['y'],
                     residue['Atoms'][sidechain_no]['z']), # Cbeta
             'Name': residue['Name'],
             'No': residue['No']})
    return coarse


def create_network(points, cutoff):
    dm = squareform(pdist(points))
    s = range(len(points))
    g = nx.Graph()
    g.add_nodes_from(range(len(points)))
    a, b = np.meshgrid(s, s)
    sdm = dm[a, b]

    for x, y in zip(*np.where(sdm < cutoff)):
        if x != y:
            g.add_edge(x, y)
    return g


def degree(g):
    return [v for k, v in sorted(nx.degree(g))]

def clustering(g):
    return [v for k, v in sorted(nx.clustering(g).items())]

def knn(g):
    return [v for k, v in sorted(nx.average_neighbor_degree(g).items())]

def length(g):
    l = []
    for i, lengths in nx.shortest_path_length(g):
        a = lengths.values()
        b = np.sum(a)
        l.append(sum(lengths.values())/float(len(g)-1))
    return l

def spectrum(g):
    return np.linalg.eigvalsh(nx.normalized_laplacian_matrix(g).todense())


def pdb_to_network(fname, cutoff, chain_id=0, use_cbeta=False):
    pdb = read_pdb(fname)
    chain = coarse_chain(pdb, chain_id)
    coord_key = 'Sidechain' if use_cbeta else 'Backbone'
    points = [residue[coord_key] for residue in chain]
    return chain,create_network(points, cutoff)

def n2v(gr):
    node2vec = Node2Vec(gr, dimensions=64, walk_length=30, num_walks=200, workers=1)

    # Embed nodes
    model = node2vec.fit(window=10, min_count=1,
                         batch_words=4)  # Any keywords acceptable by gensim.Word2Vec can be passed, `diemnsions` and `workers` are automatically passed (from the Node2Vec constructor)

    # Look for most similar nodes
    model.wv.most_similar('2')  # Output node names are always strings

    # Save embeddings for later use
    model.wv.save_word2vec_format("1t29.emb")

ch,n = pdb_to_network('1t29.pdb', 6.7) # Use first chain, create from Calphas
# n = pdb_to_network('1AKE.pdb', 6.7, use_cbeta=True) # Use first chain, create from Cbetas
# n = pdb_to_network('1AKE.pdb', 6.7, chain_id='B')  # Use chain with name B
# n = pdb_to_network('1AKE.pdb', 6.7, chain_id=1, use_cbeta=True) # Use second chain (0-index)

nx.draw(n)
plt.show()

n2v(n)
nx.write_edgelist(n, "1t29.edgelist")
with open('1t29-R.csv', 'w', newline='') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=',', )
    for resNo in range(len(ch)):
        l = []
        l.append(resNo)
        l.append(ch[resNo]["Name"])
        l.append(ch[resNo]["No"])
        spamwriter.writerow(l)




#plt.interactive(True)
#plt.hist(degree(n), bins=range(15))
#plt.hist(clustering(n), bins=20)
#plt.hist(knn(n))
#plt.hist(length(n))
#plt.hist(spectrum(n), bins=20)
#plt.show()
print("done")