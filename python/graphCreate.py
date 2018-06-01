from pdb2sql.pdb2sql import pdb2sql
from pdb2sql.interface import interface
from Bio import pairwise2
import numpy as np
import pickle

class GraphPSSM():
    def __init__(self,nodes_pssm_data,edges_index):
        self.nodes_pssm_data = nodes_pssm_data[2:23]
        self.nodes_info_data = nodes_pssm_data[-2]
        self.edges_index = edges_index


class graphCreate():

    def __init__(self,pdbfile,pssmfile):

        self.pdbfile = pdbfile
        self.pdb = pdb2sql(self.pdbfile)

        self.resmap = {
        'A' : 'ALA', 'R' : 'ARG', 'N' : 'ASN', 'D' : 'ASP', 'C' : 'CYS', 'E' : 'GLU', 'Q' : 'GLN',
        'G' : 'GLY', 'H' : 'HIS', 'I' : 'ILE', 'L' : 'LEU', 'K' : 'LYS', 'M' : 'MET', 'F' : 'PHE',
        'P' : 'PRO', 'S' : 'SER', 'T' : 'THR', 'W' : 'TRP', 'Y' : 'TYR', 'V' : 'VAL',
        'B' : 'ASX', 'U' : 'SEC', 'Z' : 'GLX'
        }
        self.resmap_inv = {v: k for k, v in self.resmap.items()}
        self.pssm = {}
        for chain in ['A','B']:
            self.pssm[chain] = self.read_PSSM_data(pssmfile[chain])

    def read_PSSM_data(self,fname):
        """Read the PSSM data."""

        f = open(fname,'r')
        data = f.readlines()
        f.close()

        filters = (lambda x: len(x.split())>0, lambda x: x.split()[0].isdigit())
        return list(map(lambda x: x.split(),list(filter(lambda x: all(f(x) for f in filters), data))))

    def align_sequences(self):

        self.seq_aligned = {'pdb':{},'pssm':{}}
        for chain in ['A','B']:
            pdb_seq = self._get_sequence(chain=chain)
            pssm_seq = ''.join( [data[1] for data in self.pssm[chain] ] )
            self.seq_aligned['pdb'][chain], self.seq_aligned['pssm'][chain] = self._get_aligned_seq(pdb_seq,pssm_seq)

    def get_aligned_pssm(self):

        self.align_sequences()

        self.aligned_pssm = {}
        for chain in ['A','B']:

            iResPDB,iResPSSM = 0,0
            pdbres = [(numb,name) for numb,name in self.pdb.get('resSeq,resName',chainID=chain)]
            pdbres = [v for v in dict(pdbres).items()]

            for resPDB,resPSSM in zip(self.seq_aligned['pdb'][chain], self.seq_aligned['pssm'][chain]):

                if resPSSM == '-' and resPDB != '-':
                    self.aligned_pssm[(chain,)+pdbres[iResPDB]] = None
                    iResPDB += 1

                if resPSSM != '-' and resPDB == '-':
                    iResPSSM += 1

                if resPSSM != '-' and resPDB != '-':
                    self.aligned_pssm[(chain,)+pdbres[iResPDB]] = self.pssm[chain][iResPSSM] #[list(range(2,23))+[43]]
                    iResPDB += 1
                    iResPSSM += 1

        for k,v in self.aligned_pssm.items():
            print(k,v)

    def _get_sequence(self,chain='A'):
        data = [(numb,self.resmap_inv[name]) for numb,name in self.pdb.get('resSeq,resName',chainID=chain)]
        return ''.join([v[1] for v in dict(data).items()])

    @staticmethod
    def _get_aligned_seq(seq1, seq2):
        """Align two sequnces using global alignment and return aligned sequences.
            Paramters of global alignment:
                match: 1
                mismtach: 0
                gap open: -2
                gap extend: -1

        Arguments:
            seq1 {str} -- 1st sequence.
            seq2 {str} -- 2nd sequence.

        Returns:
            [numpy array] -- seq1_ali, aligned sequence for seq1
            [numpy array] -- seq2_ali, aligned sequence for seq1
        """

        ali = pairwise2.align.globalxs(seq1, seq2, -2, -1)
        seq1_ali = np.array([i for i in ali[0][0]])
        seq2_ali = np.array([i for i in ali[0][1]])

        return seq1_ali, seq2_ali

    def construct_graph(self):

        db = interface(self.pdbfile)
        res_contact_pairs = db.get_contact_residues(cutoff = 6.0, return_contact_pairs=True)

        nodesB = []
        for k,v in res_contact_pairs.items():
            nodesB += v
        nodesB = sorted(set(nodesB))

        self.nodes = list(res_contact_pairs.keys()) + nodesB
        self.edges = []
        for key,val in res_contact_pairs.items():
            ind1 = self.nodes.index(key)
            for v in val:
                ind2 = self.nodes.index(v)
                self.edges.append([ind1,ind2])

    def export_graph(self,fname):
        nodes_pssm_data = []
        for res in self.nodes:
            pssm = self.aligned_pssm[res]
            nodes_pssm_data.append(pssm)
        graph = GraphPSSM(nodes_pssm_data,self.edges)
        pickle.dump(graph,open(fname,'wb'))

if __name__ == "__main__":

    pdb = './example_input/1E96_1w.pdb'
    pssm = {'A':'./example_input/1E96.A.pssm','B':'./example_input/1E96.B.pssm'}

    g = graphCreate(pdb,pssm)
    g.get_aligned_pssm()

    for chain in ['A','B']:
        for x,y in zip(g.seq_aligned['pdb'][chain],g.seq_aligned['pssm'][chain]):
            print(chain,x,y)

    g.construct_graph()
    g.export_graph('test.pkl')