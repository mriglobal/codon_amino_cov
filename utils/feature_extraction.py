import pandas as pd
import numpy as np
from Bio.Seq import Seq
from sklearn.base import BaseEstimator, TransformerMixin
from collections import Counter, defaultdict
from itertools import product


class CodonFrequencies(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.codons = set([''.join(p) for p in product('GTCA',repeat=3)])
        self.empty_counts = Counter({c:0 for c in self.codons})
        self.features = np.array(sorted(list(self.codons)))
    def fit(self, X, y=None):
        """Function for codon frequency counts for a collection of CDSs. Input should be a list of strings.
        """

        return self
    def transform(self,X):
        counts = []
        for f in X:
            codons = self.empty_counts.copy()
            if len(f)%3 == 0:
                codons.update(Counter([f[r:r+3] for r in range(0,len(f),3) if f[r:r+3] in self.codons]))
            else:
                raise Error("CDS sequence not divisible by three.")
            counts.append(codons)
        codon_data = pd.DataFrame(counts,columns=self.features)
        #codon_data = codon_data.reindex(sorted(codon_data.columns),axis=1)
        return codon_data.values

class AminoFrequencies(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.amino_set = set('ARNDCQEGHILKMFPSTWYV')
        self.empty_counts = Counter({c:0 for c in self.amino_set})
    def fit(self, X, y=None):
        """Function for amino acid frequency counts for a collection of CDSs. Input should be a list of strings.
        """

        return self
    def transform(self,X):
        counts = []
        for f in X:
            aminos = self.empty_counts.copy()
            if len(f)%3 == 0:
                aminos.update(Counter([a for a in str(Seq(f).translate()) if a in self.amino_set]))
            else:
                raise Error("CDS sequence not divisible by three.")
            
            counts.append(aminos)
        amino_data = pd.DataFrame(counts)
        amino_data = amino_data.reindex(sorted(amino_data.columns),axis=1)
        self.features = amino_data.columns.values
        return amino_data.values

class RSCUfeatures(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.codons = set([''.join(p) for p in product('GTCA',repeat=3)])
        self.codon_table = {
        'TTT': 'F', 'TTC': 'F', 'TTA': 'L', 'TTG': 'L',
        'CTT': 'L', 'CTC': 'L', 'CTA': 'L', 'CTG': 'L',
        'ATT': 'I', 'ATC': 'I', 'ATA': 'I', 'ATG': 'M',
        'GTT': 'V', 'GTC': 'V', 'GTA': 'V', 'GTG': 'V',
        'TCT': 'S', 'TCC': 'S', 'TCA': 'S', 'TCG': 'S',
        'CCT': 'P', 'CCC': 'P', 'CCA': 'P', 'CCG': 'P',
        'ACT': 'T', 'ACC': 'T', 'ACA': 'T', 'ACG': 'T',
        'GCT': 'A', 'GCC': 'A', 'GCA': 'A', 'GCG': 'A',
        'TAT': 'Y', 'TAC': 'Y', 'TAA': '*', 'TAG': '*',
        'CAT': 'H', 'CAC': 'H', 'CAA': 'Q', 'CAG': 'Q',
        'AAT': 'N', 'AAC': 'N', 'AAA': 'K', 'AAG': 'K',
        'GAT': 'D', 'GAC': 'D', 'GAA': 'E', 'GAG': 'E',
        'TGT': 'C', 'TGC': 'C', 'TGA': '*', 'TGG': 'W',
        'CGT': 'R', 'CGC': 'R', 'CGA': 'R', 'CGG': 'R',
        'AGT': 'S', 'AGC': 'S', 'AGA': 'R', 'AGG': 'R',
        'GGT': 'G', 'GGC': 'G', 'GGA': 'G', 'GGG': 'G'
    }
        self.empty_counts = Counter({c:0 for c in self.codons})
    def fit(self, X, y=None):
        """Function for relative synonymous codon frequency for a collection of CDSs. Input should be a list of strings.
        """

        return self
    def transform(self,X):
        rscu = []
        for f in X:
            codons = self.empty_counts.copy()
            if len(f)%3 == 0:
                codons.update(Counter([f[r:r+3] for r in range(0,len(f),3) if f[r:r+3] in self.codons]))
            else:
                raise Error("CDS sequence not divisible by three.")
            amino_acid_totals = defaultdict(int)
            for codon, aa in self.codon_table.items():
                amino_acid_totals[aa] += codons[codon]
            rscu_values = self.empty_counts.copy()
            for codon in self.codon_table:
                aa = self.codon_table[codon]
                total_codons_for_aa = amino_acid_totals[aa]
                if total_codons_for_aa > 0:
                    expected = total_codons_for_aa / len([c for c in self.codon_table if self.codon_table[c] == aa])
                    rscu_values[codon] = codons[codon] / expected
            rscu.append(rscu_values)
        codon_data = pd.DataFrame(rscu).fillna(0.0)
        codon_data = codon_data.reindex(sorted(codon_data.columns),axis=1)
        self.features = codon_data.columns.values
        return codon_data.values

