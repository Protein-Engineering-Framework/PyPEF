import os

class Aminoacid_Index:
    def __init__(self, file): 
        self.file = file
        self._d = {'H': None, 'D': None, 'R': None, 'A': None,
                  'T': None, 'J': None, '*': None, 'C': None,
                  'I': None}
        
        self._complete = True
        
        with open(self.file, 'r') as in_file:
            k_last = str()
            for line in list(in_file)[:-1]:
                try:
                    k, v = line.split(" ", 1)
                
                except ValueError: # no data for entry
                    pass
                
                if k:
                    self._d.update({k: v})
                    k_last = k
                else:
                    v = "".join([self._d[k_last], v])
                    self._d.update({k_last: v})


        data = [[], []]
        for line_idx, line in enumerate(self._d['I'].split('\n')):            
            if line_idx < 1:
                keys = [[], []]                
                for k in line.split(' '):
                    if k:
                        [keys[i].append(l) for i, l in enumerate(k.split('/'))]
                data[0].extend(keys[0] + keys[1])
                                
            else:
                try:
                    data[1].extend([float(v) for v in line.split(' ') if v])
                
                except ValueError: # due to 'NA'
                    self._complete = False
        
        self._da = {}
        for k, v in zip(data[0], data[1]):
            self._da.update({k:v})
    
    @property
    def is_complete(self):
        return self._complete
    
    @property
    def accession_number(self):
        return self._d['H'].rstrip()

    @property
    def data_description(self):
        return self._d['D'].rstrip()
        
    @property
    def pmid(self):
        return self._d['R'].rstrip()
        
    @property
    def author(self):
        return self._d['A'].rstrip()
        
    @property
    def title_of_the_article(self):
        return self._d['T'].rstrip()
    
    @property
    def journal_reference(self):
        return self._d['J'].rstrip()
    
    @property
    def comment(self):
        return self._d['*'].rstrip()
        
    @property
    def accession_number_of_similar_entries(self):
        return self._d['C'].rstrip()
    
    @property
    def amino_acid_index_data(self):
        return self._d['I'] 
        
    @property
    def value(self):
        return self._da
    
    @property
    def values(self):
        return [float(v) for v in self._da.values()]
        
    @property
    def keys(self):
        return [k for k in self._da.keys()]
    
Database = { file.split('.')[0] : Aminoacid_Index(os.path.join('aaidx', file)) for file in os.listdir('aaidx') }
