import torch
import numpy as np
cimport numpy as np
import itertools
from cython.parallel import prange
from cython.parallel import parallel
import cython
import torch
import numpy as np
cimport numpy as cnp
cimport cython
from libc.math cimport exp
from cython cimport view

# Define numpy data types
ctypedef np.int64_t DTYPE_INT
ctypedef np.float64_t DTYPE_FLOAT
cdef class ProbTable:
    cdef public object clique
    cdef public object domains
    cdef public object table

    def __cinit__(self, object clique, object domains, object table):
        self.clique = clique
        self.domains = domains
        self.table = table

    def __str__(self):
        return f"Clique: {self.clique}, \nDomains: {self.domains}, \nTable: {self.table}"

    cpdef object to_tensor(self, object device):
        self.clique = torch.tensor(self.clique, dtype=torch.int64, device=device)
        self.domains = torch.tensor(self.domains, dtype=torch.int64, device=device)
        self.table = torch.tensor(self.table, dtype=torch.float32, device=device)

    cpdef object instantiate_evidence(self, object evidence):
        """
        Instantiate evidence for the factor.
        
        :param evidence: A tensor of evidence values for all variables.
                        Use -1 for variables that are not evidence.
        :return: A new ProbTable with the instantiated factor.
        """
        cdef object new_table, mask
        cdef int i, ev
        # Ensure all tensors are on the same device

        # Create a mask tensor initialized with True
        mask = torch.ones_like(self.table, dtype=torch.bool)

        # Create the mask based on evidence
        for i, var in enumerate(self.clique):
            ev = evidence[var].item()
            if ev != -1:  # If this variable is evidence
                # Create a slice object for this dimension
                slices = [slice(None)] * len(self.domains)
                slices[i] = 1 - ev  # Mask out the non-evidence value
                mask[tuple(slices)] = False

        # Apply the mask to create the new table
        new_table = torch.exp(self.table.clone())
        new_table[~mask] = 0  # Set to 0 in probability space

        return ProbTable(self.clique, self.domains, new_table)

    def inst_evid_remove_values(self, object evidence):
        """
        Instantiate evidence for the factor, removing variables, domains, and values for instantiated evidence.
        
        :param evidence: A tensor of evidence values for all variables.
                        Use -1 for variables that are not evidence.
        :return: A new ProbTable with the instantiated factor.
        """
        cdef list new_clique = []
        cdef list new_domains = []
        cdef list evidence_indices = []
        cdef object f_clique
        cdef object f_domains
        cdef int i, ev

        # Identify variables to keep and their corresponding domains
        for i, var in enumerate(self.clique):
            ev = evidence[var].item()
            if ev == -1:  # If this variable is not evidence
                new_clique.append(var.item())
                new_domains.append(self.domains[i].item())
            else:
                evidence_indices.append((i, ev))

        # Sort evidence_indices in descending order of dimension
        evidence_indices.sort(key=lambda x: x[0], reverse=True)

        # Create a new table by selecting the appropriate slices
        new_table = self.table.clone()
        for dim, val in evidence_indices:
            new_table = torch.index_select(new_table, dim, torch.tensor([val], device=new_table.device))

        new_table = torch.exp(torch.squeeze(new_table))

        # Reshape the table if necessary
        if new_table.ndim < len(new_domains):
            new_table = new_table.reshape(new_domains)
        
        f_clique = torch.tensor(new_clique, dtype=torch.int64, device=self.table.device)    
        f_domains = torch.tensor(new_domains, dtype=torch.int64, device=self.table.device)
        f_table = new_table

        return ProbTable(f_clique, f_domains, f_table)


cdef class ProbTableSameSizeNumpy:
    cdef public cnp.ndarray variables
    cdef public cnp.ndarray domains
    cdef public cnp.ndarray tables
    cdef public cnp.ndarray exp_tables  
    cdef public tuple table_shape 
    cdef public Py_ssize_t num_factors 
    cdef public Py_ssize_t num_vars_in_clique 
    cdef public list slice_bases 

    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def __cinit__(self, cnp.ndarray[long, ndim=2] variables, cnp.ndarray[long, ndim=2] domains, cnp.ndarray tables):
        """
        Initialize the ProbTableSameSizeNumpy with the given variables, domains, and tables.
        
        :param variables: A NumPy array of variable indices for each factor. shape: (num_factors, num_vars_in_clique)
        :param domains: A NumPy array of domain sizes for each variable. shape: (num_factors, num_vars_in_clique)
        :param tables: A NumPy array of factor tables. shape: (num_factors, [2] * num_vars_in_clique)
        """
        self.variables = variables
        self.domains = domains
        self.tables = tables

        # Precompute exp(tables)
        self.exp_tables = np.exp(self.tables).astype(np.float32)
        
        # Precompute table shape
        cdef Py_ssize_t ndim = np.PyArray_NDIM(self.tables)
        cdef np.npy_intp *shape_ptr = np.PyArray_DIMS(self.tables)
        self.table_shape = tuple([shape_ptr[i] for i in range(ndim)])
        
        # Precompute number of factors and variables
        self.num_factors = self.variables.shape[0]
        self.num_vars_in_clique = self.variables.shape[1]
        
        # Precompute slice bases
        self.slice_bases = [slice(None)] * self.tables.ndim

    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef cnp.ndarray instantiate_evidence(self, long[:] evidence):
        """
        Instantiate evidence for all factors of the same size.
        
        :param evidence: A NumPy array of evidence values for all variables.
                        Use -1 for variables that are not evidence.
        :return: A tuple containing variables, domains, and new_tables with the instantiated factors.
        """
        cdef cnp.ndarray new_tables
        cdef cnp.ndarray mask
        cdef Py_ssize_t i, j, var, ev
        cdef list slices

        # Create a mask array initialized with True
        mask = np.ones(self.table_shape, dtype=bool)

        # Create the mask based on evidence
        for i in range(self.num_factors):
            for j in range(self.num_vars_in_clique):
                var = self.variables[i, j]
                ev = evidence[var]
                if ev != -1:
                    slices = self.slice_bases.copy()
                    slices[0] = i
                    slices[j + 1] = 1 - ev
                    mask[tuple(slices)] = False

        # Apply the mask to create the new tables
        new_tables = np.empty(self.table_shape, dtype=np.float32)
        it = np.nditer([self.exp_tables, mask, new_tables], flags=['multi_index'], op_flags=[['readonly'], ['readonly'], ['writeonly']])
        
        for x, m, nt in it:
            nt[...] = x * m

        return new_tables


cdef class ProbTableSameSizeTensor:
    cdef public object variables
    cdef public object domains
    cdef public object tables
    cdef public object unrolled_tables
    cdef public object precomputed_indices

    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def __cinit__(self, object variables, object domains, object tables):
        """
        Initialize the ProbTableSameSize with the given variables, domains, and tables.
        
        :param variables: A tensor of variable indices for each factor. shape: (num_factors, num_vars_in_clique)
        :param domains: A tensor of domain sizes for each variable. shape: (num_factors, num_vars_in_clique)
        :param tables: A tensor of factor tables. shape: (num_factors, [2] * num_vars_in_clique)
        """
        self.variables = variables
        self.domains = domains
        self.tables = tables


    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef object instantiate_evidence(self, object evidence):
        """
        Instantiate evidence for all factors of the same size.
        
        :param evidence: A tensor of evidence values for all variables.
                        Use -1 for variables that are not evidence.
        :return: A new ProbTableSameSize with the instantiated factors.
        """
        cdef object new_tables, mask
        cdef int i, j, ev

        # Create a mask tensor initialized with True
        mask = torch.ones_like(self.tables, dtype=torch.bool)

        # Create the mask based on evidence
        for i in range(self.variables.shape[0]):  # For each clique
            for j in range(self.variables.shape[1]):  # For each variable in the clique
                var = self.variables[i, j]
                ev = evidence[var].item()
                if ev != -1:  # If this variable is evidence
                    # Create a slice object for this dimension
                    slices = [slice(None)] * (self.tables.dim())
                    slices[0] = i  # Select the current clique
                    slices[j + 1] = 1 - ev  # Mask out the non-evidence value (+1 because the first dimension is for cliques)
                    mask[tuple(slices)] = False

        # Apply the mask to create the new tables
        new_tables = torch.exp(self.tables.clone())
        new_tables[~mask] = 0  # Set to 0 in probability space

        return self.variables, self.domains, new_tables

cdef class UAIParser:
    # Declare all attributes with cdef public for Python access
    cdef public str model_str
    cdef public str network_type
    cdef public int num_vars
    cdef public int num_cliques
    cdef public double eps
    cdef public int max_vars_in_clique
    cdef public bint pairwise_only
    cdef public bint one_d_factors
    cdef public list tokens  # List of parsed tokens
    cdef object device
    cdef public object domain_sizes
    cdef public object cliques
    cdef public Py_ssize_t num_edge_node_pairs
    
    
    # Processing lists
    cdef public list prob_tables_list
    cdef public list univariate_tables_list
    cdef public list univariate_vars_list
    cdef public list bivariate_tables_list
    cdef public list bivariate_vars_list
    cdef public dict clique_dict_parameters_list
    cdef public dict clique_dict_variables_list
    cdef public dict clique_dict_domains_list

    # Attributes for tensors
    cdef public list prob_tables
    cdef public object univariate_tables
    cdef public object univariate_vars
    cdef public object bivariate_tables
    cdef public object bivariate_vars
    cdef public dict clique_dict_class
    cdef public object clique_dict_class_np
    cdef public dict two_to_n
    cdef public dict num_cliques_per_size
    cdef public dict clique_indices
    cdef public list clique_sizes
    cdef public object all_variables
    cdef public object all_domains
    cdef public object hyperedge_index



    def __init__(self, model_str="", one_d_factors=False, device=None):
        # Initialize lists and attributes properly
        self.model_str = model_str
        self.eps = 1e-10
        self.num_edge_node_pairs = 0
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        self.max_vars_in_clique = 0
        self.pairwise_only = True
        self.one_d_factors = one_d_factors
        # Initialize the tables and variables as empty lists
        self.prob_tables_list = []
        self.univariate_tables_list = []
        self.univariate_vars_list = []
        self.bivariate_tables_list = []
        self.bivariate_vars_list = []
        self.clique_dict_parameters_list = {}
        self.clique_dict_variables_list = {}
        self.clique_dict_domains_list = {}

        # Initialize tensor attributes as None
        self.prob_tables = None
        self.univariate_tables = None
        self.univariate_vars = None
        self.bivariate_tables = None
        self.bivariate_vars = None
        self.clique_dict_class = {}
        self.clique_dict_class_np = {}
        print(f"Using 1d factors: {self.one_d_factors}")
        self.tokenize()  # Tokenize the model string
        self.parse_file()  # Parse the file contents
        self.convert_to_tensors()  # Convert lists and dictionaries to tensors
        self.precompute()

    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef void precompute(self):
        self.clique_sizes = sorted(list(self.clique_dict_class_np.keys()))
        cdef Py_ssize_t unique_num_clique_sizes = len(self.clique_sizes)
        cdef Py_ssize_t j
        cdef Py_ssize_t tmp
        cdef Py_ssize_t num_cliques
        cdef Py_ssize_t start_idx = 0

        # Initialize all_variables, all_domains, and hyperedge_index with -1
        self.all_variables = np.full((self.num_cliques, self.max_vars_in_clique), -1, dtype=np.int64)
        self.all_domains = np.full((self.num_cliques, self.max_vars_in_clique), -1, dtype=np.int64)
        self.hyperedge_index = np.full((2, self.num_edge_node_pairs), 0, dtype=np.int64)

        # Precompute two_to_n, num_cliques_per_size, and clique_indices
        self.two_to_n = {}
        self.num_cliques_per_size = {}
        self.clique_indices = {}


        for j in self.clique_sizes:
            self.two_to_n[j] = int(2 ** j)
            tmp = self.clique_dict_class_np[j].variables.shape[0]
            self.num_cliques_per_size[j] = tmp
            num_cliques = tmp
            self.clique_indices[j] = (start_idx, start_idx + num_cliques)
            start_idx += num_cliques

        cdef Py_ssize_t idx_hyperedge_index = 0
        cdef Py_ssize_t num_columns_this_iter = 0
        cdef Py_ssize_t cliques_done = 0
        for clique_size in self.clique_sizes:
            this_clique_dict = self.clique_dict_class_np[clique_size]
            num_cliques_this_size = self.num_cliques_per_size[clique_size]
            start_clq_idx, end_clq_idx = self.clique_indices[clique_size]
            domains_array = this_clique_dict.domains
            self.all_domains[start_clq_idx:end_clq_idx, :clique_size] = domains_array
            #variables
            variables_array = this_clique_dict.variables
            self.all_variables[start_clq_idx:end_clq_idx, :clique_size] = variables_array
            num_columns_this_iter = clique_size*num_cliques_this_size
            # In the first row we need to put the variable index of for each clique in order. These variables come from variables_array. Just flatten it
            first_row = variables_array.flatten()
            self.hyperedge_index[0, idx_hyperedge_index:idx_hyperedge_index+num_columns_this_iter] = first_row

            # In the second row we need to put the clique index of each clique (num_cliques_this_size), but they should repeat consecutively clique_size times, thus it should be like [0,0,1,1,2,2,3,3,4,4]
            second_row = np.repeat(np.arange(cliques_done, cliques_done+num_cliques_this_size), clique_size)
            self.hyperedge_index[1, idx_hyperedge_index:idx_hyperedge_index+num_columns_this_iter] = second_row

            idx_hyperedge_index += num_columns_this_iter
            cliques_done += num_cliques_this_size



    @cython.boundscheck(False)
    @cython.wraparound(False)
    # Helper function to check if a character is whitespace
    cdef inline bint is_whitespace(self, char):
        return char in (' ', '\n', '\r', '\t')

    @cython.boundscheck(False)
    @cython.wraparound(False)
    # Tokenize the input model string into tokens
    cdef void tokenize(self):
        self.tokens = []
        token = []
        for char in self.model_str:
            if self.is_whitespace(char):
                if token:  # if there's something in the token, add it
                    self.tokens.append(''.join(token))
                    token = []
            else:
                token.append(char)
        if token:  # add the last token if exists
            self.tokens.append(''.join(token))

    @cython.boundscheck(False)
    @cython.wraparound(False)
    # Parse the file contents
    cdef void parse_file(self):
        if not self.model_str:
            raise ValueError("No file path or model string provided")

        file_iter = iter(self.tokens)

        # Read and verify the network type (first token should be either "MARKOV" or "BAYES")
        self.network_type = self.read_next_token(file_iter)
        if self.network_type.upper() not in ["MARKOV"]:
            # if self.network_type.upper() not in ["MARKOV", "BAYES"]:

            raise ValueError(f"ERROR: UAI file does not start with MARKOV. Found: {self.network_type}")

        # Parse the number of variables
        self.num_vars = int(self.read_next_token(file_iter))

        # Parse the domain sizes for each variable
        self.domain_sizes = np.array([int(self.read_next_token(file_iter)) for _ in range(self.num_vars)])

        # Parse the number of cliques
        self.num_cliques = int(self.read_next_token(file_iter))


        # Parse cliques
        self.cliques = [
            np.array(
                [
                    int(self.read_next_token(file_iter))
                    for _ in range(int(self.read_next_token(file_iter)))
                ],
                dtype=np.int64,
            )
            for _ in range(self.num_cliques)
        ]

        for clique in self.cliques:
            self.max_vars_in_clique = max(self.max_vars_in_clique, len(clique))
            self.num_edge_node_pairs += len(clique)
            table_size = int(self.read_next_token(file_iter))
            table_values = [
                max(float(self.read_next_token(file_iter)), self.eps)
                for _ in range(table_size)
            ]
            table = np.log(np.array(table_values))
            domains_this_table = self.domain_sizes[clique]
            if not self.one_d_factors:
                table = table.reshape(domains_this_table.tolist())

            prob_table = ProbTable(clique, domains_this_table, table)
            self.prob_tables_list.append(prob_table)

            if len(clique) == 1:
                self.univariate_tables_list.append(table)
                self.univariate_vars_list.append(clique)
            elif len(clique) == 2:
                self.bivariate_tables_list.append(table)
                self.bivariate_vars_list.append(clique)
            else:
                self.pairwise_only = False

            num_vars = len(clique)
            if num_vars not in self.clique_dict_parameters_list:
                self.clique_dict_parameters_list[num_vars] = []
                self.clique_dict_variables_list[num_vars] = []
                self.clique_dict_domains_list[num_vars] = []
            self.clique_dict_parameters_list[num_vars].append(table)
            self.clique_dict_variables_list[num_vars].append(clique)
            self.clique_dict_domains_list[num_vars].append(domains_this_table)

        if self.pairwise_only:
            print("PGM is pairwise.")
        else:
            print("PGM has higher-order cliques.")

    @cython.boundscheck(False)
    @cython.wraparound(False)
    # Use cdef for read_next_token
    cdef str read_next_token(self, object tokens):
        return next(tokens, None)

    # Convert lists and dictionaries into tensors
    cpdef void convert_to_tensors(self):
        """
        Convert relevant lists and dictionaries to PyTorch tensors.
        """
        # Convert lists to tensors and store them in tensor attributes
        # only if they are not empty
        if self.univariate_tables_list:
            self.univariate_tables = torch.stack([torch.from_numpy(arr).float().to(self.device) for arr in self.univariate_tables_list], dim=0)
            self.univariate_vars = torch.stack([torch.from_numpy(arr).long().to(self.device) for arr in self.univariate_vars_list], dim=0).flatten()
        if self.bivariate_tables_list:
            self.bivariate_tables = torch.stack([torch.from_numpy(arr).float().to(self.device) for arr in self.bivariate_tables_list], dim=0)
            self.bivariate_vars = torch.stack([torch.from_numpy(arr).long().to(self.device) for arr in self.bivariate_vars_list], dim=0)

        self.prob_tables = []
        for table in self.prob_tables_list:
            table.to_tensor(self.device)
            self.prob_tables.append(table)

        # Convert dictionaries to tensors and store in tensor dictionaries
        for key in self.clique_dict_parameters_list:
            tables = torch.stack([torch.from_numpy(arr).float().to(self.device) for arr in self.clique_dict_parameters_list[key]], dim=0)
            variables = torch.stack([torch.from_numpy(arr).long().to(self.device) for arr in self.clique_dict_variables_list[key]], dim=0)
            domains = torch.stack([torch.from_numpy(arr).long().to(self.device) for arr in self.clique_dict_domains_list[key]], dim=0)
            self.clique_dict_class[key] = ProbTableSameSizeTensor(variables, domains, tables)

            # Also store the numpy version of the tables
            tables_np = np.array(self.clique_dict_parameters_list[key])
            variables_np = np.array(self.clique_dict_variables_list[key])
            domains_np = np.array(self.clique_dict_domains_list[key])
            self.clique_dict_class_np[key] = ProbTableSameSizeNumpy(variables_np, domains_np, tables_np)


    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef object instantiate_evidence(self, object batch_data, object evidence_bool, bint remove_values=False):
        instantiated_tables = []
        for example, evidence_for_example in zip(batch_data, evidence_bool):
            # if evidence_for_example is True, then we use the value at that index, otherwise we use -1
            example[~evidence_for_example] = -1
            instantiated_tables_for_example = []
            for table in self.prob_tables:
                if remove_values:
                    instantiated_table = table.inst_evid_remove_values(example)
                else:
                    instantiated_table = table.instantiate_evidence(example)
                instantiated_tables_for_example.append(instantiated_table)
            instantiated_tables.append(instantiated_tables_for_example)
        return instantiated_tables

    # cpdef object instantiate_evidence_same_size_cliques(self, object batch_data, object evidence_bool,):
    #     instantiated_tables = []
    #     for example, evidence_for_example in zip(batch_data, evidence_bool):
    #         # if evidence_for_example is True, then we use the value at that index, otherwise we use -1
    #         example[~evidence_for_example] = -1
    #         instantiated_tables_for_example = []
    #         for clique_size in self.clique_dict_class:
    #             instantiated_table = self.clique_dict_class[clique_size].optimized_instantiate_evidence(example)
    #             instantiated_tables_for_example.append(instantiated_table)
    #         instantiated_tables.append(instantiated_tables_for_example)
    #     return instantiated_tables


    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef tuple instantiate_evidence_same_size_cliques(self, object batch_data, object evidence_bool):
        cdef Py_ssize_t batch_size = batch_data.shape[0]
        cdef Py_ssize_t i, j

        
        # Pre-compute clique indices
        cdef list clique_indices = []
        cdef Py_ssize_t start_idx = 0
        for j in self.clique_sizes:
            if j in self.clique_dict_class:
                num_cliques = len(self.clique_dict_class[j].variables)
                clique_indices.append((start_idx, start_idx + num_cliques))
                start_idx += num_cliques
            else:
                clique_indices.append((start_idx, start_idx))
        
        # Pre-allocate output tensors
        cdef object all_tables = torch.zeros((batch_size, self.num_cliques, int(2**self.max_vars_in_clique)), dtype=torch.float32, device=batch_data.device)
        cdef object all_variables = torch.ones((batch_size, self.num_cliques, self.max_vars_in_clique), dtype=torch.int64, device=batch_data.device) * -1
        cdef object all_domains = torch.ones((batch_size, self.num_cliques, self.max_vars_in_clique), dtype=torch.int64, device=batch_data.device) * -1
        
        cdef object example, evidence_for_example, variables, domains, tables
        cdef Py_ssize_t num_cliques_this_size, clique_size, start_clq_idx, end_clq_idx
        
        for i in prange(batch_size, nogil=True, schedule='static'):
            with gil:
                example = batch_data[i].clone()
                evidence_for_example = evidence_bool[i]
                example[~evidence_for_example] = -1
                
                for j in range(len(self.clique_sizes)):
                    if j in self.clique_dict_class: 
                        variables, domains, tables = self.clique_dict_class[j].instantiate_evidence(example)
                        num_cliques_this_size, clique_size = variables.shape
                        start_clq_idx, end_clq_idx = clique_indices[j]
                        updated_tables = tables.view(-1, int(2**clique_size))
                        all_tables[i, start_clq_idx:end_clq_idx, :int(2**clique_size)] = updated_tables
                        all_variables[i, start_clq_idx:end_clq_idx, :clique_size] = variables
                        all_domains[i, start_clq_idx:end_clq_idx, :clique_size] = domains

        return all_tables, all_variables, all_domains


    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef tuple instantiate_evidence_same_size_cliques_np(self, long[:,:] batch_data, cnp.ndarray[cnp.uint8_t, ndim=2] evidence_bool):
        cdef Py_ssize_t batch_size = batch_data.shape[0]
        cdef Py_ssize_t i


        # Pre-allocate output arrays
        cdef cnp.ndarray[cnp.float32_t, ndim=3] all_tables = np.zeros((batch_size, self.num_cliques, int(2**self.max_vars_in_clique)), dtype=np.float32)

        cdef cnp.ndarray[long, ndim=1] example
        cdef cnp.ndarray[cnp.uint8_t, ndim=1] evidence_for_example
        cdef cnp.ndarray tables_array
        cdef cnp.ndarray updated_tables 

        cdef Py_ssize_t clique_size, start_clq_idx, end_clq_idx
        for i in prange(batch_size, nogil=True, schedule='static'):
            with gil:
                example = np.asarray(batch_data[i])
                evidence_for_example = np.asarray(evidence_bool[i])
                example[~evidence_for_example] = -1
                for clique_size in self.clique_sizes:
                    tables_array = self.clique_dict_class_np[clique_size].instantiate_evidence(example)
                    start_clq_idx, end_clq_idx = self.clique_indices[clique_size]
                    all_tables[i, start_clq_idx:end_clq_idx, :self.two_to_n[clique_size]] = tables_array.reshape(-1, self.two_to_n[clique_size])
        return all_tables, self.hyperedge_index, self.all_variables, self.all_domains


