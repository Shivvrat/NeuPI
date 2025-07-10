# distutils: language = c++
# cython: language_level=3

import numpy as np
cimport numpy as np
from libcpp.vector cimport vector
from libcpp.algorithm cimport sort
from libcpp.utility cimport pair

np.import_array()

# Define a type for a pair of (score, assignment_vector)
ctypedef pair[double, vector[int]] Assignment

# Inline comparison function for sorting assignments by score
cdef inline int compare_assignments(const Assignment& a, const Assignment& b) nogil:
    return a.first < b.first

# Core beam search algorithm implemented in C++ for performance
cdef vector[Assignment] process_assignments(const float* s_i_ptr, int N, int k) nogil:
    cdef int i, idx
    cdef double s_idx, D, D_new_0, D_new_1
    cdef vector[Assignment] L, T
    cdef Assignment current_assignment, new_assignment_0, new_assignment_1

    # Start with an empty assignment and a score of 0
    L.push_back(Assignment(0.0, vector[int]()))

    # Iterate through each variable
    for idx in range(N):
        T.clear()
        s_idx = s_i_ptr[idx]

        # For each assignment in the current beam (L)
        for i in range(L.size()):
            current_assignment = L[i]
            D = current_assignment.first

            # Create two new hypotheses: one for b_i = 0 and one for b_i = 1
            # Option 1: b_i = 0
            D_new_0 = D + s_idx
            new_assignment_0 = Assignment(D_new_0, current_assignment.second)
            new_assignment_0.second.push_back(0)

            # Option 2: b_i = 1
            D_new_1 = D + (1 - s_idx)
            new_assignment_1 = Assignment(D_new_1, current_assignment.second)
            new_assignment_1.second.push_back(1)

            T.push_back(new_assignment_0)
            T.push_back(new_assignment_1)

        # Sort all new hypotheses by score
        sort(T.begin(), T.end(), compare_assignments)
        
        # Prune the beam to keep only the top k hypotheses
        if T.size() > k:
            T.resize(k)
        
        L = T

    return L

# Python wrapper for the C++ function
def cython_process_assignments(np.ndarray[np.float32_t, ndim=1] s_i, int k):
    """
    Finds the k-best binary assignments for a given probability vector.

    Args:
        s_i (np.ndarray): A 1D NumPy array of probabilities.
        k (int): The number of best assignments to find (the beam width).

    Returns:
        tuple: A tuple containing:
            - A list of (score, assignment) pairs.
            - A 2D NumPy array of the k best binary assignments.
    """
    cdef int N = s_i.shape[0]
    cdef vector[Assignment] result = process_assignments(&s_i[0], N, k)
    
    # Convert the C++ vector of pairs to a Python list
    cdef list py_result = [(assignment.first, list(assignment.second)) for assignment in result]
    
    # Create a NumPy array for the assignments for efficient use in Python
    cdef np.ndarray[np.int32_t, ndim=2] assignments = np.zeros((len(py_result), N), dtype=np.int32)
    for i, (_, assignment_vec) in enumerate(py_result):
        # Directly create a numpy array and assign it to the row.
        # This avoids the illegal cdef statement inside the loop.
        assignments[i, :] = np.asarray(assignment_vec, dtype=np.int32)
    
    return py_result, assignments
