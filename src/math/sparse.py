import jax
import jax.numpy as jnp
from typing import Tuple, Optional, Union, Dict, List, Any, Callable, Literal


class COOMatrix:
    """
    JAX-based Coordinate (COO) sparse matrix implementation.
    
    Stores data in coordinate format with explicit row and column indices for each value.
    All operations use only JAX functions to ensure JIT compatibility.
    
    Attributes:
        data: JAX array containing the non-zero values
        row_indices: JAX array containing row indices for each non-zero value
        col_indices: JAX array containing column indices for each non-zero value
        shape: Tuple representing the full matrix shape (rows, cols)
        dtype: Data type of the non-zero values
    """
    
    def __init__(self, 
                 data: jnp.ndarray, 
                 indices: Tuple[jnp.ndarray, jnp.ndarray], 
                 shape: Tuple[int, int],
                 dtype: Any = None):
        """
        Initialize a sparse matrix in COO format.
        
        Args:
            data: Array of non-zero values
            indices: Tuple of (row_indices, col_indices) for non-zero entries
            shape: Shape of the full matrix (rows, cols)
            dtype: Data type for the values (defaults to data.dtype)
        """
        row_indices, col_indices = indices
        self.data = jnp.asarray(data, dtype=dtype)
        self.row_indices = jnp.asarray(row_indices, dtype=jnp.int32)
        self.col_indices = jnp.asarray(col_indices, dtype=jnp.int32)
        self.shape = shape
        
        if dtype is None:
            self.dtype = self.data.dtype
        else:
            self.dtype = dtype
    
    @classmethod
    def from_dense(cls, dense_matrix: jnp.ndarray) -> 'COOMatrix':
        """
        Create a sparse matrix from a dense matrix using only JAX operations.
        
        Args:
            dense_matrix: Dense matrix to convert
            
        Returns:
            A COOMatrix representation of the input
        """
        dense_array = jnp.asarray(dense_matrix)
        # Find non-zero elements
        non_zero_mask = dense_array != 0
        
        # Instead of using jnp.nonzero which doesn't work with JAX transformations,
        # create indices manually in a JAX-compatible way
        shape = dense_array.shape
        indices = jnp.mgrid[:shape[0], :shape[1]].reshape(2, -1)
        row_indices_all, col_indices_all = indices[0], indices[1]
        
        # Flatten the mask and use it to select valid indices
        flat_mask = non_zero_mask.flatten()
        row_indices = jnp.compress(flat_mask, row_indices_all)
        col_indices = jnp.compress(flat_mask, col_indices_all)
        
        # Get the corresponding data values
        data = jnp.compress(flat_mask, dense_array.flatten())
        
        return cls(data, (row_indices, col_indices), dense_array.shape, dense_array.dtype)

    @classmethod
    def zeros(cls, shape: Tuple[int, int], dtype: Any = jnp.float32) -> 'COOMatrix':
        """
        Create an empty sparse matrix (all zeros).
        
        Args:
            shape: Shape of the matrix (rows, cols)
            dtype: Data type for the values
            
        Returns:
            A COOMatrix representation with zero elements
        """
        return cls(
            jnp.array([], dtype=dtype),
            (jnp.array([], dtype=jnp.int32), jnp.array([], dtype=jnp.int32)),
            shape,
            dtype
        )
    
    @classmethod
    def random(cls, shape: Tuple[int, int], density: float = 0.01, 
               key: Optional[jax.random.PRNGKey] = None, 
               dtype: Any = jnp.float32) -> 'COOMatrix':
        """
        Create a random sparse matrix with the given density.
        
        Args:
            shape: Shape of the matrix (rows, cols)
            density: Fraction of non-zero elements (between 0 and 1)
            key: JAX random key (if None, a new one will be created)
            dtype: Data type for the values
            
        Returns:
            A random COOMatrix
        """
        if key is None:
            key = jax.random.PRNGKey(0)
        
        rows, cols = shape
        nnz = int(rows * cols * density)  # Number of non-zeros
        
        # Generate random indices
        key1, key2, key3 = jax.random.split(key, 3)
        row_indices = jax.random.randint(key1, (nnz,), 0, rows)
        col_indices = jax.random.randint(key2, (nnz,), 0, cols)
        
        # Generate random values
        data = jax.random.normal(key3, (nnz,), dtype=dtype)
        
        return cls(data, (row_indices, col_indices), shape, dtype)
    
    def to_dense(self) -> jnp.ndarray:
        """
        Convert sparse matrix to dense format.
        
        Returns:
            Dense representation of the sparse matrix
        """
        dense = jnp.zeros(self.shape, dtype=self.dtype)
        return dense.at[self.row_indices, self.col_indices].add(self.data)
    
    def __matmul__(self, other: Union['COOMatrix', jnp.ndarray]) -> Union['COOMatrix', jnp.ndarray]:
        """
        Matrix multiplication with another matrix or vector.
        
        Args:
            other: Another matrix (sparse or dense) or vector
            
        Returns:
            The result of the matrix multiplication
        """
        if isinstance(other, COOMatrix):
            # For sparse @ sparse, convert to dense for now
            return self.to_dense() @ other.to_dense()
        else:
            # For sparse @ dense
            return self.to_dense() @ other
    
    def __add__(self, other: 'COOMatrix') -> 'COOMatrix':
        """
        Add another sparse matrix.
        
        Args:
            other: Another sparse matrix
            
        Returns:
            The sum as a new COOMatrix
        """
        # Convert to dense, add, and convert back to sparse
        # A more efficient JAX-based implementation would combine the sparse representations
        result_dense = self.to_dense() + other.to_dense()
        return COOMatrix.from_dense(result_dense)
    
    def __sub__(self, other: 'COOMatrix') -> 'COOMatrix':
        """
        Subtract another sparse matrix.
        
        Args:
            other: Another sparse matrix
            
        Returns:
            The difference as a new COOMatrix
        """
        result_dense = self.to_dense() - other.to_dense()
        return COOMatrix.from_dense(result_dense)
    
    def __mul__(self, scalar: Union[float, int]) -> 'COOMatrix':
        """
        Multiply by a scalar.
        
        Args:
            scalar: A scalar value
            
        Returns:
            The scaled matrix as a new COOMatrix
        """
        scaled_data = self.data * scalar
        return COOMatrix(scaled_data, (self.row_indices, self.col_indices), self.shape, self.dtype)
    
    def transpose(self) -> 'COOMatrix':
        """
        Transpose the matrix.
        
        Returns:
            The transposed matrix as a new COOMatrix
        """
        return COOMatrix(
            self.data,
            (self.col_indices, self.row_indices),
            (self.shape[1], self.shape[0]),
            self.dtype
        )
    
    def __repr__(self) -> str:
        """
        String representation of the sparse matrix.
        
        Returns:
            A string description of the matrix
        """
        nnz = len(self.data)
        density = nnz / (self.shape[0] * self.shape[1]) * 100
        return f"COOMatrix(shape={self.shape}, non-zeros={nnz}, density={density:.2f}%)"
    
    @property
    def T(self) -> 'COOMatrix':
        """Transpose property."""
        return self.transpose()
    
    def apply_function(self, func: Callable[[jnp.ndarray], jnp.ndarray]) -> 'COOMatrix':
        """
        Apply a function to each non-zero element.
        
        Args:
            func: A function to apply to the data values
            
        Returns:
            A new COOMatrix with transformed values
        """
        new_data = func(self.data)
        return COOMatrix(new_data, (self.row_indices, self.col_indices), self.shape, new_data.dtype)

    def to_csr(self) -> 'CSRMatrix':
        """
        Convert COO format to CSR format.
        
        Returns:
            A CSRMatrix equivalent to this COOMatrix
        """
        # Use JAX's lexsort equivalent to sort by row and then column
        sort_keys = jnp.stack([self.row_indices, self.col_indices])
        sort_order = jnp.lexsort(sort_keys[::-1])  # Sort by row first, then column
        
        # Sort data and indices
        sorted_data = self.data[sort_order]
        sorted_row_indices = self.row_indices[sort_order]
        sorted_col_indices = self.col_indices[sort_order]
        
        # Compute row pointers
        rows = self.shape[0]
        
        # Create vector for counting entries in each row
        row_counts = jnp.zeros(rows, dtype=jnp.int32)
        
        # Count entries in each row using a scatter_add operation
        # Note: Manually doing this is tricky in pure JAX, so using a simpler
        # approach for clarity
        def count_rows(row_counts, row_index):
            # Add 1 to the count for each row
            return row_counts.at[row_index].add(1)
        
        row_counts = jax.lax.fori_loop(
            0, len(sorted_row_indices),
            lambda i, rc: count_rows(rc, sorted_row_indices[i]),
            row_counts
        )
        
        # Compute indptr with exclusive prefix sum
        indptr = jnp.zeros(rows + 1, dtype=jnp.int32)
        indptr = indptr.at[1:].set(jnp.cumsum(row_counts))
        
        # Return CSR matrix
        return CSRMatrix(sorted_data, sorted_col_indices, indptr, self.shape, self.dtype)


class CSRMatrix:
    """
    JAX-based Compressed Sparse Row (CSR) sparse matrix implementation.
    
    Stores data in CSR format with compressed row representation.
    All operations use only JAX functions to ensure JIT compatibility.
    
    Attributes:
        data: JAX array containing the non-zero values
        indices: JAX array containing column indices for each non-zero value
        indptr: JAX array containing row pointers (start indices for each row)
        shape: Tuple representing the full matrix shape (rows, cols)
        dtype: Data type of the non-zero values
    """
    
    def __init__(self, 
                 data: jnp.ndarray, 
                 indices: jnp.ndarray,
                 indptr: jnp.ndarray,
                 shape: Tuple[int, int],
                 dtype: Any = None):
        """
        Initialize a sparse matrix in CSR format.
        
        Args:
            data: Array of non-zero values
            indices: Column indices for each non-zero value
            indptr: Row pointers indicating the start positions of each row in data
            shape: Shape of the full matrix (rows, cols)
            dtype: Data type for the values (defaults to data.dtype)
        """
        self.data = jnp.asarray(data, dtype=dtype)
        self.indices = jnp.asarray(indices, dtype=jnp.int32)
        self.indptr = jnp.asarray(indptr, dtype=jnp.int32)
        self.shape = shape
        
        if dtype is None:
            self.dtype = self.data.dtype
        else:
            self.dtype = dtype
    
    @classmethod
    def from_dense(cls, dense_matrix: jnp.ndarray) -> 'CSRMatrix':
        """
        Create a CSR sparse matrix from a dense matrix.
        
        Args:
            dense_matrix: Dense matrix to convert
            
        Returns:
            A CSRMatrix representation of the input
        """
        # First create a COO matrix
        coo_matrix = COOMatrix.from_dense(dense_matrix)
        
        # Then convert to CSR
        return coo_matrix.to_csr()

    @classmethod
    def zeros(cls, shape: Tuple[int, int], dtype: Any = jnp.float32) -> 'CSRMatrix':
        """
        Create an empty sparse matrix (all zeros).
        
        Args:
            shape: Shape of the matrix (rows, cols)
            dtype: Data type for the values
            
        Returns:
            A CSRMatrix representation with zero elements
        """
        rows = shape[0]
        return cls(
            jnp.array([], dtype=dtype),
            jnp.array([], dtype=jnp.int32),
            jnp.zeros(rows + 1, dtype=jnp.int32),
            shape,
            dtype
        )
    
    @classmethod
    def random(cls, shape: Tuple[int, int], density: float = 0.01, 
               key: Optional[jax.random.PRNGKey] = None, 
               dtype: Any = jnp.float32) -> 'CSRMatrix':
        """
        Create a random CSR sparse matrix with the given density.
        
        Args:
            shape: Shape of the matrix (rows, cols)
            density: Fraction of non-zero elements (between 0 and 1)
            key: JAX random key (if None, a new one will be created)
            dtype: Data type for the values
            
        Returns:
            A random CSRMatrix
        """
        # Create a random COO matrix first
        coo_matrix = COOMatrix.random(shape, density, key, dtype)
        
        # Convert to CSR
        return coo_matrix.to_csr()
    
    def to_dense(self) -> jnp.ndarray:
        """
        Convert CSR sparse matrix to dense format.
        
        Returns:
            Dense representation of the sparse matrix
        """
        dense = jnp.zeros(self.shape, dtype=self.dtype)
        
        # Create a function to process each row
        def process_row(i, dense):
            start, end = self.indptr[i], self.indptr[i+1]
            values = self.data[start:end]
            cols = self.indices[start:end]
            
            # For each element in the row, add value to the dense matrix
            row_array = jnp.zeros((self.shape[1],), dtype=self.dtype)
            row_array = row_array.at[cols].add(values)
            dense = dense.at[i].set(row_array)
            
            return dense
        
        # Process all rows
        return jax.lax.fori_loop(0, self.shape[0], process_row, dense)
    
    def __matmul__(self, other: Union['CSRMatrix', jnp.ndarray]) -> Union['CSRMatrix', jnp.ndarray]:
        """
        Matrix multiplication with another matrix or vector.
        
        Args:
            other: Another matrix (sparse or dense) or vector
            
        Returns:
            The result of the matrix multiplication
        """
        if isinstance(other, CSRMatrix):
            # For sparse @ sparse, convert to dense for now
            return self.to_dense() @ other.to_dense()
        else:
            # For sparse @ dense
            if other.ndim == 1:
                return self._matvec(other)
            else:
                return self.to_dense() @ other
    
    def _matvec(self, vec: jnp.ndarray) -> jnp.ndarray:
        """
        Efficient sparse matrix-vector multiplication.
        
        Args:
            vec: Dense vector to multiply with
            
        Returns:
            Result of multiplication as a dense vector
        """
        rows = self.shape[0]
        result = jnp.zeros(rows, dtype=jnp.promote_types(self.dtype, vec.dtype))
        
        # Define a function for a single row multiplication
        def row_mul(i, result):
            start, end = self.indptr[i], self.indptr[i+1]
            # Sum product of row elements and vector elements
            row_sum = jnp.sum(self.data[start:end] * vec[self.indices[start:end]])
            return result.at[i].set(row_sum)
        
        # Use a fori_loop to process each row (more JAX friendly)
        result = jax.lax.fori_loop(0, rows, row_mul, result)
        
        return result
    
    def __add__(self, other: 'CSRMatrix') -> 'CSRMatrix':
        """
        Add another sparse matrix.
        
        Args:
            other: Another CSR sparse matrix
            
        Returns:
            The sum as a new CSRMatrix
        """
        # For simplicity, convert to dense, add, and convert back to sparse
        result_dense = self.to_dense() + other.to_dense()
        return CSRMatrix.from_dense(result_dense)
    
    def __sub__(self, other: 'CSRMatrix') -> 'CSRMatrix':
        """
        Subtract another sparse matrix.
        
        Args:
            other: Another CSR sparse matrix
            
        Returns:
            The difference as a new CSRMatrix
        """
        result_dense = self.to_dense() - other.to_dense()
        return CSRMatrix.from_dense(result_dense)
    
    def __mul__(self, scalar: Union[float, int]) -> 'CSRMatrix':
        """
        Multiply by a scalar.
        
        Args:
            scalar: A scalar value
            
        Returns:
            The scaled matrix as a new CSRMatrix
        """
        scaled_data = self.data * scalar
        return CSRMatrix(scaled_data, self.indices, self.indptr, self.shape, self.dtype)
    
    def transpose(self) -> 'CSRMatrix':
        """
        Transpose the matrix.
        
        Returns:
            The transposed matrix (as CSR)
        """
        # Convert to dense and transpose (a pure JAX CSR->CSC would be more efficient)
        return CSRMatrix.from_dense(self.to_dense().T)
    
    def __repr__(self) -> str:
        """
        String representation of the sparse matrix.
        
        Returns:
            A string description of the matrix
        """
        nnz = len(self.data)
        density = nnz / (self.shape[0] * self.shape[1]) * 100
        return f"CSRMatrix(shape={self.shape}, non-zeros={nnz}, density={density:.2f}%)"
    
    @property
    def T(self) -> 'CSRMatrix':
        """Transpose property."""
        return self.transpose()
    
    def apply_function(self, func: Callable[[jnp.ndarray], jnp.ndarray]) -> 'CSRMatrix':
        """
        Apply a function to each non-zero element.
        
        Args:
            func: A function to apply to the data values
            
        Returns:
            A new CSRMatrix with transformed values
        """
        new_data = func(self.data)
        return CSRMatrix(new_data, self.indices, self.indptr, self.shape, new_data.dtype)
    
    def get_row(self, row_idx: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Get the non-zero values and column indices for a specific row.
        
        Args:
            row_idx: Row index
            
        Returns:
            Tuple of (values, column_indices) for the specified row
        """
        start, end = self.indptr[row_idx], self.indptr[row_idx + 1]
        return self.data[start:end], self.indices[start:end]
    
    def row_slice(self, start_row: int, end_row: int) -> 'CSRMatrix':
        """
        Extract a slice of rows from the matrix.
        
        Args:
            start_row: Starting row index (inclusive)
            end_row: Ending row index (exclusive)
            
        Returns:
            A CSRMatrix containing only the specified rows
        """
        if start_row < 0 or end_row > self.shape[0] or start_row >= end_row:
            raise ValueError(f"Invalid row slice: [{start_row}:{end_row}]")
        
        start_ptr = self.indptr[start_row]
        end_ptr = self.indptr[end_row]
        
        # Extract slices of data and indices
        new_data = self.data[start_ptr:end_ptr]
        new_indices = self.indices[start_ptr:end_ptr]
        
        # Create new indptr array
        old_indptr_slice = self.indptr[start_row:end_row+1]
        offset = self.indptr[start_row]
        new_indptr = old_indptr_slice - offset
        
        return CSRMatrix(
            new_data,
            new_indices,
            new_indptr,
            (end_row - start_row, self.shape[1]),
            self.dtype
        )
    
    def to_coo(self) -> COOMatrix:
        """
        Convert CSR format to COO format.
        
        Returns:
            A COOMatrix equivalent to this CSRMatrix
        """
        # Total number of non-zeros
        nnz = self.indptr[-1]
        
        # Create row indices by repeating each row index according to number of elements
        row_indices = jnp.zeros(nnz, dtype=jnp.int32)
        
        # Define a function to fill row indices
        def fill_row_indices(i, row_indices):
            start, end = self.indptr[i], self.indptr[i+1]
            count = end - start
            if count > 0:
                row_indices = row_indices.at[start:end].set(jnp.ones(count, dtype=jnp.int32) * i)
            return row_indices
        
        # Fill row indices for all rows
        row_indices = jax.lax.fori_loop(0, self.shape[0], fill_row_indices, row_indices)
        
        return COOMatrix(self.data, (row_indices, self.indices), self.shape, self.dtype)


# Utility functions for JAX transformations

# @jax.jit
def coo_to_dense(data, row_indices, col_indices, shape):
    """
    Convert COO format arrays to a dense matrix.
    JIT-compatible function for use in transformed code.
    
    Args:
        data: Non-zero values
        row_indices: Row indices for non-zeros
        col_indices: Column indices for non-zeros
        shape: Shape of the matrix
        
    Returns:
        Dense matrix
    """
    dense = jnp.zeros(shape, dtype=data.dtype)
    return dense.at[row_indices, col_indices].add(data)


# @jax.jit
def csr_matvec(data, indices, indptr, vec):
    """
    Matrix-vector multiplication for CSR format.
    JIT-compatible function for use in transformed code.
    
    Args:
        data: Non-zero values
        indices: Column indices for non-zeros
        indptr: Row pointers
        vec: Vector to multiply with
        
    Returns:
        Result vector
    """
    rows = indptr.shape[0] - 1
    result = jnp.zeros(rows, dtype=jnp.promote_types(data.dtype, vec.dtype))
    
    def row_mul(i, result):
        start, end = indptr[i], indptr[i+1]
        row_sum = jnp.sum(data[start:end] * vec[indices[start:end]])
        return result.at[i].set(row_sum)
    
    return jax.lax.fori_loop(0, rows, row_mul, result)


# Example for using with JAX transformations
# @jax.jit
def sparse_matrix_vector_product(csr_data, csr_indices, csr_indptr, vector):
    """
    Example of a JIT-compatible sparse matrix operation.
    """
    return csr_matvec(csr_data, csr_indices, csr_indptr, vector)


# Example for using with JAX's grad
def jittable_objective(csr_data, csr_indices, csr_indptr, x):
    """
    Example objective function for optimization with JAX.
    """
    result = csr_matvec(csr_data, csr_indices, csr_indptr, x)
    return jnp.sum(result ** 2)


# Example to create a sparse matrix and use JAX transformations
def example_jax_sparse():
    # Create a random sparse matrix
    key = jax.random.PRNGKey(42)
    sparse_mat = CSRMatrix.random((5, 5), density=0.5, key=key)
    
    # Create a test vector
    x = jnp.ones(5)
    
    # Direct multiplication
    result1 = sparse_mat @ x
    
    # Using JIT-compatible functions
    result2 = sparse_matrix_vector_product(
        sparse_mat.data, sparse_mat.indices, sparse_mat.indptr, x
    )
    
    # Should be the same
    return jnp.allclose(result1, result2), result1, result2


def segment_csr(
    src: jnp.ndarray,
    indptr: jnp.ndarray,
    reduction: Literal["mean", "sum"] = "sum",
):
    """
    JAX implementation of segment_csr that reduces all entries of a CSR-formatted
    matrix by summing or averaging over neighbors.
    
    Used to reduce features over neighborhoods in integral transforms or graph operations.
    
    Parameters
    ----------
    src : jnp.ndarray
        Tensor of features for each point
    indptr : jnp.ndarray
        Splits representing start and end indices of each neighborhood in src
    reduction : Literal['mean', 'sum'], optional
        How to reduce a neighborhood. If 'mean',
        reduce by taking the average of all neighbors.
        Otherwise take the sum.
        
    Returns
    -------
    jnp.ndarray
        Reduced tensor with shape matching src except for the dimension
        that was reduced (which will have length len(indptr)-1)
    """
    if reduction not in ["mean", "sum"]:
        raise ValueError("reduction must be one of 'mean', 'sum'")

    # Check if batched (3D) or unbatched (2D)
    batched = src.ndim == 3
    
    # Calculate output shape - number of output points is len(indptr)-1
    n_out = indptr.shape[1] - 1 if batched else indptr.shape[0] - 1
    
    # Initialize output with the right shape
    if batched:
        feature_dim = src.shape[2]
        batch_size = src.shape[0]
        out_shape = (batch_size, n_out, feature_dim)
    else:
        feature_dim = src.shape[1] if src.ndim > 1 else 1
        out_shape = (n_out, feature_dim) if src.ndim > 1 else (n_out,)
    
    out = jnp.zeros(out_shape, dtype=src.dtype)
    
    # Define a function to sum a segment without using direct slicing
    def sum_segment(i_pt, out):
        # Get segment bounds
        start_idx = indptr[i_pt] if not batched else indptr[0, i_pt]
        end_idx = indptr[i_pt + 1] if not batched else indptr[0, i_pt + 1]
        segment_size = end_idx - start_idx
        
        # Skip empty segments
        def process_nonempty():
            if batched:
                # For each batch, we need to accumulate separately
                def process_batch(b_idx, b_out):
                    # Initialize accumulator for this batch and point
                    acc = jnp.zeros((feature_dim,), dtype=src.dtype)
                    
                    # Define a loop to accumulate values in the segment
                    def accumulate_loop(j, acc):
                        # Only include indices that are within the segment
                        valid_idx = (start_idx <= j) & (j < end_idx)
                        # Get the value at this index if valid
                        val = jnp.where(valid_idx, src[b_idx, j], jnp.zeros_like(src[b_idx, 0]))
                        # Add to accumulator only if valid
                        return acc + jnp.where(valid_idx, val, jnp.zeros_like(val))
                    
                    # Run the accumulation loop over all possible source indices
                    # We need to use a fixed range for JIT compatibility
                    max_src_idx = src.shape[1]  # Maximum possible source index
                    acc = jax.lax.fori_loop(0, max_src_idx, accumulate_loop, acc)
                    
                    # Average if needed
                    if reduction == "mean" and segment_size > 0:
                        acc = acc / segment_size
                    
                    # Update output for this batch
                    return b_out.at[b_idx, i_pt].set(acc)
                
                # Process each batch
                return jax.lax.fori_loop(0, batch_size, process_batch, out)
            else:
                # For unbatched, accumulate directly
                acc = jnp.zeros((feature_dim,) if src.ndim > 1 else (), dtype=src.dtype)
                
                # Define accumulation loop
                def accumulate_loop(j, acc):
                    # Only include indices that are within the segment
                    valid_idx = (start_idx <= j) & (j < end_idx)
                    # Get the value at this index if valid
                    val = jnp.where(valid_idx, 
                                   src[j] if src.ndim == 1 else src[j],
                                   jnp.zeros_like(src[0]))
                    # Add to accumulator only if valid
                    return acc + jnp.where(valid_idx, val, jnp.zeros_like(val))
                
                # Run the accumulation loop
                max_src_idx = src.shape[0]  # Maximum possible source index
                acc = jax.lax.fori_loop(0, max_src_idx, accumulate_loop, acc)
                
                # Average if needed
                if reduction == "mean" and segment_size > 0:
                    acc = acc / segment_size
                
                # Update output
                return out.at[i_pt].set(acc)
        
        # Handle empty segments
        return jax.lax.cond(
            segment_size > 0,
            lambda _: process_nonempty(),
            lambda _: out,
            operand=None
        )
    
    # Process all segments
    out = jax.lax.fori_loop(0, n_out, sum_segment, out)
    
    return out


