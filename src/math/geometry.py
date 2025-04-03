import jax
import jax.numpy as jnp
from typing import NamedTuple, Optional, Tuple, List
from functools import partial
from scipy.spatial import KDTree as KDTree_scipy
import numpy as np
from src.math.sparse import COOMatrix, CSRMatrix
from jax.experimental.sparse import BCSR
@jax.jit
def get_rotation_matrix(theta):
    return jnp.array([[jnp.cos(theta), -jnp.sin(theta)], [jnp.sin(theta), jnp.cos(theta)]])

@jax.jit
def get_rotation_angle(rotation_matrix):
    return jnp.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])


class KDTree:
    def __init__(self, points):
        """
        Implement KD tree using JAX
        
        Parameters:
            points: shape (n, k), where n is the number of points and k is the dimension
        """
        self.points = points
        self.n, self.k = points.shape
        self.tree = self._build_tree(jnp.arange(self.n), 0)
        
    def _build_tree(self, indices, depth):
        """
        Recursively build KD tree
        
        Parameters:
            indices: indices of points in the current subtree
            depth: current depth (used to determine the split axis)
            
        Returns:
            dictionary containing the tree structure
        """
        if len(indices) == 0:
            return None
            
        # select axis based on depth
        axis = depth % self.k
        
        # sort points based on the selected axis
        sorted_idx = jnp.argsort(self.points[indices, axis])
        sorted_indices = indices[sorted_idx]
        
        # select median as split point
        median_idx = len(sorted_indices) // 2
        node_idx = sorted_indices[median_idx]
        
        # recursively build left and right subtrees
        left_indices = sorted_indices[:median_idx]
        right_indices = sorted_indices[median_idx+1:]
        
        return {
            'index': node_idx,
            'point': self.points[node_idx],
            'axis': axis,
            'left': self._build_tree(left_indices, depth + 1),
            'right': self._build_tree(right_indices, depth + 1)
        }
    
    def radius_neighbors(self, query_point, radius):
        """
        Find all points within a radius of the query point
        
        Parameters:
            query_point: query point, shape (k,)
            radius: search radius
            
        Returns:
            list of indices of points within the radius
        """
        radius_squared = radius ** 2
        found_indices = []
        
        def search_recursive(node):
            if node is None:
                return
                
            # calculate the distance to the current node point
            point_idx = node['index']
            point = node['point']
            dist_squared = jnp.sum((point - query_point) ** 2)
            
            # if within radius, add to results
            if dist_squared <= radius_squared:
                found_indices.append(int(point_idx))
                
            # calculate the distance to the split hyperplane
            axis = node['axis']
            dist_to_plane = query_point[axis] - point[axis]
            
            # based on the distance to the split hyperplane, decide which subtree to search
            if dist_to_plane <= radius:
                search_recursive(node['left'])
                
            if dist_to_plane >= -radius:
                search_recursive(node['right'])
        
        # start searching from the root node
        search_recursive(self.tree)
        return jnp.array(found_indices)


# @jax.jit
# def radius_neighbors_vectorized(points, query_point, radius):
#     """
#     Radius neighbor search using JAX vectorized operations
    
#     This function is more efficient when JIT compilation and GPU acceleration are needed
    
#     Parameters:
#         points: shape (n, k), where n is the number of points and k is the dimension
#         query_point: shape (k,)
#         radius: search radius
        
#     Returns:
#         indices of points within the radius
#     """
#     # calculate the squared distance to all points
#     squared_distances = jnp.sum((points - query_point) ** 2, axis=1)
    
#     # find points within the radius
#     mask = squared_distances <= radius ** 2
#     indices = jnp.where(mask)[0]
    
#     return indices


def batch_radius_neighbors(points, queries, radius):
    """
    Batch radius neighbor search for multiple query points
    
    Parameters:
        points: shape (n, k), where n is the number of points and k is the dimension
        queries: shape (m, k), where m is the number of query points
        radius: search radius
        
    Returns:
        list of indices of points within the radius for each query point 
        to be noted that the length for each query is different
    """
    # vmap_radius_search = jax.vmap(lambda q: radius_neighbors_vectorized(points, q, radius))

    
    # return vmap_radius_search(queries)
    # use scipy KDTree
    tree = KDTree_scipy(points)

    indices = tree.query_ball_point(queries, radius)
    total_neighbors = sum(len(idx_list) for idx_list in indices)
    jax.debug.print(f"Total neighbors: {total_neighbors}")

    return indices


def jax_batch_radius_neighbors(points, queries, radius):
    """
    Batch radius neighbor search for multiple query points using JAX.
    
    Parameters:
    points: shape (n, k), where n is the number of points and k is the dimension
    queries: shape (m, k), where m is the number of query points
    radius: search radius
    
    Returns:
    list of indices of points within the radius for each query point
    to be noted that the length for each query is different
    """
    # Calculate squared distances between queries and points
    # We use squared distance to avoid unnecessary sqrt operations
    # and compare with squared radius later
    squared_radius = radius ** 2
    
    # Compute squared distances for all query-point pairs
    # reshape queries to (m, 1, k) and points to (1, n, k) for broadcasting
    queries_expanded = jnp.expand_dims(queries, axis=1)  # Shape (m, 1, k)
    points_expanded = jnp.expand_dims(points, axis=0)    # Shape (1, n, k)
    
    # Calculate squared Euclidean distances
    squared_distances = jnp.sum((queries_expanded - points_expanded) ** 2, axis=2)  # Shape (m, n)
    
    # For each query, find which points are within radius
    # This creates a boolean mask of shape (m, n)
    mask = squared_distances <= squared_radius
    
    # Use vmap to process each query independently
    def process_single_query(mask_row):
        # Get indices where mask is True
        indices = jnp.where(mask_row)[0]
        return indices
    
    # Map the function over each row of the mask
    result = jax.vmap(process_single_query)(mask)
    
    return result

def find_nearest_neighbors_in_ball(query_points, reference_points, radius, sparse=False):
    """
    Jit version of find_nearest_neighbors_in_ball
    
    Args:
        query_points: shape (n_queries, dim)
        reference_points: shape (n_refs, dim)
        radius: search radius
        
    Returns:
        neighbors_mask: boolean matrix, shape (n_queries, n_refs), indicating the neighbor relation between points
    """
    # 计算查询点与所有参考点之间的平方欧氏距离
    # 使用 (a-b)^2 = a^2 - 2ab + b^2 展开计算
    
    # [n_queries, 1]
    query_norm_sq = jnp.sum(query_points**2, axis=1, keepdims=True)
    
    # [1, n_refs]
    ref_norm_sq = jnp.sum(reference_points**2, axis=1, keepdims=True).T
    
    # [n_queries, n_refs]
    dot_product = jnp.dot(query_points, reference_points.T)
    
    # [n_queries, n_refs]
    squared_distances = query_norm_sq + ref_norm_sq - 2 * dot_product
    
    # 使用数值稳定性修正，防止数值误差导致的负距离
    squared_distances = jnp.maximum(squared_distances, 0.0)
    
    # 对于在球内的点，标记为True
    neighbors_mask = squared_distances < (radius ** 2)
    print("query_points.shape", query_points.shape)
    print("reference_points.shape", reference_points.shape)
    print("neighbors_mask.shape", neighbors_mask.shape)
    if sparse:
        sparse_matrix_csr = CSRMatrix.from_dense(neighbors_mask)
        return sparse_matrix_csr
    else:
        return neighbors_mask