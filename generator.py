import random

def generate_symmetric_matrix(size):
    """
    Generates a symmetric adjacency matrix of size 'size'.

    Args:
        size: The size of the square matrix.

    Returns:
        A symmetric adjacency matrix represented as a 2D list.
    """
    # Create an empty list to store the matrix
    matrix = [[0 for _ in range(size)] for _ in range(size)]

    # Fill the lower triangular part of the matrix with random connections
    for i in range(size):
        for j in range(i, size):
            if i == j:
                continue  # Skip the diagonal (no self-connections)
            connection_probability = 0.4  # Adjust for desired sparsity
            if random.random() < connection_probability:
                connection_value = random.randint(1, 9)  # Generate weight if there's a connection
            else:
                connection_value = 0  # Set to zero if no connection # Random weight
            matrix[i][j] = connection_value
            matrix[j][i] = connection_value  # Copy to upper triangular part for symmetry

    return matrix

# Example usage: Define the matrix size
size = 9

# Generate the symmetric adjacency matrix
symmetric_matrix = generate_symmetric_matrix(size)

# Print the matrix
for row in symmetric_matrix:
  print(*row)