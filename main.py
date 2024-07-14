# TODO: Import libraries
import numpy as np
import tracemalloc
import time
from heapq import heappush, heappop

# 1. Search Strategies Implementation
# 1.1. Breadth-first search (BFS)
def bfs(arr, source, destination):
    """
    BFS algorithm:
    Parameters:
    ---------------------------
    arr: list / numpy array 
        The graph's adjacency matrix
    source: integer
        Starting node
    destination: integer
        Ending node
    
    Returns
    ---------------------
    visited: dictionary
        The dictionary contains visited nodes, each key is a visited node,
        each value is the adjacent node visited before it.
    path: list
        Founded path
    """
    # TODO
    path = []
    visited = {source: None} 
    queue = [source]
    num_nodes = len(arr) 

    while queue:
        current_node = queue.pop(0)

        # Check if the current node is the destination
        if current_node == destination:
            break 

        # Explore unvisited neighbors
        for neighbor in range(num_nodes):
            if arr[current_node][neighbor] != 0 and neighbor not in visited:
                visited[neighbor] = current_node
                queue.append(neighbor)  

    # Reconstruct the path
    if destination in visited:
        current_node = destination
        while current_node is not None:
            path.append(current_node)
            current_node = visited[current_node]
        path.reverse()  # Reverse the path to get the correct order (source to destination)  

    return visited, path


# 1.2. Depth-first search (DFS)
def dfs(arr, source, destination):
    """
    DFS algorithm:
    Parameters:
    ---------------------------
    arr: list / numpy array 
        The graph's adjacency matrix
    source: integer
        Starting node
    destination: integer
        Ending node
    
    Returns
    ---------------------
    visited: dictionary
        The dictionary contains visited nodes, each key is a visited node,
        each value is the adjacent node visited before it.
    path: list
        Founded path
    """
    # TODO
    path = []
    visited = {source: None} 
    stack = [source]
    num_nodes = len(arr) 

    while stack:
        current_node = stack.pop()  # pop the last node

        # Check if the current node is the destination
        if current_node == destination:
            break  # Path found, break the loop

        # Explore unvisited neighbors
        for neighbor in range(num_nodes):
            if arr[current_node][neighbor] != 0 and neighbor not in visited:
                visited[neighbor] = current_node
                stack.append(neighbor)
                
    # Reconstruct the path
    if destination in visited:
        current_node = destination
        while current_node is not None:
            path.append(current_node)
            current_node = visited[current_node]
        path.reverse()  # Reverse the path to get the correct order (source to destination)

    return visited, path


# 1.3. Uniform-cost search (UCS)
def ucs(arr, source, destination): # done
    """
    UCS algorithm:
    Parameters:
    ---------------------------
    arr: list / numpy array 
        The graph's adjacency matrix
    source: integer
        Starting node
    destination: integer
        Ending node
    
    Returns
    ---------------------
    visited: dictionary
        The dictionary contains visited nodes, each key is a visited node,
        each value is the adjacent node visited before it.
    path: list
        Founded path
    """
    # TODO
    path = []
    visited = {source: None}
    costs = {source: 0}
    num_nodes = len(arr)

    priority_queue = [(0, source)]

    while priority_queue:
        current_cost, current = heappop(priority_queue)
        
        # Check if the current node is the destination
        if current == destination:
            break
        
        # Explore unvisited neighbors
        for neighbor in range(num_nodes):
            if arr[current][neighbor] != 0:
                total_cost = current_cost + arr[current][neighbor]
                if neighbor not in costs or total_cost < costs[neighbor]:
                    costs[neighbor] = total_cost
                    visited[neighbor] = current
                    heappush(priority_queue, (total_cost, neighbor))

    # Reconstruct the path
    if destination in visited:
        current_node = destination
        while current_node is not None:
            path.append(current_node)
            current_node = visited[current_node]
        path.reverse()  # Reverse the path to get the correct order (source to destination)

    return visited, path


# 1.4. Iterative deepening search (IDS)
# 1.4.a. Depth-limited search
def dls(arr, source, destination, depth_limit):
    """
    DLS algorithm:
    Parameters:
    ---------------------------
    arr: list / numpy array 
        The graph's adjacency matrix
    source: integer
        Starting node
    destination: integer
        Ending node
    depth_limit: integer
        Maximum depth for search
    
    Returns
    ---------------------
    visited: dictionary
        The dictionary contains visited nodes, each key is a visited node,
        each value is the adjacent node visited before it.
    path: list
        Founded path
    """
    # TODO
    path = []
    visited = {source: None}
    stack = [(source, 0)]  # (node, depth)
    num_nodes = len(arr)

    while stack:
        current_node, current_depth = stack.pop()

        # Check if current_depth < depth_limit
        if current_depth < depth_limit:
            if current_node == destination: # Check if the current node is the destination
                while current_node is not None:
                    path.append(current_node)
                    current_node = visited[current_node]
                path.reverse()
                return visited, path
            
            # Explore unvisited neighbors
            for neighbor in range(num_nodes):
                if arr[current_node][neighbor] != 0 and neighbor not in visited:
                    visited[neighbor] = current_node
                    stack.append((neighbor, current_depth + 1))

    return visited, path  # No path found

# 1.4.b. IDS
def ids(arr, source, destination):
    """
    IDS algorithm:
    Parameters:
    ---------------------------
    arr: list / numpy array 
        The graph's adjacency matrix
    source: integer
        Starting node
    destination: integer
        Ending node
    
    Returns
    ---------------------
    visited: dictionary
        The dictionary contains visited nodes, each key is a visited node,
        each value is the adjacent node visited before it.
    path: list
        Founded path
    """
    # TODO
    path = []
    visited = {source: None}
    depth_limit = 0

    while True:
        visited, path = dls(arr, source, destination, depth_limit)
        if path:
            return visited, path

        depth_limit += 1


# 1.5. Greedy best first search (GBFS)
def gbfs(arr, source, destination, heuristic):
    """
    GBFS algorithm:
    Parameters:
    ---------------------------
    arr: list / numpy array 
        The graph's adjacency matrix
    source: integer
        Starting node
    destination: integer
        Ending node
    heuristic: list / numpy array
        The heuristic value from the current node to the goal
    
    Returns
    ---------------------
    visited: dictionary
        The dictionary contains visited nodes, each key is a visited node,
        each value is the adjacent node visited before it.
    path: list
        Founded path
    """
    # TODO
    path = []
    visited = {source: None}
    num_nodes = len(arr)
    priority_queue = [(heuristic[source], source)]

    while priority_queue:
        current_heuristic, current = heappop(priority_queue)

        # Check if the current node is the destination
        if current == destination:
            break
        
        # Explore unvisited neighbors
        for neighbor in range(num_nodes):
            if arr[current][neighbor] != 0 and neighbor not in visited:
                    visited[neighbor] = current
                    heappush(priority_queue, (heuristic[neighbor], neighbor))

    # Reconstruct the path
    if destination in visited:
        current_node = destination
        while current_node is not None:
            path.append(current_node)
            current_node = visited[current_node]
        path.reverse()  # Reverse the path to get the correct order (source to destination) 

    return visited, path


# 1.6. Graph-search A* (AStar)
def astar(arr, source, destination, heuristic): # Done
    """
    A* algorithm:
    Parameters:
    ---------------------------
    arr: list / numpy array 
        The graph's adjacency matrix
    source: integer
        Starting node
    destination: integer
        Ending node
    heuristic: list / numpy array
        The heuristic value from the current node to the goal
    
    Returns
    ---------------------
    visited: dictionary
        The dictionary contains visited nodes, each key is a visited node,
        each value is the adjacent node visited before it.
    path: list
        Founded path
    """
    # TODO
    path = []
    visited = {source: None}
    costs = {source: heuristic[0]}
    num_nodes = len(arr)

    priority_queue = [(heuristic[source], source)]

    while priority_queue:
        current_cost, current = heappop(priority_queue)
        
        # Check if the current node is the destination
        if current == destination:
            break
        
        # Explore unvisited neighbors
        for neighbor in range(num_nodes):
            if arr[current][neighbor] != 0:
                total_cost = current_cost - heuristic[current] + arr[current][neighbor] + heuristic[neighbor]
                if neighbor not in costs or total_cost < costs[neighbor]:
                    costs[neighbor] = total_cost
                    visited[neighbor] = current
                    heappush(priority_queue, (total_cost, neighbor))

    # Reconstruct the path
    if destination in visited:
        current_node = destination
        while current_node is not None:
            path.append(current_node)
            current_node = visited[current_node]
        path.reverse()  # Reverse the path to get the correct order (source to destination)

    return visited, path


# 1.7. Hill-climbing First-choice (HC)
def hc(arr, source, destination, heuristic):
    """
    HC algorithm:
    Parameters:
    ---------------------------
    arr: list / numpy array 
        The graph's adjacency matrix
    source: integer
        Starting node
    destination: integer
        Ending node
    heuristic: list / numpy array
        The heuristic value from the current node to the goal
    
    Returns
    ---------------------
    visited: dictionary
        The dictionary contains visited nodes, each key is a visited node,
        each value is the adjacent node visited before it.
    path: list
        Founded path
    """
    # TODO
    path = []
    visited = {source: None}
    current = source
    
    while current != destination:
        neighbors = [(neighbor, heuristic[neighbor]) for neighbor, connected in enumerate(arr[current]) if connected != 0] # Get all neighbor of current
        if not neighbors:
            break
        
        next_node = min(neighbors, key=lambda x: x[1])[0] # Find next_node
        
        if heuristic[next_node] >= heuristic[current]:
            break
        
        visited[next_node] = current
        current = next_node
    
    # Reconstruct the path
    if current == destination:
        current_node = destination
        while current_node is not None:
            path.append(current_node)
            current_node = visited[current_node]
        path.reverse()
    
    return visited, path

def readInputData(file_path):
    """
    Function to read the input txt file from a file_path
    Parameters:
    ---------------------------
    file_path: String
        The file path of input file
    
    Returns
    ---------------------
    startNode: int
        Starting node
    goalNode: int
        Ending node
    matrix: list / numpy array
        The graph's adjacency matrix
    heuristic: list / numpy array
        The heuristic value from the current node to the goal
    """
    # Open input file
    f = open(file_path, "r")

    # Read the first line for number of nodes
    numberOfNodes = int(f.readline()) 

    # Read second line for start and goal node
    startAndGoal = f.readline() 
    startAndGoal = startAndGoal.split(' ') 
    startNode = int(startAndGoal[0])
    goalNode = int(startAndGoal[1][0: len(startAndGoal[1])-1])

    # Read adjacency matrix
    matrix = np.zeros((numberOfNodes, numberOfNodes))
    for i in range(0,numberOfNodes,1):
        temp = f.readline()
        vector = np.fromstring(temp[0:len(temp)-1], sep=' ')
        matrix[i] += vector

    # Read heuristic
    temp = f.readline()
    heuristic = np.fromstring(temp[0:len(temp)], sep=' ')

    # Close input file
    f.close()

    return startNode, goalNode, matrix, heuristic

# 2. Main function
if __name__ == "__main__":
    # TODO: Read the input data

    # Get file_path
    file_path = input('Enter input file\'s path: ') 
    print("-----------------------")

    # Parse information from file_path
    name_index = file_path.rfind('.')
    file_name = file_path[0:name_index].split("\\")[-1]

    # Read data from input file
    startNode, goalNode, matrix, heuristic = readInputData(file_path) 

    # Open output file and remove any existing content
    result_file = "output_" + file_name + ".txt" 
    f = open(result_file, "a")
    f.seek(0)
    f.truncate()

    # Perform evey search strategies
    for algoChoice in range(7):

    # TODO: Start measuring
        tracemalloc.start()
        start_time = time.time()

    # TODO: Call a function to execute the path finding process
        display_path = ''
        currentAlgo = ''
        if algoChoice == 0:
            currentAlgo = 'BFS:'
            visited, path = bfs(matrix, startNode, goalNode)
        elif algoChoice == 1:
            currentAlgo = 'DFS:'
            visited, path = dfs(matrix, startNode, goalNode)
        elif algoChoice == 2:
            currentAlgo = 'UCS:'
            visited, path = ucs(matrix, startNode, goalNode)
        elif algoChoice == 3:
            currentAlgo = 'IDS:'
            visited, path = ids(matrix, startNode, goalNode)
        elif algoChoice == 4:
            currentAlgo = 'GBFS:'
            visited, path = gbfs(matrix, startNode, goalNode, heuristic)
        elif algoChoice == 5:
            currentAlgo = 'A*:'
            visited, path = astar(matrix, startNode, goalNode, heuristic)
        elif algoChoice == 6:
            currentAlgo = 'HC:'
            visited, path = hc(matrix, startNode, goalNode, heuristic)

    # TODO: Stop measuring
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        endTime = time.time() - start_time

    # TODO: Show the output data
        for i in range(len(path)):
            if (i == len(path)-1):
                display_path = display_path + str(path[i])
            else:
                display_path = display_path + str(path[i]) + " -> "
        
        f.write(f"{currentAlgo}\n")
        print(f"Current algorithm {currentAlgo}")
        print(f"Visited: {visited}")

        if path:
            f.write(f"Path: {display_path}\n")
            print(f"Path: {display_path}")
        else:
            f.write(f"Path: -1\n")
            print("Path: -1")
        
        f.write(f"Time: {endTime:.6f} seconds\n")
        print(f"Time: {endTime:.6f} seconds")
        f.write(f"Memory: {peak/1024:.2f} KB\n")
        print(f"Memory: {peak/1024:.2f} KB")
        f.write("\n")
        print("-----------------------")
        
    # Close output file
    f.close() 
    pass