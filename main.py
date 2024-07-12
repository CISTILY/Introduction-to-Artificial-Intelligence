# TODO: Import libraries
import numpy as np
import tracemalloc
import time
from heapq import heappush, heappop


# 1. Search Strategies Implementation
# 1.1. Breadth-first search (BFS)
def bfs(arr, source, destination): # done
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

    # Initialize visited dictionary (key: node, value: predecessor node), path and queue
    path = []
    visited = {source: None}   # Mark source as visited with no predecessor
    queue = [source]

    num_nodes = len(arr)  # Get the number of nodes from the matrix size

    while queue:
        current_node = queue.pop(0)  # Dequeue the first node

        # Check if the current node is the destination
        if current_node == destination:
            break  # Path found, break the loop

        # Explore unvisited neighbors
        for neighbor in range(num_nodes):
            if arr[current_node][neighbor] != 0 and neighbor not in visited:
                visited[neighbor] = current_node  # Update predecessor for neighbor
                queue.append(neighbor)  # Enqueue unvisited neighbor

    # Construct the path (if found) by backtracking through predecessors
    if destination in visited:
        current_node = destination
        while current_node is not None:
            path.append(current_node)
            current_node = visited[current_node]
        path.reverse()  # Reverse the path to get the correct order (source to destination)
    else:
        return visited, path

    return visited, path


# 1.2. Depth-first search (DFS)
def dfs(arr, source, destination): # done
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

    # Initialize visited dictionary (key: node, value: predecessor node), path and queue
    path = []
    visited = {source: None}   # Mark source as visited with no predecessor
    stack = [source]
    
    num_nodes = len(arr)  # Get the number of nodes from the matrix size

    while stack:
        current_node = stack.pop()  # pop the last node

        # Check if the current node is the destination
        if current_node == destination:
            break  # Path found, break the loop

        # Explore unvisited neighbors
        for neighbor in range(num_nodes):
            if arr[current_node][neighbor] != 0 and neighbor not in visited:
                visited[neighbor] = current_node # Update predecessor for neighbor
                stack.append(neighbor)
    # Construct the path (if found) by backtracking through predecessors
    if destination in visited:
        current_node = destination
        while current_node is not None:
            path.append(current_node)
            current_node = visited[current_node]
        path.reverse()  # Reverse the path to get the correct order (source to destination)
    else:
        return visited, path

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
        
        if current == destination:
            break

        for neighbor in range(num_nodes):
            if arr[current][neighbor] != 0:
                total_cost = current_cost + arr[current][neighbor]
                if neighbor not in costs or total_cost < costs[neighbor]:
                    costs[neighbor] = total_cost
                    visited[neighbor] = current
                    heappush(priority_queue, (total_cost, neighbor))

    if destination in visited:
        current_node = destination
        while current_node is not None:
            path.append(current_node)
            current_node = visited[current_node]
        path.reverse()  # Reverse the path to get the correct order (source to destination)
    else:
        return visited, path    

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

    while stack:
        current_node, current_depth = stack.pop()

        if current_depth < depth_limit:
            for neighbor in range(len(arr)):
                if arr[current_node][neighbor] != 0 and neighbor not in visited:
                    visited[neighbor] = current_node
                    stack.append((neighbor, current_depth + 1))

        if current_node == destination:
            while current_node is not None:
                path.append(current_node)
                current_node = visited[current_node]
            path.reverse()
            return visited, path

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
            print(depth_limit)
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

        if current == destination:
            break
        
        for neighbor in range(num_nodes):
            if arr[current][neighbor] != 0:
                if neighbor not in visited:
                    visited[neighbor] = current
                    heappush(priority_queue, (heuristic[neighbor], neighbor))
            print(priority_queue)

    if destination in visited:
        current_node = destination
        while current_node is not None:
            path.append(current_node)
            # print(visited[current_node])
            current_node = visited[current_node]
        path.reverse()  # Reverse the path to get the correct order (source to destination)
    else:
        return visited, path    

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
        
        if current == destination:
            break

        for neighbor in range(num_nodes):
            if arr[current][neighbor] != 0:
                total_cost = current_cost - heuristic[current] + arr[current][neighbor] + heuristic[neighbor]
                if neighbor not in costs or total_cost < costs[neighbor]:
                    costs[neighbor] = total_cost
                    visited[neighbor] = current
                    heappush(priority_queue, (total_cost, neighbor))

    if destination in visited:
        current_node = destination
        while current_node is not None:
            path.append(current_node)
            current_node = visited[current_node]
        path.reverse()  # Reverse the path to get the correct order (source to destination)
    else:
        return visited, path    

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
    path = []
    visited = {source: None}
    current = source
    
    while current != destination:
        neighbors = [(neighbor, heuristic[neighbor]) for neighbor, connected in enumerate(arr[current]) if connected]
        if not neighbors:
            break
        
        next_node = min(neighbors, key=lambda x: x[1])[0]
        
        if heuristic[next_node] >= heuristic[current]:
            break
        
        visited[next_node] = current
        current = next_node
    
    # Reconstruct the path
    if current == destination:
        node = destination
        while node is not None:
            path.append(node)
            node = visited[node]
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
    f = open(file_path, "r")
    numberOfNodes = int(f.readline())
    startAndGoal = f.readline()
    startAndGoal = startAndGoal.split(' ')
    startNode = int(startAndGoal[0])
    goalNode = int(startAndGoal[1][0: len(startAndGoal[1])-1])
    matrix = np.zeros((numberOfNodes, numberOfNodes))
    
    for i in range(0,numberOfNodes,1):
        temp = f.readline()
        vector = np.fromstring(temp[0:len(temp)-1], sep=' ')

        matrix[i] += vector

    temp = f.readline()
    heuristic = np.fromstring(temp[0:len(temp)], sep=' ')
    f.close()
    return startNode, goalNode, matrix, heuristic

# 2. Main function
if __name__ == "__main__":
    # TODO: Read the input data
    file_path = input('Enter input file\'s path: ')
    name_index = file_path.rfind('.')
    file_name = file_path[0:name_index].split("\\")[-1]

    startNode, goalNode, matrix, heuristic = readInputData(file_path)
    result_file = "output_" + file_name + ".txt"
    f = open(result_file, "a")
    f.seek(0)
    f.truncate()
    for algoChoice in range(7):
    # TODO: Start measuring
        tracemalloc.start()
        start_time = time.time()

    # TODO: Call a function to execute the path finding process
        display_path = ''
        currentAlgo = ''
        if algoChoice == 0:
            currentAlgo = 'BFS:\n'
            visited, path = bfs(matrix, startNode, goalNode)
            # print("BFS")
            # print(visited)
            # print(path)
        elif algoChoice == 1:
            currentAlgo = 'DFS:\n'
            visited, path = dfs(matrix, startNode, goalNode)
            # print("DFS")
            # print(visited)
            # print(path)
        elif algoChoice == 2:
            currentAlgo = 'UCS:\n'
            visited, path = ucs(matrix, startNode, goalNode)
            # print("UCS")
            # print(visited)
            # print(path)
        elif algoChoice == 3:
            currentAlgo = 'IDS:\n'
            visited, path = ids(matrix, startNode, goalNode)
            # print("IDS")
            # print(visited)
            # print(path)
        elif algoChoice == 4:
            currentAlgo = 'GBFS:\n'
            visited, path = gbfs(matrix, startNode, goalNode, heuristic)
            # print("GBFS")
            # print(visited)
            # print(path)
        elif algoChoice == 5:
            currentAlgo = 'A*:\n'
            visited, path = astar(matrix, startNode, goalNode, heuristic)
            # print("A*")
            # print(visited)
            # print(path)
        elif algoChoice == 6:
            currentAlgo = 'HC:\n'
            visited, path = hc(matrix, startNode, goalNode, heuristic)
            # print("HC")
            # print(visited)
            # print(path)
    # TODO: Stop measuring s
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        endTime = time.time() - start_time
    # TODO: Show the output data
        for i in range(len(path)):
            if (i == len(path)-1):
                display_path = display_path + str(path[i])
            else:
                display_path = display_path + str(path[i]) + " -> "
            
        f.write(currentAlgo)
        if path:
            f.write(f"Path: {display_path}\n")
        else:
            f.write(f"Path: -1\n")
        f.write(f"Time: {endTime:.6f} seconds\n")
        f.write(f"Memory: {peak/1024:.2f} KB\n")
        f.write("\n")
    f.close()
    pass