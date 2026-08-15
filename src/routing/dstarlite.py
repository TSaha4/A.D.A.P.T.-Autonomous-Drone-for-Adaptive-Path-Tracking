import heapq
from typing import List, Tuple, Dict, Set
import numpy as np

class DStarLite:
    """
    Simplified D* Lite algorithm for 2D grid pathfinding.
    Supports efficient replanning when obstacles are discovered.
    """
    def __init__(self, grid_shape: Tuple[int, int], start: Tuple[int, int], goal: Tuple[int, int]):
        self.rows, self.cols = grid_shape
        self.start = start
        self.goal = goal
        self.km = 0.0
        
        # rhs and g values
        self.rhs = {}
        self.g = {}
        
        # Priority queue and a set to keep track of elements in it
        self.U = []
        self.U_set = set()
        
        # Obstacles
        self.obstacles = set()
        
        # Initialize
        self.rhs[self.goal] = 0.0
        self.insert(self.goal, self.calculate_key(self.goal))

    def heuristic(self, p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
        # Chebyshev distance for 8-way movement
        return max(abs(p1[0] - p2[0]), abs(p1[1] - p2[1]))

    def calculate_key(self, u: Tuple[int, int]) -> Tuple[float, float]:
        g_rhs_min = min(self.g.get(u, float('inf')), self.rhs.get(u, float('inf')))
        return (g_rhs_min + self.heuristic(self.start, u) + self.km, g_rhs_min)

    def insert(self, u: Tuple[int, int], key: Tuple[float, float]):
        heapq.heappush(self.U, (key, u))
        self.U_set.add(u)

    def top_key(self) -> Tuple[float, float]:
        while self.U:
            key, u = self.U[0]
            if u in self.U_set:
                return key
            heapq.heappop(self.U)
        return (float('inf'), float('inf'))

    def pop(self) -> Tuple[int, int]:
        while self.U:
            _, u = heapq.heappop(self.U)
            if u in self.U_set:
                self.U_set.remove(u)
                return u
        return None

    def get_neighbors(self, u: Tuple[int, int]) -> List[Tuple[int, int]]:
        neighbors = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx, ny = u[0] + dx, u[1] + dy
                if 0 <= nx < self.cols and 0 <= ny < self.rows:
                    neighbors.append((nx, ny))
        return neighbors

    def cost(self, u: Tuple[int, int], v: Tuple[int, int]) -> float:
        if u in self.obstacles or v in self.obstacles:
            return float('inf')
        return np.sqrt((u[0]-v[0])**2 + (u[1]-v[1])**2)

    def update_vertex(self, u: Tuple[int, int]):
        if u != self.goal:
            min_rhs = float('inf')
            for n in self.get_neighbors(u):
                min_rhs = min(min_rhs, self.cost(u, n) + self.g.get(n, float('inf')))
            self.rhs[u] = min_rhs
            
        if u in self.U_set:
            self.U_set.remove(u)
            
        if self.g.get(u, float('inf')) != self.rhs.get(u, float('inf')):
            self.insert(u, self.calculate_key(u))

    def compute_shortest_path(self):
        while self.U_set and (self.top_key() < self.calculate_key(self.start) or 
                              self.rhs.get(self.start, float('inf')) != self.g.get(self.start, float('inf'))):
            k_old = self.top_key()
            u = self.pop()
            if u is None:
                break
                
            k_new = self.calculate_key(u)
            if k_old < k_new:
                self.insert(u, k_new)
            elif self.g.get(u, float('inf')) > self.rhs.get(u, float('inf')):
                self.g[u] = self.rhs.get(u, float('inf'))
                for n in self.get_neighbors(u):
                    self.update_vertex(n)
            else:
                self.g[u] = float('inf')
                self.update_vertex(u)
                for n in self.get_neighbors(u):
                    self.update_vertex(n)

    def plan_path(self, obstacles: Set[Tuple[int, int]] = None) -> List[Tuple[int, int]]:
        """
        Plans and returns the path from start to goal.
        """
        if obstacles:
            for obs in obstacles:
                if obs not in self.obstacles:
                    self.obstacles.add(obs)
                    self.update_vertex(obs)
                    for n in self.get_neighbors(obs):
                        self.update_vertex(n)
                        
        self.compute_shortest_path()
        
        path = [self.start]
        current = self.start
        
        if self.g.get(self.start, float('inf')) == float('inf'):
            return [] # No path found
            
        while current != self.goal:
            best_n = None
            best_cost = float('inf')
            for n in self.get_neighbors(current):
                cost = self.cost(current, n) + self.g.get(n, float('inf'))
                if cost < best_cost:
                    best_cost = cost
                    best_n = n
            
            if best_n is None or best_cost == float('inf'):
                break
                
            current = best_n
            path.append(current)
            
            # Prevent infinite loops
            if len(path) > self.rows * self.cols:
                break
                
        return path
