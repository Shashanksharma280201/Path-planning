import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point, LineString, MultiPolygon
from shapely import make_valid
import os
import json
from typing import List, Tuple, Optional
from pathlib import Path

class ObstacleAvoidancePlanner:
    def __init__(self, coverage_path_file: str, obstacles_file: str, safety_margin: float = 0.5):
        """
        Initialize the obstacle avoidance planner
        Args:
            coverage_path_file: Path to the JSON file containing the original coverage path
            obstacles_file: Path to JSON file containing obstacle boundaries
            safety_margin: Safety margin around obstacles in meters
        """
        self.safety_margin = safety_margin
        self.path = self.load_path(coverage_path_file)
        self.obstacles = self.load_obstacles(obstacles_file)
        self.buffered_obstacles = [obs.buffer(safety_margin) for obs in self.obstacles]

    def load_path(self, path_file: str) -> np.ndarray:
        """Load the original coverage path from JSON"""
        try:
            with open(path_file, 'r') as f:
                data = json.load(f)
            
            # Expect JSON structure: {"path": [{"x": x1, "y": y1}, {"x": x2, "y": y2}, ...]}
            if 'path' not in data:
                raise ValueError("JSON must contain 'path' key")
            
            path_points = [(point['x'], point['y']) for point in data['path']]
            return np.array(path_points)
        except Exception as e:
            print(f"Error loading path: {e}")
            raise

    def load_obstacles(self, obstacles_file: str) -> List[Polygon]:
        """Load all obstacle boundaries from JSON file"""
        obstacles = []
        try:
            with open(obstacles_file, 'r') as f:
                data = json.load(f)
            
            # Expect JSON structure: {"obstacles": [{"boundary": [{"x": x1, "y": y1}, ...]}, ...]}
            if 'obstacles' not in data:
                raise ValueError("JSON must contain 'obstacles' key")

            for obstacle_data in data['obstacles']:
                try:
                    if 'boundary' not in obstacle_data:
                        print(f"Skipping obstacle: missing boundary")
                        continue

                    coords = [(point['x'], point['y']) for point in obstacle_data['boundary']]
                    coords = np.array(coords)

                    if not np.array_equal(coords[0], coords[-1]):
                        coords = np.vstack([coords, coords[0]])

                    polygon = Polygon(coords)
                    if not polygon.is_valid:
                        polygon = make_valid(polygon)
                    
                    if isinstance(polygon, MultiPolygon):
                        # Take the largest polygon if we have multiple
                        polygon = max(polygon.geoms, key=lambda p: p.area)
                    
                    obstacles.append(polygon)
                    print(f"Loaded obstacle {len(obstacles)}")
                except Exception as e:
                    print(f"Error loading obstacle: {e}")

            return obstacles
        except Exception as e:
            print(f"Error loading obstacles: {e}")
            return []

    def check_intersection(self, start: Tuple[float, float], end: Tuple[float, float]) -> Optional[Polygon]:
        """
        Check if a line segment intersects with any obstacle
        Returns the first intersecting obstacle or None
        """
        line = LineString([start, end])
        for obstacle in self.buffered_obstacles:
            if line.intersects(obstacle):
                return obstacle
        return None

    def generate_avoidance_points(self, 
                                start: Tuple[float, float], 
                                end: Tuple[float, float], 
                                obstacle: Polygon,
                                right_turn: bool = True) -> List[Tuple[float, float]]:
        """Generate points to avoid an obstacle with right turns"""
        try:
            # Create a buffered version of the obstacle
            buffered = obstacle.buffer(self.safety_margin)
            
            # Get the exterior coordinates of the buffered obstacle
            coords = list(buffered.exterior.coords)
            
            # Find the closest point on obstacle boundary to start point
            start_idx = min(range(len(coords)), 
                          key=lambda i: Point(coords[i]).distance(Point(start)))
            
            # Find the closest point on obstacle boundary to end point
            end_idx = min(range(len(coords)), 
                         key=lambda i: Point(coords[i]).distance(Point(end)))
            
            # Generate avoidance points based on turn direction
            avoidance_points = []
            if right_turn:
                # Go clockwise around obstacle
                if end_idx <= start_idx:
                    end_idx += len(coords) - 1
                points = coords[start_idx:end_idx+1]
            else:
                # Go counter-clockwise around obstacle
                if start_idx <= end_idx:
                    start_idx += len(coords) - 1
                points = coords[end_idx:start_idx+1][::-1]
            
            # Add intermediate points for smoother path
            for i in range(len(points)-1):
                pt1, pt2 = points[i], points[i+1]
                dist = Point(pt1).distance(Point(pt2))
                num_points = max(int(dist / self.safety_margin), 2)
                
                for j in range(num_points):
                    t = j / (num_points - 1)
                    point = (
                        pt1[0] + (pt2[0] - pt1[0]) * t,
                        pt1[1] + (pt2[1] - pt1[1]) * t
                    )
                    avoidance_points.append(point)
            
            return avoidance_points
            
        except Exception as e:
            print(f"Error generating avoidance points: {e}")
            return [start, end]

    def replan_path(self) -> np.ndarray:
        """Replan the path to avoid obstacles"""
        try:
            new_path = []
            current_pos = tuple(self.path[0])
            new_path.append(current_pos)
            
            for next_point in self.path[1:]:
                next_point = tuple(next_point)
                
                # Check if current segment intersects any obstacle
                obstacle = self.check_intersection(current_pos, next_point)
                
                if obstacle:
                    # Generate avoidance points (right turn by default)
                    avoidance_points = self.generate_avoidance_points(
                        current_pos, next_point, obstacle, right_turn=True
                    )
                    
                    # Add avoidance points to path
                    new_path.extend(avoidance_points)
                else:
                    new_path.append(next_point)
                
                current_pos = new_path[-1]
            
            return np.array(new_path)
            
        except Exception as e:
            print(f"Error replanning path: {e}")
            return self.path

    def plot_path(self, original_path: np.ndarray, new_path: np.ndarray):
        """Plot original path, obstacles, and new path with enhanced visualization"""
        try:
            # Set up the figure with a clean, modern style
            plt.style.use('seaborn')
            fig, ax = plt.subplots(figsize=(15, 15))
            
            # Plot original path with better styling
            plt.plot(original_path[:,0], original_path[:,1], 
                    color='#A0AEC0', linestyle='--', linewidth=2, 
                    label='Original Path', alpha=0.6)
            
            # Plot obstacles with distinct colors and styling
            for i, obstacle in enumerate(self.obstacles):
                x, y = obstacle.exterior.xy
                plt.fill(x, y, color='#FC8181', alpha=0.2)  # Light red fill
                plt.plot(x, y, color='#E53E3E', linewidth=2.5, 
                        label='Obstacle' if i == 0 else "")  # Solid red border
            
            # Plot buffered obstacles with distinct styling
            for i, obstacle in enumerate(self.buffered_obstacles):
                x, y = obstacle.exterior.xy
                plt.plot(x, y, color='#F56565', linestyle='--', 
                        linewidth=1.5, alpha=0.7,
                        label='Safety Margin' if i == 0 else "")
            
            # Plot new path with improved arrows and styling
            path_color = '#3182CE'  # Nice blue color
            arrow_color = '#2B6CB0'  # Slightly darker blue for arrows
            
            # Plot continuous path line first
            plt.plot(new_path[:,0], new_path[:,1], 
                    color=path_color, linewidth=3, 
                    label='Replanned Path', zorder=5)
            
            # Add direction arrows with improved spacing
            arrow_spacing = max(len(new_path) // 20, 1)  # Adaptive arrow spacing
            for i in range(0, len(new_path)-1, arrow_spacing):
                dx = new_path[i+1,0] - new_path[i,0]
                dy = new_path[i+1,1] - new_path[i,1]
                
                # Calculate midpoint
                mid_x = (new_path[i,0] + new_path[i+1,0]) / 2
                mid_y = (new_path[i,1] + new_path[i+1,1]) / 2
                
                # Add direction arrow
                plt.arrow(mid_x - dx*0.1, mid_y - dy*0.1,
                         dx*0.2, dy*0.2,
                         head_width=0.2,
                         head_length=0.3,
                         fc=arrow_color, ec=arrow_color,
                         length_includes_head=True,
                         zorder=6)
            
            # Plot start and end points with improved styling
            plt.plot(new_path[0,0], new_path[0,1], 'o', 
                    color='#38A169', markersize=12, label='Start',
                    zorder=7)
            plt.plot(new_path[-1,0], new_path[-1,1], 'o', 
                    color='#E53E3E', markersize=12, label='End',
                    zorder=7)
            
            # Improve grid and background
            plt.grid(True, linestyle='--', alpha=0.6)
            ax.set_facecolor('#F7FAFC')  # Light gray background
            
            # Enhance legend
            plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1),
                      frameon=True, fancybox=True, shadow=True,
                      fontsize=10)
            
            # Add title and labels with improved styling
            plt.title('Path Replanning with Obstacle Avoidance',
                     pad=20, fontsize=16, fontweight='bold')
            plt.xlabel('X Distance (meters)', labelpad=10, fontsize=12)
            plt.ylabel('Y Distance (meters)', labelpad=10, fontsize=12)
            
            # Ensure equal aspect ratio
            plt.axis('equal')
            
            # Add margins around the plot
            plt.margins(0.1)
            
            # Adjust layout to prevent label cutoff
            plt.tight_layout()
            
            plt.show()
            
        except Exception as e:
            print(f"Error plotting: {e}")

    def save_path(self, path: np.ndarray, output_file: str):
        """
        Save the replanned path to JSON
        Args:
            path: Path points to save
            output_file: Output file path
        """
        try:
            # Convert path to list of dictionaries
            path_points = [{'x': float(x), 'y': float(y)} for x, y in path]
            
            # Create output JSON structure
            output_data = {
                'path': path_points,
                'metadata': {
                    'num_points': len(path_points),
                    'safety_margin': self.safety_margin,
                    'num_obstacles': len(self.obstacles)
                }
            }
            
            # Save to JSON file with nice formatting
            with open(output_file, 'w') as f:
                json.dump(output_data, f, indent=2)
            print(f"Path saved to {output_file}")
        except Exception as e:
            print(f"Error saving path: {e}")
            raise

def main():
    try:
        # Input/output paths
        coverage_path = "coverage_path.json"
        obstacles_file = "obstacles.json"
        output_path = "replanned_path.json"
        
        # Safety margin in meters
        safety_margin = 0.4
        
        # Create planner
        planner = ObstacleAvoidancePlanner(
            coverage_path_file=coverage_path,
            obstacles_file=obstacles_file,
            safety_margin=safety_margin
        )
        
        print("Replanning path...")
        original_path = planner.path
        new_path = planner.replan_path()
        
        print("Plotting result...")
        planner.plot_path(original_path, new_path)
        
        print("Saving replanned path...")
        planner.save_path(new_path, output_path)
        
        print("Done!")
        return 0
        
    except Exception as e:
        print(f"Error in main: {e}")
        return 1

if __name__ == "__main__":
    exit(main())