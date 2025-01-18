import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point, LineString, MultiPolygon
from shapely import make_valid
import os
from typing import List, Tuple, Optional
from pathlib import Path

class ObstacleAvoidancePlanner:
    def __init__(self, coverage_path_file: str, obstacles_folder: str, safety_margin: float = 0.5):
        """
        Initialize the obstacle avoidance planner
        Args:
            coverage_path_file: Path to the CSV file containing the original coverage path
            obstacles_folder: Path to folder containing obstacle boundary CSV files
            safety_margin: Safety margin around obstacles in meters
        """
        self.safety_margin = safety_margin
        self.path = self.load_path(coverage_path_file)
        self.obstacles = self.load_obstacles(obstacles_folder)
        self.buffered_obstacles = [obs.buffer(safety_margin) for obs in self.obstacles]

    def load_path(self, path_file: str) -> np.ndarray:
        """Load the original coverage path"""
        try:
            df = pd.read_csv(path_file)
            if 'x' not in df.columns or 'y' not in df.columns:
                raise ValueError("Path CSV must contain 'x' and 'y' columns")
            return np.array(list(zip(df['x'], df['y'])))
        except Exception as e:
            print(f"Error loading path: {e}")
            raise

    def load_obstacles(self, obstacles_folder: str) -> List[Polygon]:
        """Load all obstacle boundaries from CSV files"""
        obstacles = []
        try:
            folder = Path(obstacles_folder)
            if not folder.exists():
                raise FileNotFoundError(f"Obstacles folder not found: {obstacles_folder}")

            for file in folder.glob("*.csv"):
                try:
                    df = pd.read_csv(file)
                    if 'x' not in df.columns or 'y' not in df.columns:
                        print(f"Skipping {file}: missing required columns")
                        continue

                    coords = np.array(list(zip(df['x'], df['y'])))
                    if not np.array_equal(coords[0], coords[-1]):
                        coords = np.vstack([coords, coords[0]])

                    polygon = Polygon(coords)
                    if not polygon.is_valid:
                        polygon = make_valid(polygon)
                    
                    if isinstance(polygon, MultiPolygon):
                        # Take the largest polygon if we have multiple
                        polygon = max(polygon.geoms, key=lambda p: p.area)
                    
                    obstacles.append(polygon)
                    print(f"Loaded obstacle from {file.name}")
                except Exception as e:
                    print(f"Error loading obstacle {file}: {e}")

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
        """Plot original path, obstacles, and new path"""
        try:
            plt.figure(figsize=(12, 12))
            
            # Plot original path
            plt.plot(original_path[:,0], original_path[:,1], 
                    'b--', linewidth=1, label='Original Path')
            
            # Plot obstacles
            for obstacle in self.obstacles:
                x, y = obstacle.exterior.xy
                plt.plot(x, y, 'r-', linewidth=2)
            
            # Plot buffered obstacles
            for obstacle in self.buffered_obstacles:
                x, y = obstacle.exterior.xy
                plt.plot(x, y, 'r--', linewidth=1)
            
            # Plot new path with arrows
            for i in range(len(new_path)-1):
                dx = new_path[i+1,0] - new_path[i,0]
                dy = new_path[i+1,1] - new_path[i,1]
                
                plt.plot([new_path[i,0], new_path[i+1,0]], 
                        [new_path[i,1], new_path[i+1,1]], 
                        'g-', linewidth=2)
                
                # Add direction arrow at midpoint
                mid_x = (new_path[i,0] + new_path[i+1,0]) / 2
                mid_y = (new_path[i,1] + new_path[i+1,1]) / 2
                
                plt.arrow(mid_x - dx*0.1, mid_y - dy*0.1,
                         dx*0.2, dy*0.2,
                         head_width=0.3,
                         head_length=0.3,
                         fc='g', ec='g',
                         length_includes_head=True)
            
            # Plot start and end points
            plt.plot(new_path[0,0], new_path[0,1], 'go', markersize=10, label='Start')
            plt.plot(new_path[-1,0], new_path[-1,1], 'ro', markersize=10, label='End')
            
            plt.grid(True)
            plt.legend()
            plt.axis('equal')
            plt.title('Path Replanning with Obstacle Avoidance')
            plt.show()
            
        except Exception as e:
            print(f"Error plotting: {e}")

    def save_path(self, path: np.ndarray, output_file: str):
        """
        Save the replanned path to CSV with index column
        Args:
            path: Path points to save
            output_file: Output file path
        """
        try:
            # Create DataFrame with x and y columns
            df = pd.DataFrame(path, columns=['x', 'y'])
            
            # Reset index to create sequential numbering starting from 0
            df = df.reset_index()
            
            # Round coordinates to 12 decimal places to match format
            df['x'] = df['x'].round(12)
            df['y'] = df['y'].round(12)
            
            # Save to CSV with specific formatting
            df.to_csv(output_file, index=False, float_format='%.12f')
            print(f"Path saved to {output_file}")
        except Exception as e:
            print(f"Error saving path: {e}")
            raise

def main():
    try:
        # Get the current script's directory
        script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
        
        # Input/output paths
        coverage_path = script_dir / "/home/flo/mmr_ws/codes/coverage_path.csv"
        obstacles_folder = script_dir / "/home/flo/mmr_ws/paths/obstacles"
        output_path = script_dir / "replanned_path.csv"
        
        # Safety margin in meters
        safety_margin = 0.1
        
        # Create planner
        planner = ObstacleAvoidancePlanner(
            coverage_path_file=str(coverage_path),
            obstacles_folder=str(obstacles_folder),
            safety_margin=safety_margin
        )
        
        print("Replanning path...")
        original_path = planner.path
        new_path = planner.replan_path()
        
        print("Plotting result...")
        planner.plot_path(original_path, new_path)
        
        print("Saving replanned path...")
        planner.save_path(new_path, str(output_path))
        
        print("Done!")
        return 0
        
    except Exception as e:
        print(f"Error in main: {e}")
        return 1

if __name__ == "__main__":
    exit(main())