import numpy as np
import pandas as pd
from shapely.geometry import Point, LineString, Polygon
from shapely.ops import nearest_points
import os
from pathlib import Path
import matplotlib.pyplot as plt

class PathReplanner:
    def __init__(self, planned_path_csv, obstacles_folder, safety_margin=0.5):
        self.safety_margin = safety_margin
        self.planned_path = self.load_planned_path(planned_path_csv)
        self.obstacles = self.load_obstacles(obstacles_folder)
        self.buffered_obstacles = self.create_buffered_obstacles()
        
    def load_planned_path(self, csv_path):
        try:
            df = pd.read_csv(csv_path)
            path = np.array(list(zip(df['x'], df['y'])))
            print(f"Loaded planned path: {len(path)} points")
            print(f"Path bounds: X({path[:,0].min():.2f}, {path[:,0].max():.2f}), Y({path[:,1].min():.2f}, {path[:,1].max():.2f})")
            return path
            
        except Exception as e:
            print(f"Error loading planned path: {e}")
            raise
            
    def load_obstacles(self, folder_path):
        obstacles = []
        try:
            for file in Path(folder_path).glob('*.csv'):
                print(f"Loading obstacle file: {file}")
                df = pd.read_csv(file)
                coords = np.array(list(zip(df['x'], df['y'])))
                
                if not np.array_equal(coords[0], coords[-1]):
                    coords = np.vstack([coords, coords[0]])
                
                obstacle = Polygon(coords)
                print(f"Obstacle bounds: {obstacle.bounds}")
                obstacles.append(obstacle)
                
            print(f"Loaded {len(obstacles)} obstacles")
            return obstacles
            
        except Exception as e:
            print(f"Error loading obstacles: {e}")
            raise
            
    def create_buffered_obstacles(self):
        return [obstacle.buffer(self.safety_margin) for obstacle in self.obstacles]

    def check_intersection(self, start_point, end_point, obstacle):
        """
        Check if line segment intersects with obstacle and get intersection points,
        ensuring correct ordering of entry/exit points
        """
        line = LineString([start_point, end_point])
        if line.intersects(obstacle):
            intersection = line.intersection(obstacle)
            points = []
            
            # Convert intersection to points
            if intersection.geom_type == 'Point':
                points = [intersection]
            elif intersection.geom_type == 'MultiPoint':
                points = list(intersection.geoms)
            elif intersection.geom_type in ['LineString', 'MultiLineString']:
                points = [Point(p) for p in intersection.coords]
            
            # Sort points by distance along the original line
            if points:
                base_line = LineString([start_point, end_point])
                points.sort(key=lambda p: base_line.project(p))
                return points
                
        return []

    def get_clockwise_boundary_points(self, obstacle, entry_point, exit_point):
        """
        Get obstacle boundary points in clockwise direction, ensuring shortest path
        between entry and exit points
        """
        boundary_coords = list(obstacle.exterior.coords)
        
        # Find nearest points on boundary
        entry_idx = min(range(len(boundary_coords)), 
                    key=lambda i: Point(boundary_coords[i]).distance(entry_point))
        exit_idx = min(range(len(boundary_coords)), 
                    key=lambda i: Point(boundary_coords[i]).distance(exit_point))
        
        # Get both possible paths (clockwise and counterclockwise)
        path1 = []  # Going forward from entry to exit
        path2 = []  # Going backward from entry to exit
        
        # Path going forward (handling wrap-around)
        curr_idx = entry_idx
        while curr_idx != exit_idx:
            path1.append(boundary_coords[curr_idx])
            curr_idx = (curr_idx + 1) % len(boundary_coords)
        path1.append(boundary_coords[exit_idx])
        
        # Path going backward (handling wrap-around)
        curr_idx = entry_idx
        while curr_idx != exit_idx:
            path2.append(boundary_coords[curr_idx])
            curr_idx = (curr_idx - 1) % len(boundary_coords)
        path2.append(boundary_coords[exit_idx])
        
        # Return the shorter path
        if len(path1) <= len(path2):
            return path1
        else:
            return path2

    def replan_path(self):
        replanned_path = []
        i = 0
        
        print("Starting path replanning...")
        intersections_found = 0
        
        while i < len(self.planned_path) - 1:
            start_point = self.planned_path[i]
            end_point = self.planned_path[i + 1]
            added_bypass = False
            
            for obstacle in self.buffered_obstacles:
                intersection_points = self.check_intersection(start_point, end_point, obstacle)
                
                if intersection_points and len(intersection_points) >= 2:
                    intersections_found += 1
                    print(f"Found intersection {intersections_found} at segment {i}")
                    
                    # Sort intersection points by distance from start
                    intersection_points.sort(key=lambda p: Point(start_point).distance(p))
                    entry_point = intersection_points[0]
                    exit_point = intersection_points[-1]
                    
                    # Get bypass points
                    bypass_points = self.get_clockwise_boundary_points(
                        obstacle, entry_point, exit_point
                    )
                    
                    if bypass_points:
                        replanned_path.extend(bypass_points)
                        added_bypass = True
                        break
            
            if not added_bypass:
                replanned_path.append(start_point)
            
            i += 1
        
        # Add final point if needed
        if not np.array_equal(replanned_path[-1], self.planned_path[-1]):
            replanned_path.append(self.planned_path[-1])
            
        print(f"Path replanning complete. Found {intersections_found} intersections.")
        return np.array(replanned_path)

    def visualize_path(self, replanned_path=None):
        """Visualize original path, obstacles, and replanned path"""
        plt.figure(figsize=(15, 15))
        
        # Plot original path
        plt.plot(self.planned_path[:,0], self.planned_path[:,1], 
                'b-', alpha=0.5, label='Original Path')
        
        # Plot obstacles
        for obstacle in self.buffered_obstacles:
            x, y = obstacle.exterior.xy
            plt.plot(x, y, 'r-', linewidth=2)
            
        # Plot replanned path if available
        if replanned_path is not None:
            plt.plot(replanned_path[:,0], replanned_path[:,1], 
                    'g-', label='Replanned Path')
            
        plt.axis('equal')
        plt.legend()
        plt.title('Path Planning Visualization')
        plt.grid(True)
        
        # Save plot
        plt.savefig('path_visualization.png')
        print("Visualization saved as 'path_visualization.png'")
        plt.close()

    def save_replanned_path(self, output_file):
        """Save replanned path to CSV file"""
        replanned_path = self.replan_path()
        
        # Visualize the paths
        self.visualize_path(replanned_path)
        
        df = pd.DataFrame(replanned_path, columns=['x', 'y'])
        df = df.reset_index()
        df.to_csv(output_file, index=False)
        print(f"Replanned path saved to {output_file}")
        return replanned_path

def main():
    try:
        # Get paths from user
        planned_path_csv = input("Enter path to planned path CSV: ")
        obstacles_folder = input("Enter path to obstacles folder: ")
        output_file = input("Enter path for output CSV: ")
        safety_margin = float(input("Enter safety margin in meters (default: 0.5): ") or 0.5)
        
        # Create and run replanner
        replanner = PathReplanner(
            planned_path_csv=planned_path_csv,
            obstacles_folder=obstacles_folder,
            safety_margin=safety_margin
        )
        
        # Generate and save replanned path
        replanned_path = replanner.save_replanned_path(output_file)
        
        print(f"\nPath replanning completed:")
        print(f"- Original path points: {len(replanner.planned_path)}")
        print(f"- Replanned path points: {len(replanned_path)}")
        print(f"- Number of obstacles: {len(replanner.obstacles)}")
        print(f"- Output saved to: {output_file}")
        print("- Visualization saved as 'path_visualization.png'")
        
        return 0
        
    except Exception as e:
        print(f"Error in main: {e}")
        return 1

if __name__ == "__main__":
    exit(main())