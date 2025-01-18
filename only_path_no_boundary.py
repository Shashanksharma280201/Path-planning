import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point, LineString, MultiLineString, MultiPolygon
from shapely import make_valid
import cv2
import os


class SinglePassCoverage:
    def __init__(self, csv_file, wheel_separation=0.25):
        """
        Initialize coverage planner with robot constraints
        wheel_separation: distance between robot wheels in meters
        """
        self.wheel_separation = wheel_separation
        self.step_size = 0.3 * wheel_separation  # 80% overlap for complete coverage
        self.boundary_coords = self.load_boundary(csv_file)
        # Store starting point from boundary
        self.start_point = tuple(self.boundary_coords[0])
        self.create_valid_polygon()
        
    def load_boundary(self, csv_file):
        """Load boundary coordinates from CSV file"""
        try:
            if not os.path.exists(csv_file):
                raise FileNotFoundError(f"CSV file not found: {csv_file}")
                
            df = pd.read_csv(csv_file)
            
            if 'x' not in df.columns or 'y' not in df.columns:
                raise ValueError("CSV must contain 'x' and 'y' columns")
                
            coords = np.array(list(zip(df['x'], df['y'])))
            
            # Ensure the polygon is closed
            if not np.array_equal(coords[0], coords[-1]):
                coords = np.vstack([coords, coords[0]])
            
            return coords
            
        except Exception as e:
            print(f"Error loading CSV file: {e}")
            raise  # Re-raise to stop execution

    def create_valid_polygon(self):
        """Create a valid polygon from boundary coordinates"""
        try:
            polygon = Polygon(self.boundary_coords)
            if not polygon.is_valid:
                polygon = make_valid(polygon)
            
            self.boundary_polygon = polygon
            self.min_x, self.min_y, self.max_x, self.max_y = polygon.bounds
            print(f"Polygon bounds: ({self.min_x}, {self.min_y}) to ({self.max_x}, {self.max_y})")
        except Exception as e:
            print(f"Error creating polygon: {e}")
            raise

    def is_point_inside(self, point):
        """Check if point is inside boundary"""
        point_obj = Point(point)
        return point_obj.within(self.boundary_polygon)

    def get_vertical_intersection(self, x):
        """Get intersection points of a vertical line with the polygon"""
        try:
            vertical_line = LineString([(x, self.min_y - 1), (x, self.max_y + 1)])
            intersection = vertical_line.intersection(self.boundary_polygon)
            
            if intersection.is_empty:
                return []
            
            points = []
            if intersection.geom_type == 'MultiLineString':
                for line in intersection:
                    points.extend(list(line.coords))
            elif intersection.geom_type == 'LineString':
                points.extend(list(intersection.coords))
            elif intersection.geom_type == 'Point':
                points.append((intersection.x, intersection.y))
            
            # Remove duplicates and sort by y-coordinate
            points = list(set(points))
            points.sort(key=lambda p: p[1])
            
            return points
        except Exception as e:
            print(f"Error getting intersection: {e}")
            return []

    def generate_single_pass_path(self):
        """Generate coverage path starting from boundary start point"""
        try:
            path = []
            visited_segments = set()
            
            # Start from the boundary starting point
            path.append(self.start_point)
            current_pos = self.start_point
            
            # Calculate stripe positions
            stripe_width = self.step_size * 0.9
            num_stripes = int((self.max_x - self.min_x) / stripe_width) + 1
            x_positions = np.linspace(self.min_x, self.max_x, num_stripes)
            
            # Find closest stripe to start point
            start_x_idx = np.argmin(np.abs(x_positions - self.start_point[0]))
            
            # Reorder x_positions to start from closest stripe
            if start_x_idx > len(x_positions) // 2:
                # If closer to right side, work right to left
                x_positions = x_positions[::-1]
                start_x_idx = len(x_positions) - start_x_idx - 1
            
            # Reorder positions to start from start_x_idx
            x_positions = np.concatenate([x_positions[start_x_idx:], x_positions[:start_x_idx]])
            
            moving_up = self.start_point[1] < np.mean([self.min_y, self.max_y])
            
            # Process each vertical position
            for i, x in enumerate(x_positions):
                intersections = self.get_vertical_intersection(x)
                
                if len(intersections) < 2:
                    continue
                
                # Sort intersections based on current direction
                if not moving_up:
                    intersections.reverse()
                
                # Find closest intersection to current position for first stripe
                if i == 0:
                    start_idx = min(range(len(intersections)), 
                                  key=lambda j: Point(current_pos).distance(Point(intersections[j])))
                    intersections = intersections[start_idx:] + intersections[:start_idx]
                
                # Create vertical segment points
                for j in range(len(intersections)-1):
                    start_point = intersections[j]
                    end_point = intersections[j+1]
                    
                    segment_id = f"{x:.3f}_{min(start_point[1], end_point[1]):.3f}_{max(start_point[1], end_point[1]):.3f}"
                    
                    if segment_id not in visited_segments:
                        # Add connecting path from current position
                        if len(path) > 0:
                            last_point = path[-1]
                            connection = self.generate_connection(last_point, start_point)
                            path.extend(connection[1:])
                        
                        # Add vertical segment points
                        height = abs(end_point[1] - start_point[1])
                        num_points = max(int(height / (self.step_size/2)), 2)
                        for k in range(num_points):
                            t = k / (num_points - 1)
                            point = (
                                x,
                                start_point[1] + (end_point[1] - start_point[1]) * t
                            )
                            path.append(point)
                        
                        visited_segments.add(segment_id)
                        current_pos = path[-1]
                
                # Switch directions for next stripe
                moving_up = not moving_up
            
            return np.array(path)
            
        except Exception as e:
            print(f"Error generating path: {e}")
            raise

    def generate_connection(self, start, end):
        """Generate smooth connection between points"""
        try:
            direct_line = LineString([start, end])
            
            # If direct line is valid, use it
            if direct_line.within(self.boundary_polygon):
                distance = Point(start).distance(Point(end))
                num_points = max(int(distance / (self.step_size/2)), 2)
                return [
                    (
                        start[0] + (end[0] - start[0]) * i / (num_points - 1),
                        start[1] + (end[1] - start[1]) * i / (num_points - 1)
                    )
                    for i in range(num_points)
                ]
            
            # Otherwise, use curved path
            control_point = (
                (start[0] + end[0])/2,
                (start[1] + end[1])/2 + self.wheel_separation
            )
            
            num_points = max(int(Point(start).distance(Point(end)) / (self.step_size/2)), 2)
            t = np.linspace(0, 1, num_points)
            
            x = (1-t)**2 * start[0] + 2*(1-t)*t*control_point[0] + t**2*end[0]
            y = (1-t)**2 * start[1] + 2*(1-t)*t*control_point[1] + t**2*end[1]
            
            return list(zip(x, y))
            
        except Exception as e:
            print(f"Error generating connection: {e}")
            return [start, end]

    def plot_path(self, path):
        """Plot the boundary and coverage path with directional arrows"""
        try:
            plt.figure(figsize=(12, 12))
            
            # Plot boundary
            x, y = self.boundary_polygon.exterior.xy
            plt.plot(x, y, 'k-', linewidth=2, label='Boundary')
            
            # Convert path to numpy array
            path = np.array(path)
            
            # Plot path with arrows
            for i in range(len(path)-1):
                # Calculate arrow properties
                dx = path[i+1,0] - path[i,0]
                dy = path[i+1,1] - path[i,1]
                arrow_length = np.sqrt(dx**2 + dy**2)
                
                if arrow_length > self.step_size * 0.5:
                    # Plot line segment
                    plt.plot([path[i,0], path[i+1,0]], 
                            [path[i,1], path[i+1,1]], 
                            'b-', linewidth=1)
                    
                    # Add arrow at midpoint
                    mid_x = (path[i,0] + path[i+1,0]) / 2
                    mid_y = (path[i,1] + path[i+1,1]) / 2
                    
                    plt.arrow(mid_x - dx*0.1, mid_y - dy*0.1,
                             dx*0.2, dy*0.2,
                             head_width=self.step_size*0.3,
                             head_length=self.step_size*0.3,
                             fc='r', ec='r',
                             length_includes_head=True)
            
            # Plot start and end points
            plt.plot(path[0,0], path[0,1], 'go', markersize=10, label='Start')
            plt.plot(path[-1,0], path[-1,1], 'ro', markersize=10, label='End')
            
            # Legend arrow
            plt.plot([], [], 'b-', label='Path Direction', 
                    marker='>', markersize=10,
                    markerfacecolor='r',
                    markeredgecolor='r')
            
            plt.grid(True)
            plt.legend()
            plt.axis('equal')
            plt.title('Single-Pass Coverage Path')
            plt.show()
            
        except Exception as e:
            print(f"Error plotting path: {e}")

    def save_path_to_csv(self, path, output_file):
        """Save the generated path to a CSV file"""
        try:
            df = pd.DataFrame(path, columns=['x', 'y'])
            df = df.reset_index()
            df.to_csv(output_file, index=False)
            print(f"Path saved to {output_file}")
        except Exception as e:
            print(f"Error saving path: {e}")

def main():
    try:
        # Get the current script's directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Use the provided input file path
        input_file = "/home/flo/mmr_ws/paths/half_butterfly.csv"
        output_file = os.path.join(script_dir, "coverage_path.csv")
        wheel_separation = 0.62
        
        # Verify input file exists
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Input file not found: {input_file}")
        
        planner = SinglePassCoverage(
            csv_file=input_file,
            wheel_separation=wheel_separation
        )
        
        print(f"Starting point from boundary: {planner.start_point}")
        print("Generating path...")
        path = planner.generate_single_pass_path()
        
        if len(path) < 2:
            raise ValueError("Generated path is too short or invalid")
        
        print("Plotting path...")
        planner.plot_path(path)
        
        print("Saving path...")
        planner.save_path_to_csv(path, output_file)
        
        print("Done!")
        return 0
        
    except FileNotFoundError as e:
        print(f"File error: {e}")
        return 1
    except ValueError as e:
        print(f"Value error: {e}")
        return 1
    except Exception as e:
        print(f"Error in main: {e}")
        return 1

if __name__ == "__main__":
    exit(main())