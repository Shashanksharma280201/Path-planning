import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point, LineString, MultiLineString, MultiPolygon
from shapely import make_valid
from shapely.ops import nearest_points
import cv2
import os

class SinglePassCoverage:
    def __init__(self, csv_file, wheel_separation=0.25, direction='vertical', safety_margin=0.1):
        """
        Initialize coverage planner with robot constraints
        wheel_separation: distance between robot wheels in meters
        direction: 'vertical' or 'horizontal' path planning direction
        safety_margin: minimum distance to maintain from boundary (in meters)
        """
        self.wheel_separation = wheel_separation
        self.step_size = 0.5 * wheel_separation
        self.safety_margin = safety_margin
        self.direction = direction.lower()
        
        if self.direction not in ['vertical', 'horizontal']:
            raise ValueError("Direction must be either 'vertical' or 'horizontal'")
            
        self.boundary_coords = self.load_boundary(csv_file)
        self.start_point = tuple(self.boundary_coords[0])
        self.create_valid_polygon()
        self.safe_boundary = self.create_buffered_boundary()
        
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
            raise

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

    def create_buffered_boundary(self):
        """Create a boundary polygon with safety margin"""
        try:
            buffered = self.boundary_polygon.buffer(-self.safety_margin)
            if isinstance(buffered, MultiPolygon):
                # Take the largest polygon if multiple are created
                areas = [p.area for p in buffered.geoms]
                buffered = buffered.geoms[np.argmax(areas)]
            return buffered
        except Exception as e:
            print(f"Error creating buffered boundary: {e}")
            return self.boundary_polygon

    def is_point_inside_safe(self, point):
        """Check if point is inside safe boundary"""
        return Point(point).within(self.safe_boundary)

    def get_line_intersection(self, coord, is_vertical=True):
        """Get intersection points of a line with the safe boundary"""
        try:
            if is_vertical:
                line = LineString([(coord, self.min_y - 1), (coord, self.max_y + 1)])
            else:
                line = LineString([(self.min_x - 1, coord), (self.max_x + 1, coord)])
                
            intersection = line.intersection(self.safe_boundary)
            
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
            
            # Remove duplicates and sort
            points = list(set(points))
            if is_vertical:
                points.sort(key=lambda p: p[1])
            else:
                points.sort(key=lambda p: p[0])
            
            return points
        except Exception as e:
            print(f"Error getting intersection: {e}")
            return []

    def generate_single_pass_path(self):
        """Generate coverage path starting from boundary start point"""
        try:
            path = []
            visited_segments = set()
            
            # Adjust start point to be within safe boundary if needed
            if not self.is_point_inside_safe(self.start_point):
                nearest_point = nearest_points(Point(self.start_point), self.safe_boundary)[1]
                self.start_point = (nearest_point.x, nearest_point.y)
            
            path.append(self.start_point)
            current_pos = self.start_point
            
            # Calculate stripe positions with safety margin
            stripe_width = self.step_size * 0.9
            safe_min_x = self.min_x + self.safety_margin
            safe_max_x = self.max_x - self.safety_margin
            safe_min_y = self.min_y + self.safety_margin
            safe_max_y = self.max_y - self.safety_margin
            
            if self.direction == 'vertical':
                num_stripes = int((safe_max_x - safe_min_x) / stripe_width) + 1
                positions = np.linspace(safe_min_x, safe_max_x, num_stripes)
                start_idx = np.argmin(np.abs(positions - self.start_point[0]))
            else:
                num_stripes = int((safe_max_y - safe_min_y) / stripe_width) + 1
                positions = np.linspace(safe_min_y, safe_max_y, num_stripes)
                start_idx = np.argmin(np.abs(positions - self.start_point[1]))
            
            # Reorder positions to start from closest stripe
            if start_idx > len(positions) // 2:
                positions = positions[::-1]
                start_idx = len(positions) - start_idx - 1
            
            positions = np.concatenate([positions[start_idx:], positions[:start_idx]])
            
            # Set initial direction based on start point position
            if self.direction == 'vertical':
                moving_positive = self.start_point[1] < np.mean([safe_min_y, safe_max_y])
            else:
                moving_positive = self.start_point[0] < np.mean([safe_min_x, safe_max_x])
            
            # Process each stripe position
            for i, pos in enumerate(positions):
                intersections = self.get_line_intersection(pos, is_vertical=(self.direction == 'vertical'))
                
                if len(intersections) < 2:
                    continue
                
                # Sort intersections based on current direction
                if not moving_positive:
                    intersections.reverse()
                
                # Find closest intersection for first stripe
                if i == 0:
                    start_idx = min(range(len(intersections)), 
                                  key=lambda j: Point(current_pos).distance(Point(intersections[j])))
                    intersections = intersections[start_idx:] + intersections[:start_idx]
                
                # Create segment points
                for j in range(len(intersections)-1):
                    start_point = intersections[j]
                    end_point = intersections[j+1]
                    
                    if self.direction == 'vertical':
                        segment_id = f"{pos:.3f}_{min(start_point[1], end_point[1]):.3f}_{max(start_point[1], end_point[1]):.3f}"
                    else:
                        segment_id = f"{min(start_point[0], end_point[0]):.3f}_{pos:.3f}_{max(start_point[0], end_point[0]):.3f}"
                    
                    if segment_id not in visited_segments:
                        # Add connecting path
                        if len(path) > 0:
                            last_point = path[-1]
                            connection = self.generate_connection(last_point, start_point)
                            path.extend(connection[1:])
                        
                        # Add stripe segment points
                        segment_points = self.generate_segment_points(start_point, end_point, pos)
                        path.extend(segment_points)
                        
                        visited_segments.add(segment_id)
                        current_pos = path[-1]
                
                # Switch directions for next stripe
                moving_positive = not moving_positive
            
            return np.array(path)
            
        except Exception as e:
            print(f"Error generating path: {e}")
            raise

    def generate_segment_points(self, start_point, end_point, pos):
        """Generate points along a segment with safety checks"""
        if self.direction == 'vertical':
            height = abs(end_point[1] - start_point[1])
            num_points = max(int(height / (self.step_size/2)), 2)
            return [
                (pos, start_point[1] + (end_point[1] - start_point[1]) * k / (num_points - 1))
                for k in range(num_points)
                if self.is_point_inside_safe((pos, start_point[1] + (end_point[1] - start_point[1]) * k / (num_points - 1)))
            ]
        else:
            width = abs(end_point[0] - start_point[0])
            num_points = max(int(width / (self.step_size/2)), 2)
            return [
                (start_point[0] + (end_point[0] - start_point[0]) * k / (num_points - 1), pos)
                for k in range(num_points)
                if self.is_point_inside_safe((start_point[0] + (end_point[0] - start_point[0]) * k / (num_points - 1), pos))
            ]

    def generate_connection(self, start, end):
        """Generate smooth connection between points"""
        try:
            direct_line = LineString([start, end])
            
            # If direct line is valid within safe boundary
            if direct_line.within(self.safe_boundary):
                distance = Point(start).distance(Point(end))
                num_points = max(int(distance / (self.step_size/2)), 2)
                return [
                    (
                        start[0] + (end[0] - start[0]) * i / (num_points - 1),
                        start[1] + (end[1] - start[1]) * i / (num_points - 1)
                    )
                    for i in range(num_points)
                ]
            
            # Try curved path
            control_point = (
                (start[0] + end[0])/2,
                (start[1] + end[1])/2 + self.wheel_separation
            )
            
            num_points = max(int(Point(start).distance(Point(end)) / (self.step_size/2)), 2)
            t = np.linspace(0, 1, num_points)
            
            x = (1-t)**2 * start[0] + 2*(1-t)*t*control_point[0] + t**2*end[0]
            y = (1-t)**2 * start[1] + 2*(1-t)*t*control_point[1] + t**2*end[1]
            
            points = list(zip(x, y))
            return [p for p in points if self.is_point_inside_safe(p)]
            
        except Exception as e:
            print(f"Error generating connection: {e}")
            return [start, end]

    def plot_path(self, path):
        """Plot the boundary and coverage path with directional arrows"""
        try:
            plt.figure(figsize=(12, 12))
            
            # Plot original boundary
            x, y = self.boundary_polygon.exterior.xy
            plt.plot(x, y, 'k-', linewidth=2, label='Boundary')
            
            # Plot safety margin boundary
            safe_x, safe_y = self.safe_boundary.exterior.xy
            plt.plot(safe_x, safe_y, 'k--', linewidth=1, 
                    label=f'Safety Margin ({self.safety_margin}m)')
            
            # Convert path to numpy array
            path = np.array(path)
            
            # Plot path with arrows
            for i in range(len(path)-1):
                dx = path[i+1,0] - path[i,0]
                dy = path[i+1,1] - path[i,1]
                arrow_length = np.sqrt(dx**2 + dy**2)
                
                if arrow_length > self.step_size * 0.5:
                    plt.plot([path[i,0], path[i+1,0]], 
                            [path[i,1], path[i+1,1]], 
                            'b-', linewidth=1)
                    
                    mid_x = (path[i,0] + path[i+1,0]) / 2
                    mid_y = (path[i,1] + path[i+1,1]) / 2
                    
                    plt.arrow(mid_x - dx*0.1, mid_y - dy*0.1,
                             dx*0.2, dy*0.2,
                             head_width=self.step_size*0.3,
                             head_length=self.step_size*0.3,
                             fc='r', ec='r',
                             length_includes_head=True)
            
            plt.plot(path[0,0], path[0,1], 'go', markersize=10, label='Start')
            plt.plot(path[-1,0], path[-1,1], 'ro', markersize=10, label='End')
            
            plt.plot([], [], 'b-', label='Path Direction', 
                    marker='>', markersize=10,
                    markerfacecolor='r',
                    markeredgecolor='r')
            
            plt.grid(True)
            plt.legend()
            plt.axis('equal')
            plt.title(f'Single-Pass Coverage Path ({self.direction.capitalize()} Pattern)')
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
        script_dir = os.path.dirname(os.path.abspath(__file__))
        input_file = "/home/flo/Music/mmr_old_ws/paths/bigmap.csv"
        output_file = os.path.join(script_dir, "new_coverage_path.csv")
        
        # Get user inputs with validation
        while True:
            try:
                wheel_separation = float(input("Enter wheel separation in meters (default: 0.5): ") or 0.5)
                if wheel_separation <= 0:
                    print("Wheel separation must be positive")
                    continue
                break
            except ValueError:
                print("Please enter a valid number")
        
        while True:
            try:
                safety_margin = float(input("Enter safety margin in meters (default: 0.1): ") or 0.1)
                if safety_margin < 0:
                    print("Safety margin cannot be negative")
                    continue
                break
            except ValueError:
                print("Please enter a valid number")
        
        while True:
            direction = input("Enter path planning direction (vertical/horizontal): ").lower()
            if direction in ['vertical', 'horizontal']:
                break
            print("Invalid input. Please enter either 'vertical' or 'horizontal'.")
        
        # Create and run the planner
        planner = SinglePassCoverage(
            csv_file=input_file,
            wheel_separation=wheel_separation,
            direction=direction,
            safety_margin=safety_margin
        )
        
        print(f"\nConfiguration:")
        print(f"- Input file: {input_file}")
        print(f"- Wheel separation: {wheel_separation}m")
        print(f"- Safety margin: {safety_margin}m")
        print(f"- Direction: {direction}")
        print(f"- Starting point: {planner.start_point}")
        
        print("\nGenerating path...")
        path = planner.generate_single_pass_path()
        
        if len(path) < 2:
            raise ValueError("Generated path is too short or invalid")
        
        print("Plotting path...")
        planner.plot_path(path)
        
        print("Saving path...")
        planner.save_path_to_csv(path, output_file)
        
        print("\nDone!")
        print(f"- Path points generated: {len(path)}")
        print(f"- Output saved to: {output_file}")
        
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