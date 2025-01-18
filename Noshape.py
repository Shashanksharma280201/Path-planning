import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point, LineString, MultiLineString, MultiPolygon
from shapely import make_valid
import cv2
import os
from typing import List, Tuple, Optional, Union

class SinglePassCoverage:
    def __init__(self, csv_file: str, wheel_separation: float = 0.25):
        """
        Initialize coverage planner with robot constraints
        Args:
            csv_file: Path to CSV file containing boundary coordinates
            wheel_separation: Distance between robot wheels in meters
        """
        self.wheel_separation = wheel_separation
        self.step_size = 0.6 * wheel_separation
        self.boundary_coords = self.load_boundary(csv_file)
        self.start_point = tuple(self.boundary_coords[0])
        self.create_valid_polygon()
        self.subdivisions = []

    def load_boundary(self, csv_file: str) -> np.ndarray:
        """
        Load boundary coordinates from CSV file
        Args:
            csv_file: Path to CSV file
        Returns:
            np.ndarray: Array of boundary coordinates
        """
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
            
            if isinstance(polygon, MultiPolygon):
                # Take the largest polygon if we have multiple
                polygon = max(polygon.geoms, key=lambda p: p.area)
            
            self.boundary_polygon = polygon
            self.min_x, self.min_y, self.max_x, self.max_y = polygon.bounds
            print(f"Polygon bounds: ({self.min_x}, {self.min_y}) to ({self.max_x}, {self.max_y})")
        except Exception as e:
            print(f"Error creating polygon: {e}")
            raise

    def subdivide_area(self) -> List[Polygon]:
        """
        Subdivide the area if it's complex or causing errors
        Returns:
            List[Polygon]: List of subdivided polygons
        """
        try:
            centroid = self.boundary_polygon.centroid
            
            # Try vertical split first
            vertical_line = LineString([
                (centroid.x, self.min_y - 1),
                (centroid.x, self.max_y + 1)
            ])
            split_polys = self.split_polygon_by_line(self.boundary_polygon, vertical_line)
            
            if not split_polys:
                # Try horizontal split if vertical fails
                horizontal_line = LineString([
                    (self.min_x - 1, centroid.y),
                    (self.max_x + 1, centroid.y)
                ])
                split_polys = self.split_polygon_by_line(self.boundary_polygon, horizontal_line)
            
            if split_polys:
                self.subdivisions = split_polys
                return split_polys
            
            # If both splits fail, try diagonal split
            diagonal_line = LineString([
                (self.min_x - 1, self.min_y - 1),
                (self.max_x + 1, self.max_y + 1)
            ])
            split_polys = self.split_polygon_by_line(self.boundary_polygon, diagonal_line)
            
            self.subdivisions = split_polys if split_polys else [self.boundary_polygon]
            return self.subdivisions
            
        except Exception as e:
            print(f"Error in subdivision: {e}")
            self.subdivisions = [self.boundary_polygon]
            return self.subdivisions

    def split_polygon_by_line(self, polygon: Polygon, line: LineString) -> List[Polygon]:
        """
        Split a polygon using a line
        Args:
            polygon: Polygon to split
            line: Line to split with
        Returns:
            List[Polygon]: List of split polygons
        """
        try:
            if not polygon.is_valid:
                polygon = make_valid(polygon)
            
            line_buffer = line.buffer(self.step_size / 10)
            difference = polygon.difference(line_buffer)
            
            if isinstance(difference, Polygon):
                return [difference]
            elif isinstance(difference, MultiPolygon):
                return [poly for poly in difference.geoms if poly.area > self.step_size]
            
            return []
            
        except Exception as e:
            print(f"Error in splitting: {e}")
            return []

    def get_vertical_intersection(self, x: float) -> List[Tuple[float, float]]:
        """
        Get intersection points of a vertical line with the polygon
        Args:
            x: X-coordinate of vertical line
        Returns:
            List[Tuple[float, float]]: List of intersection points
        """
        try:
            vertical_line = LineString([
                (x, self.min_y - 1),
                (x, self.max_y + 1)
            ])
            intersection = vertical_line.intersection(self.boundary_polygon)
            
            if intersection.is_empty:
                return []
            
            points = []
            if isinstance(intersection, MultiLineString):
                for line in intersection.geoms:
                    points.extend(list(line.coords))
            elif isinstance(intersection, LineString):
                points.extend(list(intersection.coords))
            elif isinstance(intersection, Point):
                points.append((intersection.x, intersection.y))
            
            # Remove duplicates and sort by y-coordinate
            points = list(set(points))
            points.sort(key=lambda p: p[1])
            
            return points
            
        except Exception as e:
            print(f"Error getting intersection: {e}")
            return []

    def generate_single_pass_path(self, start_point: Optional[Tuple[float, float]] = None) -> np.ndarray:
        """
        Generate coverage path
        Args:
            start_point: Optional starting point
        Returns:
            np.ndarray: Generated path points
        """
        try:
            if start_point is not None:
                self.start_point = start_point
            
            path = []
            try:
                path = self._generate_basic_path()
            except Exception as e:
                print(f"Error in basic path generation: {e}")
                print("Attempting subdivision...")
                
                subdivisions = self.subdivide_area()
                if not subdivisions:
                    raise ValueError("Failed to subdivide area")
                
                current_point = self.start_point
                for poly in subdivisions:
                    sub_path = self._generate_subdivision_path(poly, current_point)
                    if len(sub_path) > 0:
                        if len(path) > 0:
                            connection = self.generate_connection(path[-1], sub_path[0])
                            path = np.vstack([path, connection[1:], sub_path])
                        else:
                            path = sub_path
                        current_point = tuple(path[-1])
            
            return np.array(path)
            
        except Exception as e:
            print(f"Error in path generation: {e}")
            return np.array([])

    def _generate_basic_path(self) -> np.ndarray:
        """Generate basic coverage path"""
        path = []
        visited_segments = set()
        
        path.append(self.start_point)
        current_pos = self.start_point
        
        stripe_width = self.step_size * 0.9
        num_stripes = int((self.max_x - self.min_x) / stripe_width) + 1
        x_positions = np.linspace(self.min_x, self.max_x, num_stripes)
        
        start_x_idx = np.argmin(np.abs(x_positions - self.start_point[0]))
        
        if start_x_idx > len(x_positions) // 2:
            x_positions = x_positions[::-1]
            start_x_idx = len(x_positions) - start_x_idx - 1
        
        x_positions = np.concatenate([x_positions[start_x_idx:], x_positions[:start_x_idx]])
        moving_up = self.start_point[1] < np.mean([self.min_y, self.max_y])
        
        for i, x in enumerate(x_positions):
            intersections = self.get_vertical_intersection(x)
            
            if len(intersections) < 2:
                continue
            
            if not moving_up:
                intersections.reverse()
            
            if i == 0:
                start_idx = min(range(len(intersections)), 
                              key=lambda j: Point(current_pos).distance(Point(intersections[j])))
                intersections = intersections[start_idx:] + intersections[:start_idx]
            
            for j in range(len(intersections)-1):
                start_point = intersections[j]
                end_point = intersections[j+1]
                
                segment_id = f"{x:.3f}_{min(start_point[1], end_point[1]):.3f}_{max(start_point[1], end_point[1]):.3f}"
                
                if segment_id not in visited_segments:
                    if len(path) > 0:
                        last_point = path[-1]
                        connection = self.generate_connection(last_point, start_point)
                        path.extend(connection[1:])
                    
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
            
            moving_up = not moving_up
        
        return np.array(path)

    def _generate_subdivision_path(self, polygon: Polygon, start_point: Tuple[float, float]) -> np.ndarray:
        """
        Generate path for a subdivision
        Args:
            polygon: Subdivision polygon
            start_point: Starting point
        Returns:
            np.ndarray: Generated path points
        """
        original_boundary = self.boundary_polygon
        self.boundary_polygon = polygon
        self.min_x, self.min_y, self.max_x, self.max_y = polygon.bounds
        
        path = self._generate_basic_path()
        
        self.boundary_polygon = original_boundary
        return path

    def generate_connection(self, start: Tuple[float, float], end: Tuple[float, float]) -> List[Tuple[float, float]]:
        """
        Generate smooth connection between points
        Args:
            start: Starting point
            end: Ending point
        Returns:
            List[Tuple[float, float]]: Connection points
        """
        try:
            direct_line = LineString([start, end])
            
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

    def plot_path(self, path: np.ndarray):
        """
        Plot the boundary and coverage path
        Args:
            path: Path points to plot
        """
        try:
            plt.figure(figsize=(12, 12))
            
            # Plot boundary and subdivisions
            if self.subdivisions:
                for i, poly in enumerate(self.subdivisions):
                    x, y = poly.exterior.xy
                    plt.plot(x, y, '--', linewidth=1, label=f'Subdivision {i+1}')
            else:
                x, y = self.boundary_polygon.exterior.xy
                plt.plot(x, y, 'k-', linewidth=2, label='Boundary')
            
            # Plot path with arrows
            if len(path) > 1:
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
                
                # Plot start and end points
                plt.plot(path[0,0], path[0,1], 'go', markersize=10, label='Start')
                plt.plot(path[-1,0], path[-1,1], 'ro', markersize=10, label='End')
                
                # Add direction indicator to legend
                plt.plot([], [], 'b-', label='Path Direction',
                        marker='>', markersize=10,
                        markerfacecolor='r',
                        markeredgecolor='r')
            
            plt.grid(True)
            plt.legend()
            plt.axis('equal')
            plt.title('Coverage Path with Subdivisions')
            plt.show()
            
        except Exception as e:
            print(f"Error plotting: {e}")

    def save_path_to_csv(self, path: np.ndarray, output_file: str):
        """
        Save the generated path to a CSV file
        Args:
            path: Path points to save
            output_file: Output file path
        """
        try:
            df = pd.DataFrame(path, columns=['x', 'y'])
            df = df.reset_index()
            df.to_csv(output_file, index=False)
            print(f"Path saved to {output_file}")
        except Exception as e:
            print(f"Error saving path: {e}")

def main():
    """Main function to run the coverage planner"""
    try:
        # Get the current script's directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Use the provided input file path
        input_file = "/home/flo/mmr_ws/paths/bigmap.csv"
        output_file = os.path.join(script_dir, "coverage_path.csv")
        wheel_separation = 0.62  # Robot specific parameter
        
        # Verify input file exists
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Input file not found: {input_file}")
        
        # Create planner instance
        planner = SinglePassCoverage(
            csv_file=input_file,
            wheel_separation=wheel_separation
        )
        
        print(f"Starting point from boundary: {planner.start_point}")
        print("Generating path...")
        
        # Generate coverage path
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