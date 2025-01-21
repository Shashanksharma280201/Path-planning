import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import argparse

class PointVisualizer:
    def __init__(self, csv_path, speed=1.0):
        """
        Initialize the visualizer
        
        Parameters:
        - csv_path: path to CSV file
        - speed: animation speed (points per second)
        """
        # Read CSV with the specific format
        self.df = pd.read_csv(csv_path)
        
        # Initialize plot
        self.fig, (self.ax, self.ax_zoom) = plt.subplots(1, 2, figsize=(15, 7))
        
        # Main plot
        self.line, = self.ax.plot([], [], 'b-', label='Path')
        self.point, = self.ax.plot([], [], 'ro', markersize=10, label='Current Point')
        
        # Zoomed plot
        self.line_zoom, = self.ax_zoom.plot([], [], 'b-', label='Path')
        self.point_zoom, = self.ax_zoom.plot([], [], 'ro', markersize=10, label='Current Point')
        
        self.speed = speed
        self.points_x = []
        self.points_y = []
        
        # Setup both plots
        self.setup_plots()
        
    def setup_plots(self):
        """Setup both main and zoomed plots"""
        margin = 0.1  # 10% margin
        x_min, x_max = self.df['x'].min(), self.df['x'].max()
        y_min, y_max = self.df['y'].min(), self.df['y'].max()
        
        # Calculate ranges
        x_range = x_max - x_min
        y_range = y_max - y_min
        
        # Setup main plot
        self.ax.set_xlim(x_min - margin * x_range, x_max + margin * x_range)
        self.ax.set_ylim(y_min - margin * y_range, y_max + margin * y_range)
        self.ax.set_xlabel('X Coordinate')
        self.ax.set_ylabel('Y Coordinate')
        self.ax.set_title('Full View')
        self.ax.grid(True)
        self.ax.legend()
        
        # Store limits for zoomed plot updates
        self.zoom_window = min(x_range, y_range) * 0.1
        
        self.fig.suptitle('Point Visualization', fontsize=14)
        plt.tight_layout()
        
    def update_zoom_plot(self, current_x, current_y):
        """Update the zoomed plot limits"""
        self.ax_zoom.set_xlim(current_x - self.zoom_window/2, current_x + self.zoom_window/2)
        self.ax_zoom.set_ylim(current_y - self.zoom_window/2, current_y + self.zoom_window/2)
        self.ax_zoom.set_title('Zoomed View')
        self.ax_zoom.grid(True)
        self.ax_zoom.legend()
        
    def animate(self, frame):
        """Animation function called for each frame"""
        # Add new point
        current_x = self.df.iloc[frame]['x']
        current_y = self.df.iloc[frame]['y']
        
        self.points_x.append(current_x)
        self.points_y.append(current_y)
        
        # Update both plots
        self.line.set_data(self.points_x, self.points_y)
        self.point.set_data([current_x], [current_y])
        
        self.line_zoom.set_data(self.points_x, self.points_y)
        self.point_zoom.set_data([current_x], [current_y])
        
        # Update zoom plot limits
        self.update_zoom_plot(current_x, current_y)
        
        # Calculate distance from previous point if available
        distance = 0
        if frame > 0:
            prev_x = self.df.iloc[frame-1]['x']
            prev_y = self.df.iloc[frame-1]['y']
            distance = np.sqrt((current_x - prev_x)**2 + (current_y - prev_y)**2)
        
        # Update title with detailed information
        self.fig.suptitle(
            f'Point {frame+1}/{len(self.df)}\n' +
            f'Coordinates: ({current_x:.3f}, {current_y:.3f})\n' +
            f'Distance from previous: {distance:.3f}',
            fontsize=10
        )
        
        return self.line, self.point, self.line_zoom, self.point_zoom
    
    def visualize(self):
        """Start the visualization"""
        interval = int(1000 / self.speed)  # Convert speed to milliseconds interval
        anim = FuncAnimation(
            self.fig, 
            self.animate,
            frames=len(self.df),
            interval=interval,
            blit=True,
            repeat=False
        )
        plt.show()

def main():
    parser = argparse.ArgumentParser(description='Visualize points from CSV file')
    parser.add_argument('csv_path', type=str, help='Path to CSV file')
    parser.add_argument('--speed', type=float, default=1.0, help='Animation speed (points per second)')
    
    args = parser.parse_args()
    
    visualizer = PointVisualizer(
        args.csv_path,
        speed=args.speed
    )
    visualizer.visualize()

if __name__ == "__main__":
    main()