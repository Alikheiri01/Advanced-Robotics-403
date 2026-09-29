import pandas as pd
import matplotlib.pyplot as plt
import os

def visualize_robot_tracks(csv_file_name='robot_position_log.csv'):
    """
    Reads robot position data from a CSV file and plots the true and
    odometry-based trajectories.

    Args:
        csv_file_name (str): The name of the CSV file to read. The script
                             expects this file to be in the user's home directory.
    """
    # Construct the full path to the CSV file in the user's home directory
    file_path = os.path.join(os.path.expanduser('~'), csv_file_name)

    # Check if the file exists before attempting to read it
    if not os.path.exists(file_path):
        print(f"Error: The file '{file_path}' was not found.")
        print("Please make sure you have run the ROS recorder node first to generate the data.")
        return

    print(f"Reading data from '{file_path}'...")
    
    try:
        # Read the CSV data into a pandas DataFrame
        df = pd.read_csv(file_path)

        # --- Plotting the Trajectories ---

        # Set up the plot with a specific size and a single subplot
        fig, ax = plt.subplots(figsize=(10, 8))

        # Plot the ground truth trajectory from Gazebo
        ax.plot(df['true_x'].to_numpy(), df['true_y'].to_numpy(), label='True Path (Gazebo)', color='blue', linewidth=2.5)

        # Plot the noisy odometry trajectory
        ax.plot(df['odom_x'].to_numpy(), df['odom_y'].to_numpy(), label='Odometry Path (/odom)', color='red', 
                linestyle='--', linewidth=2)

        # Mark the starting and ending points for clarity
        # Start point (first row of the dataframe)
        ax.scatter(df['true_x'].iloc[0], df['true_y'].iloc[0], marker='o', color='green', s=150, zorder=5, label='Start')
        # End point (last row of the dataframe)
        ax.scatter(df['true_x'].iloc[-1], df['true_y'].iloc[-1], marker='X', color='purple', s=150, zorder=5, label='End')

        # --- Formatting the Plot ---

        # Add a title and labels for the axes
        ax.set_title('Robot Trajectory Comparison: True vs. Odometry', fontsize=16)
        ax.set_xlabel('X Position (meters)', fontsize=12)
        ax.set_ylabel('Y Position (meters)', fontsize=12)
        
        # Ensure the aspect ratio is equal to prevent distortion of the path's shape
        ax.set_aspect('equal', adjustable='box')

        # Add a legend to identify the different lines
        ax.legend(fontsize=10)

        # Add a grid for better readability
        ax.grid(True)
        
        # Display the plot
        print("Displaying plot. Close the plot window to exit the script.")
        plt.show()

    except pd.errors.EmptyDataError:
        print(f"Error: The CSV file '{file_path}' is empty. No data to plot.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    visualize_robot_tracks()

