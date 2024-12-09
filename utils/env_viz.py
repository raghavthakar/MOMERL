import matplotlib.pyplot as plt
import yaml

def parse_and_plot_env_config(file_path):
    # Load the YAML configuration
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    
    # Extract environment dimensions
    env_width, env_height = config['Environment']['dimensions'][1], config['Environment']['dimensions'][0]
    
    # Extract POIs and Agents
    pois = config['Environment']['pois']
    agents = config['Agents']
    
    # Initialize figure
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_xlim(0, env_width)
    ax.set_ylim(0, env_height)
    ax.set_aspect('equal', adjustable='box')
    ax.set_title("Environment Visualizer", fontsize=14)
    ax.set_xlabel("Width")
    ax.set_ylabel("Height")

    # Set grid to repeat at every unit
    ax.set_xticks(range(env_width + 1))
    ax.set_yticks(range(env_height + 1))
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

    # Draw POIs and their observation radii
    for poi in pois:
        x, y = poi["location"]
        radius = poi["radius"]
        circle = plt.Circle((y, x), radius, color="blue", alpha=0.3, label="POI" if pois.index(poi) == 0 else "")
        ax.add_patch(circle)
        ax.plot(y, x, 'bo', label="")

    # Draw Agents and their observation radii
    for idx, (loc, obs_radius) in enumerate(zip(agents['starting_locs'], agents['observation_radii'])):
        x, y = loc
        circle = plt.Circle((y, x), obs_radius, color="green", alpha=0.2, label="Agent Observation" if idx == 0 else "")
        ax.add_patch(circle)
        ax.plot(y, x, 'go', label="Agent" if idx == 0 else "")

    # Legend and final adjustments
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper right')

    plt.show()

# Example usage
if __name__ == "__main__":
    import argparse

    # Command-line argument parsing
    parser = argparse.ArgumentParser(description="Visualize an environment configuration from a YAML file.")
    parser.add_argument("file", type=str, help="Path to the YAML configuration file.")
    args = parser.parse_args()

    # Parse and plot the environment
    parse_and_plot_env_config(args.file)
