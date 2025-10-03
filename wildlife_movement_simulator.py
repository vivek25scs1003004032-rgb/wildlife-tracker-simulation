import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import deque
import matplotlib.patches as patches

# --- Configuration ---
MAX_POINTS = 50          # Max points to store for the path
PREDICTION_WINDOW = 5    # Use the last 5 points for prediction
TICK_INTERVAL_MS = 300   # Update every 0.3 seconds

# --- Hotspot Configuration ---
HOTSPOT_RADIUS = 80
HOTSPOT_PULL_STRENGTH = 0.05

# --- Map Dimensions (will be set from plot size) ---
MAP_WIDTH = 800
MAP_HEIGHT = 500

# --- Animal State and History ---
# Using a deque is efficient for fixed-length histories
animal = {
    'x': MAP_WIDTH / 2,
    'y': MAP_HEIGHT / 2,
    'history': deque(maxlen=MAX_POINTS),
    'prediction': {'x': MAP_WIDTH / 2, 'y': MAP_HEIGHT / 2},
    'velocity': {'x': 0, 'y': 0}
}

# --- Hotspot State ---
hotspot = {'x': MAP_WIDTH * 0.75, 'y': MAP_HEIGHT * 0.25}

def predict_next_location(history):
    """Predicts the next location based on simple extrapolation of recent velocity."""
    if len(history) < PREDICTION_WINDOW:
        return {'x': animal['x'], 'y': animal['y']} # Not enough data

    # Get the recent movement window
    recent_history = list(history)[-PREDICTION_WINDOW:]
    
    # Calculate average velocity vector
    velocities = np.diff(np.array([(p['x'], p['y']) for p in recent_history]), axis=0)
    avg_vx, avg_vy = np.mean(velocities, axis=0)

    # Extrapolate the next position with a slight momentum factor
    predicted_x = animal['x'] + (avg_vx * 1.2)
    predicted_y = animal['y'] + (avg_vy * 1.2)
    
    return {'x': predicted_x, 'y': predicted_y}

def update_animal_location():
    """Simulates the 'true' animal movement, including random walk and behavior."""
    last_x, last_y = animal['x'], animal['y']

    # 1. Add a small random component (Random Walk)
    random_walk = (np.random.rand(2) - 0.5) * 5

    # 2. Hotspot Attraction Force (Behavioral Model)
    dx_hotspot = hotspot['x'] - animal['x']
    dy_hotspot = hotspot['y'] - animal['y']
    distance_to_hotspot = np.sqrt(dx_hotspot*2 + dy_hotspot*2)
    
    attraction_force = np.zeros(2)
    # Apply attraction if within a certain range
    if distance_to_hotspot > 1 and distance_to_hotspot < MAP_WIDTH / 3:
        # Attraction is stronger when closer
        attraction_factor = HOTSPOT_PULL_STRENGTH / distance_to_hotspot
        attraction_force = np.array([dx_hotspot, dy_hotspot]) * attraction_factor
        
    # 3. Combine forces: previous velocity (momentum), random walk, and attraction
    # Damping factor of 0.8 reduces velocity over time
    animal['velocity']['x'] = (animal['velocity']['x'] * 0.8) + random_walk[0] + attraction_force[0]
    animal['velocity']['y'] = (animal['velocity']['y'] * 0.8) + random_walk[1] + attraction_force[1]

    # 4. Limit max velocity to prevent chaotic movement
    velocity_magnitude = np.sqrt(animal['velocity']['x']*2 + animal['velocity']['y']*2)
    max_v = 15
    if velocity_magnitude > max_v:
        animal['velocity']['x'] = (animal['velocity']['x'] / velocity_magnitude) * max_v
        animal['velocity']['y'] = (animal['velocity']['y'] / velocity_magnitude) * max_v
        
    # 5. Apply new position
    animal['x'] += animal['velocity']['x']
    animal['y'] += animal['velocity']['y']

    # 6. Keep the animal within map boundaries (Bounce effect)
    if not (10 < animal['x'] < MAP_WIDTH - 10):
        animal['x'] = np.clip(animal['x'], 10, MAP_WIDTH - 10)
        animal['velocity']['x'] *= -0.5 # Reverse and dampen velocity
    if not (10 < animal['y'] < MAP_HEIGHT - 10):
        animal['y'] = np.clip(animal['y'], 10, MAP_HEIGHT - 10)
        animal['velocity']['y'] *= -0.5

    # Store the actual new location in history
    animal['history'].append({'x': animal['x'], 'y': animal['y']})
    
    # Calculate speed for display
    speed = np.sqrt((animal['x'] - last_x)*2 + (animal['y'] - last_y)*2)
    return speed

# --- Visualization Setup ---
fig, ax = plt.subplots(figsize=(10, 6.25)) # 800x500 aspect ratio

def setup_plot():
    """Initializes the plot elements and styles."""
    ax.set_facecolor('#f7fee7') # Light green background
    ax.set_xlim(0, MAP_WIDTH)
    ax.set_ylim(0, MAP_HEIGHT)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.canvas.manager.set_window_title('Wanderer Deer Predictive Tracker (Python + Matplotlib)')
    
# Draw static hotspot circle
hotspot_circle_fill = patches.Circle((hotspot['x'], hotspot['y']), HOTSPOT_RADIUS, 
                                     facecolor='rgba(251, 191, 36, 0.2)', zorder=0)
hotspot_circle_border = patches.Circle((hotspot['x'], hotspot['y']), HOTSPOT_RADIUS, 
                                       edgecolor='#f59e0b', facecolor='none', linewidth=2, zorder=0)
ax.add_patch(hotspot_circle_fill)
ax.add_patch(hotspot_circle_border)
ax.text(hotspot['x'], hotspot['y'], 'Habitat Hotspot', ha='center', va='center', 
        color='#b45309', fontsize=10, weight='bold')

# Initialize dynamic plot elements
path_line, = ax.plot([], [], lw=2, color='#10b981', label='Actual Path (GPS)')
current_pos_dot, = ax.plot([], [], 'o', markersize=8, color='#3b82f6', label='Current Location')
predicted_pos_marker, = ax.plot([], [], 'x', markersize=12, mew=2, color='#ef4444', label='Predicted Location (AI)')

# Setup for the info text box
info_text = ax.text(0.01, 0.98, '', transform=ax.transAxes, va='top', ha='left',
                    bbox=dict(boxstyle='round,pad=0.5', fc='#f0fdf4', alpha=0.8))

ax.legend(loc='lower left', frameon=True, facecolor='white', framealpha=0.8)
plt.tight_layout()

# --- Main Animation Loop ---
def animate(frame):
    """This function is called for each frame of the animation."""
    # 1. Move the animal (Simulate real world)
    speed = update_animal_location()
    
    # 2. Predict the next location (AI model)
    animal['prediction'] = predict_next_location(animal['history'])

    # 3. Update plot data
    if animal['history']:
        hist_x, hist_y = zip(*[(p['x'], p['y']) for p in animal['history']])
        path_line.set_data(hist_x, hist_y)
        current_pos_dot.set_data([animal['x']], [animal['y']])
    
    predicted_pos_marker.set_data([animal['prediction']['x']], [animal['prediction']['y']])
    
    # 4. Update UI text
    error = np.sqrt((animal['x'] - animal['prediction']['x'])**2 + 
                    (animal['y'] - animal['prediction']['y'])**2)
                    
    text_content = (
        f"--- Actual GPS ---\n"
        f"Coord (X, Y): ({animal['x']:.2f}, {animal['y']:.2f})\n"
        f"Speed: {speed:.2f} units/step\n"
        f"\n--- AI Prediction ---\n"
        f"Pred. (X, Y): ({animal['prediction']['x']:.2f}, {animal['prediction']['y']:.2f})\n"
        f"Prediction Error: {error:.2f} units"
    )
    info_text.set_text(text_content)
    
    # Return the artists that have been updated
    return path_line, current_pos_dot, predicted_pos_marker, info_text

# --- Run the Simulation ---
if _name_ == '_main_':
    setup_plot()
    # Create the animation object
    ani = FuncAnimation(fig, animate, frames=200, interval=TICK_INTERVAL_MS, blit=True)
    plt.show()