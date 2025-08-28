import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import tkinter as tk

# -------------------------------
# Simulation Parameters
# -------------------------------
dt = 0.01
total_time = 10
steps = int(total_time / dt)

# Target orientation
target = np.array([0.0, 0.0, 0.0])

# Initial state
state = np.zeros(3)
state_history = []

# -------------------------------
# Generate Training Data for ML Control
# -------------------------------
def generate_training_data(samples=500):
    X = []
    y = []
    for _ in range(samples):
        # Random initial orientation
        init_state = np.random.uniform(-0.5, 0.5, 3)
        state_sim = init_state.copy()
        # Random “ideal control signals” (simulated)
        control_signal_history = []
        for _ in range(steps):
            error = target - state_sim
            # Here we simulate “optimal control” for training
            control_signal = 2*error + np.random.normal(0, 0.05, 3)
            state_sim += control_signal*dt
            control_signal_history.append(control_signal.copy())
        X.append(np.tile(init_state, (steps,1)))  # replicate for each timestep
        y.append(control_signal_history)
    return np.vstack(X), np.vstack(y)

print("Generating training data...")
X_train, y_train = generate_training_data(samples=50)  # fewer for faster demo

# Train ML model to predict control signals directly
print("Training ML model...")
model = RandomForestRegressor(n_estimators=50)
model.fit(X_train, y_train)

# -------------------------------
# ML-based Simulation
# -------------------------------
def simulate_ml_control():
    global state
    state_history.clear()
    state = np.random.uniform(-0.3,0.3,3)
    for _ in range(steps):
        # ML predicts the control signal directly
        control_signal = model.predict(state.reshape(1,-1))[0]
        state += control_signal*dt + np.random.normal(0, 0.01, 3)
        state_history.append(state.copy())
    return np.array(state_history)

# -------------------------------
# Visualization
# -------------------------------
def plot_simulation():
    data = simulate_ml_control()
    plt.figure(figsize=(8,5))
    plt.plot(data[:,0], label="Roll")
    plt.plot(data[:,1], label="Pitch")
    plt.plot(data[:,2], label="Yaw")
    plt.title("ML-Controlled Quadcopter Simulation")
    plt.xlabel("Time step")
    plt.ylabel("Angle (rad)")
    plt.legend()
    plt.show()

# -------------------------------
# GUI to Run Simulation
# -------------------------------
def run_gui():
    root = tk.Tk()
    root.title("ML-Controlled Quadcopter Simulator")
    tk.Label(root, text="Press button to run ML-based simulation").pack(pady=10)
    tk.Button(root, text="Run Simulation", command=plot_simulation).pack(pady=20)
    root.mainloop()

if __name__ == "__main__":
    run_gui()




    import numpy as np
import matplotlib.pyplot as plt

t = np.linspace(0, 10, 1000)

roll_ideal = np.zeros_like(t)             # Hover: roll = 0
pitch_ideal = 0.1*np.sin(2*np.pi*0.5*t)  # Small forward-back tilt
yaw_ideal = 0.05*t                        # Slow yaw rotation

plt.plot(t, roll_ideal, label="Roll")
plt.plot(t, pitch_ideal, label="Pitch")
plt.plot(t, yaw_ideal, label="Yaw")
plt.xlabel("Time (s)")
plt.ylabel("Angle (rad)")
plt.title("Ideal Roll, Pitch, Yaw for simple maneuver")
plt.legend()
plt.show()

