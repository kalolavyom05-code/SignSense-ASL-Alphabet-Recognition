import os
import numpy as np
import pandas as pd

DATA_DIR = "data"
CSV_PATH = os.path.join(DATA_DIR, "landmarks.csv")

# Standard MediaPipe 21 hand landmarks base templates (approximated canonical configurations)
def get_base_skeleton(gesture):
    # Base neutral hand joints: wrist at (0, 0, 0)
    # 21 points x 3 (x, y, z)
    coords = np.zeros((21, 3), dtype=np.float32)
    
    # Palm joints MCPs
    coords[1]  = [0.20, 0.20, 0.0]   # Thumb CMC
    coords[2]  = [0.35, 0.35, 0.0]   # Thumb MCP
    coords[5]  = [0.25, 0.50, 0.0]   # Index MCP
    coords[9]  = [0.00, 0.55, 0.0]   # Middle MCP
    coords[13] = [-0.20, 0.50, 0.0]  # Ring MCP
    coords[17] = [-0.35, 0.40, 0.0]  # Pinky MCP

    if gesture == "A":  # Fist with thumb pointing upward along index
        # Curled fingers
        coords[3] = [0.40, 0.45, 0.1]
        coords[4] = [0.25, 0.60, 0.1]  # Thumb resting along fist
        for base, tip_idx in [(5, 8), (9, 12), (13, 16), (17, 20)]:
            coords[base+1] = coords[base] + [0.0, 0.15, -0.1]
            coords[base+2] = coords[base] + [0.0, 0.05, -0.2]
            coords[base+3] = coords[base] + [0.0, -0.05, -0.15]
            
    elif gesture == "B": # Open flat palm, thumb curled inward across palm
        # Fingers extended up
        for base, tip_idx in [(5, 8), (9, 12), (13, 16), (17, 20)]:
            coords[base+1] = coords[base] + [0.0, 0.15, 0.0]
            coords[base+2] = coords[base] + [0.0, 0.30, 0.0]
            coords[base+3] = coords[base] + [0.0, 0.45, 0.0]
        # Thumb folded across palm
        coords[3] = [0.10, 0.30, -0.1]
        coords[4] = [-0.05, 0.25, -0.1]

    elif gesture == "L": # L shape: thumb and index extended at 90 deg
        coords[3] = [0.50, 0.35, 0.0]
        coords[4] = [0.70, 0.35, 0.0]  # Thumb sticking out
        coords[6] = [0.25, 0.65, 0.0]
        coords[7] = [0.25, 0.80, 0.0]
        coords[8] = [0.25, 0.95, 0.0]  # Index pointing straight up
        # Curled middle, ring, pinky
        for base in [9, 13, 17]:
            coords[base+1] = coords[base] + [0.0, 0.10, -0.1]
            coords[base+2] = coords[base] + [0.0, 0.00, -0.15]
            coords[base+3] = coords[base] + [0.0, -0.05, -0.1]

    elif gesture == "V": # Peace sign: Index and Middle extended in V shape
        # Index tilted slightly right
        coords[6] = [0.35, 0.65, 0.0]
        coords[7] = [0.45, 0.80, 0.0]
        coords[8] = [0.55, 0.95, 0.0]
        # Middle tilted slightly left
        coords[10] = [-0.10, 0.70, 0.0]
        coords[11] = [-0.18, 0.85, 0.0]
        coords[12] = [-0.25, 1.00, 0.0]
        # Curled thumb, ring, pinky
        coords[3] = [0.20, 0.30, -0.1]
        coords[4] = [0.10, 0.25, -0.1]
        for base in [13, 17]:
            coords[base+1] = coords[base] + [0.0, 0.10, -0.1]
            coords[base+2] = coords[base] + [0.0, 0.00, -0.15]
            coords[base+3] = coords[base] + [0.0, -0.05, -0.1]

    elif gesture == "Y": # Thumb and pinky extended, middle three curled
        # Thumb extended outward
        coords[3] = [0.50, 0.35, 0.0]
        coords[4] = [0.70, 0.40, 0.0]
        # Pinky extended outward
        coords[18] = [-0.45, 0.50, 0.0]
        coords[19] = [-0.55, 0.60, 0.0]
        coords[20] = [-0.65, 0.70, 0.0]
        # Middle three curled
        for base in [5, 9, 13]:
            coords[base+1] = coords[base] + [0.0, 0.10, -0.1]
            coords[base+2] = coords[base] + [0.0, 0.00, -0.15]
            coords[base+3] = coords[base] + [0.0, -0.05, -0.1]

    # Normalize scale: max distance from wrist
    dists = np.linalg.norm(coords, axis=1)
    max_d = np.max(dists) if np.max(dists) > 0 else 1.0
    return coords / max_d


def generate_starter_dataset():
    os.makedirs(DATA_DIR, exist_ok=True)
    gestures = ["A", "B", "L", "V", "Y"]
    num_samples_per_gesture = 60
    rows = []

    for g in gestures:
        base = get_base_skeleton(g)
        for _ in range(num_samples_per_gesture):
            # Add small random noise to simulate natural hand variance
            noise = np.random.normal(0, 0.025, base.shape).astype(np.float32)
            noisy_sample = (base + noise).flatten()
            rows.append([g] + noisy_sample.tolist())

    cols = ["label"] + [f"f_{i}" for i in range(63)]
    df = pd.DataFrame(rows, columns=cols)
    df.to_csv(CSV_PATH, index=False)
    print(f"[SUCCESS] Generated starter dataset with {len(df)} samples at '{CSV_PATH}'")


if __name__ == "__main__":
    generate_starter_dataset()
