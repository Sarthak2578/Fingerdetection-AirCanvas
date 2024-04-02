import cv2
import mediapipe as mp
import numpy as np
import time
import matplotlib.pyplot as plt

# Initialize Mediapipe Hand module
mp_hand = mp.solutions.hands
hands = mp_hand.Hands()

# Open video capture
cap = cv2.VideoCapture(0)

# Initialize finger trajectory storage
finger_trajectories = [[] for _ in range(21)]  # One list for each finger landmark

# Initialize text storage
text = ""

# Initialize a file to save the data
file_path = "finger_trajectory.txt"

record_trajectory = False  # Flag to indicate whether to record trajectory

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    # Convert the frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame using Mediapipe Hand tracking
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Get landmarks for the index finger
            index_finger_landmarks = [hand_landmarks.landmark[i] for i in [8]]

            # Get current timestamp
            current_time = time.time()

            # Process and track the index finger landmarks
            for idx, landmark in enumerate(index_finger_landmarks):
                x, y = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])

                if record_trajectory:
                    # Store the landmark coordinates and timestamp for finger trajectory tracking
                    finger_trajectories[idx].append((x, y, current_time))

                # Draw the landmarks on the frame
                cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)

    # Remove outdated points from trajectories if not recording
    if not record_trajectory:
        finger_trajectories = [[] for _ in range(21)]

    # Draw finger trajectories if recording
    if record_trajectory:
        for trajectory in finger_trajectories:
            if len(trajectory) > 1:
                pts = np.array([(x, y) for x, y, _ in trajectory], np.int32)  # Convert to NumPy array of points
                pts = pts.reshape((-1, 1, 2))  # Reshape for polylines
                cv2.polylines(frame, [pts], isClosed=False, color=(230, 216, 173), thickness=2)

    # Display the frame
    cv2.imshow("Finger Trajectory Tracker", frame)

    # Save the finger trajectories and text to a file if 's' key is pressed
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break
    elif k == ord('s'):
        with open(file_path, "w") as file:
            # Save finger trajectories
            for idx, trajectory in enumerate(finger_trajectories):
                file.write(f"Finger {idx} Trajectory:\n")
                for x, y, timestamp in trajectory:
                    file.write(f"{x},{y},{timestamp}\n")
                file.write("\n")

            # Save text
            file.write("Text:\n")
            file.write(text)
    elif k == ord('d'):
        record_trajectory = not record_trajectory  # Toggle recording trajectory

# Release video capture and close windows
cap.release()
cv2.destroyAllWindows()

# Read the saved finger trajectory data from the file
file_path = "finger_trajectory.txt"

finger_trajectories = [[] for _ in range(21)]  # One list for each finger landmark

with open(file_path, "r") as file:
    current_finger_index = None
    for line in file:
        line = line.strip()
        if line.startswith("Finger"):
            # Extract the finger index from the line
            current_finger_index = int(line.split()[1])
        elif line.startswith("Text:"):
            # Skip lines that contain "Text:"
            continue
        elif line:
            # Split the line into x, y, and timestamp values
            x, y, timestamp = map(float, line.split(","))
            finger_trajectories[current_finger_index].append((x, y, timestamp))

# Plot the finger trajectories with inverted y-coordinates
plt.figure(figsize=(8, 6))
for idx, trajectory in enumerate(finger_trajectories):
    if trajectory:
        x_values, y_values, _ = zip(*trajectory)
        y_values = [-y for y in y_values]  # Invert y-coordinates
        plt.plot(x_values, y_values, label=f"Finger {idx}")

plt.title("Finger Trajectories")
plt.xlabel("X Coordinate")
plt.ylabel("Y Coordinate (Inverted)")
plt.legend()
plt.grid(True)

# Show the plot
plt.show()
