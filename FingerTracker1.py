import cv2
import mediapipe as mp
import numpy as np
import time

# Initialize Mediapipe Hand module
mp_hand = mp.solutions.hands
hands = mp_hand.Hands()

cap = cv2.VideoCapture(0)

# Initialize finger trajectory storage
finger_trajectories = [[] for _ in range(21)]  # One list for each finger landmark

# Initialize text storage
text = ""

# Initialize a file to save the data
file_path = "finger_trajectory.txt"

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame using Mediapipe Hand tracking
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            index_finger_landmarks = [hand_landmarks.landmark[i] for i in [8]]

            current_time = time.time()

            for idx, landmark in enumerate(index_finger_landmarks):
                x, y = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])

                finger_trajectories[idx].append((x, y, current_time))

                cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)

    # Remove outdated points from trajectories
    for trajectory in finger_trajectories:
        current_time = time.time()
        while trajectory and current_time - trajectory[0][2] > 2:
            trajectory.pop(0)

    # Draw finger trajectories
    for trajectory in finger_trajectories:
        if len(trajectory) > 1:
            pts = np.array([(x, y) for x, y, _ in trajectory], np.int32)  # Convert to NumPy array of points
            pts = pts.reshape((-1, 1, 2))  # Reshape for polylines
            cv2.polylines(frame, [pts], isClosed=False, color=(230, 216, 173), thickness=2)

    # Display the frame
    cv2.imshow("Finger Trajectory Tracker", frame)

    k = cv2.waitKey(1) & 0xFF

    if k == 27:
        break

# Release video capture and close windows
cap.release()
cv2.destroyAllWindows()
