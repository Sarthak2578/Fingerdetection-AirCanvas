import cv2
import mediapipe as mp
import pyautogui

# Initialize MediaPipe Hands model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, max_num_hands=1)

# Initialize MediaPipe drawing utilities for visualization
mp_draw = mp.solutions.drawing_utils


def check_fingers_up(hand_landmarks, hand_world_landmarks):
    """
    Checks which fingers are up. Returns a list of booleans.
    """
    tip_ids = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky finger tip ids
    fingers_up = []
    for id in tip_ids:
        # For thumb, check x coordinate (special case)
        if id == 4:
            fingers_up.append(hand_landmarks.landmark[id].x < hand_landmarks.landmark[id - 1].x)
        else:
            fingers_up.append(hand_landmarks.landmark[id].y < hand_landmarks.landmark[id - 2].y)
    return fingers_up


def interpret_gesture(fingers_up):
    """
    Interprets the gesture based on which fingers are up.
    """
    if fingers_up == [False, True, False, False, False]:
        return 'next_slide'
    elif fingers_up == [False, True, True, False, False]:
        return 'prev_slide'
    elif fingers_up == [True, False, False, False, False]:
        return 'scroll_up'
    elif fingers_up == [False, False, False, False, True]:
        return 'scroll_down'
    else:
        return None


# Capture video from webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        continue

    # Convert the image color from BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process the image and find hands
    results = hands.process(image)

    # Convert the image color back to BGR for OpenCV
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Check if any hand is detected
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks
            mp_draw.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Check fingers up
            fingers_up = check_fingers_up(hand_landmarks, results.multi_hand_world_landmarks[0])

            # Interpret gesture
            gesture = interpret_gesture(fingers_up)

            # Perform action based on gesture
            if gesture == 'next_slide':
                pyautogui.press('right')
            elif gesture == 'prev_slide':
                pyautogui.press('left')
            elif gesture == 'scroll_up':
                pyautogui.scroll(100)
            elif gesture == 'scroll_down':
                pyautogui.scroll(-100)

    # Show the image
    cv2.imshow('Hand Tracking', image)
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
