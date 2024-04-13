import cv2
import mediapipe as mp
import pyautogui
import time
from datetime import datetime

# Initialize MediaPipe Hands model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5, max_num_hands=2, static_image_mode=False)

# Initialize MediaPipe drawing utilities for visualization
mp_draw = mp.solutions.drawing_utils

last_slide_change_time = 0  # Track the last slide change time
slide_change_delay = 1  # Delay between slide changes
last_screenshot_time = 0  # Track the last screenshot time
screenshot_delay = 1  # Delay between screenshots

def display_instructions(image):
    instructions = [
        "",
        "GUIDE -",
        "Thumbs up: Left hand - scroll up, Right hand - scroll down",
        "Index finger: Left hand - previous slide, Right hand - next slide",
        "All fingers: Left hand - volume down, Right hand - volume up",
        "Thumb + index of both hands: Screenshot"
    ]
    font_scale = 0.58
    font_color = (255, 255, 255)  # White
    font_thickness = 1
    font = cv2.FONT_HERSHEY_SIMPLEX
    x, y = 10, 10  # Starting position
    rectangle_bkg_color = (0,0,0)  # Black background for better readability
    cv2.rectangle(image, (0, 1), (655, 150), rectangle_bkg_color, -1)  # Adjust size as needed

    for line in instructions:
        cv2.putText(image, line, (x, y), font, font_scale, font_color, font_thickness)
        y += 25  # Move to the next line

def are_all_fingers_up(hand_landmarks):
    """Check if all fingers are up."""
    tip_ids = [4, 8, 12, 16, 20]
    return all(hand_landmarks.landmark[tip_id].y < hand_landmarks.landmark[tip_id - 2].y for tip_id in tip_ids)

def interpret_gesture(hand_landmarks, handedness_label, current_time):
    global last_slide_change_time, last_screenshot_time
    # Implement existing functionalities...
    
    # Check for volume control and slide changing
    if are_all_fingers_up(hand_landmarks):
        if handedness_label == 'Left':
            pyautogui.press('volumeup')
        elif handedness_label == 'Right':
            pyautogui.press('volumedown')
        return
    
    thumb_tip_y = hand_landmarks.landmark[4].y
    wrist_y = hand_landmarks.landmark[0].y
    index_finger_tip_y = hand_landmarks.landmark[8].y
    index_finger_pip_y = hand_landmarks.landmark[6].y

    # Scroll based on thumb position
    if thumb_tip_y < wrist_y - 0.05:
        if handedness_label == 'Right':
            pyautogui.scroll(100)  # Scroll up
        elif handedness_label == 'Left':
            pyautogui.scroll(-100)  # Scroll down
    
    # Slide change based on index finger
    if index_finger_tip_y < index_finger_pip_y and (current_time - last_slide_change_time > slide_change_delay):
        if handedness_label == 'Right':
            pyautogui.press('left')
            last_slide_change_time = current_time
        elif handedness_label == 'Left':
            pyautogui.press('right')
            last_slide_change_time = current_time

    
def check_screenshot_gesture(left_hand_landmarks, right_hand_landmarks, current_time):
    global last_screenshot_time
    # Check for thumb and index up, other fingers down
    conditions = [
        left_hand_landmarks.landmark[4].y < left_hand_landmarks.landmark[3].y,  # Left thumb
        left_hand_landmarks.landmark[8].y < left_hand_landmarks.landmark[6].y,  # Left index
        right_hand_landmarks.landmark[4].y < right_hand_landmarks.landmark[3].y,  # Right thumb
        right_hand_landmarks.landmark[8].y < right_hand_landmarks.landmark[6].y,  # Right index
    ]
    if all(conditions) and (current_time - last_screenshot_time > screenshot_delay):
        filename = f"screenshot_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.png"
        pyautogui.screenshot(filename)
        print(f"Screenshot saved as {filename}")
        last_screenshot_time = current_time


# Capture video from webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        continue

    # Convert the image color from BGR to RGB and process it
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    current_time = time.time()

    if results.multi_hand_landmarks and len(results.multi_hand_landmarks) == 2:
        # Assuming the first detected hand is left and the second is right for the screenshot gesture
        left_hand_landmarks = results.multi_hand_landmarks[0]
        right_hand_landmarks = results.multi_hand_landmarks[1]
        check_screenshot_gesture(left_hand_landmarks, right_hand_landmarks, current_time)

    # Handle other gestures
    if results.multi_hand_landmarks:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            mp_draw.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            interpret_gesture(hand_landmarks, handedness.classification[0].label, current_time)
    
    # Display instructions on the frame
    display_instructions(image)

    # Show the image
    cv2.imshow('Hand Tracking', image)
    if cv2.waitKey(5) & 0xFF == 27:  # ESC key to exit
        break

cap.release()
cv2.destroyAllWindows()




