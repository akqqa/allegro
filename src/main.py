import cv2
import mediapipe as mp
import math
import time

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

DEBOUNCE_THRESHOLD = 2
MAX_HANDS = 1
MODEL_COMPLEXITY = 0
MIN_DETECTION_CONFIDENCE = 0.6
MIN_TRACKING_CONFIDENCE = 0.4

# PLAN: first hand controls play/dont play
# Second hand controls pitch. 

# Sound lib: https://python-sounddevice.readthedocs.io/en/0.5.1/examples.html - to generate sine wave on the fly 
# Use sine by default, but allow overwriting with custom waves. read from sample array instead of generating, and loop. pitch altered by sample rate.

# Potentially use mediapipe solely for left hand gestures, and track a colour for pitch. any coloured object can be used, eg a green card, etc. will improve speed at which pitch can change!

# Takes mediapipe results, which hand (inverted), and a threshold for normalised distance to count as a pinch. True if pinch, False if open, None if no hands detected
def detectPinching(results, hand, threshold):

  if not results.multi_hand_landmarks or not results.multi_handedness:
    return None  # No hands detected

  for handLabel, handLandmarks in zip(results.multi_handedness, results.multi_hand_landmarks):

    label = handLabel.classification[0].label.lower()

    if label == hand:

      thumbTip = handLandmarks.landmark[4]
      indexTip = handLandmarks.landmark[8]
      wrist = handLandmarks.landmark[0] # Base of wrist
      mcp = handLandmarks.landmark[9] # Base of middle finger

      pinchDistance = distance3d(thumbTip, indexTip)
      handSize = distance3d(wrist, mcp)

      # Normalise pinch distance based on handsize
      normalisedPinchDistance = pinchDistance / handSize

      if normalisedPinchDistance < threshold:
        return True
      else:
        return False
  
def distance3d(landmark1, landmark2):
  dx = landmark1.x - landmark2.x
  dy = landmark1.y - landmark2.y
  dz = landmark1.z - landmark2.z
  distance = math.sqrt(dx*dx + dy*dy + dz*dz)

  return distance
    
pinchCounter = 0
notPinchCounter = 0
pinching = False

prevTime = 0

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
with mp_hands.Hands(
    max_num_hands=MAX_HANDS,
    model_complexity=MODEL_COMPLEXITY,
    min_detection_confidence=MIN_DETECTION_CONFIDENCE,
    min_tracking_confidence=MIN_TRACKING_CONFIDENCE) as hands:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_hand_landmarks:
      for hand_landmarks in results.multi_hand_landmarks:
        mp_drawing.draw_landmarks(
            image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())

    pinchDetected = detectPinching(results, "right", 0.18)

    # Possibility of debounce, but makes it unresponsive
    if pinchDetected == True: # actually left, inverted
      pinchCounter += 1
      notPinchCounter = 0
      if (pinchCounter >= DEBOUNCE_THRESHOLD):
        pinching = True
        notPinchCounter = 0
    elif pinchDetected == False:
      notPinchCounter += 1
      pinchCounter = 0
      if (notPinchCounter >= DEBOUNCE_THRESHOLD):
        pinching = False
        pinchCounter = 0
    else:
      pass

    currTime = time.time()
    fps = 1/ (currTime - prevTime)
    prevTime = currTime

    image = cv2.flip(image, 1)

    cv2.putText(image, f'FPS: {int(fps)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    if pinching:
      cv2.circle(image, (30, 50), 8, (0, 0, 255), -1)

    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('MediaPipe Hands', image)
    if cv2.waitKey(5) & 0xFF == 27:
      break


cap.release()




