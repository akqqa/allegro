import cv2
import mediapipe as mp
import math
import time
import numpy as np
import sounddevice as sd

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

DEBOUNCE_THRESHOLD = 2
MAX_HANDS = 1
MODEL_COMPLEXITY = 0
MIN_DETECTION_CONFIDENCE = 0.8
MIN_TRACKING_CONFIDENCE = 0.8

PINCH_THRESHOLD = 0.6

FREQUENCY = 440
AMPLITUDE = 0.2
SAMPLERATE = 44100

SAMPLES = 1024
waveform = np.sin(2 * np.pi * np.arange(SAMPLES) / SAMPLES).astype(np.float32)
step = FREQUENCY * SAMPLES / SAMPLERATE
fade_duration = int(0.01 * SAMPLERATE)
start_idx = 0

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
      mcp = handLandmarks.landmark[1] # Base of thumb

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

def sineCallback(outdata, frames, time, status):
  global start_idx
  out = np.zeros((frames, 1), dtype=np.float32)

  for i in range(frames):
    out[i] = waveform[int(start_idx) % SAMPLES]
    start_idx += step

  outdata[:] = out

# From https://stackoverflow.com/questions/44588279/find-and-draw-the-largest-contour-in-opencv-on-a-specific-color-python
def detectColour(image):
  hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

  # Define HSV range for orange
  lower = np.array([10, 20, 20])
  upper = np.array([30, 255, 255])

  # find the colors within the specified boundaries and apply
  # the mask
  mask = cv2.inRange(hsv, lower, upper)
  output = cv2.bitwise_and(image, image, mask=mask)

  ret,thresh = cv2.threshold(mask, 40, 255, 0)
  major_version = int(cv2.__version__.split('.')[0])
  if major_version >= 4:
      contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
  else:
      _, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

  if len(contours) != 0:
    # draw in blue the contours that were founded
    cv2.drawContours(output, contours, -1, (255, 0, 0), 3)  # Blue

    # find the biggest countour (c) by the area
    c = max(contours, key = cv2.contourArea)
    x,y,w,h = cv2.boundingRect(c)

    # draw the biggest contour (c) in green
    cv2.rectangle(output,(x,y),(x+w,y+h),(0,255,0),2)
  
  # show the images
  cv2.imshow("Result", np.hstack([image, output]))

  return output


    
pinchCounter = 0
notPinchCounter = 0
pinching = False

prevTime = 0

cap = cv2.VideoCapture(0)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
with mp_hands.Hands(
    max_num_hands=MAX_HANDS,
    model_complexity=MODEL_COMPLEXITY,
    min_detection_confidence=MIN_DETECTION_CONFIDENCE,
    min_tracking_confidence=MIN_TRACKING_CONFIDENCE) as hands:
  
  soundStream = sd.OutputStream(channels=1, callback=sineCallback, samplerate=SAMPLERATE, blocksize=256, latency=0.035)

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

    #detectColour(image)


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

    pinchDetected = detectPinching(results, "left", PINCH_THRESHOLD)

    # Possibility of debounce, but makes it unresponsive
    if pinchDetected == True:
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


    # Generate a sine wave


    currTime = time.time()
    fps = 1/ (currTime - prevTime)
    prevTime = currTime

    image = cv2.flip(image, 1)

    cv2.putText(image, f'FPS: {int(fps)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    if pinching:
      cv2.circle(image, (30, 50), 8, (0, 0, 255), -1)
      if not soundStream.active:
        soundStream.start()
    else:
      print(start_idx % SAMPLES)
      if soundStream.active:
        soundStream.stop()
        start_idx = 0

    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('MediaPipe Hands', image)
    if cv2.waitKey(5) & 0xFF == 27:
      break


cap.release()




