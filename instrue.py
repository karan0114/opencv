import cv2
import pygame
import numpy as np
import mediapipe as mp

# 1. Initialize Audio Mixer
pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)

def generate_chord(frequencies, duration=0.5):
    """Synthesizes multiple frequencies into a single harmonic chord."""
    sample_rate = 44100
    n_samples = int(sample_rate * duration)
    t = np.linspace(0, duration, n_samples, False)
    
    # Combine sine waves for each note in the chord
    wave = np.zeros(n_samples)
    for freq in frequencies:
        wave += np.sin(2 * np.pi * freq * t)
    
    # Normalize volume to avoid audio clipping distortion
    wave = (wave / len(frequencies)) * 0.5
    
    # Convert to C-contiguous stereo array
    sound_array = np.asarray([wave, wave]).T * 32767
    sound_array = np.ascontiguousarray(sound_array, dtype=np.int16)
    
    return pygame.sndarray.make_sound(sound_array)

# 2. Define Chord Frequencies (Hz)
CHORDS = [
    {
        "name": "D Major",
        "sound": generate_chord([293.66, 369.99, 440.00]),  # D4, F#4, A4
        "color": (0, 255, 0)
    },
    {
        "name": "A Major",
        "sound": generate_chord([220.00, 277.18, 329.63]),  # A3, C#4, E4
        "color": (255, 255, 0)
    },
    {
        "name": "E Major",
        "sound": generate_chord([329.63, 415.30, 493.88]),  # E4, G#4, B4
        "color": (255, 0, 255)
    },
    {
        "name": "F# Minor",
        "sound": generate_chord([369.99, 440.00, 554.37]),  # F#4, A4, C#5
        "color": (0, 165, 255)
    }
]

# 3. Setup MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

# 4. Main Application Loop
# Try finding an active camera across standard indices (0, 1, 2)
cap = None
for cam_index in [0, 1, 2]:
    cap = cv2.VideoCapture(cam_index, cv2.CAP_V4L2)
    if cap.isOpened():
        print(f"Successfully opened camera at index {cam_index}")
        break

if not cap or not cap.isOpened():
    print("Error: Could not open any camera. Check if another app is using it.")
    exit()

chord_index = 0
was_pinched = False

print("Gesture Synth Running! Pinch thumb & index finger to advance chords.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, c = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    results = hands.process(rgb_frame)

    current_chord = CHORDS[chord_index]

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get Index Tip (8) and Thumb Tip (4) positions
            index_tip = hand_landmarks.landmark[8]
            thumb_tip = hand_landmarks.landmark[4]

            ix, iy = int(index_tip.x * w), int(index_tip.y * h)
            tx, ty = int(thumb_tip.x * w), int(thumb_tip.y * h)

            # Calculate distance between fingertips
            distance = np.hypot(ix - tx, iy - ty)

            # Trigger chord pinch
            if distance < 40:
                cv2.circle(frame, (ix, iy), 15, current_chord["color"], cv2.FILLED)
                
                if not was_pinched:
                    # Play current chord
                    current_chord["sound"].play()
                    # Move to next chord in loop (D -> A -> E -> F#m -> D ...)
                    chord_index = (chord_index + 1) % len(CHORDS)
                    was_pinched = True
            else:
                was_pinched = False

    # Overlay current chord onto screen UI
    cv2.putText(
        frame,
        f"Next Chord: {current_chord['name']}",
        (40, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        current_chord["color"],
        3
    )

    cv2.imshow("Gesture Synth - Avicii Mode", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
