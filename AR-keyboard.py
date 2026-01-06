import cv2
import numpy as np
import time
from collections import defaultdict, deque

try:
    import mediapipe as mp
except ImportError:
    print("ERROR: MediaPipe not installed!")
    print("Please install: pip install mediapipe")
    exit(1)

class ARKeyboard:
    def __init__(self, debug_mode=True):
        try:
            self.mp_hands = mp.solutions.hands
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=1,
                min_detection_confidence=0.7,
                min_tracking_confidence=0.5
            )
            self.mp_draw = mp.solutions.drawing_utils
        except Exception as e:
            print(f"Error initializing MediaPipe: {e}")
            raise

        self.keys = [
            ['Q', 'W', 'E', 'R', 'T', 'Y', 'U', 'I', 'O', 'P'],
            ['A', 'S', 'D', 'F', 'G', 'H', 'J', 'K', 'L'],
            ['Z', 'X', 'C', 'V', 'B', 'N', 'M'],
            ['CLEAR', 'SPACE', 'DEL']
        ]

        self.key_width = 55
        self.key_height = 55
        self.key_margin = 8
        self.keyboard_y = 200

        self.typed_text = ""
        self.last_key_time = 0
        self.key_delay = 0.5  # Increased to prevent accidental repeats

        # Store recent fingertip positions: key = (hand_idx, tip_id)
        self.finger_history = defaultdict(lambda: deque(maxlen=12))
        
        # Track when finger entered key zone
        self.finger_hover_start = {}
        self.finger_in_zone = {}
        
        # Require stable hover before allowing tap
        self.min_hover_time = 0.2  # Must hover for 200ms first

        # Last tapped key for visual feedback
        self.current_pressed_key = None

        # Per-key cooldown to avoid jitter
        self.key_last_tap = {}

        # Increased tap zone for easier targeting
        self.tap_proximity = 60

        # Debug mode
        self.debug_mode = debug_mode
        self.debug_info = ""

    def get_key_rect(self, row, col, frame_width):
        # Calculate even-sized keys for all rows, scaled to frame width
        total_keys = len(self.keys[row])
        
        if row == 3:  # Bottom row with special keys
            # Make all three keys equal width
            key_width = int(self.key_width * 2.5)  # Wider keys for bottom row
            total_width = 3 * key_width + 2 * self.key_margin
            start_x = (frame_width - total_width) // 2
            x = start_x + col * (key_width + self.key_margin)
            w = key_width
        else:
            # Regular keys - all same size
            total_width = total_keys * self.key_width + (total_keys - 1) * self.key_margin
            start_x = (frame_width - total_width) // 2
            x = start_x + col * (self.key_width + self.key_margin)
            w = self.key_width

        y = self.keyboard_y + row * (self.key_height + self.key_margin)
        h = self.key_height
        return int(x), int(y), int(w), int(h)

    def draw_transparent_rect(self, frame, x, y, w, h, color, alpha=0.6, border_color=(200, 200, 200)):
        # Ensure all coordinates are integers
        x, y, w, h = int(x), int(y), int(w), int(h)
        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y), (x + w, y + h), color, -1)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        cv2.rectangle(frame, (x, y), (x + w, y + h), border_color, 1)

    def draw_keyboard(self, frame):
        frame_width = frame.shape[1]
        for row_idx, row in enumerate(self.keys):
            for col_idx, key in enumerate(row):
                x, y, w, h = self.get_key_rect(row_idx, col_idx, frame_width)
                if y + h > frame.shape[0] or x + w < 0 or x > frame.shape[1]:
                    continue

                color = (0, 220, 0) if key == self.current_pressed_key else (25, 25, 25)
                alpha = 0.75 if key == self.current_pressed_key else 0.5

                self.draw_transparent_rect(frame, x, y, w, h, color, alpha=alpha)

                display_text = {'SPACE': 'Space', 'DEL': 'Del', 'CLEAR': 'Clear'}.get(key, key)
                font_scale = 0.6 if key in ['SPACE', 'CLEAR', 'DEL'] else 0.7
                thickness = 2
                (text_w, text_h), _ = cv2.getTextSize(display_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                text_x = x + (w - text_w) // 2
                text_y = y + (h + text_h) // 2
                cv2.putText(frame, display_text, (text_x, text_y),
                           cv2.FONT_HERSHEY_SIMPLEX, font_scale, (240, 240, 240), thickness)

    def get_key_at_point(self, px, py, frame_width):
        """Return key name if (px, py) is close to any key center."""
        for row_idx, row in enumerate(self.keys):
            for col_idx, key in enumerate(row):
                x, y, w, h = self.get_key_rect(row_idx, col_idx, frame_width)
                cx, cy = x + w // 2, y + h // 2
                dist = ((px - cx)**2 + (py - cy)**2) ** 0.5
                if dist <= self.tap_proximity:
                    return key, (cx, cy), dist
        return None, None, None

    def detect_tap_gesture(self, history, finger_id):
        """
        More deliberate tap detection:
        - Finger must hover stably first (initial frames stable)
        - Then make a definite downward motion (pressing)
        - Motion should be 12-35 pixels downward
        - Must have good acceleration (deliberate press)
        """
        if len(history) < 6:
            return False

        points = list(history)[-10:]  # Look at last 10 frames
        if len(points) < 6:
            return False

        y_vals = [p[1] for p in points]
        x_vals = [p[0] for p in points]

        # First, check that early frames were stable (hovering)
        early_y = y_vals[:3]
        early_stability = max(early_y) - min(early_y)
        if early_stability > 8:  # Too much movement in hover phase
            if self.debug_mode:
                self.debug_info = f"No stable hover: {early_stability:.1f}px"
            return False

        # Check for downward motion in recent frames
        y_change = y_vals[-1] - y_vals[0]
        
        # Looking for a press: downward motion between 12-35 pixels
        if y_change < 12 or y_change > 35:
            if self.debug_mode:
                self.debug_info = f"Y change: {y_change:.1f} (need 12-35)"
            return False

        # Check horizontal stability (less than 20 pixels drift)
        x_range = max(x_vals) - min(x_vals)
        if x_range > 20:
            if self.debug_mode:
                self.debug_info = f"X unstable: {x_range:.1f}"
            return False

        # Check for strong acceleration (deliberate press, not drift)
        if len(points) >= 6:
            early_motion = abs(y_vals[3] - y_vals[0])
            late_motion = abs(y_vals[-1] - y_vals[-4])
            
            # Late motion should be significantly greater (accelerating press)
            if late_motion < early_motion * 1.2 or late_motion < 8:
                if self.debug_mode:
                    self.debug_info = f"Not deliberate enough"
                return False

        if self.debug_mode:
            self.debug_info = f"TAP! Y:{y_change:.1f} X:{x_range:.1f}"
        
        return True

    def process_frame(self, hand_idx, tip_id, fx, fy, frame_width):
        finger_id = (hand_idx, tip_id)
        
        key, key_center, dist = self.get_key_at_point(fx, fy, frame_width)
        
        if key is None:
            # Finger left the zone
            self.finger_history[finger_id].clear()
            self.finger_in_zone[finger_id] = None
            self.finger_hover_start.pop(finger_id, None)
            return

        # Track which key we're hovering over
        if self.finger_in_zone.get(finger_id) != key:
            # Switched to a new key - reset
            self.finger_in_zone[finger_id] = key
            self.finger_hover_start[finger_id] = time.time()
            self.finger_history[finger_id].clear()
            return
        
        # Check if we've hovered long enough
        hover_time = time.time() - self.finger_hover_start.get(finger_id, time.time())
        if hover_time < self.min_hover_time:
            # Still in initial hover period
            self.finger_history[finger_id].append((fx, fy))
            return
        
        # Add position to history (after hover period)
        self.finger_history[finger_id].append((fx, fy))
        
        # Only check for taps if we have enough history
        if len(self.finger_history[finger_id]) >= 6:
            if self.detect_tap_gesture(self.finger_history[finger_id], finger_id):
                current_time = time.time()
                last_time = self.key_last_tap.get(key, 0)
                
                if current_time - last_time > self.key_delay:
                    self.key_last_tap[key] = current_time
                    self.current_pressed_key = key
                    self.apply_key_action(key)
                    # Clear history after successful tap
                    self.finger_history[finger_id].clear()
                    self.finger_hover_start[finger_id] = time.time()  # Reset hover timer
                    print(f"Key pressed: {key}")  # Console feedback

    def apply_key_action(self, key):
        if key == 'SPACE':
            self.typed_text += ' '
        elif key == 'DEL':
            self.typed_text = self.typed_text[:-1]
        elif key == 'CLEAR':
            self.typed_text = ''
        else:
            self.typed_text += key

    def run(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("ERROR: Cannot open webcam!")
            return

        # Set higher resolution for better detection
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        print("=" * 60)
        print("AR KEYBOARD - Improved Tap Detection")
        print("=" * 60)
        print("Instructions:")
        print("  1. Position your index finger over a key")
        print("  2. Make a quick downward 'press' motion (like poking)")
        print("  3. The key will briefly turn green when pressed")
        print()
        print("Tips:")
        print("  - Use only ONE hand with your INDEX finger")
        print("  - Keep your hand relatively still while hovering")
        print("  - Make a deliberate downward motion to 'press'")
        print("  - Don't move too fast or too slow")
        print()
        print("Press 'q' to quit, 'd' to toggle debug mode")
        print("=" * 60)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            h, w = frame.shape[:2]
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)

            # Draw text area
            cv2.rectangle(frame, (0, 0), (w, 90), (0, 0, 0), -1)
            cv2.putText(frame, "Typed:", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(frame, self.typed_text[-35:], (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 100), 2)
            
            # Debug info
            if self.debug_mode and self.debug_info:
                cv2.putText(frame, f"Debug: {self.debug_info}", (10, 75), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

            self.current_pressed_key = None  # reset unless a tap occurs
            self.debug_info = ""

            if results.multi_hand_landmarks and results.multi_handedness:
                for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                    
                    # Use only index finger tip (landmark 8)
                    tip_id = 8
                    tip = hand_landmarks.landmark[tip_id]
                    fx = int(tip.x * w)
                    fy = int(tip.y * h)
                    
                    # Draw fingertip with color based on proximity to keys
                    key, _, dist = self.get_key_at_point(fx, fy, w)
                    if key:
                        # Green when over a key
                        cv2.circle(frame, (fx, fy), 12, (0, 255, 0), -1)
                        cv2.circle(frame, (fx, fy), 15, (0, 255, 0), 3)
                    else:
                        # Yellow when not over a key
                        cv2.circle(frame, (fx, fy), 10, (0, 255, 255), -1)
                    
                    self.process_frame(hand_idx, tip_id, fx, fy, w)

            self.draw_keyboard(frame)

            cv2.imshow('AR Keyboard', frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('d'):
                self.debug_mode = not self.debug_mode
                print(f"Debug mode: {'ON' if self.debug_mode else 'OFF'}")

        cap.release()
        cv2.destroyAllWindows()
        self.hands.close()


if __name__ == "__main__":
    try:
        keyboard = ARKeyboard(debug_mode=True)
        keyboard.run()
    except KeyboardInterrupt:
        print("\nExited by user.")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()