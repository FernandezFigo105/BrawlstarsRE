import subprocess
import gymnasium as gym
import numpy as np
import logging
import cv2
from collections import deque
import time
from ultralytics import YOLO


yolo_model = YOLO("C:/Users/Figo/Desktop/BrawlStars/runs/detect/train13/weights/best.onnx")

class BrawlStarsEnv (gym.Env):
    
    CLASS_NAMES = ["player", "enemy", "power_box", "damage", "damage_taken",
                "Defeat", "Kill", "Play", "Play_again", "Exit", "Proceed", "Loading",
                "grass", "cloud_region", "projectile", "super", "gadget", "power_up",
                "obstacle", "jump_pad", "Victory", ]
    def __init__(self):
        super().__init__()
        # initalize logger and set to a path
        timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        # logging.basicConfig(filename=f"./debugLogs/{timestamp}episode.log"  ,encoding="utf_8",level=logging.INFO)
        logging.info("Initializing Brawl Stars environment.")


        # allowed action to the agent
        self.action_space = gym.spaces.Box(low=np.array([-1, -1, 0], dtype=np.float32),
                                           high=np.array([1, 1, 1], dtype=np.float32))

        # environment definition
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(3,640,640), dtype=np.uint8)
            
        # initalizing member variables
        self.player_pos = None
        self.player_position_history = deque(maxlen=50)
        self.target_pos = None
        self.target_type = None
        self.episode_count = 0
        self.cumulative_reward = 0
        self.move_x_prev = 0
        self.move_y_prev = 0
        self.prev_distance = None
        self.victory_defeat_counter = 0

        # Initialize video capture
        try:
            self.cap = cv2.VideoCapture(1)  # Adjust camera index if needed
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

            # Verify camera is opened correctly
            if not self.cap.isOpened():
                raise Exception("Failed to open camera")

            # Test capture
            ret, _ = self.cap.read()
            if not ret:
                raise Exception("Failed to capture frame from camera")

        except Exception as e:
            logging.error(f"Camera initialization failed: {e}")
            raise
        
    def _get_resized_frame(self):
        logging.debug("Capturing frame...")
        """Capture the current frame from the game."""
        # try 3 times to capture a frame else return nothing
        for _ in range(3):  
            ret, frame = self.cap.read()
            if ret:
                resized_frame = cv2.resize(frame, (640,640))
                normalized_frame = resized_frame.astype(np.float32) / 255.0
                # converting to chw format for opencv to process this frame
                chwImage = np.transpose(normalized_frame, (2, 0, 1))
                self.render()
                return chwImage
            time.sleep(0.05)

        logging.warning("Failed to capture frame after multiple attempts.")
        return None
    
    def _detect_objects(self, frame):
        logging.debug("Running object detection...")
        """Detect objects using the YOLO model and filter by explicitly defined classes."""
        if frame is None:
            return []

        # Convert frame to HWC format and scale pixel values
        frame_hwc = np.transpose(frame, (1, 2, 0)) * 255.0
        frame_hwc = frame_hwc.astype(np.uint8)

        # Run inference using the YOLO model
        results = yolo_model(frame_hwc)

        # Initialize a list to store filtered detections
        detected = []

        victory_detected = False
        defeat_detected = False

        # Process each detection
        for result in results:
            """Extracts boxes field from each result"""
            boxes = result.boxes
            if len(boxes) == 0:
                continue

            try:
                xyxy_boxes = [box.xyxy[0].tolist() for box in boxes]
                confidences = [float(box.conf[0].item()) for box in boxes]

                # Skip if no boxes
                if not xyxy_boxes:
                    continue

                keep_indices = cv2.dnn.NMSBoxes(
                    xyxy_boxes,
                    confidences,
                    score_threshold=0.3,
                    nms_threshold=0.7
                )

                for idx in keep_indices:
                    box = boxes[idx]
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cls = int(box.cls[0].item())
                    class_name = yolo_model.names[cls]  # Get the class name
                    confidence = float(box.conf[0].item())

                    # Filter detections based on explicitly defined CLASS_NAMES (case-insensitive)
                    if class_name in self.CLASS_NAMES:
                        center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
                        detected.append((class_name, center_x, center_y, confidence))

                        # Automatically tap Play button if detected with high confidence
                    if class_name == "Play" and confidence >= 0.3:
                        self.adb_tap(1500, 1000)

                    elif class_name == "power_box" and confidence >= 0.3:
                        if self.player_pos:
                            distance = np.sqrt(
                                (center_x - self.player_pos[0]) ** 2 + (center_y - self.player_pos[1]) ** 2)
                            if distance < 100:  # Adjust this threshold based on gameplay
                                logging.info(
                                    "Power box detected and player is close enough. Performing attack.")
                                self.adb_tap(1500, 1000)

                    elif class_name == "Defeat" and not defeat_detected and confidence >= 0.3:
                        defeat_detected = True
                        detected.append(
                            class_name, center_x, center_y, confidence)
                        logging.info(
                            "Detected 'Defeat' class. Tapping below it at hardcoded position (1000, 1000).")
                        self.adb_tap(1000, 1000)
                    elif class_name == "Victory" and not victory_detected and confidence >= 0.3:
                        victory_detected = True
                        detected.append(
                            class_name, center_x, center_y, confidence)
                        logging.info(
                            "Detected 'Victory' class. Tapping below it at hardcoded position (1000, 1000).")

                    elif class_name == "Proceed" and confidence >= 0.3:
                        proceed_button_x, proceed_button_y = center_x, center_y
                        logging.info(
                            f"Detected 'Proceed' button at position: ({proceed_button_x},{proceed_button_y})")
                        self.adb_tap(1000, 1300)

                    elif class_name == "Play_again" and confidence >= 0.5:
                        play_again_button_x, play_again_button_y = center_x, center_y
                        logging.info(
                            f"Detected 'Play_again' button at position: ({play_again_button_x}, {play_again_button_y})")
                        self.adb_tap(1000, 1500)
                        break
            except Exception as e:
                logging.error(f"Error during object detection: {e}")

        return detected
    
    def adb_tap(self, x, y):
            logging.debug(f"Simulating tap at ({x}, {y}) using ADB...")
            """Simulate a tap action using ADB."""
            try:
                subprocess.run(
                    ["adb", "shell", "input", "tap", str(x), str(y)],
                    check=True,
                    timeout=5
                )
                logging.info(f"Tapped at ({x}, {y})")
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
                logging.error(f"ADB tap failed: {e}")
                
    # def _get_state(self):
    #     logging.debug("Capturing frame...")
    #     """Capture the current frame from the game."""
    #     for _ in range(3):  # Retry up to 3 times
    #         ret, frame = self.cap.read()
    #         if ret:
    #             resized_frame = cv2.resize(frame, (640, 640))
    #             normalized_frame = resized_frame.astype(np.float32) / 255.0
    #             state = np.transpose(normalized_frame, (2, 0, 1))
    #             cv2.imshow("Preprocessed Frame", resized_frame)
    #             cv2.waitKey(1)  # Convert to CHW format
    #             return state
    #         time.sleep(0.05)

        # logging.warning("Failed to capture frame after multiple attempts.")
        # return None

    def _perform_action(self,action):
        logging.debug(f"Performing action: {action}")
        print("perform action not implemented!")
        
        #Heres the code to perform tap action on screen using adb
        """
        Perform an attack action by spamming the attack button whenever an enemy or power box is detected.
        """
        logging.debug("Starting attack loop...")
        max_attempts = 20  # Maximum number of attempts to prevent infinite loops
        attempt = 0
        while attempt < max_attempts:
            frame = self._get_resized_frame()
            if frame is None:
                logging.warning("No frame captured. Skipping attack.")
                break
            detected_objects = self._detect_objects(frame)
            target_found = False
            for obj in detected_objects:
                class_name, center_x, center_y, confidence = obj
                # Confidence threshold
                if class_name in ["enemy", "power_box"] and confidence >= 0.3:
                    target_found = True
                    try:
                        # Tap at a hardcoded position (1500, 1000)
                        self.adb_tap(1500, 1000)
                        logging.info(
                            f"Attacking {class_name} at ({1500}, {1000})")
                    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
                        logging.error(f"ADB tap failed during attack: {e}")
                    break  # Exit the loop once the target is attacked
            if not target_found:
                logging.info(
                    "Target destroyed or no longer detected. Stopping attack.")
                break
            attempt += 1
            # Short delay between attacks to avoid spamming too quickly
            time.sleep(0.2)
        if attempt == max_attempts:
            logging.warning(
                "Reached maximum attack attempts. Stopping attack loop.")
        
        pass

    def step(self, action):
        # perform action
        move_x,move_y, attack = action
        move_x=np.clip(move_x,-1,1) # clip movement to valid range
        move_y=np.clip(move_y,-1,1) # clip movement to valid range
        # Threshold attack to binary (True or False)
        attack = float(attack) > 0.5
        if attack :
            self._perform_action(action)
            
        

        # get next frame after action
        state = self._get_resized_frame()
        if state is None:
            logging.warning("Frame capture failed during episode.")
            # Return neutral step result with minimal penalty
            return state, -1, False, False, {}
        detected_objects = self._detect_objects(state)
        
        self._smooth_move_player(move_x, move_y)

        # calculate reward
        reward = self._calculate_reward(detected_objects, attack)

        # check if episode is over 
        terminated = self._check_termination(detected_objects)
        return state, reward, terminated, False, {}



        # logging.debug(f"Performing action: {action}")
        # """Perform a single step in the environment."""
        # move_x, move_y, attack = action
        # move_x = np.clip(move_x, -1, 1)  # Clip movement to valid range
        # move_y = np.clip(move_y, -1, 1)
        # # Threshold attack to binary (True or False)
        # attack = float(attack) > 0.5

        # # Capture the current state
        # state = self._get_resized_frame()
        # if state is None:
        #     logging.warning("Frame capture failed during episode.")
        #     # Return neutral step result with minimal penalty
        #     return state, -1, False, False, {}

        # # Detect objects in the current frame
        # detected_objects = self._detect_objects(state)
        # self._update_positions(detected_objects)

        # # Execute movement
        # self._smooth_move_player(move_x, move_y)

        # # Execute attack if needed
        # if attack and self.target_pos:
        #     self._perform_attack()

        # # Get updated state after action
        # new_state = self._get_resized_frame()
        # if new_state is None:
        #     new_state = state  # If capture fails, use previous state

        # # Get updated detections
        # new_detected_objects = self._detect_objects(new_state)
        # self._update_positions(new_detected_objects)

        # # Calculate reward
        # reward = self._calculate_reward(detected_objects, attack)
        # self.cumulative_reward += reward
        # logging.info(
        #     f"Step reward: {reward:.2f}, Cumulative reward: {self.cumulative_reward:.2f}")

        # # Check termination conditions
        # terminated = self._check_termination(detected_objects)
        # truncated = False  # Not used in this environment

        # return new_state, reward, terminated, truncated, {}
        
    def adb_move_player(self, dx, dy):
        logging.debug(f"Simulating player movement using ADB swipe...")
        """Simulate moving the player using ADB swipe."""
        max_retries = 3
        swipe_duration = 2000  # Duration of swipe in milliseconds

        # Center of the screen
        start_x, start_y = 300, 600   # Midpoint of 1920x1080

        # Calculate end coordinates
        # Scale to actual screen dimensions
        end_x = start_x + int(dx * (1920 / 448))
        end_y = start_y + int(dy * (1080 / 448))

        # Ensure coordinates are within screen bounds
        end_x = max(0, min(end_x, 1920))
        end_y = max(0, min(end_y, 1080))

        for attempt in range(max_retries):
            try:
                subprocess.run(
                    ["adb", "shell", "input", "swipe",
                     str(start_x), str(start_y),
                     str(end_x), str(end_y),
                     str(swipe_duration)],
                    check=True,
                    timeout=swipe_duration/1000 + 5  # Add 5 seconds buffer
                )
                logging.debug(
                    f"ADB swipe from ({start_x}, {start_y}) to ({end_x}, {end_y})")
                return True
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
                logging.error(f"ADB move attempt {attempt + 1} failed: {e}")
                time.sleep(1)  # Wait before retrying

        return False
        
    def _smooth_move_player(self, move_x, move_y):
        logging.debug("Smoothing player movement...")
        """Smoothly move the player toward the target."""
        if self.target_pos is None or self.player_pos is None:
            return  # Can't move if we don't know positions

        # Calculate actual movement
        dx = self.target_pos[0] - self.player_pos[0]
        dy = self.target_pos[1] - self.player_pos[1]
        distance = np.sqrt(dx**2 + dy**2)

        # Normalize direction vector
        if distance > 0:
            dx /= distance
            dy /= distance

        # Blend with previous movement (smoothing factor)
        alpha = 0.2
        move_x = alpha * dx + (1 - alpha) * self.move_x_prev
        move_y = alpha * dy + (1 - alpha) * self.move_y_prev

        # Update previous movement
        self.move_x_prev = move_x
        self.move_y_prev = move_y

        # Move toward the target
        # Move toward the target
        if distance < 150:  # If the target is too close, perform attack
            logging.info(
                f"Target is too close ({distance:.2f}). Performing attack action.")
            subprocess.run(["adb", "shell", "input", "tap",
                           str(1000), str(1000)], check=True)
        else:  # Otherwise, move toward the target with a swipe
            start_x, start_y = 100, 300  # Center of the screen
            end_x = int(start_x + move_x * 500)  # Scale movement
            end_y = int(start_y + move_y * 500)
            logging.info(
                f"Moving toward target with swipe: ({start_x}, {start_y}) -> ({end_x}, {end_y})")
            self.adb_move_player(move_x * 700, move_y * 700)
    
    def _check_termination(self, detected_objects):
        logging.debug("Checking termination conditions...")
        """Check if the episode should terminate."""
        # Check for Victory or Defeat
        for obj in detected_objects:
            class_name = obj[0]

            if class_name in ["Victory", "Defeat"]:
                logging.info(f"{class_name} detected. Ending episode.")
                return True

        # Reset counter if no victory/defeat detected
        self.victory_defeat_counter = 0

        # Check for Exit button
        for obj in detected_objects:
            class_name, _, _, _ = obj
        if self.player_pos and (
            self.player_pos[0] < 20 or self.player_pos[0] > 204 or
            self.player_pos[1] < 20 or self.player_pos[1] > 204
        ):
            logging.info("Episode ended: Player near edge.")
            return True

        return False
    
    def _calculate_reward(self, detected_objects, attack):
        logging.debug("Calculating reward...")
        reward = 0

        # Survival bonus (small positive reward for staying alive)
        reward += 0.1

        # Distance-based reward (toward target)
        if self.target_pos and self.player_pos:
            distance_to_target = np.linalg.norm(
                np.array(self.target_pos) - np.array(self.player_pos), ord=2)

            # Store previous distance for progress tracking
            if self.prev_distance is not None:
                # Reward for moving closer to target
                distance_change = self.prev_distance - distance_to_target
                reward += distance_change * 0.5

            # Update previous distance
            self.prev_distance = distance_to_target

            # Base distance reward (inverse distance to encourage moving closer)
            reward += 1 / (distance_to_target + 1)

        # Event-based penalties
        for obj in detected_objects:
            class_name, _, _, _ = obj
            if class_name == "damage_taken":
                logging.info(f"Damage taken: -50")
                reward -= 50
            elif class_name == "Defeat":
                logging.info("Defeat detected: large negative reward (-500)")
                reward -= 500

        # Event rewards
        for obj in detected_objects:
            class_name = obj[0]
            if class_name == "Kill":
                reward += 200
                logging.info("Kill detected: large positive reward (+200)")
            elif class_name == "damage":
                reward += 50
            elif class_name == "Victory":
                reward += 500
                logging.info("Victory detected: large positive reward (+500)")
            elif class_name == "power_up":
                reward += 150
                logging.info(
                    "Power-up collected: large positive reward (+150)")

        # Attack logic - only reward if enemy is in range
        if attack and self.target_type == "enemy" and self.target_pos and self.player_pos:
            distance_to_target = np.linalg.norm(
                np.array(self.target_pos) - np.array(self.player_pos), ord=2)
            if distance_to_target < 120:  # Adjust this threshold based on gameplay
                reward += 100  # Reward for attacking enemies when close
            else:
                reward -= 10  # Small penalty for attacking when enemy is too far

        # Inactivity penalty
        if len(self.player_position_history) >= 10:  # Check last 10 positions
            # Calculate variance in x and y coordinates
            x_positions = [pos[0] for pos in self.player_position_history]
            y_positions = [pos[1] for pos in self.player_position_history]

            x_variance = np.var(x_positions)
            y_variance = np.var(y_positions)

            movement_threshold = 100  # Minimum variance to consider movement meaningful
            if x_variance < movement_threshold and y_variance < movement_threshold:
                logging.info(
                    "Bot is not moving enough. Applying inactivity penalty.")
                reward -= 10  # Penalize for staying in one place

        # Log total reward for debugging
        logging.debug(f"Total reward for this step: {reward:.2f}")
        return reward
    
    
    def _update_positions(self, detected_objects):
        logging.debug("Updating player and target positions...")
        """Update player and target positions based on detected objects."""
        self.player_pos = None
        self.target_pos = None
        self.target_type = None
        min_distance = float("inf")

        for obj in detected_objects:
            class_name, x, y, _ = obj
            if class_name == "player":
                self.player_pos = (x, y)
                logging.info(f"Player detected at position: {self.player_pos}")
            elif class_name in ["enemy", "power_box"]:
                # Only calculate distance if player position is known
                if self.player_pos:
                    distance = np.sqrt(
                        (x - self.player_pos[0]) ** 2 + (y - self.player_pos[1]) ** 2)
                    if distance < min_distance:
                        min_distance = distance
                        self.target_pos = (x, y)
                        self.target_type = class_name
                        logging.debug(
                            f"New target: {self.target_type} at {self.target_pos}, distance: {distance:.2f}")

            for obj in detected_objects:
                class_name, x, y, _ = obj
                if class_name == "Defeat":
                    self.adb_tap(1000, 1000)
                    """To terminate the game immediately once the defeated class is detected."""
                    self._check_termination(detected_objects)
                    logging.info(f"Tapping at the the exit button")

            for obj in detected_objects:
                class_name, x, y, _ = obj
                if class_name == "Victory":
                    self._check_termination(detected_objects)
                    logging.info(
                        f"Victory button detcted terminating the episode")


    def reset(self, seed=None, options=None):
        logging.debug("Resetting environment...")
        """
        Reset the environment for a new episode.
        """
        super().reset(seed=seed, options=options)

        # Reset variables for the new episode
        self.cumulative_reward = 0
        self.player_pos = None
        self.target_pos = None
        self.target_type = None
        self.prev_distance = None
        self.move_x_prev = 0
        self.move_y_prev = 0
        self.victory_defeat_counter = 0
        self.episode_count += 1

        logging.info(f"Starting episode {self.episode_count}")

        # Check for common buttons
        state = self._get_resized_frame()
        if state is None:
            logging.error("Failed to get initial state during reset.")
            # Fallback to empty state
            state = np.zeros((3, 640,640), dtype=np.float32)
            return state, {}

        # Wait for player detection to confirm the episode has started
        logging.info("Waiting for player detection...")
        start_time = time.time()
        timeout = 60  # 60 seconds timeout
        while (time.time() - start_time) < timeout:
            state = self._get_resized_frame()
            if state is None:
                time.sleep(0.5)
                continue

            detected_objects = self._detect_objects(state)
            self._update_positions(detected_objects)

            for obj in detected_objects:
                class_name, _, _, _ = obj
                if class_name in ["Victory", "Defeat"]:
                    logging.info(f"{class_name} detected. Ending episode.")
                    return state, {}

            if self.player_pos is not None:
                logging.info(
                    f"Player detected at {self.player_pos}. Starting episode.")
                break

            time.sleep(0.5)  # Prevent busy waiting

        if self.player_pos is None:
            logging.warning(
                "Player not detected within timeout. Proceeding with episode start.")

        return state, {}

    def render(chwImage):
        # cv2.imshow("Preprocessed Frame", chwImage) # tell opencv to display the image
        # cv2.waitKey(1)
        pass  

    def close(self):
        logging.info("Closing environment...")
        """Release resources when the environment is closed."""
        if hasattr(self, 'cap') and self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()
        logging.info("Environment resources released.")
