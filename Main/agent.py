import time
from tkinter import CENTER
import cv2
import numpy as np
from ultralytics import YOLO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import PPO
import subprocess
import gymnasium as gym
import logging
from collections import deque
import os

# =========================
# Logging
# =========================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# =========================
# YOLO model
# =========================
# NOTE: make sure this path is correct on your machine.
yolo_model = YOLO("C:/Users/Figo/Desktop/BrawlStars/runs/detect/train13/weights/best.pt")
yolo_model.to('cuda')


class BrawlStarsEnv(gym.Env):
    """
    Custom Gymnasium environment for Brawl Stars controlled via ADB + YOLO detections.
    Observation: RGB image resized to (3, 448, 448) uint8 (channel-first).
    Action: [move_x, move_y, attack] -> in range [-1..1], [-1..1], [0..1].
    """
    metadata = {"render_modes": []}

    CLASS_NAMES = [
        "player", "enemy", "power_box", "damage", "damage_taken", "Defeat", "Kill",
        "Play", "Play_again", "Exit", "Proceed", "Loading", "grass", "cloud_region",
        "projectile", "super", "gadget", "power_up", "obstacle", "jump_pad", "Victory",
    ]

    def __init__(self):
        super().__init__()

        logging.info("Initializing Brawl Stars environment.")

        # Action space: [move_x, move_y, attack]
        self.action_space = gym.spaces.Box(
            low=np.array([-1.0, -1.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        # Observation space: (C, H, W) with channel-first for SB3
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(3, 448, 448),
            dtype=np.uint8,
        )

        # Runtime state
        self.width: int = 1920
        self.screen_width: int = 1920
        self.screen_height: int = 1080
        self.height: int = 1080
        self.player_pos: tuple | None = None
        self.player_position_history = deque(maxlen=50)
        self.target_pos: tuple | None = None
        self.target_type: str | None = None
        self.enemy_pos: tuple | None = None
        self.powerbox_pos: tuple | None = None

        self.episode_count = 0
        self.cumulative_reward = 0.0
        self.move_x_prev = 0.0
        self.move_y_prev = 0.0
        self.prev_distance: float | None = None

        # Video capture
        try:
            self.cap = cv2.VideoCapture(1)  # adjust index if needed
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

            if not self.cap.isOpened():
                raise RuntimeError("Failed to open camera")

            ok, _ = self.cap.read()
            if not ok:
                raise RuntimeError("Failed to capture frame from camera")
        except Exception as e:
            logging.error(f"Camera initialization failed: {e}")
            raise

    # --------------- Utils ---------------
    
    def normalize_and_tap(self, x, y, frame_width=448, frame_height=448, screen_width=1920, screen_height=1080):
        """
        Normalize detection coordinates (from resized frame) to actual phone screen
        and perform adb tap.
        """
        norm_x = int((x / frame_width) * screen_width)
        norm_y = int((y / frame_height) * screen_height)
        self.adb_tap(norm_x, norm_y)
    
    
    def _safe_zero_obs(self) -> np.ndarray:
        """A valid dummy observation when frame capture fails."""
        return np.zeros((3, 448, 448), dtype=np.uint8)

    def _get_state(self) -> np.ndarray | None:
        """Capture and preprocess current frame (returns CHW or None)."""
        for _ in range(3):  # retry a few times
            ok, frame = self.cap.read()
            if ok and frame is not None:
                try:
                    resized = cv2.resize(frame, (448, 448), interpolation=cv2.INTER_LINEAR)
                    # HWC -> CHW
                    state = np.transpose(resized, (2, 0, 1)).astype(np.uint8)
                    # Optional preview (disable if headless)
                    cv2.imshow("Preprocessed Frame", resized)
                    cv2.waitKey(1)
                    return state
                except Exception as e:
                    logging.warning(f"Preprocess failed, retrying: {e}")
            time.sleep(0.03)
        logging.warning("Failed to capture frame after multiple attempts.")
        return None

    def _detect_objects(self, frame_chw: np.ndarray | None):
        """
        Run YOLO on CHW frame, return list of tuples:
        (class_name, center_x, center_y, confidence)
        """
        if frame_chw is None:
            return []

        # CHW -> HWC (uint8)
        frame_hwc = np.transpose(frame_chw, (1, 2, 0)).astype(np.uint8)

        try:
            results = yolo_model(frame_hwc)
        except Exception as e:
            logging.error(f"YOLO inference failed: {e}")
            return []

        detected = []

        # flags to avoid repeated taps
        defeat_detected = False
        victory_detected = False

        for res in results:
            boxes = getattr(res, "boxes", None)
            if boxes is None or len(boxes) == 0:
                continue

            try:
                xyxy_boxes = [box.xyxy[0].tolist() for box in boxes]
                confidences = [float(box.conf[0]) for box in boxes]
                # cv2.dnn.NMSBoxes expects [x, y, w, h]
                xywh_boxes = [[x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)]
                              for (x1, y1, x2, y2) in xyxy_boxes]

                if len(xywh_boxes) == 0:
                    continue

                indices = cv2.dnn.NMSBoxes(
                    bboxes=xywh_boxes,
                    scores=confidences,
                    score_threshold=0.30,
                    nms_threshold=0.70
                )

                # Normalize indices to a flat list
                if indices is None or len(indices) == 0:
                    keep_ids = []
                else:
                    # OpenCV can return Nx1 or a Python list
                    keep_ids = np.array(indices).flatten().tolist()

                for idx in keep_ids:
                    box = boxes[idx]
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cls = int(box.cls[0])
                    confidence = float(box.conf[0])
                    class_name = yolo_model.names.get(cls, str(cls)) if hasattr(yolo_model, "names") else str(cls)

                    if class_name not in self.CLASS_NAMES:
                        continue

                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    detected.append((class_name, cx, cy, confidence))
                    img_w, img_h = 640, 640      # YOLO input resolution
                    screen_w, screen_h = 1920, 1080

                    # Side-effect taps for UI classes (guards included)
                    if class_name == "Play" and confidence >= 0.30:
                        self.normalize_and_tap(cx, cy)

                    elif class_name == "power_box" and confidence >= 0.30:
                        if self.player_pos is not None:
                            dist = float(np.hypot(cx - self.player_pos[0], cy - self.player_pos[1]))
                            if dist < 100.0:
                                logging.info(f"Power box close at ({cx},{cy}). Attacking center.")
                                self.normalize_and_tap(cx, cy)

                    elif class_name == "Defeat" and not defeat_detected and confidence >= 0.30:
                        defeat_detected = True
                        logging.info(f"Detected 'Defeat' at ({cx},{cy}). Tapping center.")
                        self.normalize_and_tap(1500, 1000)

                    elif class_name == "Victory" and not victory_detected and confidence >= 0.30:
                        victory_detected = True
                        logging.info(f"Detected 'Victory' at ({cx},{cy}). Tapping center.")
                        self.normalize_and_tap(cx, cy)

                    elif class_name == "Proceed" and confidence >= 0.30:
                        
                        logging.info(f"Detected 'Proceed' at ({cx},{cy}). Tapping normalized center.")
                        self.normalize_and_tap(cx, cy)

                    elif class_name == "Play_again" and confidence >= 0.50:
                        
                        logging.info(f"Detected 'Play_again' at ({cx},{cy}). Tapping normalized center.")
                        self.normalize_and_tap(cx, cy)


            except Exception as e:
                logging.error(f"Error during object parsing/NMS: {e}")
                continue

        return detected

    def _update_positions(self, detected_objects):
        """Update tracked positions and choose current target."""
        self.player_pos = None
        self.enemy_pos = None
        self.powerbox_pos = None
        self.target_pos = None
        self.target_type = None

        for class_name, x, y, _ in detected_objects:
            if class_name == "player":
                self.player_pos = (int(x), int(y))
            elif class_name == "enemy":
                self.enemy_pos = (int(x), int(y))
            elif class_name == "power_box":
                self.powerbox_pos = (int(x), int(y))

        if self.player_pos is not None:
            enemy_distance = float("inf")
            powerbox_distance = float("inf")
            if self.enemy_pos is not None:
                enemy_distance = float(np.linalg.norm(np.array(self.enemy_pos) - np.array(self.player_pos)))
            if self.powerbox_pos is not None:
                powerbox_distance = float(np.linalg.norm(np.array(self.powerbox_pos) - np.array(self.player_pos)))
            self._prioritize_target(enemy_distance, powerbox_distance)

        # track movement for inactivity penalty
        if self.player_pos is not None:
            self.player_position_history.append(self.player_pos)
    
    def tap_object(self, cx, cy, input_w=640, input_h=640, screen_w=1920, screen_h=1080):
        """
        Normalize YOLO detection coordinates to screen resolution and tap.
        cx, cy -> detection center coordinates from YOLO
        input_w, input_h -> YOLO model input size (default 640x640)
        screen_w, screen_h -> phone screen resolution
        """
        # Normalize to screen size
        screen_x = int(cx * screen_w / input_w)
        screen_y = int(cy * screen_h / input_h)

        logging.info(f"Tapping normalized center at ({screen_x}, {screen_y})")
        self.adb_tap(screen_x, screen_y)
    
    def tap(self, x, y):
        """Tap at raw screen coordinates"""
        cmd = f"adb shell input tap {int(x)} {int(y)}"
        os.system(cmd)
        
    def normalized_tap(self, x_norm, y_norm):
        """Tap at normalized coordinates (0–1 range)"""
        x = int(x_norm * self.screen_width)
        y = int(y_norm * self.screen_height)
        self.tap(x, y)
    
    def adb_tap(self, x: int, y: int):
        """Tap at absolute screen coordinates via ADB."""
        try:
            subprocess.run(
                ["adb", "shell", "input", "tap", str(int(x)), str(int(y))],
                check=True,
                timeout=5
            )
            logging.info(f"Tapped at ({x}, {y})")
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            logging.error(f"ADB tap failed: {e}")

    def adb_move_player(self, dx: float, dy: float) -> bool:
        """Swipe from fixed origin toward (dx,dy) direction (scaled) via ADB."""
        start_x, start_y = 300, 600
        end_x = start_x + int(dx * (1920 / 448))
        end_y = start_y + int(dy * (1080 / 448))
        end_x = max(0, min(end_x, 1920))
        end_y = max(0, min(end_y, 1080))

        for attempt in range(3):
            try:
                subprocess.run(
                    ["adb", "shell", "input", "swipe",
                     str(start_x), str(start_y), str(end_x), str(end_y), "2000"],
                    check=True,
                    timeout=7  # 2s swipe + buffer
                )
                logging.debug(f"Swipe: ({start_x},{start_y}) -> ({end_x},{end_y})")
                return True
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
                logging.error(f"Swipe attempt {attempt + 1} failed: {e}")
                time.sleep(0.5)
        return False

    # --------------- RL helpers ---------------

    def _calculate_reward(self, detected_objects, attack_bool: bool) -> float:
        reward = 0.1  # small survival bonus

        # Distance-based shaping
        if self.target_pos is not None and self.player_pos is not None:
            distance_to_target = float(np.linalg.norm(np.array(self.target_pos) - np.array(self.player_pos)))
            if self.prev_distance is not None:
                reward += (self.prev_distance - distance_to_target) * 0.5
            self.prev_distance = distance_to_target
            reward += 1.0 / (distance_to_target + 1.0)

        # Penalties and rewards from events
        for class_name, _, _, _ in detected_objects:
            if class_name == "damage_taken":
                reward -= 50.0
                logging.info("Damage taken: -50")
            elif class_name == "Defeat":
                reward -= 500.0
                logging.info("Defeat: -500")
            elif class_name == "Kill":
                reward += 200.0
                logging.info("Kill: +200")
            elif class_name == "damage":
                reward += 50.0
            elif class_name == "Victory":
                reward += 500.0
                logging.info("Victory: +500")
            elif class_name == "power_up":
                reward += 150.0
                logging.info("Power-up: +150")

        # Attack efficiency: reward if enemy close when attacking
        if (attack_bool and
                self.target_type == "enemy" and
                self.target_pos is not None and
                self.player_pos is not None):
            dist = float(np.linalg.norm(np.array(self.target_pos) - np.array(self.player_pos)))
            reward += 100.0 if dist < 120.0 else -10.0

        # Inactivity penalty
        if len(self.player_position_history) >= 10:
            xs = np.array([p[0] for p in self.player_position_history], dtype=np.float32)
            ys = np.array([p[1] for p in self.player_position_history], dtype=np.float32)
            if float(np.var(xs)) < 100.0 and float(np.var(ys)) < 100.0:
                reward -= 10.0
                logging.info("Inactivity penalty: -10")

        return float(reward)

    def _perform_attack(self):
        logging.debug("Starting attack loop...")
        max_attempts = 20

        for attempt in range(max_attempts):
            frame = self._get_state()
            if frame is None:
                logging.warning("No frame captured. Skipping attack.")
                break

            detected_objects = self._detect_objects(frame)

            # find closest enemy/power_box
            target = None
            min_d = float("inf")

            if self.player_pos is None:
                logging.warning("Player position not set, skipping attack...")
                break

            px, py = self.player_pos  # player center
            for cls, cx, cy, conf in detected_objects:
                if cls.lower() in ("enemy", "power_box") and conf >= 0.30:
                    d = np.hypot(cx - px, cy - py)  # Euclidean distance
                    if d < min_d:
                        min_d = d
                        target = (cx, cy, cls)

            if target is None:
                logging.info("No valid target found; stopping attack loop.")
                break

            tx, ty, cls = target
            logging.info(f"Target close ({min_d:.2f}). Attacking {cls} at ({tx},{ty})")
            self.normalized_tap(tx, ty)

            time.sleep(0.2)


    def _smooth_move_player(self, move_x: float, move_y: float):
        """Blend movement toward target and issue swipe or close-range tap."""
        if self.target_pos is None or self.player_pos is None:
            return

        dx = float(self.target_pos[0] - self.player_pos[0])
        dy = float(self.target_pos[1] - self.player_pos[1])
        dist = float(np.hypot(dx, dy))

        if dist > 0.0:
            dx /= dist
            dy /= dist

        alpha = 0.2
        move_x = alpha * dx + (1.0 - alpha) * self.move_x_prev
        move_y = alpha * dy + (1.0 - alpha) * self.move_y_prev
        self.move_x_prev = move_x
        self.move_y_prev = move_y

        if dist < 150.0:
            logging.info(f"Target close ({dist:.2f}). Attacking.")
            self.adb_tap(1000, 1000)
        else:
            self.adb_move_player(move_x * 700.0, move_y * 700.0)

    def _check_termination(self, detected_objects) -> bool:
        for class_name, _, _, _ in detected_objects:
            if class_name in ("Victory", "Defeat"):
                logging.info(f"{class_name} detected. Ending episode.")
                return True
        return False

    def _prioritize_target(self, enemy_distance: float, powerbox_distance: float):
        """Prefer power boxes if any; else enemy; else none."""
        if np.isfinite(powerbox_distance):
            self.target_pos = self.powerbox_pos
            self.target_type = "power_box"
        elif np.isfinite(enemy_distance):
            self.target_pos = self.enemy_pos
            self.target_type = "enemy"
        else:
            self.target_pos = None
            self.target_type = None

    # --------------- Gym API ---------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)

        self.cumulative_reward = 0.0
        self.player_pos = None
        self.enemy_pos = None
        self.powerbox_pos = None
        self.target_pos = None
        self.target_type = None
        self.prev_distance = None
        self.move_x_prev = 0.0
        self.move_y_prev = 0.0
        self.player_position_history.clear()

        self.episode_count += 1
        logging.info(f"Starting episode {self.episode_count}")

        state = self._get_state()
        if state is None:
            logging.error("Initial frame missing; returning zero observation.")
            return self._safe_zero_obs(), {}

        # Wait up to 60s for player detection (non-blocking fallback)
        start_time = time.time()
        while (time.time() - start_time) < 60.0:
            frame = self._get_state()
            if frame is None:
                time.sleep(0.2)
                continue
            detected = self._detect_objects(frame)
            self._update_positions(detected)

            if any(cls in ("Victory", "Defeat") for (cls, _, _, _) in detected):
                logging.info("Victory/Defeat on reset; continuing anyway.")
                break

            if self.player_pos is not None:
                logging.info(f"Player detected at {self.player_pos}.")
                state = frame
                break

            time.sleep(0.2)

        if self.player_pos is None:
            logging.warning("Player not detected within timeout. Starting anyway.")

        return state if state is not None else self._safe_zero_obs(), {}

    def step(self, action):
        # Unpack & clip action
        move_x, move_y, attack = map(float, action)
        move_x = float(np.clip(move_x, -1.0, 1.0))
        move_y = float(np.clip(move_y, -1.0, 1.0))
        attack_bool = bool(attack > 0.5)

        # Current state
        state = self._get_state()
        if state is None:
            logging.warning("Frame capture failed in step; returning safe fallback and small penalty.")
            obs = self._safe_zero_obs()
            return obs, -1.0, False, False, {}

        # Detect & update positions
        detections = self._detect_objects(state)
        self._update_positions(detections)

        # Movement & optional attack
        self._smooth_move_player(move_x, move_y)
        if attack_bool and self.target_pos is not None:
            self._perform_attack()

        # New observation after acting
        new_state = self._get_state()
        if new_state is None:
            new_state = state  # keep previous if capture failed

        new_detections = self._detect_objects(new_state)
        self._update_positions(new_detections)

        # Reward & termination
        reward = self._calculate_reward(detections, attack_bool)
        self.cumulative_reward += reward
        logging.info(f"Step reward: {reward:.2f}, Cumulative: {self.cumulative_reward:.2f}")

        terminated = self._check_termination(new_detections)
        truncated = False

        return new_state, float(reward), bool(terminated), bool(truncated), {}

    def close(self):
        logging.info("Closing environment...")
        try:
            if hasattr(self, "cap") and self.cap is not None:
                self.cap.release()
            cv2.destroyAllWindows()
        finally:
            logging.info("Environment resources released.")


# =========================
# MAIN TRAINING LOOP
# =========================
if __name__ == "__main__":
    try:
        env = DummyVecEnv([lambda: BrawlStarsEnv()])
        ppo_model = PPO(
            "CnnPolicy",
            env,
            verbose=1,
            device="cuda",  # change to "auto" if you want it to fall back to CPU automatically
            policy_kwargs={"normalize_images": False},
            n_steps=512,
            learning_rate=1e-4,
            batch_size=64,
            gamma=0.99,
            tensorboard_log="./brawlstars_tensorboard/",
        )
        logging.info("Starting training...")
        ppo_model.learn(total_timesteps=500_000)

    except KeyboardInterrupt:
        logging.info("Training interrupted by user.")

    except Exception as e:
        logging.error(f"Training failed: {e}")

    finally:
        logging.info("Saving model and closing environment...")
        try:
            if "ppo_model" in locals():
                ppo_model.save("ppo_brawlstars_model")
        finally:
            if "env" in locals():
                env.close()
        logging.info("Model saved and environment closed.")
