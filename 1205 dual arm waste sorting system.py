import cv2  
import numpy as np  
import time  
import json  
from pyniryo import *  
import onnxruntime as ort  
from pyniryo.api.enums_communication import ToolID, PinID  
import threading  
import signal
import sys

class DualArmSortingRobot:  
    def __init__(self, robot_ips, onnx_model_path, calibration_configs, workspace_configs):  
        # Setting up signal handling 
        signal.signal(signal.SIGINT, self.signal_handler)
        
        # Robotic Arm Connection  
        print("Connecting to robots...")  
        self.gamma_robot = NiryoRobot(robot_ips[0]) # Gamma robotic arm
        self.alpha_robot = NiryoRobot(robot_ips[1]) # Alpha robotic arm
        
        # robot calibration  
        print("Calibrating Gamma robot...")  
        self.gamma_robot.calibrate_auto()  
        print("Calibrating Alpha robot...")  
        self.alpha_robot.calibrate_auto()  

        # Assign a target type to each robotic arm for picking  
        print("\n" + "="*50)  
        print("=== Object Type Selection ===")  
        self.gamma_target_class = self.select_target_class("Gamma")  
        self.alpha_target_class = self.select_target_class("Alpha")  
        
        # Choose the gripper type for both robot  
        print("\n" + "="*50)  
        print("=== Gripper Selection ===")  
        print(f"\n--- Gamma Robot ({self.gamma_target_class}) Gripper Selection ---")  
        self.gamma_tool = self.select_gripper(self.gamma_robot)  
        print(f"\n--- Alpha Robot ({self.alpha_target_class}) Gripper Selection ---")  
        self.alpha_tool = self.select_gripper(self.alpha_robot)  

        # Load ONNX model  
        print("Loading ONNX model...")  
        self.session = ort.InferenceSession(onnx_model_path, providers=['CPUExecutionProvider'])  
        self.input_name = self.session.get_inputs()[0].name  

        # Model Parameters  
        self.input_size = 640  
        self.conf_thresh = 0.5  
        self.nms_thresh = 0.45  

        # Waste sorting categories  
        self.class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic']  

        # === Load calibration matrics for both arms ===  
        print("Loading calibration matrices for both arms...")  
        # Gamma calibration matrics
        self.gamma_fixed_transform_matrix = np.load(calibration_configs['gamma_fixed'])
        self.gamma_conveyor_transform_matrix = np.load(calibration_configs['gamma_conveyor'])
        # Alpha calibration matrics
        self.alpha_fixed_transform_matrix = np.load(calibration_configs['alpha_fixed'])
        self.alpha_conveyor_transform_matrix = np.load(calibration_configs['alpha_conveyor'])
        print("Calibration matrices loaded successfully.")  

        # === Load workspace for both arms ===  
        print("Loading workspaces for both arms...")  
        # Gamma workspace
        with open(workspace_configs['gamma_fixed'], "r") as f:  
            gamma_workspace = json.load(f)  
            gamma_x_list = [p["x"] for p in gamma_workspace["points"]]  
            gamma_y_list = [p["y"] for p in gamma_workspace["points"]]  
            self.GAMMA_X_MIN, self.GAMMA_X_MAX = min(gamma_x_list), max(gamma_x_list)  
            self.GAMMA_Y_MIN, self.GAMMA_Y_MAX = min(gamma_y_list), max(gamma_y_list)  
        print(f"Gamma Fixed Workspace X range: {self.GAMMA_X_MIN:.3f} ~ {self.GAMMA_X_MAX:.3f}")  
        print(f"Gamma Fixed Workspace Y range: {self.GAMMA_Y_MIN:.3f} ~ {self.GAMMA_Y_MAX:.3f}")  

        # Alpha workspace
        with open(workspace_configs['alpha_fixed'], "r") as f:  
            alpha_workspace = json.load(f)  
            alpha_x_list = [p["x"] for p in alpha_workspace["points"]]  
            alpha_y_list = [p["y"] for p in alpha_workspace["points"]]  
            self.ALPHA_X_MIN, self.ALPHA_X_MAX = min(alpha_x_list), max(alpha_x_list)  
            self.ALPHA_Y_MIN, self.ALPHA_Y_MAX = min(alpha_y_list), max(alpha_y_list)  
        print(f"Alpha Fixed Workspace X range: {self.ALPHA_X_MIN:.3f} ~ {self.ALPHA_X_MAX:.3f}")  
        print(f"Alpha Fixed Workspace Y range: {self.ALPHA_Y_MIN:.3f} ~ {self.ALPHA_Y_MAX:.3f}")  

        # === Loading conveyor workspaces for both arms === 
        print("Loading conveyor workspaces for both arms...") 
        # Gamma conveyor belt workspace
        with open(workspace_configs['gamma_conveyor'], "r") as f:  
            gamma_conveyor_workspace = json.load(f)  
            gamma_conveyor_x_list = [p["x"] for p in gamma_conveyor_workspace["points"]]  
            gamma_conveyor_y_list = [p["y"] for p in gamma_conveyor_workspace["points"]]  
            self.GAMMA_CONVEYOR_X_MIN, self.GAMMA_CONVEYOR_X_MAX = min(gamma_conveyor_x_list), max(gamma_conveyor_x_list)  
            self.GAMMA_CONVEYOR_Y_MIN, self.GAMMA_CONVEYOR_Y_MAX = min(gamma_conveyor_y_list), max(gamma_conveyor_y_list)  
        print(f"Gamma Conveyor Workspace X range: {self.GAMMA_CONVEYOR_X_MIN:.3f} ~ {self.GAMMA_CONVEYOR_X_MAX:.3f}")  
        print(f"Gamma Conveyor Workspace Y range: {self.GAMMA_CONVEYOR_Y_MIN:.3f} ~ {self.GAMMA_CONVEYOR_Y_MAX:.3f}")  

        # Alpha conveyor belt workspace 
        with open(workspace_configs['alpha_conveyor'], "r") as f:  
            alpha_conveyor_workspace = json.load(f)  
            alpha_conveyor_x_list = [p["x"] for p in alpha_conveyor_workspace["points"]]  
            alpha_conveyor_y_list = [p["y"] for p in alpha_conveyor_workspace["points"]]  
            self.ALPHA_CONVEYOR_X_MIN, self.ALPHA_CONVEYOR_X_MAX = min(alpha_conveyor_x_list), max(alpha_conveyor_x_list)  
            self.ALPHA_CONVEYOR_Y_MIN, self.ALPHA_CONVEYOR_Y_MAX = min(alpha_conveyor_y_list), max(alpha_conveyor_y_list)  
        print(f"Alpha Conveyor Workspace X range: {self.ALPHA_CONVEYOR_X_MIN:.3f} ~ {self.ALPHA_CONVEYOR_X_MAX:.3f}")  
        print(f"Alpha Conveyor Workspace Y range: {self.ALPHA_CONVEYOR_Y_MIN:.3f} ~ {self.ALPHA_CONVEYOR_Y_MAX:.3f}")  

        # === conveyor parameters ===
        print("Setting up conveyor parameters...")
        # The conveyor is connected to the Alpha robotic arm
        self.conveyor_speed = 50  # conveyor belt speed
        self.pixel_to_mm = 0.5000  # Pixel-to-millimeter conversion factor
        
        # === Individual arm delay compensation parameters === 
        print("Setting up delay compensation for both arms...") 
        # Gamma delay compensation 
        self.gamma_total_delay_ms = -670
        self.setup_delay_compensation('gamma') 
        # Alpha delay compensation  
        self.alpha_total_delay_ms = 700 # The delay time can be adjusted based on actual conditions
        self.setup_delay_compensation('alpha') 

        # Height Parameters  
        self.FIXED_Z_PICK = 0.09  
        self.FIXED_Z_APPROACH = 0.15  
        self.FIXED_Z_SAFE = 0.20  
        self.CONVEYOR_Z_PICK = 0.11 
        self.CONVEYOR_Z_APPROACH = 0.18  
        self.CONVEYOR_Z_SAFE = 0.23  

        # Position definition  
        # Bin location - Set different positions based on sorting target type. 
        self.GAMMA_TRASH_BIN_POS = self.get_trash_bin_position('gamma')
        self.ALPHA_TRASH_BIN_POS = self.get_trash_bin_position('alpha')

        # === Fixed workspace observation position === 
        self.GAMMA_FIXED_OBSERVE_POS = [0.191, -0.010, 0.298, -3.127, 1.333, -3.112]
        self.ALPHA_FIXED_OBSERVE_POS = [0.191, -0.010, 0.298, -3.131, 1.339, -3.115]

        # === Dedicated per-arm observation position over the conveyor workspace === 
        # Observation position at the upstream conveyor for the Gamma robotic arm 
        self.GAMMA_CONVEYOR_OBSERVE_POS = [0.251, 0.005, 0.254, 2.978, 1.311, 3.003]
        # Observation position at the downstream conveyor for the Alpha robotic arm 
        self.ALPHA_CONVEYOR_OBSERVE_POS = [0.191, -0.010, 0.298, -3.131, 1.339, -3.115]

        # Safe pose
        self.GAMMA_SAFE_POS = [0.140, -0.000, 0.203, 0.000, 0.753, -0.001]
        self.ALPHA_SAFE_POS = [0.140, -0.000, 0.203, -0.003, 0.757, -0.001]

        # conveyor parameters - connected to alpha arm
        self.conveyor_id = self.alpha_robot.set_conveyor()
        # synchronized control
        self.operation_lock = threading.Lock()
        self.current_working_arm = None # 'gamma' 或 'alpha' 或 None 
        self.auto_detection_running = False
        self.shutdown_flag = False  # Set the shutdown flag
        self.conveyor_running = False  # Monitor Conveyor Status
        self.conveyor_stop_requested = False  # conveyor stop request
        self.pending_pick_operations = 0  #Pending Pick Count
        self.batch_pick_count = 0  #Batch Pick Count
        
        #Display Window Parameters 
        self.gamma_window_name = "Gamma Arm Detection"
        self.alpha_window_name = "Alpha Arm Detection"
        self.window_size = (800, 600)
        
        # Image Processing Parameters
        self.display_size = (640, 480)  # Uniform Display Size
        
        # Create a display window and set its position
        self.setup_windows()
        
        print("Dual Arm System initialization complete")
        print(f"\n=== Configuration Summary ===")
        print(f"Gamma Arm: {self.gamma_target_class}")
        print(f"Alpha Arm: {self.alpha_target_class}")

    def setup_windows(self):
        """Set up and arrange the display windows"""
        # Get Screen Size
        try:
            # Try to get the screen resolution
            import subprocess
            result = subprocess.run(['xrandr'], capture_output=True, text=True)
            lines = result.stdout.split('\n')
            for line in lines:
                if 'current' in line:
                    parts = line.split(',')
                    for part in parts:
                        if 'current' in part:
                            resolution = part.split()[1]
                            screen_width, screen_height = map(int, resolution.split('x'))
                            break
                    break
            else:
                # If unable to get the screen resolution, use the default value
                screen_width, screen_height = 1920, 1080
        except:
            # If any error occurs, use the default value
            screen_width, screen_height = 1920, 1080
        
        print(f"Screen resolution: {screen_width}x{screen_height}")
        
        # Determine Window Placement
        window_width, window_height = self.window_size
        
        # Determine the position of the left window（Gamma）
        gamma_x = 50  # Left Margin
        gamma_y = 100  # Top Margin
        
        # Determine the position of the right window（Alpha）
        alpha_x = screen_width - window_width - 50  # Right Margin
        alpha_y = 100  # Top Margin
        
        # Create the Windows
        cv2.namedWindow(self.gamma_window_name, cv2.WINDOW_NORMAL)
        cv2.namedWindow(self.alpha_window_name, cv2.WINDOW_NORMAL)
        
        # Set Window Size
        cv2.resizeWindow(self.gamma_window_name, window_width, window_height)
        cv2.resizeWindow(self.alpha_window_name, window_width, window_height)
        
        # Set Window Position
        cv2.moveWindow(self.gamma_window_name, gamma_x, gamma_y)
        cv2.moveWindow(self.alpha_window_name, alpha_x, alpha_y)
        
        print(f"Windows positioned: Gamma at ({gamma_x}, {gamma_y}), Alpha at ({alpha_x}, {alpha_y})")

    def signal_handler(self, sig, frame):
        """Handle the Ctrl+C Signal"""
        print("\nCtrl+C pressed! Shutting down...")
        self.shutdown_flag = True
        self.auto_detection_running = False
        self.cleanup()
        sys.exit(0)

    def undistort_image(self, img, mtx, dist):
        """
        Correct Image Distortion
        Args:
            img: Raw Image
            mtx: Camera Intrinsic Matrix
            dist: Distortion Coefficients
        Returns:
            undistorted_img: Undistorted Image
        """
        try:
            h, w = img.shape[:2]
            
            # Use undistort directly
            undistorted_img = cv2.undistort(img, mtx, dist, None, mtx)
            
            return undistorted_img
            
        except Exception as e:
            print(f"Image undistortion failed: {e}")
            # Return the raw image as a fallback
            return img

    def get_gamma_camera_image(self):
        """Acquire an image from the Gamma camera and correct its distortion"""
        try:
            # Get raw image
            img_compressed = self.gamma_robot.get_img_compressed()
            img_raw = uncompress_image(img_compressed)
            
            # Obtain the camera intrinsics and perform distortion correction
            mtx, dist = self.gamma_robot.get_camera_intrinsics()
            img_undistort = self.undistort_image(img_raw, mtx, dist)
            
            return img_undistort
            
        except Exception as e:
            print(f"Failed to get gamma camera image: {e}")
            return None

    def get_alpha_camera_image(self):
        """Capture an image with the Alpha camera"""
        try:
            img_compressed = self.alpha_robot.get_img_compressed()
            img_raw = uncompress_image(img_compressed)
            return img_raw
        except Exception as e:
            print(f"Failed to get alpha camera image: {e}")
            return None

    def select_target_class(self, arm_name):
        """Select the target waste type for robotic arm sorting"""
        print(f"\nSelect target class for {arm_name} Arm:")
        print("1 - cardboard")
        print("2 - glass")  
        print("3 - metal")
        print("4 - paper")
        print("5 - plastic")

        class_mapping = {
            '1': 'cardboard',
            '2': 'glass',
            '3': 'metal',  
            '4': 'paper',
            '5': 'plastic'
        }

        while True:
            choice = input(f"Enter your choice (1-5) for {arm_name} Arm: ").strip()
            if choice in class_mapping:
                selected_class = class_mapping[choice]
                print(f"{arm_name} Arm will sort: {selected_class}")
                return selected_class
            else:
                print("Invalid choice, please try again.")

    def select_gripper(self, robot):
        """Select gripper type"""
        print("\nSelect gripper type:")  
        print("1 - GRIPPER_1")  
        print("2 - ELECTROMAGNET_1")  
        print("3 - VACUUM_PUMP_1")  

        while True:  
            choice = input("Enter your choice (1-3): ").strip()  
            if choice == "1":  
                tool = ToolID.GRIPPER_1  
                print("Selected GRIPPER_1")  
                return tool
            elif choice == "2":  
                tool = ToolID.ELECTROMAGNET_1  
                pin_electromagnet = PinID.DO4  
                robot.setup_electromagnet(pin_electromagnet)  
                print("Selected ELECTROMAGNET_1")  
                return tool
            elif choice == "3":  
                tool = ToolID.VACUUM_PUMP_1  
                print("Selected VACUUM_PUMP_1")  
                return tool
            else:  
                print("Invalid choice, please try again.")

    def get_trash_bin_position(self, arm_type):
        """Get the bin position based on the target type"""
        if arm_type == 'gamma':
            target_class = self.gamma_target_class
        else:
            target_class = self.alpha_target_class
        # Set different bin positions for different types of waste 
        bin_positions = {
            'cardboard': [0.030, 0.232, 0.253, -1.658, 1.477, -1.625], # robot GAMMA 79 
            'glass': [0.079, 0.256, 0.215, -2.852, 1.435, -1.555], # robot ALPHA 81 
            'metal': [0.079, 0.256, 0.215, -2.852, 1.435, -1.555], # robot ALPHA 81 
            'paper': [0.079, 0.256, 0.215, -2.852, 1.435, -1.555], # robot ALPHA 81 
            'plastic': [0.030, 0.232, 0.253, -1.658, 1.477, -1.625] # robot GAMMA 79 
        }
        return bin_positions.get(target_class, [0.079, 0.256, 0.215, -2.852, 1.435, -1.555])

    def setup_delay_compensation(self, arm_type):
        """Set the delay compensation parameters"""
        if arm_type == 'gamma':
            delay_seconds = self.gamma_total_delay_ms / 1000
            
            self.gamma_compensation_mm = self.conveyor_speed * delay_seconds
            self.gamma_compensation_pixels = self.gamma_compensation_mm / self.pixel_to_mm
            print(f"Gamma arm compensation: {self.gamma_compensation_pixels:.1f} px")
        else:
            delay_seconds = self.alpha_total_delay_ms / 1000
            
            self.alpha_compensation_mm = self.conveyor_speed * delay_seconds
            self.alpha_compensation_pixels = self.alpha_compensation_mm / self.pixel_to_mm
            print(f"Alpha arm compensation: {self.alpha_compensation_pixels:.1f} px")

    def activate_gripper(self, robot, tool):
        """Activate the gripper"""
        print(f"Activating gripper...")  
        try:  
            robot.grasp_with_tool()  
            print("Gripper activated")  
            time.sleep(1)  
        except Exception as e:  
            print(f"Gripper activation failed: {e}")

    def deactivate_gripper(self, robot, tool):
        """Release the gripper"""
        print(f"Releasing gripper...")  
        try:  
            robot.release_with_tool()  
            print("Gripper released")  
            time.sleep(0.5)  
        except Exception as e:  
            print(f"Gripper release failed: {e}")

    def get_z_heights(self, workspace_type):
        """Get the height parameter"""
        if workspace_type == "fixed":  
            return {  
                "pick": self.FIXED_Z_PICK,  
                "approach": self.FIXED_Z_APPROACH,  
                "safe": self.FIXED_Z_SAFE  
            }  
        else: # conveyor  
            return {  
                "pick": self.CONVEYOR_Z_PICK,  
                "approach": self.CONVEYOR_Z_APPROACH,  
                "safe": self.CONVEYOR_Z_SAFE  
            }  

    def preprocess(self, img):  
        """Image Preprocessing for Model Inference"""  
        img_resized = cv2.resize(img, (self.input_size, self.input_size))  
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)  
        img_input = img_rgb.transpose(2, 0, 1)[None, :, :, :] / 255.0  
        return img_input.astype(np.float32)  

    def resize_for_display(self, img, target_size=None):
        """Resize the image for display while maintaining the aspect ratio"""
        if target_size is None:
            target_size = self.display_size
            
        h, w = img.shape[:2]
        target_w, target_h = target_size
        
        # Calculate the scaling factor
        scale = min(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Resize
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Create a canvas of the target size
        canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        
        # Place the resized image at the center of the canvas
        y_offset = (target_h - new_h) // 2
        x_offset = (target_w - new_w) // 2
        canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        
        return canvas

    def detect_objects(self, img, workspace_type="fixed"):  
        """Detect all objects"""  
        h0, w0 = img.shape[:2]  
        input_tensor = self.preprocess(img)  
        outputs = self.session.run(None, {self.input_name: input_tensor})  
        preds = outputs[0][0]  
        boxes, scores, class_ids = [], [], []  

        for det in preds:  
            conf = det[4]  
            if conf < self.conf_thresh:  
                continue  

            cls_confidences = det[5:]  
            cls_id = np.argmax(cls_confidences)  
            cls_conf = cls_confidences[cls_id] * conf  
            if cls_conf < self.conf_thresh:  
                continue  

            cx, cy, w, h = det[0:4]  
            x1 = int((cx - w / 2) * w0 / self.input_size)  
            y1 = int((cy - h / 2) * h0 / self.input_size)  
            x2 = int((cx + w / 2) * w0 / self.input_size)  
            y2 = int((cy + h / 2) * h0 / self.input_size)  

            boxes.append([x1, y1, x2, y2])  
            scores.append(float(cls_conf))  
            class_ids.append(cls_id)  

        final_boxes, final_scores, final_class_ids = [], [], []  
        indices = cv2.dnn.NMSBoxes(boxes, scores, self.conf_thresh, self.nms_thresh)  

        if len(indices) > 0:  
            for idx in indices:  
                if isinstance(idx, (list, np.ndarray)):  
                    idx = idx[0]  
                final_boxes.append(boxes[idx])  
                final_scores.append(scores[idx])  
                final_class_ids.append(class_ids[idx])  

        detections = []  
        for i in range(len(final_boxes)):  
            x1, y1, x2, y2 = final_boxes[i]  
            class_id = final_class_ids[i]  
            class_name = self.class_names[class_id]  
            confidence = final_scores[i]  
            center_x = (x1 + x2) / 2  
            center_y = (y1 + y2) / 2  

            # Calculate coordinates for each of the two robotic arms 
            gamma_x, gamma_y, _ = self.pixel_to_robot(center_x, center_y, 'gamma', workspace_type)
            alpha_x, alpha_y, _ = self.pixel_to_robot(center_x, center_y, 'alpha', workspace_type)

            # Check if they are within their respective workspaces
            if workspace_type == "fixed":  
                in_gamma_workspace = self.in_workspace(gamma_x, gamma_y, 'gamma')
                in_alpha_workspace = self.in_workspace(alpha_x, alpha_y, 'alpha')
                in_workspace = in_gamma_workspace or in_alpha_workspace
                in_conveyor = False  
            else:  
                in_gamma_workspace = self.in_conveyor_workspace(gamma_x, gamma_y, 'gamma')
                in_alpha_workspace = self.in_conveyor_workspace(alpha_x, alpha_y, 'alpha')
                in_workspace = in_gamma_workspace or in_alpha_workspace
                in_conveyor = in_workspace

            detections.append({  
                "class": class_name,  
                "confidence": confidence,  
                "bbox": [x1, y1, x2, y2],  
                "center_x": center_x,  
                "center_y": center_y,  
                "gamma_robot_x": gamma_x,  
                "gamma_robot_y": gamma_y,  
                "alpha_robot_x": alpha_x,  
                "alpha_robot_y": alpha_y,  
                "in_gamma_workspace": in_gamma_workspace,
                "in_alpha_workspace": in_alpha_workspace,
                "in_workspace": in_workspace,  
                "in_conveyor": in_conveyor  
            })  

        return detections  

    def draw_detections_on_image(self, img, detections, arm_type, workspace_type="fixed"):
        """Visualize detection results on the image - show all waste items but highlight only the target category"""
        # First, resize the image for display
        display_img = self.resize_for_display(img)
        h, w = display_img.shape[:2]
        
        # Calculate the scaling ratio from the raw image to the displayed image
        orig_h, orig_w = img.shape[:2]
        scale_x = w / orig_w
        scale_y = h / orig_h

        # Draw the detection results
        for det in detections:
            # Adjust the bounding box coordinates to the display dimensions
            x1_orig, y1_orig, x2_orig, y2_orig = det["bbox"]
            x1 = int(x1_orig * scale_x)
            y1 = int(y1_orig * scale_y)
            x2 = int(x2_orig * scale_x)
            y2 = int(y2_orig * scale_y)
            
            class_name = det["class"]
            confidence = det["confidence"]
            
            # Determine the target area - check if it lies within the corresponding workspace based on the robot arm type
            if arm_type == 'gamma':
                in_target_area = det["in_gamma_workspace"]
            else:  # alpha
                in_target_area = det["in_alpha_workspace"]

            # Set colors: red for the target category, blue for other categories, and gray for items outside the target area.
            if not in_target_area: 
                color = (128, 128, 128)  # Gray - Not in workspace
                thickness = 2
            elif class_name == (self.gamma_target_class if arm_type == 'gamma' else self.alpha_target_class):
                color = (0, 0, 255)  # Red - Target Category
                thickness = 3
            else:
                color = (255, 0, 0)  # Blue - Other Categories
                thickness = 2

            # Draw bounding box
            cv2.rectangle(display_img, (x1, y1), (x2, y2), color, thickness)

            # Draw labels
            status = "" if in_target_area else " (Outside)" 
            label = f"{class_name}: {confidence:.2f}{status}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            # Label background
            cv2.rectangle(display_img, (x1, y1 - label_size[1] - 10),
                         (x1 + label_size[0], y1), color, -1)
            cv2.putText(display_img, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            # Draw the center fo the bounding box
            cx_orig, cy_orig = det["center_x"], det["center_y"]
            cx = int(cx_orig * scale_x)
            cy = int(cy_orig * scale_y)
            cv2.circle(display_img, (cx, cy), 5, color, -1)

            # highlight valid targets
            if class_name == (self.gamma_target_class if arm_type == 'gamma' else self.alpha_target_class) and in_target_area:
                cv2.putText(display_img, "TARGET", (cx - 20, cy - 15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Display Statistics
        total_count = len(detections)
        target_class = self.gamma_target_class if arm_type == 'gamma' else self.alpha_target_class
        target_count = len([d for d in detections if d["class"] == target_class])
        
        if workspace_type == "fixed":
            # Count objects within the corresponding workspace for each robot arm in fixed workspace
            if arm_type == 'gamma':
                in_target_area_count = len([d for d in detections if d["in_gamma_workspace"]])
                target_in_area = len([d for d in detections if d["class"] == target_class and d["in_gamma_workspace"]])
            else:
                in_target_area_count = len([d for d in detections if d["in_alpha_workspace"]])
                target_in_area = len([d for d in detections if d["class"] == target_class and d["in_alpha_workspace"]])
            workspace_name = "Fixed Workspace"
        else:
            # Count objects within the corresponding workspace for each robot arm type in conveyor belt workspace
            if arm_type == 'gamma':
                in_target_area_count = len([d for d in detections if d["in_gamma_workspace"]])
                target_in_area = len([d for d in detections if d["class"] == target_class and d["in_gamma_workspace"]])
            else:
                in_target_area_count = len([d for d in detections if d["in_alpha_workspace"]])
                target_in_area = len([d for d in detections if d["class"] == target_class and d["in_alpha_workspace"]])
            workspace_name = "Conveyor Workspace"

        cv2.putText(display_img, f"{arm_type.upper()} Arm - {workspace_name}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(display_img, f"Target: {target_class}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(display_img, f"Total objects: {total_count}", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(display_img, f"In target area: {in_target_area_count}", (10, 120),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(display_img, f"Target objects: {target_count} (In area: {target_in_area})", (10, 150),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        return display_img

    def show_detection_window(self, gamma_image, alpha_image, gamma_detections, alpha_detections, workspace_type="fixed"):
        """Display the detection windows for both robotic arms"""
        # Ensure neither image is None; if any is None, create a black image
        if gamma_image is None:
            print("Warning: Gamma image is None, creating placeholder")
            gamma_image = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(gamma_image, "GAMMA CAMERA NOT AVAILABLE", (50, 240), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        if alpha_image is None:
            print("Warning: Alpha image is None, creating placeholder")
            alpha_image = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(alpha_image, "ALPHA CAMERA NOT AVAILABLE", (50, 240), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Display the detection results for the Gamma robotic arm
        gamma_display = self.draw_detections_on_image(gamma_image, gamma_detections, 'gamma', workspace_type)
        # Display the detection results for the Alpha robotic arm
        alpha_display = self.draw_detections_on_image(alpha_image, alpha_detections, 'alpha', workspace_type)

        # Display the windows
        cv2.imshow(self.gamma_window_name, gamma_display)
        cv2.imshow(self.alpha_window_name, alpha_display)

        # Window Refresh 
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:  # Exit on pressing the 'q' or ESC key
            print("Exit key pressed!")
            self.shutdown_flag = True
            self.auto_detection_running = False
            return False
        return True

    def filter_detections_by_class_and_workspace(self, detections, target_class, arm_type, workspace_type="fixed"):
        """Filter detection results by category and workspace - return only target category objects located within their corresponding workspace"""
        filtered_detections = []
        for det in detections:
            if det["class"] == target_class:
                # Check if it is within the corresponding robotic arm's workspace
                if arm_type == 'gamma':
                    in_workspace = det["in_gamma_workspace"]
                else:  # alpha
                    in_workspace = det["in_alpha_workspace"]
                
                if in_workspace:
                    filtered_detections.append(det)
        return filtered_detections

    def specialized_dual_camera_detection(self, workspace_type="fixed"):
        """Division of Detection Tasks - Each camera focuses on its own target"""
        if self.shutdown_flag:
            return {'gamma_tasks': [], 'alpha_tasks': []}
            
        print("Starting specialized dual camera detection...")
        
        # Initialize the variables
        gamma_image = None
        alpha_image = None
        all_gamma_detections = []
        all_alpha_detections = []
        gamma_target_detections = []
        alpha_target_detections = []
        
        # The Gamma camera focuses on its own target - using the undistorted image
        try:
            # Get the undistorted image
            gamma_image = self.get_gamma_camera_image()
            
            if gamma_image is not None:
                print(f"Gamma undistorted image shape: {gamma_image.shape}")
                # Perform detection on the undistorted image
                all_gamma_detections = self.detect_objects(gamma_image, workspace_type)
                # Only pick the target category located within the Gamma workspace
                gamma_target_detections = self.filter_detections_by_class_and_workspace(
                    all_gamma_detections, self.gamma_target_class, 'gamma', workspace_type)
                print(f"Gamma camera detected {len(all_gamma_detections)} objects, {len(gamma_target_detections)} {self.gamma_target_class} objects in Gamma workspace")
            else:
                print("Failed to get Gamma camera image")
        except Exception as e:
            print(f"Gamma camera detection error: {e}")

        # The Alpha camera focuses on its own target - using the original (uncompressed) image
        try:
            alpha_image = self.get_alpha_camera_image()
            if alpha_image is not None:
                print(f"Alpha image shape: {alpha_image.shape}")
                # Detect all objects
                all_alpha_detections = self.detect_objects(alpha_image, workspace_type)
                # Only pick the target category located within the Alpha workspace
                alpha_target_detections = self.filter_detections_by_class_and_workspace(
                    all_alpha_detections, self.alpha_target_class, 'alpha', workspace_type)
                print(f"Alpha camera detected {len(all_alpha_detections)} objects, {len(alpha_target_detections)} {self.alpha_target_class} objects in Alpha workspace")
            else:
                print("Failed to get Alpha camera image")
        except Exception as e:
            print(f"Alpha camera detection error: {e}")

        # Display the detection windows
        if not self.show_detection_window(gamma_image, alpha_image, all_gamma_detections, all_alpha_detections, workspace_type):
            return {'gamma_tasks': [], 'alpha_tasks': []}

        return {
            'gamma_tasks': gamma_target_detections,
            'alpha_tasks': alpha_target_detections
        }

    def pixel_to_robot(self, u, v, arm_type, workspace_type="fixed"):
        """Convert pixel coordinates to robot coordinates - using their respective transformation matrices"""
        pixel = np.array([u, v, 1])
        if arm_type == 'gamma':
            if workspace_type == "fixed":
                world = np.dot(self.gamma_fixed_transform_matrix, pixel)
            else:
                world = np.dot(self.gamma_conveyor_transform_matrix, pixel)
        else: # alpha
            if workspace_type == "fixed":
                world = np.dot(self.alpha_fixed_transform_matrix, pixel)
            else:
                world = np.dot(self.alpha_conveyor_transform_matrix, pixel)
        z = self.get_z_heights(workspace_type)["pick"]
        return world[0], world[1], z

    def in_workspace(self, x, y, arm_type):  
        """Check if the coordinates are within the fixed workspace"""
        if arm_type == 'gamma':
            return self.GAMMA_X_MIN <= x <= self.GAMMA_X_MAX and self.GAMMA_Y_MIN <= y <= self.GAMMA_Y_MAX  
        else: # alpha
            return self.ALPHA_X_MIN <= x <= self.ALPHA_X_MAX and self.ALPHA_Y_MIN <= y <= self.ALPHA_Y_MAX  

    def in_conveyor_workspace(self, x, y, arm_type):  
        """Check if the coordinates are within the conveyor belt workspace"""
        if arm_type == 'gamma':
            return self.GAMMA_CONVEYOR_X_MIN <= x <= self.GAMMA_CONVEYOR_X_MAX and self.GAMMA_CONVEYOR_Y_MIN <= y <= self.GAMMA_CONVEYOR_Y_MAX  
        else: # alpha
            return self.ALPHA_CONVEYOR_X_MIN <= x <= self.ALPHA_CONVEYOR_X_MAX and self.ALPHA_CONVEYOR_Y_MIN <= y <= self.ALPHA_CONVEYOR_Y_MAX  

    def move_to_observe(self, workspace_type="fixed", arm_type=None):  
        """Move to the observation position"""
        if self.shutdown_flag:
            return
            
        print(f" Moving to {workspace_type} workspace observe position...")  
        try:  
            if workspace_type == "fixed":  
                if arm_type == 'gamma' or arm_type is None:
                    gamma_observe_pose = PoseObject(*self.GAMMA_FIXED_OBSERVE_POS)
                    self.gamma_robot.move_pose(gamma_observe_pose)
                if arm_type == 'alpha' or arm_type is None:
                    alpha_observe_pose = PoseObject(*self.ALPHA_FIXED_OBSERVE_POS)
                    self.alpha_robot.move(alpha_observe_pose)
            else: # conveyor  
                if arm_type == 'gamma' or arm_type is None:
                    gamma_observe_pose = PoseObject(*self.GAMMA_CONVEYOR_OBSERVE_POS)
                    self.gamma_robot.move_pose(gamma_observe_pose)
                if arm_type == 'alpha' or arm_type is None:
                    alpha_observe_pose = PoseObject(*self.ALPHA_CONVEYOR_OBSERVE_POS)
                    self.alpha_robot.move(alpha_observe_pose)
            time.sleep(1)  
        except Exception as e:  
            print(f" Move to observe position failed: {e}")  

    def control_conveyor(self, action="start"):  
        """Control the conveyor belt"""  
        if self.shutdown_flag:
            return False
            
        try:  
            if action == "start":  
                self.alpha_robot.run_conveyor(self.conveyor_id, speed=self.conveyor_speed, direction=ConveyorDirection.FORWARD)  
                print(f" Conveyor started at speed {self.conveyor_speed}")  
                self.conveyor_running = True
                self.conveyor_stop_requested = False
            elif action == "stop":  
                self.alpha_robot.stop_conveyor(self.conveyor_id)  
                print(" Conveyor stopped")  
                self.conveyor_running = False
                self.conveyor_stop_requested = True
            time.sleep(1)  
            return True  
        except Exception as e:  
            print(f" Conveyor control failed: {e}")  
            return False  

    def start_batch_pick_operation(self, total_tasks, workspace_type="conveyor"):
        """Start the batch picking operation"""
        if workspace_type == "conveyor" and total_tasks > 0:
            print(f"Starting batch pick operation with {total_tasks} tasks")
            self.batch_pick_count = total_tasks
            # Stop the conveyor belt
            self.control_conveyor("stop")
            return True
        return False

    def complete_batch_pick_operation(self, workspace_type="conveyor"):
        """Complete the batch picking operation"""
        if workspace_type == "conveyor" and self.batch_pick_count > 0:
            self.batch_pick_count -= 1
            print(f"Batch pick operation completed. Remaining tasks: {self.batch_pick_count}")
            
            # If all batch picking tasks are complete, restart the conveyor
            if self.batch_pick_count == 0:
                print("All batch pick operations completed. Restarting conveyor...")
                self.control_conveyor("start")
                return True
        return False

    def pick_and_place_object(self, detection, arm_type, workspace_type="fixed"):  
        """Pick and Place"""  
        if self.shutdown_flag:
            return False
            
        with self.operation_lock:
            self.current_working_arm = arm_type

            if arm_type == 'gamma':
                robot = self.gamma_robot
                tool = self.gamma_tool
                x, y = detection["gamma_robot_x"], detection["gamma_robot_y"]
                trash_bin_pos = self.GAMMA_TRASH_BIN_POS
                compensation_pixels = self.gamma_compensation_pixels
            else:
                robot = self.alpha_robot
                tool = self.alpha_tool
                x, y = detection["alpha_robot_x"], detection["alpha_robot_y"]
                trash_bin_pos = self.ALPHA_TRASH_BIN_POS
                compensation_pixels = self.alpha_compensation_pixels

            # Conveyor Delay Compensation
            if workspace_type == "conveyor":
                original_pixel_x = detection["center_x"]
                compensated_pixel_x = original_pixel_x - compensation_pixels
                # Recompute the robot coordinates
                if arm_type == 'gamma':
                    x, y, _ = self.pixel_to_robot(compensated_pixel_x, detection["center_y"], 'gamma', workspace_type)
                else:
                    x, y, _ = self.pixel_to_robot(compensated_pixel_x, detection["center_y"], 'alpha', workspace_type)
                print(f" {arm_type.capitalize()} arm latency compensation:")
                print(f" Compensation distance: +{compensation_pixels:.1f} pixels")
            print(f"{arm_type.capitalize()} Arm targeting {detection['class']} at coordinates: ({x:.3f}, {y:.3f})")  

            z_heights = self.get_z_heights(workspace_type)  

            try:  
                # Set the gripper
                self.deactivate_gripper(robot, tool)
                time.sleep(0.5)

                # The Pick Action
                approach_pose = PoseObject(x, y, z_heights["approach"], 0.0, 1.57, 0.0)
                pick_pose = PoseObject(x, y, z_heights["pick"], 0.0, 1.57, 0.0)
                safe_pose = PoseObject(x, y, z_heights["safe"], 0.0, 1.57, 0.0)

                print(" Moving to approach position...")
                robot.move_pose(approach_pose)

                print(" Moving to pick position...")
                robot.move_pose(pick_pose)

                # Pick
                self.activate_gripper(robot, tool)
                time.sleep(1)

                print(" Moving to safe height...")
                robot.move_pose(safe_pose)

                # Move to the trash bin position
                print(" Moving to trash bin...")
                trash_pose = PoseObject(*trash_bin_pos)
                robot.move_pose(trash_pose)

                # Place
                self.deactivate_gripper(robot, tool)
                time.sleep(0.5)

                # Move to the safe height
                safe_trash_pose = PoseObject(
                    trash_bin_pos[0],
                    trash_bin_pos[1],
                    z_heights["safe"],
                    0.0, 1.57, 0.0
                )
                robot.move_pose(safe_trash_pose)
                print(f"{detection['class']} placed successfully by {arm_type} arm.")

                # Complete the batch picking operation
                if workspace_type == "conveyor":
                    self.complete_batch_pick_operation(workspace_type)

                # Return to the observation position
                if workspace_type == "conveyor":
                    self.move_to_observe("conveyor", arm_type)

                self.current_working_arm = None
                return True

            except Exception as e:
                print(f"Pick and place failed: {e}")
                # Complete the batch picking operation (even if it failed)
                if workspace_type == "conveyor":
                    self.complete_batch_pick_operation(workspace_type)
                self.current_working_arm = None
                return False

    def move_to_safe_position(self, arm_type):
        """Move to the safe position"""
        try:
            if arm_type == 'gamma':
                safe_pose = PoseObject(*self.GAMMA_SAFE_POS)
                self.gamma_robot.move_pose(safe_pose)
                print("Gamma arm moved to safe position")
            else:
                safe_pose = PoseObject(*self.ALPHA_SAFE_POS)
                self.alpha_robot.move(safe_pose)
                print("Alpha arm moved to safe position")
        except Exception as e:
            print(f"Move to safe position failed: {e}")

    def coordinate_division_operation(self, detection_results, workspace_type="fixed"):
        """Coordinated Division operation"""
        if self.shutdown_flag:
            return 0
            
        gamma_tasks = detection_results['gamma_tasks']
        alpha_tasks = detection_results['alpha_tasks']

        print(f"Gamma tasks ({self.gamma_target_class}): {len(gamma_tasks)}")
        print(f"Alpha tasks ({self.alpha_target_class}): {len(alpha_tasks)}")

        # Calculate the total task count
        total_tasks = len(gamma_tasks) + len(alpha_tasks)
        print(f"Total tasks to process: {total_tasks}")

        # If in conveyor mode, start the batch picking operation
        if workspace_type == "conveyor" and total_tasks > 0:
            self.start_batch_pick_operation(total_tasks, workspace_type)

        processed_count = 0
        
        # Process Gamma tasks
        for i, task in enumerate(gamma_tasks):
            if self.shutdown_flag:
                break
            print(f"\nGamma Arm processing {self.gamma_target_class} object {i+1}/{len(gamma_tasks)}")
            success = self.pick_and_place_object(task, 'gamma', workspace_type)
            if success:
                processed_count += 1
            time.sleep(1)

        # Process Alpha tasks
        for i, task in enumerate(alpha_tasks):
            if self.shutdown_flag:
                break
            print(f"\nAlpha Arm processing {self.alpha_target_class} object {i+1}/{len(alpha_tasks)}")
            success = self.pick_and_place_object(task, 'alpha', workspace_type)
            if success:
                processed_count += 1
            time.sleep(1)

        return processed_count

    def auto_detect_and_sort_division(self, workspace_type="fixed"):
        """Automated Sorting"""
        if self.shutdown_flag:
            return
            
        print(f"\n=== Dual Arm {workspace_type.title()} Workspace: Division Detection ===")
        print(f"Gamma Arm: {self.gamma_target_class}")
        print(f"Alpha Arm: {self.alpha_target_class}")
        # Move to observation position
        self.move_to_observe(workspace_type)

        # Task-Specific Detection
        detection_results = self.specialized_dual_camera_detection(workspace_type)
        # Coordinate Task Division
        processed_count = self.coordinate_division_operation(detection_results, workspace_type)

        print(f"\nDivision sorting completed! Processed {processed_count} objects.")

    def auto_detect_and_sort_division_conveyor(self):
        """Distributed Detection and Automated Sorting - conveyor"""
        if self.shutdown_flag:
            return
            
        print(f"\n=== Conveyor Division Detection ===")
        print(f"Gamma Arm: {self.gamma_target_class}")
        print(f"Alpha Arm: {self.alpha_target_class}")
        # start the conveyor belt
        if not self.control_conveyor("start"):
            print("Failed to start conveyor")
            return

        # Move to the observation position
        self.move_to_observe("conveyor")

        self.auto_detection_running = True
        total_processed = 0
        cycle_count = 0

        try:
            while self.auto_detection_running and not self.shutdown_flag:
                cycle_count += 1
                print(f"\n--- Division Detection Cycle {cycle_count} ---")
                
                # Ensure the conveyor is running unless a batch picking operation is in progress
                if not self.conveyor_running and self.batch_pick_count == 0:
                    print("No batch operations, ensuring conveyor is running...")
                    self.control_conveyor("start")
                
                # Task-Specific Detection
                detection_results = self.specialized_dual_camera_detection("conveyor")
                
                # If a target is detected, execute the pick operation
                if detection_results['gamma_tasks'] or detection_results['alpha_tasks']:
                    print(f"Targets detected! Gamma: {len(detection_results['gamma_tasks'])}, Alpha: {len(detection_results['alpha_tasks'])}")
                    # Coordinated Task Division (the conveyor stops automatically within pick_and_place_object)
                    processed_this_cycle = self.coordinate_division_operation(detection_results, "conveyor")
                    total_processed += processed_this_cycle
                    print(f"Cycle {cycle_count}: Processed {processed_this_cycle} objects (Total: {total_processed})")
                else:
                    print(f"Cycle {cycle_count}: No targets detected, continuing conveyor operation...")
                    processed_this_cycle = 0
                
                # Brief pause (1s)
                time.sleep(1)
                
        except KeyboardInterrupt:
            print("Division detection interrupted by user")
        except Exception as e:
            print(f"Division detection error: {e}")
        finally:
            self.auto_detection_running = False
            # Ensure the conveyor is stopped
            self.emergency_stop_conveyor()
            print(f"Conveyor division sorting finished. Total processed: {total_processed}")

    def emergency_stop_conveyor(self):
        """Emergency Stop the Conveyor"""
        print("Emergency stopping conveyor...")
        try:
            # Make multiple attempts to stop the conveyor
            for i in range(3):
                try:
                    self.alpha_robot.stop_conveyor(self.conveyor_id)
                    print(f"Conveyor stop attempt {i+1}: Success")
                    self.conveyor_running = False
                    self.batch_pick_count = 0  # Reset the batch count
                    break
                except Exception as e:
                    print(f"Conveyor stop attempt {i+1} failed: {e}")
                    time.sleep(0.5)
        except Exception as e:
            print(f"Emergency conveyor stop failed: {e}")

    def test_camera_connection(self):  
        """Camera Connection Test"""  
        print("Testing camera connections...")  
        gamma_success = False
        alpha_success = False
        try:  
            gamma_image = self.get_gamma_camera_image()  
            if gamma_image is not None:  
                print(f" Gamma camera connection successful! Image size: {gamma_image.shape}")  
                gamma_success = True
            else:  
                print(" Gamma camera connection failed")  
        except Exception as e:  
            print(f" Gamma camera test failed: {e}")  

        try:  
            alpha_image = self.get_alpha_camera_image()  
            if alpha_image is not None:  
                print(f" Alpha camera connection successful! Image size: {alpha_image.shape}")  
                alpha_success = True
            else:  
                print(" Alpha camera connection failed")  
        except Exception as e:  
            print(f" Alpha camera test failed: {e}")  

        return gamma_success and alpha_success

    def debug_camera_connections(self):
        """Debug camera connection"""
        print("\n=== Debugging Camera Connections ===")
        
        # Test the Gamma camera
        print("Testing Gamma camera...")
        try:
            gamma_img = self.get_gamma_camera_image()
            if gamma_img is not None:
                print(f"Gamma camera OK - Image shape: {gamma_img.shape}")
                # Display basic information about the Gamma image
                cv2.imshow("Gamma Debug", gamma_img)
                cv2.waitKey(1000)  # Display 1 sec
                cv2.destroyWindow("Gamma Debug")
            else:
                print("Gamma camera FAILED - No image received")
        except Exception as e:
            print(f"Gamma camera ERROR: {e}")
        
        # Test the Alpha camera
        print("Testing Alpha camera...")
        try:
            alpha_img = self.get_alpha_camera_image()
            if alpha_img is not None:
                print(f"Alpha camera OK - Image shape: {alpha_img.shape}")
                # Display basic information about the Alpha image
                cv2.imshow("Alpha Debug", alpha_img)
                cv2.waitKey(1000)  # display 1 sec
                cv2.destroyWindow("Alpha Debug")
            else:
                print("Alpha camera FAILED - No image received")
        except Exception as e:
            print(f"Alpha camera ERROR: {e}")

    def cleanup(self):  
        """clean up"""  
        print("Cleaning up resources...")
        self.auto_detection_running = False
        self.shutdown_flag = True
        
        # first, stop the conveyor belt
        print("Stopping conveyor...")
        self.emergency_stop_conveyor()
        
        # close windows
        print("Closing windows...")
        try:
            cv2.destroyAllWindows()
            cv2.waitKey(1)  # Ensure the window is closed
        except:
            pass
            
        # Disconnect from the robot
        print("Closing robot connections...")
        try:  
            self.gamma_robot.close_connection()  
            print("Gamma robot connection closed")
        except Exception as e:  
            print(f"Error closing Gamma robot: {e}")
            
        try:
            self.alpha_robot.close_connection()  
            print("Alpha robot connection closed")  
        except Exception as e:
            print(f"Error closing Alpha robot: {e}")

def main():  
    # Configuration Parameters
    ROBOT_IPS = ["192.168.0.109", "192.168.0.106"] # Gamma和Alpha机械臂的IP地址 
    ONNX_MODEL_PATH = "/home/ned2/Desktop/Waste sorting/code/best.onnx"
    # Calibration Configuration File 
    CALIBRATION_CONFIGS = {
        'gamma_fixed': "/home/ned2/Desktop/Waste sorting/code/1124 GAMMAconveyor_calibration_matrix.npy",
        'gamma_conveyor': "/home/ned2/Desktop/Waste sorting/code/1203 conveyorcalibration_matrix1.npy",
        'alpha_fixed': "/home/ned2/Desktop/Waste sorting/code/1124 ALPHAconveyor_calibration_matrix.npy",
        'alpha_conveyor': "/home/ned2/Desktop/Waste sorting/code/1114robot81conveyor_calibration_matrix.npy"
    }
    # Workspace Configuration File 
    WORKSPACE_CONFIGS = {
        'gamma_fixed': "/home/ned2/Desktop/Waste sorting/code/1124robot79workspace_conveyor.json",
        'alpha_fixed': "/home/ned2/Desktop/Waste sorting/code/1124robot81workspace_conveyor.json",
        'gamma_conveyor': "/home/ned2/Desktop/Waste sorting/code/1124robot79workspace_conveyor.json", # Gamma传送带工作区 
        'alpha_conveyor': "/home/ned2/Desktop/Waste sorting/code/1124robot81workspace_conveyor.json" # Alpha传送带工作区 
    }

    print("=== Gamma & Alpha Dual Arm Sorting Robot System ===")

    # Initialize the dual-arm robotic system
    robot_system = DualArmSortingRobot(ROBOT_IPS, ONNX_MODEL_PATH, CALIBRATION_CONFIGS, WORKSPACE_CONFIGS)

    try:
        # Test camera connection
        if robot_system.test_camera_connection():  
            # Run Debug
            robot_system.debug_camera_connections()
            
            while not robot_system.shutdown_flag:  
                print("\n" + "="*50)  
                print("Select mode:")  
                print("1 - Fixed Workspace: Division Detection & Sort")  
                print("2 - Conveyor: Continuous Division Detection & Sort")  
                print("3 - Exit")  
                print("Press 'q' in any window to stop current operation")
                print("Press Ctrl+C to emergency stop")

                choice = input("Enter your choice (1-3): ").strip()  

                if choice == "1":  
                    robot_system.auto_detect_and_sort_division("fixed")  
                elif choice == "2":  
                    robot_system.auto_detect_and_sort_division_conveyor()  
                elif choice == "3":  
                    print(" Exiting...")  
                    break  
                else:  
                    print(" Invalid choice, please try again.")  
        else:  
            print(" Unable to connect to both cameras")

    except KeyboardInterrupt:
        print("\nProgram interrupted by user")
    except Exception as e:  
        print(f" Program runtime error: {e}")  
    finally:  
        robot_system.cleanup()  
        print(" Program finished")  

if __name__ == "__main__":  
    main()