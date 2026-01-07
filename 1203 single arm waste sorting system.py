import cv2  
import numpy as np  
import time  
import json  
from pyniryo import *  
import onnxruntime as ort  
import threading  
import signal
import sys

class SingleArmSortingRobot:  
    def __init__(self, robot_ip, onnx_model_path, calibration_configs, workspace_configs):  
        # Set up signal handling
        signal.signal(signal.SIGINT, self.signal_handler)
        
        # Robotic Arm Connection  
        print("Connecting to robot...")  
        self.robot = NiryoRobot(robot_ip)
        
        # Calibrate robotic arm  
        print("Calibrating robot...")  
        self.robot.calibrate_auto()  

        # Select target type to sort  
        print("\n" + "="*50)  
        print("=== Object Type Selection ===")  
        self.target_class = self.select_target_class()  
        
        # Select gripper type  
        print("\n" + "="*50)  
        print("=== Gripper Selection ===")  
        print(f"\n--- Robot ({self.target_class}) Gripper Selection ---")  
        self.tool = self.select_gripper(self.robot)  

        # Load ONNX model  
        print("Loading ONNX model...")  
        self.session = ort.InferenceSession(onnx_model_path, providers=['CPUExecutionProvider'])  
        self.input_name = self.session.get_inputs()[0].name  

        # Model Parameters  
        self.input_size = 640  
        self.conf_thresh = 0.25  
        self.nms_thresh = 0.45  

        # Waste sorting categories  
        self.class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic']  

        # === Load calibration matrix ===  
        print("Loading calibration matrix...")  
        self.fixed_transform_matrix = np.load(calibration_configs['fixed'])
        self.conveyor_transform_matrix = np.load(calibration_configs['conveyor'])
        print("Calibration matrices loaded successfully.")  

        # === Load workspaces ===  
        print("Loading workspaces...")  
        with open(workspace_configs['fixed'], "r") as f:  
            workspace = json.load(f)  
            x_list = [p["x"] for p in workspace["points"]]  
            y_list = [p["y"] for p in workspace["points"]]  
            self.X_MIN, self.X_MAX = min(x_list), max(x_list)  
            self.Y_MIN, self.Y_MAX = min(y_list), max(y_list)  
            # Store workspace points for visualization
            self.fixed_workspace_points = workspace["points"]
        print(f"Fixed Workspace X range: {self.X_MIN:.3f} ~ {self.X_MAX:.3f}")  
        print(f"Fixed Workspace Y range: {self.Y_MIN:.3f} ~ {self.Y_MAX:.3f}")  

        # === Conveyor workspace === 
        print("Loading conveyor workspace...") 
        with open(workspace_configs['conveyor'], "r") as f:  
            conveyor_workspace = json.load(f)  
            conveyor_x_list = [p["x"] for p in conveyor_workspace["points"]]  
            conveyor_y_list = [p["y"] for p in conveyor_workspace["points"]]  
            self.CONVEYOR_X_MIN, self.CONVEYOR_X_MAX = min(conveyor_x_list), max(conveyor_x_list)  
            self.CONVEYOR_Y_MIN, self.CONVEYOR_Y_MAX = min(conveyor_y_list), max(conveyor_y_list)  
            # Store conveyor workspace points for visualization
            self.conveyor_workspace_points = conveyor_workspace["points"]
        print(f"Conveyor Workspace X range: {self.CONVEYOR_X_MIN:.3f} ~ {self.CONVEYOR_X_MAX:.3f}")  
        print(f"Conveyor Workspace Y range: {self.CONVEYOR_Y_MIN:.3f} ~ {self.CONVEYOR_Y_MAX:.3f}")  

        # === Conveyor parameters ===
        print("Setting up conveyor parameters...")
        self.conveyor_speed = 50
        self.pixel_to_mm = 0.5000
        
        # === Delay compensation parameters === 
        print("Setting up delay compensation...") 
        self.total_delay_ms = 600
        self.setup_delay_compensation()

        # Height Parameters  
        self.FIXED_Z_PICK = 0.05  
        self.FIXED_Z_APPROACH = 0.15  
        self.FIXED_Z_SAFE = 0.20  
        self.CONVEYOR_Z_PICK = 0.11 
        self.CONVEYOR_Z_APPROACH = 0.18  
        self.CONVEYOR_Z_SAFE = 0.23  

        # Position definition  
        # Trash bin position
        self.TRASH_BIN_POS = self.get_trash_bin_position()
        # === Fixed workspace observation position === 
        self.FIXED_OBSERVE_POS = [0.006, 0.162, 0.253, 3.034, 1.327, -1.712]
        # === Conveyor workspace observation position === 
        self.CONVEYOR_OBSERVE_POS = [0.251, 0.005, 0.254, 2.978, 1.311, 3.003]
        # Safe position
        self.SAFE_POS = [0.140, -0.000, 0.203, 0.000, 0.753, -0.001]

        # Conveyor setup
        self.conveyor_id = self.robot.set_conveyor()
        # Synchronization control
        self.operation_lock = threading.Lock()
        self.auto_detection_running = False
        self.shutdown_flag = False
        self.conveyor_running = False
        self.conveyor_stop_requested = False
        self.pending_pick_operations = 0
        self.batch_pick_count = 0
        
        # Display window parameters
        self.window_name = "Robot Arm Detection - Undistorted"
        self.workspace_window_name = "Workspace Region"
        self.window_size = (800, 600)
        self.workspace_window_size = (400, 300)
        
        # Image processing parameters
        self.display_size = (640, 480)

        # Create display window
        self.setup_window()
        
        print("Single Arm System initialization complete")
        print(f"\n=== Configuration Summary ===")
        print(f"Target Class: {self.target_class}")

    def extract_image_workspace(self, img, workspace_type="fixed"):
        """
        Extract and display workspace region from image
        Args:
            img: Input image (undistorted)
            workspace_type: "fixed" or "conveyor"
        Returns:
            workspace_overlay: Image with workspace boundary drawn
        """
        if img is None:
            print("Error: Cannot extract workspace from None image")
            return None
        
        # Get workspace points
        if workspace_type == "fixed":
            workspace_points = self.fixed_workspace_points
            transform_matrix = self.fixed_transform_matrix
        else:
            workspace_points = self.conveyor_workspace_points
            transform_matrix = self.conveyor_transform_matrix
        
        # Create a copy of the image for drawing
        workspace_overlay = img.copy()
        
        # Convert robot coordinates to pixel coordinates
        pixel_points = []
        for point in workspace_points:
            # Robot to pixel transformation (inverse of pixel_to_robot)
            robot_x, robot_y = point["x"], point["y"]
            
            # For affine transformation, we need to solve inverse transformation
            try:
                inv_transform = np.linalg.inv(transform_matrix)
                pixel_homogeneous = np.dot(inv_transform, np.array([robot_x, robot_y, 1]))
                u, v = pixel_homogeneous[0], pixel_homogeneous[1]
                # Ensure pixel coordinates are within image bounds
                u = max(0, min(img.shape[1] - 1, int(u)))
                v = max(0, min(img.shape[0] - 1, int(v)))
                pixel_points.append((u, v))
            except:
                # If transformation fails, skip this point
                print(f"Warning: Could not calculate inverse transform for point ({robot_x}, {robot_y})")
                continue
        
        if len(pixel_points) < 3:
            print(f"Warning: Not enough valid points for {workspace_type} workspace extraction")
            return workspace_overlay
        
        # Convert points to numpy array
        pts = np.array(pixel_points, dtype=np.int32)
        
        # Draw workspace boundary
        if len(pts) > 2:
            # Create convex hull for better visualization
            hull = cv2.convexHull(pts)
            
            # Draw filled polygon with transparency
            overlay = workspace_overlay.copy()
            cv2.fillPoly(overlay, [hull], (0, 255, 0, 50))  # Green with transparency
            
            # Blend the overlay with original image
            cv2.addWeighted(overlay, 0.3, workspace_overlay, 0.7, 0, workspace_overlay)
            
            # Draw boundary outline
            cv2.polylines(workspace_overlay, [hull], True, (0, 255, 0), 2)
            
            # Draw corner points
            for i, (px, py) in enumerate(pixel_points):
                cv2.circle(workspace_overlay, (px, py), 5, (255, 0, 0), -1)
                cv2.putText(workspace_overlay, f"P{i+1}", (px + 5, py - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Label workspace type
            cv2.putText(workspace_overlay, f"{workspace_type.upper()} WORKSPACE", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Calculate workspace center for additional info
            if len(pixel_points) >= 4:
                center_x = sum([p[0] for p in pixel_points]) // len(pixel_points)
                center_y = sum([p[1] for p in pixel_points]) // len(pixel_points)
                
                # Add area information
                area = cv2.contourArea(hull)
                cv2.putText(workspace_overlay, f"Area: {area:.0f} px^2", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return workspace_overlay

    def undistort_image(self, img, mtx, dist):
        """
        Correct image distortion - simplified version without ROI cropping
        Args:
            img: Original image
            mtx: Camera intrinsic matrix
            dist: Distortion coefficients
        Returns:
            undistorted_img: Corrected image
        """
        try:
            h, w = img.shape[:2]
            
            # Directly use undistort without ROI cropping
            undistorted_img = cv2.undistort(img, mtx, dist, None, mtx)
            
            return undistorted_img
            
        except Exception as e:
            print(f"Image undistortion failed: {e}")
            # Return original image as backup
            return img

    def setup_window(self):
        """Set up display windows"""
        # Create main detection window
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, self.window_size[0], self.window_size[1])
        
        # Create workspace window
        cv2.namedWindow(self.workspace_window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.workspace_window_name, 
                        self.workspace_window_size[0], 
                        self.workspace_window_size[1])
        
        # Center windows on screen
        try:
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
                screen_width, screen_height = 1920, 1080
        except:
            screen_width, screen_height = 1920, 1080
        
        # Position main window
        window_x = (screen_width - self.window_size[0]) // 2
        window_y = (screen_height - self.window_size[1]) // 2
        cv2.moveWindow(self.window_name, window_x, window_y)
        
        # Position workspace window next to main window
        workspace_x = window_x + self.window_size[0] + 10
        workspace_y = window_y
        cv2.moveWindow(self.workspace_window_name, workspace_x, workspace_y)
        
        print(f"Detection window positioned at ({window_x}, {window_y})")
        print(f"Workspace window positioned at ({workspace_x}, {workspace_y})")

    def signal_handler(self, sig, frame):
        """Handle Ctrl+C signal"""
        print("\nCtrl+C pressed! Shutting down...")
        self.shutdown_flag = True
        self.auto_detection_running = False
        self.cleanup()
        sys.exit(0)

    def select_target_class(self):
        """Select target class to sort"""
        print(f"\nSelect target class for Robot Arm:")
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
            choice = input(f"Enter your choice (1-5): ").strip()
            if choice in class_mapping:
                selected_class = class_mapping[choice]
                print(f"Robot Arm will sort: {selected_class}")
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

    def get_trash_bin_position(self):
        """Get trash bin position based on target type"""
        # Set different trash bin positions for different waste types
        bin_positions = {
            'cardboard': [0.017, -0.174, 0.241, 0.05, 1.5, 0],
            'glass': [0.017, -0.174, 0.241, 0.05, 1.5, 0],
            'metal': [0.017, -0.174, 0.241, 0.05, 1.5, 0],
            'paper': [0.017, -0.174, 0.241, 0.05, 1.5, 0],
            'plastic': [0.017, -0.174, 0.241, 0.05, 1.5, 0]
        }
        return bin_positions.get(self.target_class, [0.017, -0.174, 0.241, 0.05, 1.5, 0])

    def setup_delay_compensation(self):
        """Set up delay compensation parameters"""
        delay_seconds = self.total_delay_ms / 1000
        self.compensation_mm = self.conveyor_speed * delay_seconds
        self.compensation_pixels = self.compensation_mm / self.pixel_to_mm
        print(f"Compensation: {self.compensation_pixels:.1f} px")

    def activate_gripper(self):
        """Activate gripper"""
        print(f"Activating gripper...")  
        try:  
            self.robot.grasp_with_tool()  
            print("Gripper activated")  
            time.sleep(1)  
        except Exception as e:  
            print(f"Gripper activation failed: {e}")

    def deactivate_gripper(self):
        """Release gripper"""
        print(f"Releasing gripper...")  
        try:  
            self.robot.release_with_tool()  
            print("Gripper released")  
            time.sleep(0.5)  
        except Exception as e:  
            print(f"Gripper release failed: {e}")

    def get_z_heights(self, workspace_type):
        """Get height parameters"""
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
        """Image preprocessing - for model inference"""  
        img_resized = cv2.resize(img, (self.input_size, self.input_size))  
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)  
        img_input = img_rgb.transpose(2, 0, 1)[None, :, :, :] / 255.0  
        return img_input.astype(np.float32)  

    def resize_for_display(self, img, target_size=None):
        """Resize image for display, maintain aspect ratio"""
        if target_size is None:
            target_size = self.display_size
            
        h, w = img.shape[:2]
        target_w, target_h = target_size
        
        # Calculate scale factor
        scale = min(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Resize
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Create canvas of target size
        canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        
        # Place resized image in center of canvas
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

            # Calculate robot coordinates
            robot_x, robot_y, _ = self.pixel_to_robot(center_x, center_y, workspace_type)

            # Check if in workspace
            if workspace_type == "fixed":  
                in_workspace = self.in_workspace(robot_x, robot_y)
                in_conveyor = False  
            else:  
                in_workspace = self.in_conveyor_workspace(robot_x, robot_y)
                in_conveyor = in_workspace

            detections.append({  
                "class": class_name,  
                "confidence": confidence,  
                "bbox": [x1, y1, x2, y2],  
                "center_x": center_x,  
                "center_y": center_y,  
                "robot_x": robot_x,  
                "robot_y": robot_y,  
                "in_workspace": in_workspace,  
                "in_conveyor": in_conveyor  
            })  

        return detections  

    def draw_detections_on_image(self, img, detections, workspace_type="fixed"):
        """Draw detection results on image"""
        # First resize image for display
        display_img = self.resize_for_display(img)
        h, w = display_img.shape[:2]
        
        # Calculate scaling from original image to display image
        orig_h, orig_w = img.shape[:2]
        scale_x = w / orig_w
        scale_y = h / orig_h

        # Draw detection results
        for det in detections:
            # Adjust bounding box coordinates to display size
            x1_orig, y1_orig, x2_orig, y2_orig = det["bbox"]
            x1 = int(x1_orig * scale_x)
            y1 = int(y1_orig * scale_y)
            x2 = int(x2_orig * scale_x)
            y2 = int(y2_orig * scale_y)
            
            class_name = det["class"]
            confidence = det["confidence"]
            
            # Determine target area
            in_target_area = det["in_workspace"]

            # Set colors: target class - red, other classes - blue, not in target area - gray
            if not in_target_area: 
                color = (128, 128, 128)  # Gray - not in target area
                thickness = 2
            elif class_name == self.target_class:
                color = (0, 0, 255)  # Red - target class
                thickness = 3
            else:
                color = (255, 0, 0)  # Blue - other classes
                thickness = 2

            # Draw bounding box
            cv2.rectangle(display_img, (x1, y1), (x2, y2), color, thickness)

            # Draw label
            status = "" if in_target_area else " (Outside)" 
            label = f"{class_name}: {confidence:.2f}{status}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            # Label background
            cv2.rectangle(display_img, (x1, y1 - label_size[1] - 10),
                         (x1 + label_size[0], y1), color, -1)
            cv2.putText(display_img, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            # Draw center point
            cx_orig, cy_orig = det["center_x"], det["center_y"]
            cx = int(cx_orig * scale_x)
            cy = int(cy_orig * scale_y)
            cv2.circle(display_img, (cx, cy), 5, color, -1)

            # If target object and in target area, mark specially
            if class_name == self.target_class and in_target_area:
                cv2.putText(display_img, "TARGET", (cx - 20, cy - 15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Display statistics
        total_count = len(detections)
        target_count = len([d for d in detections if d["class"] == self.target_class])
        
        if workspace_type == "fixed":
            in_target_area_count = len([d for d in detections if d["in_workspace"]])
            target_in_area = len([d for d in detections if d["class"] == self.target_class and d["in_workspace"]])
            workspace_name = "Fixed Workspace"
        else:
            in_target_area_count = len([d for d in detections if d["in_workspace"]])
            target_in_area = len([d for d in detections if d["class"] == self.target_class and d["in_workspace"]])
            workspace_name = "Conveyor Workspace"

        cv2.putText(display_img, f"Robot Arm - {workspace_name}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(display_img, f"Target: {self.target_class}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(display_img, f"Total objects: {total_count}", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(display_img, f"In target area: {in_target_area_count}", (10, 120),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(display_img, f"Target objects: {target_count} (In area: {target_in_area})", (10, 150),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Add image status information
        cv2.putText(display_img, "Image: Undistorted (Corrected)", (10, 180),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        return display_img

    def show_detection_window(self, image, detections, workspace_type="fixed"):
        """Display detection window and workspace window"""
        # Ensure image is not None, create black image if None
        if image is None:
            print("Warning: Image is None, creating placeholder")
            image = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(image, "CAMERA NOT AVAILABLE", (50, 240), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow(self.window_name, image)
            cv2.imshow(self.workspace_window_name, image)
            key = cv2.waitKey(1) & 0xFF
            return key != ord('q') and key != 27
        
        try:
            # Draw detection results
            display_img = self.draw_detections_on_image(image, detections, workspace_type)
            
            # Display main detection window
            cv2.imshow(self.window_name, display_img)
            
            # Extract and display workspace overlay
            workspace_overlay = self.extract_image_workspace(image, workspace_type)
            if workspace_overlay is not None:
                # Resize workspace overlay for display
                workspace_display = self.resize_for_display(workspace_overlay, 
                                                           target_size=(400, 300))
                cv2.imshow(self.workspace_window_name, workspace_display)
            else:
                # Fallback: show the detection image
                cv2.imshow(self.workspace_window_name, display_img)
                
        except Exception as e:
            print(f"Error displaying windows: {e}")
            # Simple fallback
            if image is not None:
                cv2.imshow(self.window_name, image)
                cv2.imshow(self.workspace_window_name, image)

        # Window refresh - add exit check
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:  # 'q' or ESC key
            print("Exit key pressed!")
            self.shutdown_flag = True
            self.auto_detection_running = False
            return False
        return True

    def filter_detections_by_class_and_workspace(self, detections, workspace_type="fixed"):
        """Filter detections by class and workspace - only return target class in workspace"""
        filtered_detections = []
        for det in detections:
            if det["class"] == self.target_class and det["in_workspace"]:
                filtered_detections.append(det)
        return filtered_detections

    def camera_detection(self, workspace_type="fixed"):
        """Single camera detection using undistorted image"""
        if self.shutdown_flag:
            return []
            
        print("Starting camera detection with undistorted image...")
        
        # Initialize variables
        undistorted_image = None
        all_detections = []
        target_detections = []
        
        try:
            # Get undistorted image
            undistorted_image = self.get_camera_image()
            
            if undistorted_image is not None:
                print(f"Undistorted image shape: {undistorted_image.shape}")
                # Detect objects on undistorted image
                all_detections = self.detect_objects(undistorted_image, workspace_type)
                # Only pick target class in workspace
                target_detections = self.filter_detections_by_class_and_workspace(all_detections, workspace_type)
                print(f"Camera detected {len(all_detections)} objects, {len(target_detections)} {self.target_class} objects in workspace")
            else:
                print("Failed to get camera image")
        except Exception as e:
            print(f"Camera detection error: {e}")

        # Display detection window
        if not self.show_detection_window(undistorted_image, all_detections, workspace_type):
            return []

        return target_detections

    def pixel_to_robot(self, u, v, workspace_type="fixed"):
        """Pixel coordinates to robot coordinates"""
        pixel = np.array([u, v, 1])
        if workspace_type == "fixed":
            world = np.dot(self.fixed_transform_matrix, pixel)
        else:
            world = np.dot(self.conveyor_transform_matrix, pixel)
        z = self.get_z_heights(workspace_type)["pick"]
        return world[0], world[1], z

    def in_workspace(self, x, y):  
        """Check if coordinates are in fixed workspace"""
        return self.X_MIN <= x <= self.X_MAX and self.Y_MIN <= y <= self.Y_MAX  

    def in_conveyor_workspace(self, x, y):  
        """Check if coordinates are in conveyor workspace"""
        return self.CONVEYOR_X_MIN <= x <= self.CONVEYOR_X_MAX and self.CONVEYOR_Y_MIN <= y <= self.CONVEYOR_Y_MAX  

    def move_to_observe(self, workspace_type="fixed"):  
        """Move to observation position"""
        if self.shutdown_flag:
            return
            
        print(f" Moving to {workspace_type} workspace observe position...")  
        try:  
            if workspace_type == "fixed":  
                observe_pose = PoseObject(*self.FIXED_OBSERVE_POS)
                # Use move() instead of move_pose()
                self.robot.move_pose(observe_pose)
            else: # conveyor  
                observe_pose = PoseObject(*self.CONVEYOR_OBSERVE_POS)
                self.robot.move_pose(observe_pose)
            time.sleep(1)  
        except Exception as e:  
            print(f" Move to observe position failed: {e}")  

    def control_conveyor(self, action="start"):  
        """Control conveyor"""  
        if self.shutdown_flag:
            return False
            
        try:  
            if action == "start":  
                self.robot.run_conveyor(self.conveyor_id, speed=self.conveyor_speed, direction=ConveyorDirection.FORWARD)  
                print(f" Conveyor started at speed {self.conveyor_speed}")  
                self.conveyor_running = True
                self.conveyor_stop_requested = False
            elif action == "stop":  
                self.robot.stop_conveyor(self.conveyor_id)  
                print(" Conveyor stopped")  
                self.conveyor_running = False
                self.conveyor_stop_requested = True
            time.sleep(1)  
            return True  
        except Exception as e:  
            print(f" Conveyor control failed: {e}")  
            return False  

    def start_batch_pick_operation(self, total_tasks, workspace_type="conveyor"):
        """Start batch pick operation"""
        if workspace_type == "conveyor" and total_tasks > 0:
            print(f"Starting batch pick operation with {total_tasks} tasks")
            self.batch_pick_count = total_tasks
            # Stop conveyor
            self.control_conveyor("stop")
            return True
        return False

    def complete_batch_pick_operation(self, workspace_type="conveyor"):
        """Complete batch pick operation"""
        if workspace_type == "conveyor" and self.batch_pick_count > 0:
            self.batch_pick_count -= 1
            print(f"Batch pick operation completed. Remaining tasks: {self.batch_pick_count}")
            
            # If all batch tasks completed, restart conveyor
            if self.batch_pick_count == 0:
                print("All batch pick operations completed. Restarting conveyor...")
                self.control_conveyor("start")
                return True
        return False

    def pick_and_place_object(self, detection, workspace_type="fixed"):  
        """Pick and place object"""  
        if self.shutdown_flag:
            return False
            
        with self.operation_lock:
            x, y = detection["robot_x"], detection["robot_y"]
            trash_bin_pos = self.TRASH_BIN_POS

            # Conveyor delay compensation
            if workspace_type == "conveyor":
                original_pixel_x = detection["center_x"]
                compensated_pixel_x = original_pixel_x - self.compensation_pixels
                # Recalculate robot coordinates
                x, y, _ = self.pixel_to_robot(compensated_pixel_x, detection["center_y"], workspace_type)
                print(f"Latency compensation:")
                print(f"Compensation distance: +{self.compensation_pixels:.1f} pixels")
            print(f"Targeting {detection['class']} at coordinates: ({x:.3f}, {y:.3f})")  

            z_heights = self.get_z_heights(workspace_type)  

            try:  
                # Prepare tool
                self.deactivate_gripper()
                time.sleep(0.5)

                # Pick action
                approach_pose = PoseObject(x, y, z_heights["approach"], 0.0, 1.57, 0.0)
                pick_pose = PoseObject(x, y, z_heights["pick"], 0.0, 1.57, 0.0)
                safe_pose = PoseObject(x, y, z_heights["safe"], 0.0, 1.57, 0.0)

                print(" Moving to approach position...")
                self.robot.move_pose(approach_pose)  # Use move() instead of move_pose()

                print(" Moving to pick position...")
                self.robot.move_pose(pick_pose)  # Use move() instead of move_pose()

                # Pick
                self.activate_gripper()
                time.sleep(1)

                print(" Moving to safe height...")
                self.robot.move_pose(safe_pose)  # Use move() instead of move_pose()

                # Move to trash bin position
                print(" Moving to trash bin...")
                trash_pose = PoseObject(*trash_bin_pos)
                self.robot.move_pose(trash_pose)  # Use move() instead of move_pose()

                # Place
                self.deactivate_gripper()
                time.sleep(0.5)

                # Move to safe height
                safe_trash_pose = PoseObject(
                    trash_bin_pos[0],
                    trash_bin_pos[1],
                    z_heights["safe"],
                    0.0, 1.57, 0.0
                )
                self.robot.move_pose(safe_trash_pose)  # Use move() instead of move_pose()
                print(f"{detection['class']} placed successfully.")

                # Complete batch pick operation
                if workspace_type == "conveyor":
                    self.complete_batch_pick_operation(workspace_type)

                # Return to observation position
                if workspace_type == "conveyor":
                    self.move_to_observe("conveyor")

                return True

            except Exception as e:
                print(f"Pick and place failed: {e}")
                # Complete batch pick operation (even if failed)
                if workspace_type == "conveyor":
                    self.complete_batch_pick_operation(workspace_type)
                return False

    def move_to_safe_position(self):
        """Move to safe position"""
        try:
            safe_pose = PoseObject(*self.SAFE_POS)
            self.robot.move_pose(safe_pose)  # Use move() instead of move_pose()
            print("Robot moved to safe position")
        except Exception as e:
            print(f"Move to safe position failed: {e}")

    def coordinate_operation(self, detection_results, workspace_type="fixed"):
        """Coordinate operation"""
        if self.shutdown_flag:
            return 0
            
        tasks = detection_results

        print(f"Tasks ({self.target_class}): {len(tasks)}")

        # Calculate total tasks
        total_tasks = len(tasks)
        print(f"Total tasks to process: {total_tasks}")

        # If conveyor mode, start batch pick operation
        if workspace_type == "conveyor" and total_tasks > 0:
            self.start_batch_pick_operation(total_tasks, workspace_type)

        processed_count = 0
        
        # Process tasks
        for i, task in enumerate(tasks):
            if self.shutdown_flag:
                break
            print(f"\nProcessing {self.target_class} object {i+1}/{len(tasks)}")
            success = self.pick_and_place_object(task, workspace_type)
            if success:
                processed_count += 1
            time.sleep(1)

        return processed_count

    def auto_detect_and_sort(self, workspace_type="fixed"):
        """Automatic sorting"""
        if self.shutdown_flag:
            return
            
        print(f"\n=== Single Arm {workspace_type.title()} Workspace ===")
        print(f"Target: {self.target_class}")
        # Move to observation position
        self.move_to_observe(workspace_type)

        # Detection
        detection_results = self.camera_detection(workspace_type)
        # Coordinate operation
        processed_count = self.coordinate_operation(detection_results, workspace_type)

        print(f"\nSorting completed! Processed {processed_count} objects.")

    def auto_detect_and_sort_conveyor(self):
        """Conveyor automatic sorting"""
        if self.shutdown_flag:
            return
            
        print(f"\n=== Conveyor Sorting ===")
        print(f"Target: {self.target_class}")
        # Start conveyor
        if not self.control_conveyor("start"):
            print("Failed to start conveyor")
            return

        # Move to observation position
        self.move_to_observe("conveyor")

        self.auto_detection_running = True
        total_processed = 0
        cycle_count = 0

        try:
            while self.auto_detection_running and not self.shutdown_flag:
                cycle_count += 1
                print(f"\n--- Detection Cycle {cycle_count} ---")
                
                # Ensure conveyor is running (unless batch pick operation in progress)
                if not self.conveyor_running and self.batch_pick_count == 0:
                    print("No batch operations, ensuring conveyor is running...")
                    self.control_conveyor("start")
                
                # Detection
                detection_results = self.camera_detection("conveyor")
                
                # If targets detected, execute pick operation
                if detection_results:
                    print(f"Targets detected! Count: {len(detection_results)}")
                    # Coordinate operation (conveyor will be automatically stopped in pick_and_place_object)
                    processed_this_cycle = self.coordinate_operation(detection_results, "conveyor")
                    total_processed += processed_this_cycle
                    print(f"Cycle {cycle_count}: Processed {processed_this_cycle} objects (Total: {total_processed})")
                else:
                    print(f"Cycle {cycle_count}: No targets detected, continuing conveyor operation...")
                    processed_this_cycle = 0
                
                # Brief delay
                time.sleep(1)
                
        except KeyboardInterrupt:
            print("Detection interrupted by user")
        except Exception as e:
            print(f"Detection error: {e}")
        finally:
            self.auto_detection_running = False
            # Ensure conveyor stops
            self.emergency_stop_conveyor()
            print(f"Conveyor sorting finished. Total processed: {total_processed}")

    def emergency_stop_conveyor(self):
        """Emergency stop conveyor"""
        print("Emergency stopping conveyor...")
        try:
            # Multiple attempts to stop conveyor
            for i in range(3):
                try:
                    self.robot.stop_conveyor(self.conveyor_id)
                    print(f"Conveyor stop attempt {i+1}: Success")
                    self.conveyor_running = False
                    self.batch_pick_count = 0  # Reset batch count
                    break
                except Exception as e:
                    print(f"Conveyor stop attempt {i+1} failed: {e}")
                    time.sleep(0.5)
        except Exception as e:
            print(f"Emergency conveyor stop failed: {e}")

    def get_camera_image(self):
        """Get camera image with distortion correction"""
        try:
            # Getting image
            img_compressed = self.robot.get_img_compressed()
            
            # Uncompressing image
            img_raw = uncompress_image(img_compressed)
            
            # Get camera intrinsics and undistort image
            mtx, dist = self.robot.get_camera_intrinsics()
            img_undistort = self.undistort_image(img_raw, mtx, dist)
            
            return img_undistort
        except Exception as e:
            print(f"Failed to get camera image: {e}")
            return None

    def test_camera_connection(self):  
        """Camera connection test"""  
        print("Testing camera connection...")  
        success = False
        try:  
            image = self.get_camera_image()  
            if image is not None:  
                print(f"Camera connection successful! Image size: {image.shape}")  
                success = True
            else:  
                print("Camera connection failed")  
        except Exception as e:  
            print(f"Camera test failed: {e}")  

        return success

    def debug_camera_connection(self):
        """Debug camera connection"""
        print("\n=== Debugging Camera Connection ===")
        
        # Test camera
        print("Testing camera...")
        try:
            img = self.get_camera_image()
            if img is not None:
                print(f"Undistorted image OK - Shape: {img.shape}")
                
                # Test workspace extraction
                fixed_workspace = self.extract_image_workspace(img, "fixed")
                conveyor_workspace = self.extract_image_workspace(img, "conveyor")
                
                # Display images
                cv2.imshow("Camera Debug - Original", img)
                if fixed_workspace is not None:
                    cv2.imshow("Fixed Workspace", fixed_workspace)
                if conveyor_workspace is not None:
                    cv2.imshow("Conveyor Workspace", conveyor_workspace)
                    
                cv2.waitKey(3000)  # Display for 3 seconds
                cv2.destroyAllWindows()
            else:
                print("Camera FAILED - No image received")
        except Exception as e:
            print(f"Camera ERROR: {e}")

    def cleanup(self):  
        """Clean up resources"""  
        print("Cleaning up resources...")
        self.auto_detection_running = False
        self.shutdown_flag = True
        
        # First stop conveyor
        print("Stopping conveyor...")
        self.emergency_stop_conveyor()
        
        # Close windows
        print("Closing windows...")
        try:
            cv2.destroyAllWindows()
            cv2.waitKey(1)  # Ensure windows close
        except:
            pass
            
        # Disconnect robot
        print("Closing robot connection...")
        try:  
            self.robot.close_connection()  
            print("Robot connection closed")
        except Exception as e:  
            print(f"Error closing robot: {e}")

def main():  
    # Configuration parameters
    ROBOT_IP = "192.168.0.109"  # Single robot arm IP address
    ONNX_MODEL_PATH = "/home/ned2/Desktop/Waste sorting/code/best.onnx"
    # Calibration configuration files
    CALIBRATION_CONFIGS = {
        'fixed': "/home/ned2/Desktop/Waste sorting/code/1203 fixedcalibration_matrix.npy",
        'conveyor': "/home/ned2/Desktop/Waste sorting/code/1203 conveyorcalibration_matrix1.npy"
    }
    # Workspace configuration files
    WORKSPACE_CONFIGS = {
        'fixed': "/home/ned2/Desktop/Waste sorting/code/workspace_WasteDetector.json",
        'conveyor': "/home/ned2/Desktop/Waste sorting/code/1105robot81workspace_conveyor_WasteDetector.json"
    }

    print("=== Single Arm Sorting Robot System ===")
    print("Using undistorted images for object detection")
    print("Features:")
    print("- Workspace extraction and visualization")
    print("- Dual window display (detection + workspace)")
    print("- Press 'q' to quit current operation")
    print("- Press Ctrl+C for emergency stop")

    # Initialize single arm system
    robot_system = SingleArmSortingRobot(ROBOT_IP, ONNX_MODEL_PATH, CALIBRATION_CONFIGS, WORKSPACE_CONFIGS)

    try:
        # Test camera connection
        if robot_system.test_camera_connection():  
            # Run debugging
            robot_system.debug_camera_connection()
            
            while not robot_system.shutdown_flag:  
                print("\n" + "="*50)  
                print("Select mode:")  
                print("1 - Fixed Workspace: Detection & Sort")  
                print("2 - Conveyor: Continuous Detection & Sort")  
                print("3 - Test Workspace Visualization")  
                print("4 - Exit")  
                print("Press 'q' in window to stop current operation")
                print("Press Ctrl+C to emergency stop")

                choice = input("Enter your choice (1-4): ").strip()  

                if choice == "1":  
                    robot_system.auto_detect_and_sort("fixed")  
                elif choice == "2":  
                    robot_system.auto_detect_and_sort_conveyor()  
                elif choice == "3":
                    print("\n=== Testing Workspace Visualization ===")
                    img = robot_system.get_camera_image()
                    if img is not None:
                        # Test both workspaces
                        fixed_workspace = robot_system.extract_image_workspace(img, "fixed")
                        conveyor_workspace = robot_system.extract_image_workspace(img, "conveyor")
                        
                        cv2.imshow("Original Image", img)
                        cv2.imshow("Fixed Workspace Visualization", fixed_workspace)
                        cv2.imshow("Conveyor Workspace Visualization", conveyor_workspace)
                        
                        print("Press any key to close test windows...")
                        cv2.waitKey(0)
                        cv2.destroyAllWindows()
                elif choice == "4":  
                    print(" Exiting...")  
                    break  
                else:  
                    print(" Invalid choice, please try again.")  
        else:  
            print(" Unable to connect to camera")

    except KeyboardInterrupt:
        print("\nProgram interrupted by user")
    except Exception as e:  
        print(f" Program runtime error: {e}")  
    finally:  
        robot_system.cleanup()  
        print(" Program finished")  

if __name__ == "__main__":  
    main()