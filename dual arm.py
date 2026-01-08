import cv2  
import numpy as np  
import time  
import json  
from pyniryo import *  
import onnxruntime as ort  
import threading  
import signal
import sys

# ==================== 1. 系统初始化模块 ====================
class SystemInitializer:
    def __init__(self, robot_ips, onnx_model_path, calibration_configs, workspace_configs):
        # Set up signal handling
        signal.signal(signal.SIGINT, self.signal_handler)
        self.shutdown_flag = False
        
        # 配置参数
        self.robot_ips = robot_ips
        self.onnx_model_path = onnx_model_path
        self.calibration_configs = calibration_configs
        self.workspace_configs = workspace_configs
        
        # 初始化状态变量
        self.gamma_robot = None
        self.alpha_robot = None
        self.gamma_target_class = None
        self.alpha_target_class = None
        self.gamma_tool = None
        self.alpha_tool = None
        self.gamma_fixed_transform_matrix = None
        self.gamma_conveyor_transform_matrix = None
        self.alpha_fixed_transform_matrix = None
        self.alpha_conveyor_transform_matrix = None
        
    def initialize_robots(self):
        """初始化两台机器人连接"""
        print("Connecting to robots...")  
        self.gamma_robot = NiryoRobot(self.robot_ips[0])  # Gamma机械臂
        self.alpha_robot = NiryoRobot(self.robot_ips[1])  # Alpha机械臂
        
        print("Calibrating Gamma robot...")  
        self.gamma_robot.calibrate_auto()  
        print("Calibrating Alpha robot...")  
        self.alpha_robot.calibrate_auto()
        
        return self.gamma_robot, self.alpha_robot
    
    def select_target_classes(self):
        """为两台机器人选择目标分类"""
        print("\n" + "="*50)  
        print("=== Object Type Selection ===")  
        self.gamma_target_class = self.select_target_class("Gamma")  
        self.alpha_target_class = self.select_target_class("Alpha")  
        return self.gamma_target_class, self.alpha_target_class
    
    def select_grippers(self):
        """为两台机器人选择夹具类型"""
        print("\n" + "="*50)  
        print("=== Gripper Selection ===")  
        print(f"\n--- Gamma Robot ({self.gamma_target_class}) Gripper Selection ---")  
        self.gamma_tool = self.select_gripper(self.gamma_robot)  
        print(f"\n--- Alpha Robot ({self.alpha_target_class}) Gripper Selection ---")  
        self.alpha_tool = self.select_gripper(self.alpha_robot)
        return self.gamma_tool, self.alpha_tool
    
    def select_target_class(self, arm_name):
        """为指定机器人选择目标类别"""
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
        """为指定机器人选择夹具"""
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
    
    def load_calibration_matrices(self):
        """加载两台机器人的标定矩阵"""
        print("Loading calibration matrices for both arms...")  
        # Gamma标定矩阵
        self.gamma_fixed_transform_matrix = np.load(self.calibration_configs['gamma_fixed'])
        self.gamma_conveyor_transform_matrix = np.load(self.calibration_configs['gamma_conveyor'])
        # Alpha标定矩阵
        self.alpha_fixed_transform_matrix = np.load(self.calibration_configs['alpha_fixed'])
        self.alpha_conveyor_transform_matrix = np.load(self.calibration_configs['alpha_conveyor'])
        print("Calibration matrices loaded successfully.")
        
        return (self.gamma_fixed_transform_matrix, self.gamma_conveyor_transform_matrix,
                self.alpha_fixed_transform_matrix, self.alpha_conveyor_transform_matrix)
    
    def load_workspaces(self):
        """加载两台机器人的工作空间"""
        print("Loading workspaces for both arms...")  
        workspaces = {}
        
        # Gamma固定工作空间
        with open(self.workspace_configs['gamma_fixed'], "r") as f:  
            gamma_workspace = json.load(f)  
            gamma_x_list = [p["x"] for p in gamma_workspace["points"]]  
            gamma_y_list = [p["y"] for p in gamma_workspace["points"]]  
            GAMMA_X_MIN, GAMMA_X_MAX = min(gamma_x_list), max(gamma_x_list)  
            GAMMA_Y_MIN, GAMMA_Y_MAX = min(gamma_y_list), max(gamma_y_list)  
            workspaces['gamma_fixed'] = {
                'points': gamma_workspace["points"],
                'X_MIN': GAMMA_X_MIN,
                'X_MAX': GAMMA_X_MAX,
                'Y_MIN': GAMMA_Y_MIN,
                'Y_MAX': GAMMA_Y_MAX
            }
        print(f"Gamma Fixed Workspace X range: {GAMMA_X_MIN:.3f} ~ {GAMMA_X_MAX:.3f}")  
        print(f"Gamma Fixed Workspace Y range: {GAMMA_Y_MIN:.3f} ~ {GAMMA_Y_MAX:.3f}")  

        # Alpha固定工作空间
        with open(self.workspace_configs['alpha_fixed'], "r") as f:  
            alpha_workspace = json.load(f)  
            alpha_x_list = [p["x"] for p in alpha_workspace["points"]]  
            alpha_y_list = [p["y"] for p in alpha_workspace["points"]]  
            ALPHA_X_MIN, ALPHA_X_MAX = min(alpha_x_list), max(alpha_x_list)  
            ALPHA_Y_MIN, ALPHA_Y_MAX = min(alpha_y_list), max(alpha_y_list)  
            workspaces['alpha_fixed'] = {
                'points': alpha_workspace["points"],
                'X_MIN': ALPHA_X_MIN,
                'X_MAX': ALPHA_X_MAX,
                'Y_MIN': ALPHA_Y_MIN,
                'Y_MAX': ALPHA_Y_MAX
            }
        print(f"Alpha Fixed Workspace X range: {ALPHA_X_MIN:.3f} ~ {ALPHA_X_MAX:.3f}")  
        print(f"Alpha Fixed Workspace Y range: {ALPHA_Y_MIN:.3f} ~ {ALPHA_Y_MAX:.3f}")  

        # Gamma传送带工作空间
        print("Loading conveyor workspaces for both arms...") 
        with open(self.workspace_configs['gamma_conveyor'], "r") as f:  
            gamma_conveyor_workspace = json.load(f)  
            gamma_conveyor_x_list = [p["x"] for p in gamma_conveyor_workspace["points"]]  
            gamma_conveyor_y_list = [p["y"] for p in gamma_conveyor_workspace["points"]]  
            GAMMA_CONVEYOR_X_MIN, GAMMA_CONVEYOR_X_MAX = min(gamma_conveyor_x_list), max(gamma_conveyor_x_list)  
            GAMMA_CONVEYOR_Y_MIN, GAMMA_CONVEYOR_Y_MAX = min(gamma_conveyor_y_list), max(gamma_conveyor_y_list)  
            workspaces['gamma_conveyor'] = {
                'points': gamma_conveyor_workspace["points"],
                'X_MIN': GAMMA_CONVEYOR_X_MIN,
                'X_MAX': GAMMA_CONVEYOR_X_MAX,
                'Y_MIN': GAMMA_CONVEYOR_Y_MIN,
                'Y_MAX': GAMMA_CONVEYOR_Y_MAX
            }
        print(f"Gamma Conveyor Workspace X range: {GAMMA_CONVEYOR_X_MIN:.3f} ~ {GAMMA_CONVEYOR_X_MAX:.3f}")  
        print(f"Gamma Conveyor Workspace Y range: {GAMMA_CONVEYOR_Y_MIN:.3f} ~ {GAMMA_CONVEYOR_Y_MAX:.3f}")  

        # Alpha传送带工作空间
        with open(self.workspace_configs['alpha_conveyor'], "r") as f:  
            alpha_conveyor_workspace = json.load(f)  
            alpha_conveyor_x_list = [p["x"] for p in alpha_conveyor_workspace["points"]]  
            alpha_conveyor_y_list = [p["y"] for p in alpha_conveyor_workspace["points"]]  
            ALPHA_CONVEYOR_X_MIN, ALPHA_CONVEYOR_X_MAX = min(alpha_conveyor_x_list), max(alpha_conveyor_x_list)  
            ALPHA_CONVEYOR_Y_MIN, ALPHA_CONVEYOR_Y_MAX = min(alpha_conveyor_y_list), max(alpha_conveyor_y_list)  
            workspaces['alpha_conveyor'] = {
                'points': alpha_conveyor_workspace["points"],
                'X_MIN': ALPHA_CONVEYOR_X_MIN,
                'X_MAX': ALPHA_CONVEYOR_X_MAX,
                'Y_MIN': ALPHA_CONVEYOR_Y_MIN,
                'Y_MAX': ALPHA_CONVEYOR_Y_MAX
            }
        print(f"Alpha Conveyor Workspace X range: {ALPHA_CONVEYOR_X_MIN:.3f} ~ {ALPHA_CONVEYOR_X_MAX:.3f}")  
        print(f"Alpha Conveyor Workspace Y range: {ALPHA_CONVEYOR_Y_MIN:.3f} ~ {ALPHA_CONVEYOR_Y_MAX:.3f}")
        
        return workspaces
    
    def get_config_summary(self):
        """获取配置摘要"""
        print(f"\n=== Configuration Summary ===")
        print(f"Gamma Arm: {self.gamma_target_class}")
        print(f"Alpha Arm: {self.alpha_target_class}")
        print("Dual Arm System initialization complete")
    
    def signal_handler(self, sig, frame):
        """处理Ctrl+C信号"""
        print("\nCtrl+C pressed! Shutting down...")
        self.shutdown_flag = True
        sys.exit(0)

# ==================== 2. 视觉检测模块 ====================
class VisionDetector:
    def __init__(self, onnx_model_path, class_names, input_size=640, conf_thresh=0.5, nms_thresh=0.45):
        self.onnx_model_path = onnx_model_path
        self.class_names = class_names
        self.input_size = input_size
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        
        # 加载模型
        print("Loading ONNX model...")  
        self.session = ort.InferenceSession(onnx_model_path, providers=['CPUExecutionProvider'])  
        self.input_name = self.session.get_inputs()[0].name
        print("Model loaded successfully.")
    
    def preprocess(self, img):  
        """图像预处理"""
        img_resized = cv2.resize(img, (self.input_size, self.input_size))  
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)  
        img_input = img_rgb.transpose(2, 0, 1)[None, :, :, :] / 255.0  
        return img_input.astype(np.float32)
    
    def detect_objects(self, img):  
        """检测所有物体"""
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

            detections.append({  
                "class": class_name,  
                "confidence": confidence,  
                "bbox": [x1, y1, x2, y2],  
                "center_x": center_x,  
                "center_y": center_y
            })  

        return detections
    
    def filter_by_class(self, detections, target_class):
        """按类别过滤检测结果"""
        return [d for d in detections if d["class"] == target_class]

# ==================== 3. 坐标转换模块 ====================
class CoordinateTransformer:
    def __init__(self, gamma_fixed_matrix, gamma_conveyor_matrix, 
                 alpha_fixed_matrix, alpha_conveyor_matrix, workspaces):
        self.gamma_fixed_transform_matrix = gamma_fixed_matrix
        self.gamma_conveyor_transform_matrix = gamma_conveyor_matrix
        self.alpha_fixed_transform_matrix = alpha_fixed_matrix
        self.alpha_conveyor_transform_matrix = alpha_conveyor_matrix
        self.workspaces = workspaces
        
    def pixel_to_robot(self, u, v, arm_type, workspace_type="fixed"):
        """像素坐标转机器人坐标"""
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
        return world[0], world[1]
    
    def in_workspace(self, x, y, arm_type, workspace_type="fixed"):  
        """检查坐标是否在指定工作空间内"""
        workspace_key = f"{arm_type}_{workspace_type}"
        ws = self.workspaces[workspace_key]
        return ws['X_MIN'] <= x <= ws['X_MAX'] and ws['Y_MIN'] <= y <= ws['Y_MAX']
    
    def add_robot_coordinates(self, detections, workspace_type="fixed"):
        """为检测结果添加两台机器人的坐标和工作空间状态"""
        for det in detections:
            # Gamma机器人坐标
            gamma_x, gamma_y = self.pixel_to_robot(det["center_x"], det["center_y"], 'gamma', workspace_type)
            det["gamma_robot_x"] = gamma_x
            det["gamma_robot_y"] = gamma_y
            det["in_gamma_workspace"] = self.in_workspace(gamma_x, gamma_y, 'gamma', workspace_type)
            
            # Alpha机器人坐标
            alpha_x, alpha_y = self.pixel_to_robot(det["center_x"], det["center_y"], 'alpha', workspace_type)
            det["alpha_robot_x"] = alpha_x
            det["alpha_robot_y"] = alpha_y
            det["in_alpha_workspace"] = self.in_workspace(alpha_x, alpha_y, 'alpha', workspace_type)
            
            # 综合状态
            det["in_workspace"] = det["in_gamma_workspace"] or det["in_alpha_workspace"]
            det["in_conveyor"] = (workspace_type == "conveyor")
        return detections
    
    def filter_by_arm_workspace(self, detections, target_class, arm_type, workspace_type="fixed"):
        """按指定机器人的工作空间过滤检测结果"""
        filtered_detections = []
        for det in detections:
            if det["class"] == target_class:
                if arm_type == 'gamma':
                    in_workspace = det["in_gamma_workspace"]
                else:  # alpha
                    in_workspace = det["in_alpha_workspace"]
                
                if in_workspace:
                    filtered_detections.append(det)
        return filtered_detections

# ==================== 4. 抓取操作模块 ====================
class GraspingOperator:
    def __init__(self, gamma_robot, alpha_robot, gamma_tool, alpha_tool, 
                 gamma_target_class, alpha_target_class):
        self.gamma_robot = gamma_robot
        self.alpha_robot = alpha_robot
        self.gamma_tool = gamma_tool
        self.alpha_tool = alpha_tool
        self.gamma_target_class = gamma_target_class
        self.alpha_target_class = alpha_target_class
        
        # 高度参数
        self.FIXED_Z_PICK = 0.09  
        self.FIXED_Z_APPROACH = 0.15  
        self.FIXED_Z_SAFE = 0.20  
        self.CONVEYOR_Z_PICK = 0.11 
        self.CONVEYOR_Z_APPROACH = 0.18  
        self.CONVEYOR_Z_SAFE = 0.23  
        
        # 位置定义
        self.GAMMA_TRASH_BIN_POS = self.get_trash_bin_position('gamma')
        self.ALPHA_TRASH_BIN_POS = self.get_trash_bin_position('alpha')
        
        # 观察位置
        self.GAMMA_FIXED_OBSERVE_POS = [0.191, -0.010, 0.298, -3.127, 1.333, -3.112]
        self.ALPHA_FIXED_OBSERVE_POS = [0.191, -0.010, 0.298, -3.131, 1.339, -3.115]
        self.GAMMA_CONVEYOR_OBSERVE_POS = [0.251, 0.005, 0.254, 2.978, 1.311, 3.003]
        self.ALPHA_CONVEYOR_OBSERVE_POS = [0.191, -0.010, 0.298, -3.131, 1.339, -3.115]
        
        # 安全位置
        self.GAMMA_SAFE_POS = [0.140, -0.000, 0.203, 0.000, 0.753, -0.001]
        self.ALPHA_SAFE_POS = [0.140, -0.000, 0.203, -0.003, 0.757, -0.001]
        
        # 传送带设置
        self.conveyor_id = self.alpha_robot.set_conveyor()
        self.conveyor_speed = 50
        self.conveyor_running = False
        
        # 延迟补偿参数
        self.gamma_total_delay_ms = -670
        self.alpha_total_delay_ms = 700
        self.pixel_to_mm = 0.5000
        self.setup_delay_compensation()
    
    def get_trash_bin_position(self, arm_type):
        """根据目标类型获取垃圾桶位置"""
        target_class = self.gamma_target_class if arm_type == 'gamma' else self.alpha_target_class
        bin_positions = {
            'cardboard': [0.030, 0.232, 0.253, -1.658, 1.477, -1.625],  # robot GAMMA 79 
            'glass': [0.079, 0.256, 0.215, -2.852, 1.435, -1.555],  # robot ALPHA 81 
            'metal': [0.079, 0.256, 0.215, -2.852, 1.435, -1.555],  # robot ALPHA 81 
            'paper': [0.079, 0.256, 0.215, -2.852, 1.435, -1.555],  # robot ALPHA 81 
            'plastic': [0.030, 0.232, 0.253, -1.658, 1.477, -1.625]  # robot GAMMA 79 
        }
        return bin_positions.get(target_class, [0.079, 0.256, 0.215, -2.852, 1.435, -1.555])
    
    def setup_delay_compensation(self):
        """设置延迟补偿参数"""
        # Gamma延迟补偿
        gamma_delay_seconds = self.gamma_total_delay_ms / 1000
        self.gamma_compensation_mm = self.conveyor_speed * gamma_delay_seconds
        self.gamma_compensation_pixels = self.gamma_compensation_mm / self.pixel_to_mm
        
        # Alpha延迟补偿
        alpha_delay_seconds = self.alpha_total_delay_ms / 1000
        self.alpha_compensation_mm = self.conveyor_speed * alpha_delay_seconds
        self.alpha_compensation_pixels = self.alpha_compensation_mm / self.pixel_to_mm
        
        print(f"Gamma arm compensation: {self.gamma_compensation_pixels:.1f} px")
        print(f"Alpha arm compensation: {self.alpha_compensation_pixels:.1f} px")
    
    def get_z_heights(self, workspace_type):
        """获取高度参数"""
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
    
    def activate_gripper(self, robot, tool):
        """激活夹具"""
        print(f"Activating gripper...")  
        try:  
            robot.grasp_with_tool()  
            print("Gripper activated")  
            time.sleep(1)  
        except Exception as e:  
            print(f"Gripper activation failed: {e}")
    
    def deactivate_gripper(self, robot, tool):
        """释放夹具"""
        print(f"Releasing gripper...")  
        try:  
            robot.release_with_tool()  
            print("Gripper released")  
            time.sleep(0.5)  
        except Exception as e:  
            print(f"Gripper release failed: {e}")
    
    def move_to_observe(self, workspace_type="fixed", arm_type=None):  
        """移动到观察位置"""
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
    
    def move_to_safe_position(self, arm_type):
        """移动到安全位置"""
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
    
    def control_conveyor(self, action="start"):  
        """控制传送带"""  
        try:  
            if action == "start":  
                self.alpha_robot.run_conveyor(self.conveyor_id, speed=self.conveyor_speed, direction=ConveyorDirection.FORWARD)  
                print(f" Conveyor started at speed {self.conveyor_speed}")  
                self.conveyor_running = True
            elif action == "stop":  
                self.alpha_robot.stop_conveyor(self.conveyor_id)  
                print(" Conveyor stopped")  
                self.conveyor_running = False
            time.sleep(1)  
            return True  
        except Exception as e:  
            print(f" Conveyor control failed: {e}")  
            return False
    
    def emergency_stop_conveyor(self):
        """紧急停止传送带"""
        print("Emergency stopping conveyor...")
        try:
            # 多次尝试停止传送带
            for i in range(3):
                try:
                    self.alpha_robot.stop_conveyor(self.conveyor_id)
                    print(f"Conveyor stop attempt {i+1}: Success")
                    self.conveyor_running = False
                    break
                except Exception as e:
                    print(f"Conveyor stop attempt {i+1} failed: {e}")
                    time.sleep(0.5)
        except Exception as e:
            print(f"Emergency conveyor stop failed: {e}")
    
    def pick_and_place_object(self, detection, arm_type, workspace_type="fixed", transformer=None):  
        """抓取并放置物体"""
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

        # 传送带延迟补偿
        if workspace_type == "conveyor" and transformer:
            original_pixel_x = detection["center_x"]
            compensated_pixel_x = original_pixel_x - compensation_pixels
            # 重新计算机器人坐标
            x, y = transformer.pixel_to_robot(compensated_pixel_x, detection["center_y"], arm_type, workspace_type)
            print(f" {arm_type.capitalize()} arm latency compensation:")
            print(f" Compensation distance: +{compensation_pixels:.1f} pixels")
        print(f"{arm_type.capitalize()} Arm targeting {detection['class']} at coordinates: ({x:.3f}, {y:.3f})")  

        z_heights = self.get_z_heights(workspace_type)  

        try:  
            # 准备夹具
            self.deactivate_gripper(robot, tool)
            time.sleep(0.5)

            # 抓取动作
            approach_pose = PoseObject(x, y, z_heights["approach"], 0.0, 1.57, 0.0)
            pick_pose = PoseObject(x, y, z_heights["pick"], 0.0, 1.57, 0.0)
            safe_pose = PoseObject(x, y, z_heights["safe"], 0.0, 1.57, 0.0)

            print(" Moving to approach position...")
            robot.move_pose(approach_pose)

            print(" Moving to pick position...")
            robot.move_pose(pick_pose)

            # 抓取
            self.activate_gripper(robot, tool)
            time.sleep(1)

            print(" Moving to safe height...")
            robot.move_pose(safe_pose)

            # 移动到垃圾桶位置
            print(" Moving to trash bin...")
            trash_pose = PoseObject(*trash_bin_pos)
            robot.move_pose(trash_pose)

            # 放置
            self.deactivate_gripper(robot, tool)
            time.sleep(0.5)

            # 移动到安全高度
            safe_trash_pose = PoseObject(
                trash_bin_pos[0],
                trash_bin_pos[1],
                z_heights["safe"],
                0.0, 1.57, 0.0
            )
            robot.move_pose(safe_trash_pose)
            print(f"{detection['class']} placed successfully by {arm_type} arm.")

            return True

        except Exception as e:
            print(f"Pick and place failed: {e}")
            return False

# ==================== 5. 全程控制模块 ====================
class ProcessController:
    def __init__(self, initializer, detector, transformer, operator):
        self.initializer = initializer
        self.detector = detector
        self.transformer = transformer
        self.operator = operator
        
        # 显示窗口参数
        self.gamma_window_name = "Gamma Arm Detection"
        self.alpha_window_name = "Alpha Arm Detection"
        self.window_size = (800, 600)
        self.display_size = (640, 480)
        
        # 同步控制
        self.operation_lock = threading.Lock()
        self.auto_detection_running = False
        self.current_working_arm = None
        self.batch_pick_count = 0
        
        # 设置窗口
        self.setup_windows()
    
    def setup_windows(self):
        """设置并排列显示窗口"""
        try:
            # 获取屏幕分辨率
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
        
        print(f"Screen resolution: {screen_width}x{screen_height}")
        
        # 确定窗口位置
        window_width, window_height = self.window_size
        
        # Gamma窗口位置（左侧）
        gamma_x = 50
        gamma_y = 100
        
        # Alpha窗口位置（右侧）
        alpha_x = screen_width - window_width - 50
        alpha_y = 100
        
        # 创建窗口
        cv2.namedWindow(self.gamma_window_name, cv2.WINDOW_NORMAL)
        cv2.namedWindow(self.alpha_window_name, cv2.WINDOW_NORMAL)
        
        # 设置窗口大小
        cv2.resizeWindow(self.gamma_window_name, window_width, window_height)
        cv2.resizeWindow(self.alpha_window_name, window_width, window_height)
        
        # 设置窗口位置
        cv2.moveWindow(self.gamma_window_name, gamma_x, gamma_y)
        cv2.moveWindow(self.alpha_window_name, alpha_x, alpha_y)
        
        print(f"Windows positioned: Gamma at ({gamma_x}, {gamma_y}), Alpha at ({alpha_x}, {alpha_y})")
    
    def undistort_image(self, img, mtx, dist):
        """校正图像畸变"""
        try:
            h, w = img.shape[:2]
            undistorted_img = cv2.undistort(img, mtx, dist, None, mtx)
            return undistorted_img
        except Exception as e:
            print(f"Image undistortion failed: {e}")
            return img
    
    def get_gamma_camera_image(self):
        """获取Gamma相机图像并进行畸变校正"""
        try:
            img_compressed = self.operator.gamma_robot.get_img_compressed()
            img_raw = uncompress_image(img_compressed)
            mtx, dist = self.operator.gamma_robot.get_camera_intrinsics()
            img_undistort = self.undistort_image(img_raw, mtx, dist)
            return img_undistort
        except Exception as e:
            print(f"Failed to get gamma camera image: {e}")
            return None
    
    def get_alpha_camera_image(self):
        """获取Alpha相机图像"""
        try:
            img_compressed = self.operator.alpha_robot.get_img_compressed()
            img_raw = uncompress_image(img_compressed)
            return img_raw
        except Exception as e:
            print(f"Failed to get alpha camera image: {e}")
            return None
    
    def resize_for_display(self, img, target_size=None):
        """调整图像大小以显示，保持宽高比"""
        if target_size is None:
            target_size = self.display_size
            
        h, w = img.shape[:2]
        target_w, target_h = target_size
        
        scale = min(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        y_offset = (target_h - new_h) // 2
        x_offset = (target_w - new_w) // 2
        canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        
        return canvas
    
    def draw_detections_on_image(self, img, detections, arm_type, workspace_type="fixed"):
        """在图像上绘制检测结果"""
        display_img = self.resize_for_display(img)
        h, w = display_img.shape[:2]
        
        orig_h, orig_w = img.shape[:2]
        scale_x = w / orig_w
        scale_y = h / orig_h

        for det in detections:
            x1_orig, y1_orig, x2_orig, y2_orig = det["bbox"]
            x1 = int(x1_orig * scale_x)
            y1 = int(y1_orig * scale_y)
            x2 = int(x2_orig * scale_x)
            y2 = int(y2_orig * scale_y)
            
            class_name = det["class"]
            confidence = det["confidence"]
            
            # 确定目标区域
            if arm_type == 'gamma':
                in_target_area = det["in_gamma_workspace"]
            else:  # alpha
                in_target_area = det["in_alpha_workspace"]

            # 设置颜色
            if not in_target_area: 
                color = (128, 128, 128)  # 灰色-不在工作空间
                thickness = 2
            elif class_name == (self.initializer.gamma_target_class if arm_type == 'gamma' else self.initializer.alpha_target_class):
                color = (0, 0, 255)  # 红色-目标类别
                thickness = 3
            else:
                color = (255, 0, 0)  # 蓝色-其他类别
                thickness = 2

            # 绘制边界框
            cv2.rectangle(display_img, (x1, y1), (x2, y2), color, thickness)

            # 绘制标签
            status = "" if in_target_area else " (Outside)" 
            label = f"{class_name}: {confidence:.2f}{status}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(display_img, (x1, y1 - label_size[1] - 10),
                         (x1 + label_size[0], y1), color, -1)
            cv2.putText(display_img, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            # 绘制中心点
            cx_orig, cy_orig = det["center_x"], det["center_y"]
            cx = int(cx_orig * scale_x)
            cy = int(cy_orig * scale_y)
            cv2.circle(display_img, (cx, cy), 5, color, -1)

            # 标记有效目标
            if class_name == (self.initializer.gamma_target_class if arm_type == 'gamma' else self.initializer.alpha_target_class) and in_target_area:
                cv2.putText(display_img, "TARGET", (cx - 20, cy - 15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # 显示统计信息
        total_count = len(detections)
        target_class = self.initializer.gamma_target_class if arm_type == 'gamma' else self.initializer.alpha_target_class
        target_count = len([d for d in detections if d["class"] == target_class])
        
        if workspace_type == "fixed":
            if arm_type == 'gamma':
                in_target_area_count = len([d for d in detections if d["in_gamma_workspace"]])
                target_in_area = len([d for d in detections if d["class"] == target_class and d["in_gamma_workspace"]])
            else:
                in_target_area_count = len([d for d in detections if d["in_alpha_workspace"]])
                target_in_area = len([d for d in detections if d["class"] == target_class and d["in_alpha_workspace"]])
            workspace_name = "Fixed Workspace"
        else:
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
        """显示两台机器人的检测窗口"""
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
        
        # 显示检测结果
        gamma_display = self.draw_detections_on_image(gamma_image, gamma_detections, 'gamma', workspace_type)
        alpha_display = self.draw_detections_on_image(alpha_image, alpha_detections, 'alpha', workspace_type)

        cv2.imshow(self.gamma_window_name, gamma_display)
        cv2.imshow(self.alpha_window_name, alpha_display)

        # 窗口刷新
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:  # 'q'或ESC键退出
            print("Exit key pressed!")
            self.initializer.shutdown_flag = True
            self.auto_detection_running = False
            return False
        return True
    
    def specialized_dual_camera_detection(self, workspace_type="fixed"):
        """分工检测任务 - 每台相机专注于自己的目标"""
        if self.initializer.shutdown_flag:
            return {'gamma_tasks': [], 'alpha_tasks': []}
            
        print("Starting specialized dual camera detection...")
        
        # 初始化变量
        gamma_image = None
        alpha_image = None
        all_gamma_detections = []
        all_alpha_detections = []
        gamma_target_detections = []
        alpha_target_detections = []
        
        # Gamma相机专注于自己的目标 - 使用未畸变图像
        try:
            gamma_image = self.get_gamma_camera_image()
            
            if gamma_image is not None:
                print(f"Gamma undistorted image shape: {gamma_image.shape}")
                # 在未畸变图像上进行检测
                all_gamma_detections = self.detector.detect_objects(gamma_image)
                # 添加机器人坐标和工作空间状态
                all_gamma_detections = self.transformer.add_robot_coordinates(all_gamma_detections, workspace_type)
                # 仅选择Gamma工作空间内的目标类别
                gamma_target_detections = self.transformer.filter_by_arm_workspace(
                    all_gamma_detections, self.initializer.gamma_target_class, 'gamma', workspace_type)
                print(f"Gamma camera detected {len(all_gamma_detections)} objects, {len(gamma_target_detections)} {self.initializer.gamma_target_class} objects in Gamma workspace")
            else:
                print("Failed to get Gamma camera image")
        except Exception as e:
            print(f"Gamma camera detection error: {e}")

        # Alpha相机专注于自己的目标 - 使用原始图像
        try:
            alpha_image = self.get_alpha_camera_image()
            if alpha_image is not None:
                print(f"Alpha image shape: {alpha_image.shape}")
                # 检测所有物体
                all_alpha_detections = self.detector.detect_objects(alpha_image)
                # 添加机器人坐标和工作空间状态
                all_alpha_detections = self.transformer.add_robot_coordinates(all_alpha_detections, workspace_type)
                # 仅选择Alpha工作空间内的目标类别
                alpha_target_detections = self.transformer.filter_by_arm_workspace(
                    all_alpha_detections, self.initializer.alpha_target_class, 'alpha', workspace_type)
                print(f"Alpha camera detected {len(all_alpha_detections)} objects, {len(alpha_target_detections)} {self.initializer.alpha_target_class} objects in Alpha workspace")
            else:
                print("Failed to get Alpha camera image")
        except Exception as e:
            print(f"Alpha camera detection error: {e}")

        # 显示检测窗口
        if not self.show_detection_window(gamma_image, alpha_image, all_gamma_detections, all_alpha_detections, workspace_type):
            return {'gamma_tasks': [], 'alpha_tasks': []}

        return {
            'gamma_tasks': gamma_target_detections,
            'alpha_tasks': alpha_target_detections
        }
    
    def start_batch_pick_operation(self, total_tasks, workspace_type="conveyor"):
        """开始批量抓取操作"""
        if workspace_type == "conveyor" and total_tasks > 0:
            print(f"Starting batch pick operation with {total_tasks} tasks")
            self.batch_pick_count = total_tasks
            # 停止传送带
            self.operator.control_conveyor("stop")
            return True
        return False
    
    def complete_batch_pick_operation(self, workspace_type="conveyor"):
        """完成批量抓取操作"""
        if workspace_type == "conveyor" and self.batch_pick_count > 0:
            self.batch_pick_count -= 1
            print(f"Batch pick operation completed. Remaining tasks: {self.batch_pick_count}")
            
            # 如果所有批量任务完成，重新启动传送带
            if self.batch_pick_count == 0:
                print("All batch pick operations completed. Restarting conveyor...")
                self.operator.control_conveyor("start")
                return True
        return False
    
    def coordinate_division_operation(self, detection_results, workspace_type="fixed"):
        """协调分工操作"""
        if self.initializer.shutdown_flag:
            return 0
            
        gamma_tasks = detection_results['gamma_tasks']
        alpha_tasks = detection_results['alpha_tasks']

        print(f"Gamma tasks ({self.initializer.gamma_target_class}): {len(gamma_tasks)}")
        print(f"Alpha tasks ({self.initializer.alpha_target_class}): {len(alpha_tasks)}")

        # 计算总任务数
        total_tasks = len(gamma_tasks) + len(alpha_tasks)
        print(f"Total tasks to process: {total_tasks}")

        # 如果是传送带模式，开始批量抓取操作
        if workspace_type == "conveyor" and total_tasks > 0:
            self.start_batch_pick_operation(total_tasks, workspace_type)

        processed_count = 0
        
        # 处理Gamma任务
        for i, task in enumerate(gamma_tasks):
            if self.initializer.shutdown_flag:
                break
            print(f"\nGamma Arm processing {self.initializer.gamma_target_class} object {i+1}/{len(gamma_tasks)}")
            success = self.operator.pick_and_place_object(task, 'gamma', workspace_type, self.transformer)
            if success:
                processed_count += 1
            time.sleep(1)

        # 处理Alpha任务
        for i, task in enumerate(alpha_tasks):
            if self.initializer.shutdown_flag:
                break
            print(f"\nAlpha Arm processing {self.initializer.alpha_target_class} object {i+1}/{len(alpha_tasks)}")
            success = self.operator.pick_and_place_object(task, 'alpha', workspace_type, self.transformer)
            if success:
                processed_count += 1
            time.sleep(1)

        return processed_count
    
    def auto_detect_and_sort_division(self, workspace_type="fixed"):
        """自动分拣"""
        if self.initializer.shutdown_flag:
            return
            
        print(f"\n=== Dual Arm {workspace_type.title()} Workspace: Division Detection ===")
        print(f"Gamma Arm: {self.initializer.gamma_target_class}")
        print(f"Alpha Arm: {self.initializer.alpha_target_class}")
        # 移动到观察位置
        self.operator.move_to_observe(workspace_type)

        # 分工检测
        detection_results = self.specialized_dual_camera_detection(workspace_type)
        # 协调任务分工
        processed_count = self.coordinate_division_operation(detection_results, workspace_type)

        print(f"\nDivision sorting completed! Processed {processed_count} objects.")
    
    def auto_detect_and_sort_division_conveyor(self):
        """传送带自动分拣"""
        if self.initializer.shutdown_flag:
            return
            
        print(f"\n=== Conveyor Division Detection ===")
        print(f"Gamma Arm: {self.initializer.gamma_target_class}")
        print(f"Alpha Arm: {self.initializer.alpha_target_class}")
        # 启动传送带
        if not self.operator.control_conveyor("start"):
            print("Failed to start conveyor")
            return

        # 移动到观察位置
        self.operator.move_to_observe("conveyor")

        self.auto_detection_running = True
        total_processed = 0
        cycle_count = 0

        try:
            while self.auto_detection_running and not self.initializer.shutdown_flag:
                cycle_count += 1
                print(f"\n--- Division Detection Cycle {cycle_count} ---")
                
                # 确保传送带正在运行（除非批量抓取操作正在进行）
                if not self.operator.conveyor_running and self.batch_pick_count == 0:
                    print("No batch operations, ensuring conveyor is running...")
                    self.operator.control_conveyor("start")
                
                # 分工检测
                detection_results = self.specialized_dual_camera_detection("conveyor")
                
                # 如果检测到目标，执行抓取操作
                if detection_results['gamma_tasks'] or detection_results['alpha_tasks']:
                    print(f"Targets detected! Gamma: {len(detection_results['gamma_tasks'])}, Alpha: {len(detection_results['alpha_tasks'])}")
                    # 协调任务分工（传送带将在pick_and_place_object中自动停止）
                    processed_this_cycle = self.coordinate_division_operation(detection_results, "conveyor")
                    total_processed += processed_this_cycle
                    print(f"Cycle {cycle_count}: Processed {processed_this_cycle} objects (Total: {total_processed})")
                else:
                    print(f"Cycle {cycle_count}: No targets detected, continuing conveyor operation...")
                    processed_this_cycle = 0
                
                # 短暂延迟
                time.sleep(1)
                
        except KeyboardInterrupt:
            print("Division detection interrupted by user")
        except Exception as e:
            print(f"Division detection error: {e}")
        finally:
            self.auto_detection_running = False
            # 确保传送带停止
            self.operator.emergency_stop_conveyor()
            print(f"Conveyor division sorting finished. Total processed: {total_processed}")
    
    def test_camera_connection(self):  
        """相机连接测试"""  
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
        """调试相机连接"""
        print("\n=== Debugging Camera Connections ===")
        
        # 测试Gamma相机
        print("Testing Gamma camera...")
        try:
            gamma_img = self.get_gamma_camera_image()
            if gamma_img is not None:
                print(f"Gamma camera OK - Image shape: {gamma_img.shape}")
                cv2.imshow("Gamma Debug", gamma_img)
                cv2.waitKey(1000)
                cv2.destroyWindow("Gamma Debug")
            else:
                print("Gamma camera FAILED - No image received")
        except Exception as e:
            print(f"Gamma camera ERROR: {e}")
        
        # 测试Alpha相机
        print("Testing Alpha camera...")
        try:
            alpha_img = self.get_alpha_camera_image()
            if alpha_img is not None:
                print(f"Alpha camera OK - Image shape: {alpha_img.shape}")
                cv2.imshow("Alpha Debug", alpha_img)
                cv2.waitKey(1000)
                cv2.destroyWindow("Alpha Debug")
            else:
                print("Alpha camera FAILED - No image received")
        except Exception as e:
            print(f"Alpha camera ERROR: {e}")
    
    def cleanup(self):  
        """清理资源"""  
        print("Cleaning up resources...")
        self.auto_detection_running = False
        self.initializer.shutdown_flag = True
        
        # 首先停止传送带
        print("Stopping conveyor...")
        self.operator.emergency_stop_conveyor()
        
        # 关闭窗口
        print("Closing windows...")
        try:
            cv2.destroyAllWindows()
            cv2.waitKey(1)
        except:
            pass
            
        # 断开机器人连接
        print("Closing robot connections...")
        try:  
            self.operator.gamma_robot.close_connection()  
            print("Gamma robot connection closed")
        except Exception as e:  
            print(f"Error closing Gamma robot: {e}")
            
        try:
            self.operator.alpha_robot.close_connection()  
            print("Alpha robot connection closed")  
        except Exception as e:
            print(f"Error closing Alpha robot: {e}")

# ==================== 主程序 ====================
def main():  
    # 配置参数
    ROBOT_IPS = ["192.168.0.109", "192.168.0.106"]  # Gamma和Alpha机械臂的IP地址
    ONNX_MODEL_PATH = "/home/ned2/Desktop/Waste sorting/code/best.onnx"
    # 标定配置文件
    CALIBRATION_CONFIGS = {
        'gamma_fixed': "/home/ned2/Desktop/Waste sorting/code/1124 GAMMAconveyor_calibration_matrix.npy",
        'gamma_conveyor': "/home/ned2/Desktop/Waste sorting/code/1203 conveyorcalibration_matrix1.npy",
        'alpha_fixed': "/home/ned2/Desktop/Waste sorting/code/1124 ALPHAconveyor_calibration_matrix.npy",
        'alpha_conveyor': "/home/ned2/Desktop/Waste sorting/code/1114robot81conveyor_calibration_matrix.npy"
    }
    # 工作空间配置文件
    WORKSPACE_CONFIGS = {
        'gamma_fixed': "/home/ned2/Desktop/Waste sorting/code/1124robot79workspace_conveyor.json",
        'alpha_fixed': "/home/ned2/Desktop/Waste sorting/code/1124robot81workspace_conveyor.json",
        'gamma_conveyor': "/home/ned2/Desktop/Waste sorting/code/1124robot79workspace_conveyor.json",  # Gamma传送带工作区
        'alpha_conveyor': "/home/ned2/Desktop/Waste sorting/code/1124robot81workspace_conveyor.json"  # Alpha传送带工作区
    }
    
    # 垃圾分拣类别
    CLASS_NAMES = ['cardboard', 'glass', 'metal', 'paper', 'plastic']

    print("=== Gamma & Alpha Dual Arm Sorting Robot System ===")

    try:
        # 1. 系统初始化
        initializer = SystemInitializer(ROBOT_IPS, ONNX_MODEL_PATH, CALIBRATION_CONFIGS, WORKSPACE_CONFIGS)
        gamma_robot, alpha_robot = initializer.initialize_robots()
        gamma_target_class, alpha_target_class = initializer.select_target_classes()
        gamma_tool, alpha_tool = initializer.select_grippers()
        gamma_fixed_matrix, gamma_conveyor_matrix, alpha_fixed_matrix, alpha_conveyor_matrix = initializer.load_calibration_matrices()
        workspaces = initializer.load_workspaces()
        initializer.get_config_summary()
        
        # 2. 视觉检测模块
        detector = VisionDetector(ONNX_MODEL_PATH, CLASS_NAMES)
        
        # 3. 坐标转换模块
        transformer = CoordinateTransformer(gamma_fixed_matrix, gamma_conveyor_matrix, 
                                           alpha_fixed_matrix, alpha_conveyor_matrix, workspaces)
        
        # 4. 抓取操作模块
        operator = GraspingOperator(gamma_robot, alpha_robot, gamma_tool, alpha_tool, 
                                   gamma_target_class, alpha_target_class)
        
        # 5. 全程控制模块
        controller = ProcessController(initializer, detector, transformer, operator)
        
        # 测试相机连接
        if controller.test_camera_connection():  
            # 运行调试
            controller.debug_camera_connections()
            
            while not initializer.shutdown_flag:  
                print("\n" + "="*50)  
                print("Select mode:")  
                print("1 - Fixed Workspace: Division Detection & Sort")  
                print("2 - Conveyor: Continuous Division Detection & Sort")  
                print("3 - Exit")  
                print("Press 'q' in any window to stop current operation")
                print("Press Ctrl+C to emergency stop")

                choice = input("Enter your choice (1-3): ").strip()  

                if choice == "1":  
                    controller.auto_detect_and_sort_division("fixed")  
                elif choice == "2":  
                    controller.auto_detect_and_sort_division_conveyor()  
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
        if 'controller' in locals():
            controller.cleanup()  
        print(" Program finished")  

if __name__ == "__main__":  
    main()