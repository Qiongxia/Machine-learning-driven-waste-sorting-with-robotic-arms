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
    def __init__(self, robot_ip, onnx_model_path, calibration_configs, workspace_configs):
        # Set up signal handling
        signal.signal(signal.SIGINT, self.signal_handler)
        self.shutdown_flag = False
        
        # 配置参数
        self.robot_ip = robot_ip
        self.onnx_model_path = onnx_model_path
        self.calibration_configs = calibration_configs
        self.workspace_configs = workspace_configs
        
        # 初始化状态变量
        self.robot = None
        self.target_class = None
        self.tool = None
        self.fixed_transform_matrix = None
        self.conveyor_transform_matrix = None
        
    def initialize_robot(self):
        """初始化机器人连接"""
        print("Connecting to robot...")
        self.robot = NiryoRobot(self.robot_ip)
        print("Calibrating robot...")
        self.robot.calibrate_auto()
        return self.robot
    
    def select_target_class(self):
        """选择目标分类"""
        print("\n" + "="*50)  
        print("=== Object Type Selection ===")
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
                self.target_class = selected_class
                return selected_class
            else:
                print("Invalid choice, please try again.")
    
    def select_gripper(self, robot):
        """选择夹具类型"""
        print("\n" + "="*50)  
        print("=== Gripper Selection ===")  
        print(f"\n--- Robot ({self.target_class}) Gripper Selection ---")  
        
        print("\nSelect gripper type:")  
        print("1 - GRIPPER_1")  
        print("2 - ELECTROMAGNET_1")  
        print("3 - VACUUM_PUMP_1")  

        while True:  
            choice = input("Enter your choice (1-3): ").strip()  
            if choice == "1":  
                tool = ToolID.GRIPPER_1  
                print("Selected GRIPPER_1")  
                self.tool = tool
                return tool
            elif choice == "2":  
                tool = ToolID.ELECTROMAGNET_1  
                pin_electromagnet = PinID.DO4  
                robot.setup_electromagnet(pin_electromagnet)  
                print("Selected ELECTROMAGNET_1")  
                self.tool = tool
                return tool
            elif choice == "3":  
                tool = ToolID.VACUUM_PUMP_1  
                print("Selected VACUUM_PUMP_1")  
                self.tool = tool
                return tool
            else:  
                print("Invalid choice, please try again.")
    
    def load_calibration_matrices(self):
        """加载标定矩阵"""
        print("Loading calibration matrix...")  
        self.fixed_transform_matrix = np.load(self.calibration_configs['fixed'])
        self.conveyor_transform_matrix = np.load(self.calibration_configs['conveyor'])
        print("Calibration matrices loaded successfully.")
        return self.fixed_transform_matrix, self.conveyor_transform_matrix
    
    def load_workspaces(self):
        """加载工作空间"""
        print("Loading workspaces...")  
        workspaces = {}
        
        # Fixed workspace
        with open(self.workspace_configs['fixed'], "r") as f:  
            workspace = json.load(f)  
            x_list = [p["x"] for p in workspace["points"]]  
            y_list = [p["y"] for p in workspace["points"]]  
            X_MIN, X_MAX = min(x_list), max(x_list)  
            Y_MIN, Y_MAX = min(y_list), max(y_list)  
            workspaces['fixed'] = {
                'points': workspace["points"],
                'X_MIN': X_MIN,
                'X_MAX': X_MAX,
                'Y_MIN': Y_MIN,
                'Y_MAX': Y_MAX
            }
        print(f"Fixed Workspace X range: {X_MIN:.3f} ~ {X_MAX:.3f}")  
        print(f"Fixed Workspace Y range: {Y_MIN:.3f} ~ {Y_MAX:.3f}")  

        # Conveyor workspace
        print("Loading conveyor workspace...") 
        with open(self.workspace_configs['conveyor'], "r") as f:  
            conveyor_workspace = json.load(f)  
            conveyor_x_list = [p["x"] for p in conveyor_workspace["points"]]  
            conveyor_y_list = [p["y"] for p in conveyor_workspace["points"]]  
            CONVEYOR_X_MIN, CONVEYOR_X_MAX = min(conveyor_x_list), max(conveyor_x_list)  
            CONVEYOR_Y_MIN, CONVEYOR_Y_MAX = min(conveyor_y_list), max(conveyor_y_list)  
            workspaces['conveyor'] = {
                'points': conveyor_workspace["points"],
                'X_MIN': CONVEYOR_X_MIN,
                'X_MAX': CONVEYOR_X_MAX,
                'Y_MIN': CONVEYOR_Y_MIN,
                'Y_MAX': CONVEYOR_Y_MAX
            }
        print(f"Conveyor Workspace X range: {CONVEYOR_X_MIN:.3f} ~ {CONVEYOR_X_MAX:.3f}")  
        print(f"Conveyor Workspace Y range: {CONVEYOR_Y_MIN:.3f} ~ {CONVEYOR_Y_MAX:.3f}")  
        
        return workspaces
    
    def signal_handler(self, sig, frame):
        """处理Ctrl+C信号"""
        print("\nCtrl+C pressed! Shutting down...")
        self.shutdown_flag = True
        sys.exit(0)
    
    def get_config_summary(self):
        """获取配置摘要"""
        print(f"\n=== Configuration Summary ===")
        print(f"Target Class: {self.target_class}")
        print(f"Robot IP: {self.robot_ip}")
        print("System initialization complete")

# ==================== 2. 视觉检测模块 ====================
class VisionDetector:
    def __init__(self, onnx_model_path, class_names, input_size=640, conf_thresh=0.25, nms_thresh=0.45):
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
    
    def detect_objects(self, img, target_class=None):  
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
    def __init__(self, fixed_transform_matrix, conveyor_transform_matrix, workspaces):
        self.fixed_transform_matrix = fixed_transform_matrix
        self.conveyor_transform_matrix = conveyor_transform_matrix
        self.workspaces = workspaces
        
    def pixel_to_robot(self, u, v, workspace_type="fixed"):
        """像素坐标转机器人坐标"""
        pixel = np.array([u, v, 1])
        if workspace_type == "fixed":
            world = np.dot(self.fixed_transform_matrix, pixel)
        else:
            world = np.dot(self.conveyor_transform_matrix, pixel)
        return world[0], world[1]
    
    def in_workspace(self, x, y, workspace_type="fixed"):  
        """检查坐标是否在工作空间内"""
        ws = self.workspaces[workspace_type]
        return ws['X_MIN'] <= x <= ws['X_MAX'] and ws['Y_MIN'] <= y <= ws['Y_MAX']
    
    def add_robot_coordinates(self, detections, workspace_type="fixed"):
        """为检测结果添加机器人坐标和工作空间状态"""
        for det in detections:
            robot_x, robot_y = self.pixel_to_robot(det["center_x"], det["center_y"], workspace_type)
            det["robot_x"] = robot_x
            det["robot_y"] = robot_y
            det["in_workspace"] = self.in_workspace(robot_x, robot_y, workspace_type)
            det["in_conveyor"] = (workspace_type == "conveyor")
        return detections
    
    def extract_image_workspace(self, img, workspace_type="fixed"):
        """提取并显示工作空间区域"""
        if img is None:
            print("Error: Cannot extract workspace from None image")
            return None
        
        # 获取工作空间点
        workspace_points = self.workspaces[workspace_type]['points']
        transform_matrix = self.fixed_transform_matrix if workspace_type == "fixed" else self.conveyor_transform_matrix
        
        # 创建图像副本用于绘制
        workspace_overlay = img.copy()
        
        # 将机器人坐标转换为像素坐标
        pixel_points = []
        for point in workspace_points:
            # 机器人到像素转换（pixel_to_robot的逆变换）
            robot_x, robot_y = point["x"], point["y"]
            
            # 对于仿射变换，我们需要求解逆变换
            try:
                inv_transform = np.linalg.inv(transform_matrix)
                pixel_homogeneous = np.dot(inv_transform, np.array([robot_x, robot_y, 1]))
                u, v = pixel_homogeneous[0], pixel_homogeneous[1]
                # 确保像素坐标在图像边界内
                u = max(0, min(img.shape[1] - 1, int(u)))
                v = max(0, min(img.shape[0] - 1, int(v)))
                pixel_points.append((u, v))
            except:
                # 如果变换失败，跳过此点
                print(f"Warning: Could not calculate inverse transform for point ({robot_x}, {robot_y})")
                continue
        
        if len(pixel_points) < 3:
            print(f"Warning: Not enough valid points for {workspace_type} workspace extraction")
            return workspace_overlay
        
        # 将点转换为numpy数组
        pts = np.array(pixel_points, dtype=np.int32)
        
        # 绘制工作空间边界
        if len(pts) > 2:
            # 创建凸包以便更好地可视化
            hull = cv2.convexHull(pts)
            
            # 绘制带透明度的填充多边形
            overlay = workspace_overlay.copy()
            cv2.fillPoly(overlay, [hull], (0, 255, 0, 50))  # 绿色带透明度
            
            # 将覆盖层与原始图像混合
            cv2.addWeighted(overlay, 0.3, workspace_overlay, 0.7, 0, workspace_overlay)
            
            # 绘制边界轮廓
            cv2.polylines(workspace_overlay, [hull], True, (0, 255, 0), 2)
            
            # 绘制角点
            for i, (px, py) in enumerate(pixel_points):
                cv2.circle(workspace_overlay, (px, py), 5, (255, 0, 0), -1)
                cv2.putText(workspace_overlay, f"P{i+1}", (px + 5, py - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # 标记工作空间类型
            cv2.putText(workspace_overlay, f"{workspace_type.upper()} WORKSPACE", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return workspace_overlay

# ==================== 4. 抓取操作模块 ====================
class GraspingOperator:
    def __init__(self, robot, tool, target_class):
        self.robot = robot
        self.tool = tool
        self.target_class = target_class
        
        # 高度参数
        self.FIXED_Z_PICK = 0.05  
        self.FIXED_Z_APPROACH = 0.15  
        self.FIXED_Z_SAFE = 0.20  
        self.CONVEYOR_Z_PICK = 0.11 
        self.CONVEYOR_Z_APPROACH = 0.18  
        self.CONVEYOR_Z_SAFE = 0.23  
        
        # 位置定义
        self.TRASH_BIN_POS = self.get_trash_bin_position()
        self.FIXED_OBSERVE_POS = [0.006, 0.162, 0.253, 3.034, 1.327, -1.712]
        self.CONVEYOR_OBSERVE_POS = [0.251, 0.005, 0.254, 2.978, 1.311, 3.003]
        self.SAFE_POS = [0.140, -0.000, 0.203, 0.000, 0.753, -0.001]
        
        # 传送带设置
        self.conveyor_id = self.robot.set_conveyor()
        self.conveyor_speed = 50
        self.conveyor_running = False
        
        # 延迟补偿参数
        self.total_delay_ms = 600
        self.pixel_to_mm = 0.5000
        self.setup_delay_compensation()
    
    def get_trash_bin_position(self):
        """根据目标类型获取垃圾桶位置"""
        bin_positions = {
            'cardboard': [0.017, -0.174, 0.241, 0.05, 1.5, 0],
            'glass': [0.017, -0.174, 0.241, 0.05, 1.5, 0],
            'metal': [0.017, -0.174, 0.241, 0.05, 1.5, 0],
            'paper': [0.017, -0.174, 0.241, 0.05, 1.5, 0],
            'plastic': [0.017, -0.174, 0.241, 0.05, 1.5, 0]
        }
        return bin_positions.get(self.target_class, [0.017, -0.174, 0.241, 0.05, 1.5, 0])
    
    def setup_delay_compensation(self):
        """设置延迟补偿参数"""
        delay_seconds = self.total_delay_ms / 1000
        self.compensation_mm = self.conveyor_speed * delay_seconds
        self.compensation_pixels = self.compensation_mm / self.pixel_to_mm
        print(f"Compensation: {self.compensation_pixels:.1f} px")
    
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
    
    def activate_gripper(self):
        """激活夹具"""
        print(f"Activating gripper...")  
        try:  
            self.robot.grasp_with_tool()  
            print("Gripper activated")  
            time.sleep(1)  
        except Exception as e:  
            print(f"Gripper activation failed: {e}")
    
    def deactivate_gripper(self):
        """释放夹具"""
        print(f"Releasing gripper...")  
        try:  
            self.robot.release_with_tool()  
            print("Gripper released")  
            time.sleep(0.5)  
        except Exception as e:  
            print(f"Gripper release failed: {e}")
    
    def move_to_observe(self, workspace_type="fixed"):  
        """移动到观察位置"""
        print(f" Moving to {workspace_type} workspace observe position...")  
        try:  
            if workspace_type == "fixed":  
                observe_pose = PoseObject(*self.FIXED_OBSERVE_POS)
                self.robot.move_pose(observe_pose)
            else: # conveyor  
                observe_pose = PoseObject(*self.CONVEYOR_OBSERVE_POS)
                self.robot.move_pose(observe_pose)
            time.sleep(1)  
        except Exception as e:  
            print(f" Move to observe position failed: {e}")
    
    def move_to_safe_position(self):
        """移动到安全位置"""
        try:
            safe_pose = PoseObject(*self.SAFE_POS)
            self.robot.move_pose(safe_pose)
            print("Robot moved to safe position")
        except Exception as e:
            print(f"Move to safe position failed: {e}")
    
    def control_conveyor(self, action="start"):  
        """控制传送带"""  
        try:  
            if action == "start":  
                self.robot.run_conveyor(self.conveyor_id, speed=self.conveyor_speed, direction=ConveyorDirection.FORWARD)  
                print(f" Conveyor started at speed {self.conveyor_speed}")  
                self.conveyor_running = True
            elif action == "stop":  
                self.robot.stop_conveyor(self.conveyor_id)  
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
                    self.robot.stop_conveyor(self.conveyor_id)
                    print(f"Conveyor stop attempt {i+1}: Success")
                    self.conveyor_running = False
                    break
                except Exception as e:
                    print(f"Conveyor stop attempt {i+1} failed: {e}")
                    time.sleep(0.5)
        except Exception as e:
            print(f"Emergency conveyor stop failed: {e}")
    
    def pick_and_place_object(self, detection, workspace_type="fixed", transformer=None):  
        """抓取并放置物体"""
        x, y = detection["robot_x"], detection["robot_y"]
        trash_bin_pos = self.TRASH_BIN_POS

        # 传送带延迟补偿
        if workspace_type == "conveyor" and transformer:
            original_pixel_x = detection["center_x"]
            compensated_pixel_x = original_pixel_x - self.compensation_pixels
            # 重新计算机器人坐标
            x, y = transformer.pixel_to_robot(compensated_pixel_x, detection["center_y"], workspace_type)
            print(f"Latency compensation:")
            print(f"Compensation distance: +{self.compensation_pixels:.1f} pixels")
        print(f"Targeting {detection['class']} at coordinates: ({x:.3f}, {y:.3f})")  

        z_heights = self.get_z_heights(workspace_type)  

        try:  
            # 准备夹具
            self.deactivate_gripper()
            time.sleep(0.5)

            # 抓取动作
            approach_pose = PoseObject(x, y, z_heights["approach"], 0.0, 1.57, 0.0)
            pick_pose = PoseObject(x, y, z_heights["pick"], 0.0, 1.57, 0.0)
            safe_pose = PoseObject(x, y, z_heights["safe"], 0.0, 1.57, 0.0)

            print(" Moving to approach position...")
            self.robot.move_pose(approach_pose)

            print(" Moving to pick position...")
            self.robot.move_pose(pick_pose)

            # 抓取
            self.activate_gripper()
            time.sleep(1)

            print(" Moving to safe height...")
            self.robot.move_pose(safe_pose)

            # 移动到垃圾桶位置
            print(" Moving to trash bin...")
            trash_pose = PoseObject(*trash_bin_pos)
            self.robot.move_pose(trash_pose)

            # 放置
            self.deactivate_gripper()
            time.sleep(0.5)

            # 移动到安全高度
            safe_trash_pose = PoseObject(
                trash_bin_pos[0],
                trash_bin_pos[1],
                z_heights["safe"],
                0.0, 1.57, 0.0
            )
            self.robot.move_pose(safe_trash_pose)
            print(f"{detection['class']} placed successfully.")

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
        self.window_name = "Robot Arm Detection - Undistorted"
        self.workspace_window_name = "Workspace Region"
        self.window_size = (800, 600)
        self.workspace_window_size = (400, 300)
        self.display_size = (640, 480)
        
        # 同步控制
        self.operation_lock = threading.Lock()
        self.auto_detection_running = False
        self.batch_pick_count = 0
        
        # 设置窗口
        self.setup_window()
    
    def setup_window(self):
        """设置显示窗口"""
        # 创建主检测窗口
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, self.window_size[0], self.window_size[1])
        
        # 创建工作空间窗口
        cv2.namedWindow(self.workspace_window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.workspace_window_name, 
                        self.workspace_window_size[0], 
                        self.workspace_window_size[1])
        
        # 窗口居中
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
        
        # 定位主窗口
        window_x = (screen_width - self.window_size[0]) // 2
        window_y = (screen_height - self.window_size[1]) // 2
        cv2.moveWindow(self.window_name, window_x, window_y)
        
        # 定位工作空间窗口
        workspace_x = window_x + self.window_size[0] + 10
        workspace_y = window_y
        cv2.moveWindow(self.workspace_window_name, workspace_x, workspace_y)
        
        print(f"Detection window positioned at ({window_x}, {window_y})")
        print(f"Workspace window positioned at ({workspace_x}, {workspace_y})")
    
    def undistort_image(self, img, mtx, dist):
        """校正图像畸变"""
        try:
            h, w = img.shape[:2]
            undistorted_img = cv2.undistort(img, mtx, dist, None, mtx)
            return undistorted_img
        except Exception as e:
            print(f"Image undistortion failed: {e}")
            return img
    
    def get_camera_image(self):
        """获取相机图像并进行畸变校正"""
        try:
            # 获取图像
            img_compressed = self.operator.robot.get_img_compressed()
            
            # 解压缩图像
            img_raw = uncompress_image(img_compressed)
            
            # 获取相机内参并校正图像
            mtx, dist = self.operator.robot.get_camera_intrinsics()
            img_undistort = self.undistort_image(img_raw, mtx, dist)
            
            return img_undistort
        except Exception as e:
            print(f"Failed to get camera image: {e}")
            return None
    
    def resize_for_display(self, img, target_size=None):
        """调整图像大小以显示，保持宽高比"""
        if target_size is None:
            target_size = self.display_size
            
        h, w = img.shape[:2]
        target_w, target_h = target_size
        
        # 计算缩放因子
        scale = min(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # 调整大小
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # 创建目标大小的画布
        canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        
        # 将调整大小后的图像放在画布中央
        y_offset = (target_h - new_h) // 2
        x_offset = (target_w - new_w) // 2
        canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        
        return canvas
    
    def draw_detections_on_image(self, img, detections, workspace_type="fixed"):
        """在图像上绘制检测结果"""
        # 首先调整图像大小以显示
        display_img = self.resize_for_display(img)
        h, w = display_img.shape[:2]
        
        # 计算从原始图像到显示图像的缩放比例
        orig_h, orig_w = img.shape[:2]
        scale_x = w / orig_w
        scale_y = h / orig_h

        # 绘制检测结果
        for det in detections:
            # 将边界框坐标调整到显示大小
            x1_orig, y1_orig, x2_orig, y2_orig = det["bbox"]
            x1 = int(x1_orig * scale_x)
            y1 = int(y1_orig * scale_y)
            x2 = int(x2_orig * scale_x)
            y2 = int(y2_orig * scale_y)
            
            class_name = det["class"]
            confidence = det["confidence"]
            
            # 确定目标区域
            in_target_area = det["in_workspace"]

            # 设置颜色：目标类别-红色，其他类别-蓝色，不在目标区域-灰色
            if not in_target_area: 
                color = (128, 128, 128)  # 灰色-不在目标区域
                thickness = 2
            elif class_name == self.initializer.target_class:
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
            # 标签背景
            cv2.rectangle(display_img, (x1, y1 - label_size[1] - 10),
                         (x1 + label_size[0], y1), color, -1)
            cv2.putText(display_img, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            # 绘制中心点
            cx_orig, cy_orig = det["center_x"], det["center_y"]
            cx = int(cx_orig * scale_x)
            cy = int(cy_orig * scale_y)
            cv2.circle(display_img, (cx, cy), 5, color, -1)

            # 如果是目标物体且在目标区域，特殊标记
            if class_name == self.initializer.target_class and in_target_area:
                cv2.putText(display_img, "TARGET", (cx - 20, cy - 15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # 显示统计信息
        total_count = len(detections)
        target_count = len([d for d in detections if d["class"] == self.initializer.target_class])
        
        if workspace_type == "fixed":
            in_target_area_count = len([d for d in detections if d["in_workspace"]])
            target_in_area = len([d for d in detections if d["class"] == self.initializer.target_class and d["in_workspace"]])
            workspace_name = "Fixed Workspace"
        else:
            in_target_area_count = len([d for d in detections if d["in_workspace"]])
            target_in_area = len([d for d in detections if d["class"] == self.initializer.target_class and d["in_workspace"]])
            workspace_name = "Conveyor Workspace"

        cv2.putText(display_img, f"Robot Arm - {workspace_name}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(display_img, f"Target: {self.initializer.target_class}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(display_img, f"Total objects: {total_count}", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(display_img, f"In target area: {in_target_area_count}", (10, 120),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(display_img, f"Target objects: {target_count} (In area: {target_in_area})", (10, 150),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # 添加图像状态信息
        cv2.putText(display_img, "Image: Undistorted (Corrected)", (10, 180),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        return display_img
    
    def show_detection_window(self, image, detections, workspace_type="fixed"):
        """显示检测窗口和工作空间窗口"""
        # 确保图像不为None，如果为None则创建黑色图像
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
            # 绘制检测结果
            display_img = self.draw_detections_on_image(image, detections, workspace_type)
            
            # 显示主检测窗口
            cv2.imshow(self.window_name, display_img)
            
            # 提取并显示工作空间覆盖层
            workspace_overlay = self.transformer.extract_image_workspace(image, workspace_type)
            if workspace_overlay is not None:
                # 调整工作空间覆盖层大小以显示
                workspace_display = self.resize_for_display(workspace_overlay, 
                                                           target_size=(400, 300))
                cv2.imshow(self.workspace_window_name, workspace_display)
            else:
                # 备用：显示检测图像
                cv2.imshow(self.workspace_window_name, display_img)
                
        except Exception as e:
            print(f"Error displaying windows: {e}")
            # 简单备用方案
            if image is not None:
                cv2.imshow(self.window_name, image)
                cv2.imshow(self.workspace_window_name, image)

        # 窗口刷新 - 添加退出检查
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:  # 'q' 或 ESC 键
            print("Exit key pressed!")
            self.initializer.shutdown_flag = True
            self.auto_detection_running = False
            return False
        return True
    
    def camera_detection(self, workspace_type="fixed"):
        """相机检测"""
        if self.initializer.shutdown_flag:
            return []
            
        print("Starting camera detection with undistorted image...")
        
        # 初始化变量
        undistorted_image = None
        all_detections = []
        target_detections = []
        
        try:
            # 获取未畸变图像
            undistorted_image = self.get_camera_image()
            
            if undistorted_image is not None:
                print(f"Undistorted image shape: {undistorted_image.shape}")
                # 检测未畸变图像上的物体
                all_detections = self.detector.detect_objects(undistorted_image)
                # 添加机器人坐标和工作空间状态
                all_detections = self.transformer.add_robot_coordinates(all_detections, workspace_type)
                # 仅选择工作空间内的目标类别
                target_detections = [d for d in all_detections if d["class"] == self.initializer.target_class and d["in_workspace"]]
                print(f"Camera detected {len(all_detections)} objects, {len(target_detections)} {self.initializer.target_class} objects in workspace")
            else:
                print("Failed to get camera image")
        except Exception as e:
            print(f"Camera detection error: {e}")

        # 显示检测窗口
        if not self.show_detection_window(undistorted_image, all_detections, workspace_type):
            return []

        return target_detections
    
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
    
    def coordinate_operation(self, detection_results, workspace_type="fixed"):
        """协调操作"""
        if self.initializer.shutdown_flag:
            return 0
            
        tasks = detection_results

        print(f"Tasks ({self.initializer.target_class}): {len(tasks)}")

        # 计算总任务数
        total_tasks = len(tasks)
        print(f"Total tasks to process: {total_tasks}")

        # 如果是传送带模式，开始批量抓取操作
        if workspace_type == "conveyor" and total_tasks > 0:
            self.start_batch_pick_operation(total_tasks, workspace_type)

        processed_count = 0
        
        # 处理任务
        for i, task in enumerate(tasks):
            if self.initializer.shutdown_flag:
                break
            print(f"\nProcessing {self.initializer.target_class} object {i+1}/{len(tasks)}")
            success = self.operator.pick_and_place_object(task, workspace_type, self.transformer)
            if success:
                processed_count += 1
            time.sleep(1)

        return processed_count
    
    def auto_detect_and_sort(self, workspace_type="fixed"):
        """自动分拣"""
        if self.initializer.shutdown_flag:
            return
            
        print(f"\n=== Single Arm {workspace_type.title()} Workspace ===")
        print(f"Target: {self.initializer.target_class}")
        # 移动到观察位置
        self.operator.move_to_observe(workspace_type)

        # 检测
        detection_results = self.camera_detection(workspace_type)
        # 协调操作
        processed_count = self.coordinate_operation(detection_results, workspace_type)

        print(f"\nSorting completed! Processed {processed_count} objects.")
    
    def auto_detect_and_sort_conveyor(self):
        """传送带自动分拣"""
        if self.initializer.shutdown_flag:
            return
            
        print(f"\n=== Conveyor Sorting ===")
        print(f"Target: {self.initializer.target_class}")
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
                print(f"\n--- Detection Cycle {cycle_count} ---")
                
                # 确保传送带正在运行（除非批量抓取操作正在进行）
                if not self.operator.conveyor_running and self.batch_pick_count == 0:
                    print("No batch operations, ensuring conveyor is running...")
                    self.operator.control_conveyor("start")
                
                # 检测
                detection_results = self.camera_detection("conveyor")
                
                # 如果检测到目标，执行抓取操作
                if detection_results:
                    print(f"Targets detected! Count: {len(detection_results)}")
                    # 协调操作（传送带将在pick_and_place_object中自动停止）
                    processed_this_cycle = self.coordinate_operation(detection_results, "conveyor")
                    total_processed += processed_this_cycle
                    print(f"Cycle {cycle_count}: Processed {processed_this_cycle} objects (Total: {total_processed})")
                else:
                    print(f"Cycle {cycle_count}: No targets detected, continuing conveyor operation...")
                    processed_this_cycle = 0
                
                # 短暂延迟
                time.sleep(1)
                
        except KeyboardInterrupt:
            print("Detection interrupted by user")
        except Exception as e:
            print(f"Detection error: {e}")
        finally:
            self.auto_detection_running = False
            # 确保传送带停止
            self.operator.emergency_stop_conveyor()
            print(f"Conveyor sorting finished. Total processed: {total_processed}")
    
    def test_camera_connection(self):  
        """相机连接测试"""  
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
        """调试相机连接"""
        print("\n=== Debugging Camera Connection ===")
        
        # 测试相机
        print("Testing camera...")
        try:
            img = self.get_camera_image()
            if img is not None:
                print(f"Undistorted image OK - Shape: {img.shape}")
                
                # 测试工作空间提取
                fixed_workspace = self.transformer.extract_image_workspace(img, "fixed")
                conveyor_workspace = self.transformer.extract_image_workspace(img, "conveyor")
                
                # 显示图像
                cv2.imshow("Camera Debug - Original", img)
                if fixed_workspace is not None:
                    cv2.imshow("Fixed Workspace", fixed_workspace)
                if conveyor_workspace is not None:
                    cv2.imshow("Conveyor Workspace", conveyor_workspace)
                    
                cv2.waitKey(3000)  # 显示3秒
                cv2.destroyAllWindows()
            else:
                print("Camera FAILED - No image received")
        except Exception as e:
            print(f"Camera ERROR: {e}")
    
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
            cv2.waitKey(1)  # 确保窗口关闭
        except:
            pass
            
        # 断开机器人连接
        print("Closing robot connection...")
        try:  
            self.operator.robot.close_connection()  
            print("Robot connection closed")
        except Exception as e:  
            print(f"Error closing robot: {e}")

# ==================== 主程序 ====================
def main():  
    # 配置参数
    ROBOT_IP = "192.168.0.109"  # 单机械臂IP地址
    ONNX_MODEL_PATH = "/home/ned2/Desktop/Waste sorting/code/best.onnx"
    # 标定配置文件
    CALIBRATION_CONFIGS = {
        'fixed': "/home/ned2/Desktop/Waste sorting/code/1203 fixedcalibration_matrix.npy",
        'conveyor': "/home/ned2/Desktop/Waste sorting/code/1203 conveyorcalibration_matrix1.npy"
    }
    # 工作空间配置文件
    WORKSPACE_CONFIGS = {
        'fixed': "/home/ned2/Desktop/Waste sorting/code/workspace_WasteDetector.json",
        'conveyor': "/home/ned2/Desktop/Waste sorting/code/1105robot81workspace_conveyor_WasteDetector.json"
    }
    
    # 垃圾分拣类别
    CLASS_NAMES = ['cardboard', 'glass', 'metal', 'paper', 'plastic']

    print("=== Single Arm Sorting Robot System ===")
    print("Using undistorted images for object detection")
    print("Features:")
    print("- Workspace extraction and visualization")
    print("- Dual window display (detection + workspace)")
    print("- Press 'q' to quit current operation")
    print("- Press Ctrl+C for emergency stop")

    try:
        # 1. 系统初始化
        initializer = SystemInitializer(ROBOT_IP, ONNX_MODEL_PATH, CALIBRATION_CONFIGS, WORKSPACE_CONFIGS)
        robot = initializer.initialize_robot()
        initializer.select_target_class()
        tool = initializer.select_gripper(robot)
        fixed_matrix, conveyor_matrix = initializer.load_calibration_matrices()
        workspaces = initializer.load_workspaces()
        initializer.get_config_summary()
        
        # 2. 视觉检测模块
        detector = VisionDetector(ONNX_MODEL_PATH, CLASS_NAMES)
        
        # 3. 坐标转换模块
        transformer = CoordinateTransformer(fixed_matrix, conveyor_matrix, workspaces)
        
        # 4. 抓取操作模块
        operator = GraspingOperator(robot, tool, initializer.target_class)
        
        # 5. 全程控制模块
        controller = ProcessController(initializer, detector, transformer, operator)
        
        # 测试相机连接
        if controller.test_camera_connection():  
            # 运行调试
            controller.debug_camera_connection()
            
            while not initializer.shutdown_flag:  
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
                    controller.auto_detect_and_sort("fixed")  
                elif choice == "2":  
                    controller.auto_detect_and_sort_conveyor()  
                elif choice == "3":
                    print("\n=== Testing Workspace Visualization ===")
                    img = controller.get_camera_image()
                    if img is not None:
                        # 测试两个工作空间
                        fixed_workspace = transformer.extract_image_workspace(img, "fixed")
                        conveyor_workspace = transformer.extract_image_workspace(img, "conveyor")
                        
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
        if 'controller' in locals():
            controller.cleanup()  
        print(" Program finished")  

if __name__ == "__main__":  
    main()