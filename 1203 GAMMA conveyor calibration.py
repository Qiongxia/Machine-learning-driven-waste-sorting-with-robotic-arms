 # ============================================ 

# Niryo Ned2 4-Point Calibration 1203

# ============================================ 

 

import cv2 

import numpy as np 

from pyniryo import * 

import time 

 

ROBOT_IP = "192.168.0.109" 

SAVE_PATH = "1203 conveyorcalibration_matrix1.npy" 

 

def pixel_to_robot(u, v, transform_matrix, z=0.093): 

    """convert pixel cocrdinates (u,v) to robot coordinates (x,y,z)""" 

    pixel = np.array([u, v, 1]) 

    world = np.dot(transform_matrix, pixel) 

    return world[0], world[1], z 

 

print("Connecting to Niryo Ned2...") 

robot = NiryoRobot(ROBOT_IP) 

robot.calibrate_auto() 

print("Robot connected and calibrated.") 

 

# ====  Manually move to 4 calibration points ==== 

print("\nplease manually guide the bobot to 4 points on the calibration board.") 

world_points = [] 

for i in range(4): 

    input(f"\n移press Enter after moving to point {i+1} ...") 

    pos = robot.get_pose() 

    world_points.append([pos.x, pos.y, pos.z]) 

    print(f"Recorded robot coordinates for point {i+1} : {pos.x:.3f}, {pos.y:.3f}, {pos.z:.3f}") 

 

world_points = np.array(world_points) 

print("\n🧩4 robot coordinate points:\n", world_points) 

 
   # Move to observation position
print("Moving camera to the observation position......")
camera_pose = [0.251, 0.005, 0.254, 2.978, 1.311, 3.003]
robot.move_pose(camera_pose)
time.sleep(2)
print(" Position adjustment complete")
# ==== Capture 1 frame from the Niryo camera ==== 

print("\nCapturing image from from Niryo camera...") 

try: 

    img_compressed = robot.get_img_compressed() 

    img_raw = uncompress_image(img_compressed) 
    mtx, dist = robot.get_camera_intrinsics()
    img = undistort_image(img_raw, mtx, dist)

except Exception as e: 

    robot.close_connection() 

    raise RuntimeError(f"❌ Failed to capture image: {e}") 

 

print("✅ Image successfully captured...") 

 

clicked_points = [] 

 

def mouse_callback(event, x, y, flags, param): 

    if event == cv2.EVENT_LBUTTONDOWN: 

        clicked_points.append((x, y)) 

        print(f"✅ Clicked point {len(clicked_points)} : (u={x}, v={y})") 

 

cv2.namedWindow("Calibration Image") 

cv2.setMouseCallback("Calibration Image", mouse_callback) 

 

print("\n👉Click on the 4 points in the iamge that correspond to the robot positions.(press q to exit）。。") 

 

while True: 

    temp_img = img.copy() 

    for i, (x, y) in enumerate(clicked_points): 

        cv2.circle(temp_img, (x, y), 6, (0, 0, 255), -1) 

        cv2.putText(temp_img, f"P{i+1}", (x + 8, y - 8), 

                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2) 

    cv2.imshow("Calibration Image", temp_img) 

    key = cv2.waitKey(1) & 0xFF 

    if key == ord('q') or len(clicked_points) >= 4: 

        break 

 

cv2.destroyAllWindows() 

 

if len(clicked_points) < 4: 

    robot.close_connection() 

    raise RuntimeError("❌ Not enough points selected(need 4 points）") 

 

image_points = np.array(clicked_points) 

print("\n🖼️ Image coordinate points:\n", image_points) 

 

# ==== Calculate affine transformation matrix ==== 

A, B = [], [] 

for i in range(4): 

    u, v = image_points[i] 

    A.append([u, v, 1, 0, 0, 0]) 

    A.append([0, 0, 0, u, v, 1]) 

    B.append(world_points[i][0]) 

    B.append(world_points[i][1]) 

 

A, B = np.array(A), np.array(B) 

X, _, _, _ = np.linalg.lstsq(A, B, rcond=None) 

transform_matrix = np.array([ 

    [X[0], X[1], X[2]], 

    [X[3], X[4], X[5]], 

    [0,     0,     1] 

]) 

 

print("\n📐Affine transformation matrix (image →robot）：\n", transform_matrix) 

np.save(SAVE_PATH, transform_matrix) 

print(f"\n💾 Saved to {SAVE_PATH}") 

 

# ==== Test the calibration==== 

print("\n🧠 Test the calibration：") 

while True: 

    ans = input("test (y/n): ").lower() 

    if ans == 'n': 

        break 

    elif ans == 'y': 

        u = float(input("type the pixel u coordinate: ")) 

        v = float(input("type the pixel v coordinate: ")) 

        x, y, z = pixel_to_robot(u, v, transform_matrix) 

        print(f"pixel ({u:.1f}, {v:.1f}) → robot ({x:.3f}, {y:.3f}, {z:.3f})") 

    else: 

        print("please tyoe y or n") 

 

robot.close_connection() 

print("\n✅ Calibration complete") 
