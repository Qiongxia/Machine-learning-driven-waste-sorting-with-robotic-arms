from pyniryo import NiryoRobot
import json
from datetime import datetime

# robotic arm connection
robot_ip = "192.168.0.106"  
workspace_name = "Workspace_1124TEST robot81conveyor"
json_filename = "1124TESTrobot81workspace_conveyor.json"

# connect to the robot
robot = NiryoRobot(robot_ip)
robot.calibrate_auto()
robot.set_learning_mode(True)

print(f"=== creat workspace: {workspace_name} ===")

# get the 4 points
points = []
for i in range(4):
    
    input(f"Please move to point {i+1} and press Enter...")
    pose = robot.get_pose()
    points.append(pose)
    print(f"point {i+1}: X={pose.x:.3f}, Y={pose.y:.3f}, Z={pose.z:.3f}")

# creat the workspace
robot.save_workspace_from_robot_poses(workspace_name, *points)
print(f"Workspace '{workspace_name}' created successfully")

# prepare data to be saved to the JSON
workspace_data = {
    "workspace_name": workspace_name,
    "created_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "robot_ip": robot_ip,
    "points": []
}

# Add point data
for i, pose in enumerate(points):
    point_info = {
        "point_id": i + 1,
        "x": float(pose.x),
        "y": float(pose.y),
        "z": float(pose.z),
        "roll": float(pose.roll),
        "pitch": float(pose.pitch),
        "yaw": float(pose.yaw)
    }
    workspace_data["points"].append(point_info)

# save to JSON
with open(json_filename, 'w') as f:
    json.dump(workspace_data, f, indent=2)

print(f"Workspace data has been saved to: {json_filename}")

# close the connection
robot.close_connection()

# show the saved info
print(f"\n The saved workspace info:")
print(json.dumps(workspace_data, indent=2))
