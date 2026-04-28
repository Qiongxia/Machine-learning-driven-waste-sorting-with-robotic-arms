from pathlib import Path
from datetime import datetime
from pyniryo import *
import cv2
import time

def capture_multi_angle_photos():
    """Capture photos from 4 different observation poses"""
    print("Robot Arm Multi-Angle Photo Capture")
    print("="*50)
    
    # Get custom base name
    print("\nEnter photo base name (e.g., metal1, plastic2):")
    base_name = input("Base name: ").strip()
    
    if not base_name:
        base_name = "item"
    
    # Clean the name
    clean_base_name = base_name.replace(" ", "_").replace("/", "_").replace("\\", "_")
    
    try:
        # Connect to robot
        print(f"\n🔌 Connecting to robot arm...")
        robot = NiryoRobot("192.168.0.102")
        print("✅ Connected successfully!")
        
        # Define 4 observation poses
        poses = [
            {
                "name": "pose_1",
                "pose": PoseObject(0.004, 0.168, 0.182, -3.122, 1.198, -1.552),
                "description": "Top view"
            },
            {
                "name": "pose_2", 
                "pose": PoseObject(0.052, 0.181, 0.189, 2.563, 1.164, -2.684),
                "description": "Left side view"
            },
            {
                "name": "pose_3",
                "pose": PoseObject(-0.044, 0.191, 0.192, -2.599, 1.175, -0.553),
                "description": "Right side view"
            },
            {
                "name": "pose_4",
                "pose": PoseObject(0.004, 0.168, 0.150, -3.122, 1.198, -1.552),
                "description": "Top view (lower height)"
            }
        ]
        
        # Create save directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_dir = Path.home() / "Desktop" / "2026_WasteSorting" / f"0204Robot_Photos_{timestamp}"
        session_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize list to store captured file paths
        captured_files = []
        
        print(f"\n📁 Save directory: {session_dir}")
        print(f"📸 Will capture 4 photos from different angles")
        print(f"📷 Photo naming: {clean_base_name}, {clean_base_name}-1, {clean_base_name}-2, {clean_base_name}-3")
        print("-" * 50)
        
        # Ask for preview before starting
        print("\nDo you want to preview camera before each capture? (y/n):")
        preview_option = input("Preview option: ").strip().lower()
        enable_preview = preview_option == 'y'
        
        # Move to home position first (safety)
        print("\n🤖 Moving to home position...")
        robot.move_joints([0, 0, 0, 0, 0, 0])  # Adjust to your robot's home position
        time.sleep(1)
        
        # Capture from each pose
        for i, pose_info in enumerate(poses):
            pose_name = pose_info["name"]
            pose = pose_info["pose"]
            description = pose_info["description"]
            
            print(f"\n📍 Position {i+1}/4: {description}")
            print(f"   Target pose: {pose}")
            
            try:
                # Move to observation pose
                print(f"   Moving to position {i+1}...")
                robot.move_pose(pose)
                time.sleep(0.5)  # Short pause for stabilization
                
                # Optional preview
                if enable_preview:
                    print("   Previewing camera (Press any key to capture, ESC to skip)...")
                    try:
                        img_compressed = robot.get_img_compressed()
                        img_raw = uncompress_image(img_compressed)
                        preview_img = undistort_image(img_raw)
                        
                        # Add preview info
                        cv2.putText(preview_img, f"Position {i+1}: {description}", 
                                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        cv2.putText(preview_img, "Press any key to capture, ESC to skip", 
                                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                        
                        cv2.imshow("Camera Preview", img_raw)
                        key = cv2.waitKey(0)
                        cv2.destroyAllWindows()
                        
                        if key == 27:  # ESC key
                            print("   ⏭️ Skipping this position...")
                            continue
                            
                    except Exception as e:
                        print(f"   ⚠️ Preview failed: {e}")
                        print("   Continuing with capture...")
                
                # Countdown and capture
                print("   Preparing to capture...")
                for count in range(3, 0, -1):
                    print(f"   {count}...", end=' ', flush=True)
                    time.sleep(1)
                print("   Capture!", flush=True)
                
                # Capture image
                img_compressed = robot.get_img_compressed()
                img_raw = uncompress_image(img_compressed)
                                
                # Generate filename
                if i == 0:
                    filename = f"{clean_base_name}.jpg"
                else:
                    filename = f"{clean_base_name}-{i}.jpg"
                
                filepath = session_dir / filename
                
                # Save image
                success = cv2.imwrite(str(filepath), img_raw)
                
                if success:
                    print(f"   ✅ Photo saved: {filename}")
                    print(f"   📏 Dimensions: {img_raw.shape[1]}x{img_raw.shape[0]}")
                    
                    # Display captured photo briefly
                    display_img = img_raw.copy()
                    cv2.putText(display_img, f"Pos {i+1}: {description}", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv2.putText(display_img, f"File: {filename}", (10, 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                    
                    cv2.imshow(f"Captured: {filename}", display_img)
                    cv2.waitKey(1000)  # Display for 1 second
                    cv2.destroyAllWindows()
                    
                    captured_files.append(str(filepath))
                else:
                    print(f"   ❌ Failed to save {filename}")
                
            except Exception as e:
                print(f"   ❌ Error at position {i+1}: {e}")
                print("   Continuing to next position...")
                continue
            
            # Short pause between positions
            if i < len(poses) - 1:
                print("   Moving to next position in 2 seconds...")
                time.sleep(2)
        
        # Return to home position
        print("\n🤖 Returning to home position...")
        robot.move_joints([0, 0, 0, 0, 0, 0])
        
        # Close robot connection
        robot.close_connection()
        print("🔌 Robot connection closed")
        
        # Summary
        print("\n" + "="*50)
        print("📊 CAPTURE SESSION SUMMARY")
        print("="*50)
        print(f"Base name: {clean_base_name}")
        print(f"Session time: {timestamp}")
        print(f"Save directory: {session_dir}")
        print(f"\nCaptured {len(captured_files)} out of 4 photos:")
        
        for i, filepath in enumerate(captured_files):
            filename = Path(filepath).name
            print(f"  {i+1}. {filename}")
        
        if captured_files:
            print(f"\n✅ All photos saved in: {session_dir}")
        else:
            print("\n❌ No photos were captured successfully")
        
        return captured_files
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        
        # Ensure all OpenCV windows are closed
        try:
            cv2.destroyAllWindows()
        except:
            pass
        
        return None

def display_session_summary(files):
    """Display the captured photos in a grid"""
    if not files or len(files) == 0:
        return
    
    print("\n🖼️ Displaying captured photos in grid...")
    
    # Read all images
    images = []
    for filepath in files:
        img = cv2.imread(filepath)
        if img is not None:
            # Resize for display
            img_resized = cv2.resize(img, (400, 300))
            # Add filename
            filename = Path(filepath).name
            cv2.putText(img_resized, filename, (10, 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            images.append(img_resized)
    
    if not images:
        return
    
    # Create grid
    if len(images) == 1:
        grid = images[0]
    elif len(images) == 2:
        grid = cv2.hconcat(images)
    elif len(images) == 3:
        # Create 2x2 grid with last cell black
        row1 = cv2.hconcat([images[0], images[1]])
        black_cell = np.zeros((300, 400, 3), dtype=np.uint8)
        row2 = cv2.hconcat([images[2], black_cell])
        grid = cv2.vconcat([row1, row2])
    else:  # 4 images
        row1 = cv2.hconcat([images[0], images[1]])
        row2 = cv2.hconcat([images[2], images[3]])
        grid = cv2.vconcat([row1, row2])
    
    cv2.imshow(f"Captured Photos ({len(images)} total)", grid)
    print("Press any key to close the grid view...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Run the multi-angle capture function
    captured_files = capture_multi_angle_photos()
    
    # Display summary grid
    if captured_files:
        display_session_summary(captured_files)
    
    print("\n🎯 Multi-angle photo capture completed!")