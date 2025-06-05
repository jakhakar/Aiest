import bpy
import json
import sys
import os

# --- Get arguments passed after '--' ---
args = sys.argv
plan_filepath = None
if "--plan" in args:
    try:
        plan_filepath = args[args.index("--plan") + 1]
    except IndexError:
        print("Error: --plan argument given but no filepath provided.")
        sys.exit(1)

if not plan_filepath or not os.path.exists(plan_filepath):
    print(f"Error: Animation plan file not found or not specified: {plan_filepath}")
    # Could proceed with a default animation if no plan, or exit
    # sys.exit(1) 
    # For now, let's assume a default if no plan for simplicity
    animation_plan = {
        "scenes": [
            {"target_lat": 0, "target_lon": 0, "duration_frames": 100, "text": "Default View"}
        ],
        "output_dir": "/tmp/frames/", # Default output
        "fps": 30
    }
else:
    with open(plan_filepath, 'r') as f:
        animation_plan = json.load(f)

output_dir = animation_plan.get("output_dir", "/tmp/frames/")
fps = animation_plan.get("fps", 30)

os.makedirs(output_dir, exist_ok=True)

scene = bpy.context.scene
scene.render.image_settings.file_format = 'PNG'
scene.render.fps = fps

# --- Camera Setup ---
# Assume you have a camera named "Camera" in your .blend file
# You might have multiple cameras or animate one camera based on the plan
cam = bpy.data.objects.get("Camera")
if not cam:
    print("Error: Camera object named 'Camera' not found in .blend file.")
    sys.exit(1)

# --- Animation Loop based on animation_plan.json ---
current_frame = 1
for i, plan_scene in enumerate(animation_plan["scenes"]):
    print(f"Processing planned scene {i+1}: {plan_scene.get('text', 'No Text')}")
    
    target_lat = plan_scene.get("target_lat", 0)
    target_lon = plan_scene.get("target_lon", 0)
    # Add more params: distance, roll, pitch, yaw, text content, text animation in Blender etc.
    
    # --- Animate Camera to target (simplified example) ---
    # In a real script, you'd interpolate camera location/rotation over several frames
    # For example, set camera location to point at (lat, lon) from a certain distance/angle
    # This requires converting lat/lon to 3D coordinates on your sphere's surface
    # and then positioning the camera accordingly.
    # cam.location = (x, y, z) 
    # cam.rotation_euler = (rx, ry, rz)

    # --- Create/Update Text (Blender Text Object) ---
    # text_content = plan_scene.get("text", "")
    # Find or create a Blender text object, set its content, material, animation (keyframes for visibility/transform)
    
    # --- Render frames for this part of the plan ---
    start_render_frame = current_frame
    end_render_frame = current_frame + plan_scene.get("duration_frames", fps * 2) # Default 2 seconds

    for frame_to_render in range(start_render_frame, end_render_frame):
        scene.frame_set(frame_to_render)
        scene.render.filepath = os.path.join(output_dir, f"frame_{frame_to_render:04d}.png")
        bpy.ops.render.render(write_still=True)
        print(f"Rendered frame: {scene.render.filepath}")
    
    current_frame = end_render_frame

print("Blender script finished.")
