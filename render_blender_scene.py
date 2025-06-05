import bpy
import json
import sys
import os
import math
from mathutils import Vector, Euler

# --- Helper Functions ---
def latlon_to_cartesian(lat, lon, radius):
    """Converts Lat/Lon (degrees) to Cartesian (X,Y,Z) coordinates on a sphere."""
    lat_rad = math.radians(lat)
    lon_rad = math.radians(lon) # Blender's Z is up, X forward, Y left by default
                                # If map texture aligns with lon=0 at +X, then:
    x = radius * math.cos(lat_rad) * math.cos(lon_rad)
    y = radius * math.cos(lat_rad) * math.sin(lon_rad) # Y for longitude
    z = radius * math.sin(lat_rad)                     # Z for latitude
    return Vector((x, y, z))

def look_at(obj_camera, point_target_on_sphere, distance, earth_radius):
    """
    Positions and orients the camera to look at a point on the sphere.
    'point_target_on_sphere' is a mathutils.Vector.
    'distance' is camera distance from the *surface* of the sphere.
    """
    # Direction from sphere center to target point
    direction_to_target = point_target_on_sphere.normalized()
    
    # Camera position: along the direction_to_target, but further out
    camera_pos = direction_to_target * (earth_radius + distance)
    obj_camera.location = camera_pos
    
    # Point camera towards the sphere's center (or slightly towards the target point for better framing)
    # This uses a Track To constraint for simplicity and robustness
    # Remove existing track to constraints first
    for c in obj_camera.constraints:
        if c.type == 'TRACK_TO':
            obj_camera.constraints.remove(c)
            
    ttc = obj_camera.constraints.new(type='TRACK_TO')
    ttc.target = bpy.data.objects.get("Earth") # Assume Earth object exists and is at origin
    if ttc.target:
        ttc.track_axis = 'TRACK_NEGATIVE_Z' # Camera's -Z points forward
        ttc.up_axis = 'UP_Y' # Camera's Y is up
    else: # Fallback if Earth object not found (should not happen if created)
        # Simple look_at math (less robust than constraint for roll)
        direction_vector = point_target_on_sphere - obj_camera.location
        obj_camera.rotation_euler = direction_vector.to_track_quat('-Z', 'Y').to_euler()


def setup_scene(earth_radius, earth_texture_path, space_texture_path):
    """Creates the basic Earth, lighting, camera, and space background."""
    # --- Clear existing default scene objects ---
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

    # --- Create Earth Sphere ---
    bpy.ops.mesh.primitive_uv_sphere_add(radius=earth_radius, location=(0, 0, 0), segments=64, ring_count=32)
    earth_obj = bpy.context.active_object
    earth_obj.name = "Earth"
    bpy.ops.object.shade_smooth() # Smooth shading

    # --- Earth Material and Texture ---
    mat_earth = bpy.data.materials.new(name="EarthMaterial")
    earth_obj.data.materials.append(mat_earth)
    mat_earth.use_nodes = True
    bsdf = mat_earth.node_tree.nodes.get('Principled BSDF')
    tex_image_node = mat_earth.node_tree.nodes.new('ShaderNodeTexImage')
    
    if os.path.exists(earth_texture_path):
        tex_image_node.image = bpy.data.images.load(earth_texture_path)
        print(f"Loaded Earth texture: {earth_texture_path}")
    else:
        print(f"WARNING: Earth texture not found at {earth_texture_path}. Earth will be plain.")
        bsdf.inputs['Base Color'].default_value = (0.1, 0.2, 0.7, 1) # Blueish
        
    mat_earth.node_tree.links.new(bsdf.inputs['Base Color'], tex_image_node.outputs['Color'])
    # Add UV Map node if needed, but default sphere UVs should work with equirectangular
    tex_coord_node = mat_earth.node_tree.nodes.new('ShaderNodeTexCoord')
    mat_earth.node_tree.links.new(tex_image_node.inputs['Vector'], tex_coord_node.outputs['UV'])


    # --- Create Camera ---
    bpy.ops.object.camera_add(location=(0, - (earth_radius + 5), 0)) # Initial position
    cam_obj = bpy.context.active_object
    cam_obj.name = "SceneCamera"
    bpy.context.scene.camera = cam_obj
    cam_obj.data.lens = 35 # Focal length

    # --- Create Lighting (Sun Lamp) ---
    bpy.ops.object.light_add(type='SUN', location=(5, -5, 5))
    sun_obj = bpy.context.active_object
    sun_obj.name = "SunLight"
    sun_obj.data.energy = 3.0
    sun_obj.rotation_euler = Euler((math.radians(45), math.radians(0), math.radians(-45)), 'XYZ')


    # --- World Background (Space) ---
    world_bg = bpy.context.scene.world
    if world_bg is None:
        world_bg = bpy.data.worlds.new("SpaceBackground")
        bpy.context.scene.world = world_bg
    world_bg.use_nodes = True
    bg_tree = world_bg.node_tree
    bg_node = bg_tree.nodes.get('Background') # Default background node

    if os.path.exists(space_texture_path):
        env_tex_node = bg_tree.nodes.new('ShaderNodeTexEnvironment')
        env_tex_node.image = bpy.data.images.load(space_texture_path)
        print(f"Loaded space background texture: {space_texture_path}")
        # Connect Environment Texture to Background Color
        tex_coord_world_node = bg_tree.nodes.new('ShaderNodeTexCoord') # For generated coords
        bg_tree.links.new(env_tex_node.inputs['Vector'], tex_coord_world_node.outputs['Generated'])
        bg_tree.links.new(bg_node.inputs['Color'], env_tex_node.outputs['Color'])
        bg_node.inputs['Strength'].default_value = 0.7
    else:
        print(f"WARNING: Space background texture not found at {space_texture_path}. Using black.")
        bg_node.inputs['Color'].default_value = (0.0, 0.0, 0.0, 1.0) # Black

    return earth_obj, cam_obj

def create_country_name_text(text_content, earth_radius, target_cartesian_on_sphere):
    """Creates a 3D text object for the country name."""
    bpy.ops.object.text_add(location=(0,0,0))
    text_obj = bpy.context.active_object
    text_obj.name = f"Text_{text_content.replace(' ','_')}"
    text_obj.data.body = text_content.upper()
    text_obj.data.font = bpy.data.fonts.load(CUSTOM_FONT_PATH) if os.path.exists(CUSTOM_FONT_PATH) else None # Requires font file accessible to Blender
    
    text_obj.data.align_x = 'CENTER'
    text_obj.data.align_y = 'CENTER'
    text_obj.data.extrude = 0.02
    text_obj.data.bevel_depth = 0.005
    text_obj.data.size = 0.15 # Initial size, can be keyframed

    # Position text slightly above the target country on the sphere
    text_offset_factor = 1.05 # Slightly above surface
    text_pos = target_cartesian_on_sphere.normalized() * earth_radius * text_offset_factor
    text_obj.location = text_pos
    
    # Orient text to face outwards from sphere (away from center) and be upright
    # This is tricky. A common way is to make it billboard towards the camera or use constraints.
    # For simplicity, align its local Z with the normal of the sphere at that point.
    # And its local Y roughly "up" in world space.
    normal = target_cartesian_on_sphere.normalized()
    up_vector = Vector((0,0,1)) # World Z up
    if abs(normal.dot(up_vector)) > 0.99: # If normal is too close to Z up (poles)
        up_vector = Vector((0,1,0)) # Use world Y up instead
    
    text_obj.rotation_euler = normal.to_track_quat('Z', 'Y').to_euler() # Z aims out, Y is up for text
    # May need further rotation adjustment to make text readable from camera angle

    # Material for text
    mat_text = bpy.data.materials.new(name=f"TextMat_{text_content}")
    mat_text.use_nodes = True
    bsdf_text = mat_text.node_tree.nodes.get('Principled BSDF')
    bsdf_text.inputs['Base Color'].default_value = (1.0, 1.0, 1.0, 1.0) # White
    bsdf_text.inputs['Emission'].default_value = (0.8, 0.8, 0.8, 1) # Slight emission
    bsdf_text.inputs['Emission Strength'].default_value = 0.5
    text_obj.data.materials.append(mat_text)
    
    text_obj.hide_render = True # Initially hidden
    text_obj.scale = (0.01, 0.01, 0.01) # Start small
    return text_obj

def create_splash_effect_object(splash_color_rgb, earth_radius, target_cartesian_on_sphere):
    """Creates an object to simulate color splash (e.g., a spotlight or emissive plane)."""
    # Using a spotlight for simplicity
    bpy.ops.object.light_add(type='SPOT', radius=0.1, location=(0,0,0)) # Radius is spotlight size
    splash_light = bpy.context.active_object
    splash_light.name = "SplashLight"
    splash_light.data.color = splash_color_rgb
    splash_light.data.energy = 0 # Initially off
    splash_light.data.spot_size = math.radians(30) # Angle of the cone
    splash_light.data.spot_blend = 0.5 # Softness of edge
    splash_light.data.use_custom_distance = True
    splash_light.data.distance = earth_radius * 0.5 # How far light reaches

    # Position spotlight slightly above surface, pointing at it
    light_pos = target_cartesian_on_sphere.normalized() * (earth_radius + 0.1)
    splash_light.location = light_pos
    
    # Point light towards target on sphere (effectively towards origin if target is on sphere)
    direction_vector = target_cartesian_on_sphere - splash_light.location
    splash_light.rotation_euler = direction_vector.to_track_quat('-Z', 'Y').to_euler()
    
    return splash_light

# --- Main Script Execution ---
def run_blender_animation(plan_filepath, earth_texture_path, space_texture_path, output_dir_base):
    
    if not os.path.exists(plan_filepath):
        print(f"ERROR: Animation plan file not found: {plan_filepath}")
        sys.exit(1)
    with open(plan_filepath, 'r') as f:
        animation_plan = json.load(f)

    # --- Scene Setup ---
    bpy.ops.wm.read_factory_settings(use_empty=True) # Start with a blank scene
    scene = bpy.context.scene
    earth_obj, cam_obj = setup_scene(animation_plan["earth_radius"], earth_texture_path, space_texture_path)

    # --- Render Settings ---
    scene.render.engine = 'BLENDER_EEVEE' # Eevee is faster for this kind of work
    scene.eevee.taa_render_samples = 32 # Decent quality for Eevee
    scene.eevee.use_bloom = True # For glows
    scene.render.image_settings.file_format = 'PNG'
    scene.render.fps = animation_plan["fps"]
    scene.render.resolution_x = VIDEO_WIDTH_PX # Use global config
    scene.render.resolution_y = VIDEO_HEIGHT_PX
    scene.render.resolution_percentage = 100

    # --- Animation Loop ---
    earth_radius = animation_plan["earth_radius"]
    
    # Initial camera position (global view)
    cam_obj.location = (0, -(earth_radius + earth_radius * 3), earth_radius * 1.5) # Further out, slightly elevated
    look_at(cam_obj, Vector((0,0,0)), earth_radius * 3, earth_radius) # Look at center
    cam_obj.keyframe_insert(data_path="location", frame=1)
    cam_obj.keyframe_insert(data_path="rotation_euler", frame=1)

    # Store previous camera state for smooth transitions
    prev_cam_location = cam_obj.location.copy()
    prev_cam_rotation = cam_obj.rotation_euler.copy()

    for scene_config in animation_plan["scenes"]:
        print(f"Configuring animation for: {scene_config['target_country_name']}")
        
        start_transition_frame = scene_config["start_frame_offset"] + 1
        end_transition_frame = start_transition_frame + scene_config["transition_frames"]
        start_focus_frame = end_transition_frame
        end_focus_frame = start_focus_frame + scene_config["focus_frames"]

        # --- Target Info ---
        target_cartesian = latlon_to_cartesian(scene_config["target_lat"], scene_config["target_lon"], earth_radius)
        
        # --- Camera Animation to Target ---
        # Calculate target camera position and rotation
        # Store current cam state as start of transition
        cam_obj.keyframe_insert(data_path="location", frame=start_transition_frame -1 if start_transition_frame > 1 else 1)
        cam_obj.keyframe_insert(data_path="rotation_euler", frame=start_transition_frame -1 if start_transition_frame > 1 else 1)

        # Set camera at target position for end of transition / start of focus
        look_at(cam_obj, target_cartesian, earth_radius * scene_config["camera_distance_factor"], earth_radius)
        cam_obj.keyframe_insert(data_path="location", frame=start_focus_frame)
        cam_obj.keyframe_insert(data_path="rotation_euler", frame=start_focus_frame)
        
        # Camera holds position during focus, then will transition from this state
        cam_obj.keyframe_insert(data_path="location", frame=end_focus_frame)
        cam_obj.keyframe_insert(data_path="rotation_euler", frame=end_focus_frame)

        # --- Country Name Text Animation ---
        text_obj = create_country_name_text(scene_config["text_to_display"], earth_radius, target_cartesian)
        text_appear_start = start_focus_frame + int(scene_config["focus_frames"] * 0.2)
        text_appear_end = text_appear_start + int(FPS * 0.7) # 0.7 sec fade/scale in
        text_disappear_start = end_focus_frame - int(FPS * 0.7)
        
        text_obj.scale = (0.01, 0.01, 0.01)
        text_obj.keyframe_insert(data_path="scale", frame=start_focus_frame)
        text_obj.keyframe_insert(data_path="hide_render", frame=start_focus_frame)

        text_obj.hide_render = False
        text_obj.scale = (1.0, 1.0, 1.0) # Target scale for text
        text_obj.keyframe_insert(data_path="scale", frame=text_appear_end)
        text_obj.keyframe_insert(data_path="hide_render", frame=text_appear_start) # Becomes visible

        text_obj.keyframe_insert(data_path="scale", frame=text_disappear_start) # Hold scale
        text_obj.keyframe_insert(data_path="hide_render", frame=text_disappear_start) # Still visible

        text_obj.scale = (0.01, 0.01, 0.01)
        text_obj.hide_render = True
        text_obj.keyframe_insert(data_path="scale", frame=end_focus_frame)
        text_obj.keyframe_insert(data_path="hide_render", frame=end_focus_frame)


        # --- "Color Splash" (Spotlight) Animation ---
        splash_obj = create_splash_effect_object(scene_config["splash_color_rgb"], earth_radius, target_cartesian)
        splash_appear_start = start_focus_frame + int(scene_config["focus_frames"] * 0.1)
        splash_peak_frame = splash_appear_start + int(FPS * 0.5) # Time to full intensity
        splash_fade_start = end_focus_frame - int(FPS * 1.0) # Start fading 1s before end of focus
        splash_fade_end = end_focus_frame - int(FPS * 0.2)

        splash_obj.data.energy = 0
        splash_obj.keyframe_insert(data_path="data.energy", frame=start_focus_frame)
        
        splash_obj.data.energy = 2000 # Bright splash (adjust based on Eevee settings)
        splash_obj.keyframe_insert(data_path="data.energy", frame=splash_peak_frame)

        splash_obj.keyframe_insert(data_path="data.energy", frame=splash_fade_start) # Hold energy
        
        splash_obj.data.energy = 0
        splash_obj.keyframe_insert(data_path="data.energy", frame=splash_fade_end)
        splash_obj.keyframe_insert(data_path="data.energy", frame=end_focus_frame) # Ensure it's off

        # --- TODO: Glowing Border Simulation ---
        # This requires more complex geometry handling or advanced shader tricks.
        # For now, skipping the direct implementation of glowing border from scratch.
        # It would involve creating a curve object along the approximate border,
        # giving it an emission material, and keyframing its visibility/emission strength.

    # --- Set total animation length ---
    scene.frame_start = 1
    scene.frame_end = animation_plan["total_animation_frames"]

    # --- Render Animation ---
    output_frame_pattern = os.path.join(output_dir_base, "frame_#####") # Blender uses # for padding
    scene.render.filepath = output_frame_pattern
    print(f"Rendering animation frames to: {output_dir_base}")
    bpy.ops.render.render(animation=True)

    print("Blender rendering process finished.")


if __name__ == "__main__":
    # --- Argument Parsing for paths (when run with blender -P script.py -- args) ---
    args = sys.argv
    argv = args[args.index("--") + 1:] if "--" in args else []

    # Defaults (can be overridden by command line args if you set them up in GitHub Actions)
    # These paths are relative to where the script is run from, or absolute.
    # In GitHub Actions, you'd typically use paths relative to GITHUB_WORKSPACE
    
    # Path to the JSON file generated by generate_animation_plan.py
    plan_file = argv[0] if len(argv) > 0 else "animation_plan.json"
    
    # Path to your Earth texture image
    earth_tex = argv[1] if len(argv) > 1 else "earth_texture.jpg" # EXPECTS THIS FILE
    
    # Path to your space background image (optional)
    space_tex = argv[2] if len(argv) > 2 else "space_background.jpg" # EXPECTS THIS FILE
    
    # Output directory for rendered frames (Blender will create if not exists)
    output_frames_dir = argv[3] if len(argv) > 3 else "/tmp/blender_frames/" # Use /tmp for Actions

    # Ensure output directory exists for Blender
    if not os.path.exists(output_frames_dir):
        os.makedirs(output_frames_dir, exist_ok=True)
        print(f"Created output directory for frames: {output_frames_dir}")

    # Check if texture files exist
    if not os.path.exists(earth_tex):
        print(f"FATAL ERROR: Earth texture file not found at '{earth_tex}'. Please provide the correct path.")
        sys.exit(1)
    if space_tex and not os.path.exists(space_tex): # Only error if specified and not found
        print(f"WARNING: Space texture file '{space_tex}' not found. Will use black background.")
        space_tex = "" # Set to empty so script knows not to load it

    # --- Video dimensions (ensure these are available to Blender script) ---
    # These are defined globally at the start of this file.
    # If this script were separate, you'd pass them or read from config.
    VIDEO_WIDTH_PX = 720 # Or read from plan_json if you add it there
    VIDEO_HEIGHT_PX = 1280


    run_blender_animation(plan_file, earth_tex, space_tex, output_frames_dir)
