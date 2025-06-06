import bpy
import json
import sys
import os
import math
from mathutils import Vector, Euler # For 3D vector and rotation math

# --- Script-Level Configuration (Defaults, can be overridden by plan or args) ---
# These are used by Blender for render output if not specified elsewhere.
# The GitHub Actions workflow can also set these via environment variables if preferred.
VIDEO_WIDTH_PX = 1080
VIDEO_HEIGHT_PX = 1920

# --- Helper Functions ---
def latlon_to_cartesian(lat, lon, radius):
    """Converts Lat/Lon (degrees) to Cartesian (X,Y,Z) coordinates on a sphere."""
    lat_rad = math.radians(lat)
    lon_rad = math.radians(lon)
    # Blender's default coordinate system: Z is up, Y is typically depth or away from view, X is horizontal.
    # Standard geographic to Cartesian for Z-up:
    x = radius * math.cos(lat_rad) * math.cos(lon_rad)
    y = radius * math.cos(lat_rad) * math.sin(lon_rad)
    z = radius * math.sin(lat_rad)
    return Vector((x, y, z))

def look_at(obj_camera, point_target_on_sphere, distance_from_surface, earth_radius):
    """
    Positions and orients the camera to look at a point on the sphere.
    'point_target_on_sphere' is a mathutils.Vector representing a point on the sphere's surface.
    'distance_from_surface' is camera distance from the *surface* of the sphere.
    """
    direction_to_target = point_target_on_sphere.normalized()
    camera_total_distance_from_center = earth_radius + distance_from_surface
    camera_pos = direction_to_target * camera_total_distance_from_center
    obj_camera.location = camera_pos

    # Use Track To constraint for robust aiming
    for c in obj_camera.constraints: # Clear existing Track To constraints
        if c.type == 'TRACK_TO':
            obj_camera.constraints.remove(c)
            
    ttc = obj_camera.constraints.new(type='TRACK_TO')
    earth_obj = bpy.data.objects.get("Earth") # Assumes Earth object is named "Earth" and at origin
    if earth_obj:
        ttc.target = earth_obj
    else: # Fallback: target the origin if Earth object not found (shouldn't happen)
        # Create an empty at origin if needed for robust tracking
        empty_target = bpy.data.objects.get("LookAtTarget")
        if not empty_target:
            bpy.ops.object.empty_add(type='PLAIN_AXES', location=(0,0,0))
            empty_target = bpy.context.active_object
            empty_target.name = "LookAtTarget"
        ttc.target = empty_target

    ttc.track_axis = 'TRACK_NEGATIVE_Z' # Camera's local -Z points forward
    ttc.up_axis = 'UP_Y'                # Camera's local Y is up

def setup_scene(earth_radius, earth_texture_path, space_texture_path):
    """Creates the basic Earth, lighting, camera, and space background."""
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete() # Clear default cube, light, camera

    # --- Earth Sphere ---
    bpy.ops.mesh.primitive_uv_sphere_add(radius=earth_radius, location=(0, 0, 0), segments=128, ring_count=64) # Higher poly
    earth_obj = bpy.context.active_object
    earth_obj.name = "Earth"
    bpy.ops.object.shade_smooth()

    mat_earth = bpy.data.materials.new(name="EarthMaterial")
    earth_obj.data.materials.append(mat_earth)
    mat_earth.use_nodes = True
    nodes = mat_earth.node_tree.nodes
    bsdf = nodes.get('Principled BSDF')
    if not bsdf: bsdf = nodes.new('ShaderNodeBsdfPrincipled') # Should exist by default

    tex_image_node = nodes.new('ShaderNodeTexImage')
    if os.path.exists(earth_texture_path):
        try:
            tex_image_node.image = bpy.data.images.load(earth_texture_path)
            print(f"Loaded Earth texture: {earth_texture_path}")
            mat_earth.node_tree.links.new(bsdf.inputs['Base Color'], tex_image_node.outputs['Color'])
        except RuntimeError as e:
            print(f"ERROR loading Earth texture '{earth_texture_path}': {e}. Using blue color.")
            bsdf.inputs['Base Color'].default_value = (0.1, 0.2, 0.7, 1)
    else:
        print(f"WARNING: Earth texture not found at {earth_texture_path}. Using blue color.")
        bsdf.inputs['Base Color'].default_value = (0.1, 0.2, 0.7, 1)
    
    tex_coord_node = nodes.new('ShaderNodeTexCoord')
    mat_earth.node_tree.links.new(tex_image_node.inputs['Vector'], tex_coord_node.outputs['UV'])

    # --- Camera ---
    bpy.ops.object.camera_add(location=(0, -(earth_radius + 5), 0))
    cam_obj = bpy.context.active_object
    cam_obj.name = "SceneCamera"
    bpy.context.scene.camera = cam_obj
    cam_obj.data.lens = 35 # mm focal length

    # --- Lighting (Sun) ---
    bpy.ops.object.light_add(type='SUN', location=(earth_radius * 3, -earth_radius * 3, earth_radius * 2))
    sun_obj = bpy.context.active_object
    sun_obj.name = "SunLight"
    sun_obj.data.energy = 5.0 # Brighter sun
    sun_obj.rotation_euler = Euler((math.radians(50), math.radians(-30), math.radians(0)), 'XYZ')

    # --- World Background (Space) ---
    world_bg = bpy.context.scene.world
    if world_bg is None: world_bg = bpy.data.worlds.new("SpaceBackground"); bpy.context.scene.world = world_bg
    world_bg.use_nodes = True
    bg_tree = world_bg.node_tree
    bg_nodes = bg_tree.nodes
    # Clear default nodes if any, except Output
    for node in list(bg_nodes): # Iterate over a copy
        if node.type != 'OUTPUT_WORLD':
            bg_nodes.remove(node)
    
    output_node = bg_nodes.get('World Output')
    if not output_node: output_node = bg_nodes.new('ShaderNodeOutputWorld')

    if space_texture_path and os.path.exists(space_texture_path):
        env_tex_node = bg_nodes.new('ShaderNodeTexEnvironment')
        try:
            env_tex_node.image = bpy.data.images.load(space_texture_path)
            print(f"Loaded space background texture: {space_texture_path}")
            bg_tree.links.new(output_node.inputs['Surface'], env_tex_node.outputs['Color'])
            # Optional: control strength if it's an HDRI, or just use it as color
            # env_tex_node.inputs['Strength'].default_value = 0.5 # If needed
        except RuntimeError as e:
            print(f"ERROR loading space texture '{space_texture_path}': {e}. Using black background.")
            # Fallback to black background node
            bg_color_node = bg_nodes.new('ShaderNodeBackground')
            bg_color_node.inputs['Color'].default_value = (0.0, 0.0, 0.0, 1.0)
            bg_color_node.inputs['Strength'].default_value = 1.0
            bg_tree.links.new(output_node.inputs['Surface'], bg_color_node.outputs['Background'])
    else:
        print(f"WARNING: Space background texture not provided or found. Using black background.")
        bg_color_node = bg_nodes.new('ShaderNodeBackground')
        bg_color_node.inputs['Color'].default_value = (0.0, 0.0, 0.0, 1.0)
        bg_color_node.inputs['Strength'].default_value = 1.0
        bg_tree.links.new(output_node.inputs['Surface'], bg_color_node.outputs['Background'])

    return earth_obj, cam_obj

def create_country_name_text(text_content, earth_radius, target_cartesian_on_sphere, font_path_from_arg):
    bpy.ops.object.text_add(location=(0,0,0)) # Create at origin first
    text_obj = bpy.context.active_object
    text_obj.name = f"Text_{text_content.replace(' ','_')}"
    text_obj.data.body = text_content.upper()

    if font_path_from_arg and os.path.exists(font_path_from_arg):
        try:
            loaded_font = bpy.data.fonts.load(font_path_from_arg)
            text_obj.data.font = loaded_font
            print(f"Text object '{text_obj.name}' using custom font: {font_path_from_arg}")
        except RuntimeError as e:
            print(f"ERROR: Blender could not load font at '{font_path_from_arg}' for text '{text_content}': {e}. Using Blender default.")
    else:
        print(f"WARNING: Custom font path not provided or file not found for text '{text_content}': '{font_path_from_arg}'. Using Blender default font.")
    
    text_obj.data.align_x = 'CENTER'
    text_obj.data.align_y = 'CENTER'
    text_obj.data.extrude = 0.015 # Slightly less extrude
    text_obj.data.bevel_depth = 0.003
    text_obj.data.resolution_u = 4 # Controls text curve quality
    text_obj.data.size = 0.12 # Base size, will be keyframed

    text_offset_factor = 1.03 # Closer to surface
    text_pos = target_cartesian_on_sphere.normalized() * earth_radius * text_offset_factor
    text_obj.location = text_pos
    
    # Orient text to face outwards and be upright relative to camera (complex)
    # A simpler start: align with sphere normal, then add constraint to track camera
    normal_direction = target_cartesian_on_sphere.normalized()
    text_obj.rotation_euler = normal_direction.to_track_quat('Z', 'Y').to_euler() # Z aims along normal

    # Add a 'Track To' constraint to make text face the camera (Y-axis up)
    cam_obj = bpy.data.objects.get("SceneCamera")
    if cam_obj:
        ttc = text_obj.constraints.new(type='TRACK_TO')
        ttc.target = cam_obj
        ttc.track_axis = 'TRACK_NEGATIVE_Z' # Text's -Z points to camera
        ttc.up_axis = 'UP_Y' # Text's Y is up

    mat_text = bpy.data.materials.new(name=f"TextMat_{text_content}")
    text_obj.data.materials.append(mat_text) # Assign material to text data
    mat_text.use_nodes = True
    bsdf_text = mat_text.node_tree.nodes.get('Principled BSDF')
    if bsdf_text:
        bsdf_text.inputs['Base Color'].default_value = (1.0, 1.0, 1.0, 1.0) # White
        bsdf_text.inputs['Emission'].default_value = (0.9, 0.9, 0.9, 1) # Brighter Emission
        bsdf_text.inputs['Emission Strength'].default_value = 0.8
    
    text_obj.hide_render = True # Initially hidden
    text_obj.scale = (0.01, 0.01, 0.01) # Start very small
    return text_obj

def create_splash_effect_object(splash_color_rgb, earth_radius, target_cartesian_on_sphere, splash_id=""):
    bpy.ops.object.light_add(type='SPOT', radius=0.2, location=(0,0,0)) # Initial radius
    splash_light = bpy.context.active_object
    splash_light.name = f"SplashLight_{splash_id}"
    splash_light.data.color = splash_color_rgb
    splash_light.data.energy = 0 # Initially off
    splash_light.data.spot_size = math.radians(45) # Wider cone
    splash_light.data.spot_blend = 0.7 # Softer edge
    splash_light.data.use_custom_distance = True
    splash_light.data.distance = earth_radius * 0.6
    splash_light.data.show_cone = True # Visible in viewport

    light_pos = target_cartesian_on_sphere.normalized() * (earth_radius + 0.05) # Closer to surface
    splash_light.location = light_pos
    
    direction_vector = target_cartesian_on_sphere - splash_light.location # Point towards target
    splash_light.rotation_euler = direction_vector.to_track_quat('-Z', 'Y').to_euler()
    
    return splash_light

# --- Main Animation Logic ---
def run_blender_animation(plan_filepath, earth_texture_path, space_texture_path, font_path_from_arg, output_dir_base):
    
    if not os.path.exists(plan_filepath):
        print(f"ERROR: Animation plan file not found: {plan_filepath}"); sys.exit(1)
    with open(plan_filepath, 'r') as f:
        animation_plan = json.load(f)

    bpy.ops.wm.read_factory_settings(use_empty=True) # Clean scene
    scene = bpy.context.scene
    earth_obj, cam_obj = setup_scene(animation_plan["earth_radius"], earth_texture_path, space_texture_path)

    scene.render.engine = 'BLENDER_EEVEE'
    scene.eevee.taa_render_samples = 48 # Slightly better quality
    scene.eevee.use_bloom = True
    scene.eevee.bloom_intensity = 0.03
    scene.eevee.bloom_threshold = 0.9
    scene.render.image_settings.file_format = 'PNG'
    scene.render.fps = animation_plan["fps"]
    scene.render.resolution_x = VIDEO_WIDTH_PX # Use global from top of script
    scene.render.resolution_y = VIDEO_HEIGHT_PX
    scene.render.resolution_percentage = 100

    earth_radius = animation_plan["earth_radius"]
    
    # Initial camera position (global view)
    initial_cam_distance = earth_radius * 3.5 # Further out for global
    cam_obj.location = (0, -initial_cam_distance, earth_radius * 0.5) # Slightly elevated side view
    look_at(cam_obj, Vector((0,0,0)), initial_cam_distance - earth_radius, earth_radius) # Look at center
    
    # Keyframe initial camera state at frame 1
    cam_obj.keyframe_insert(data_path="location", frame=1)
    cam_obj.keyframe_insert(data_path="rotation_euler", frame=1) # Keyframe rotation if not using TrackTo constraint for initial view

    for scene_config in animation_plan["scenes"]:
        country_name_safe = scene_config['target_country_name'].replace(' ','_')
        print(f"Configuring animation for: {scene_config['target_country_name']}")
        
        # Frames for this scene segment
        start_transition_frame = scene_config["start_frame_offset"] + 1
        end_transition_frame = start_transition_frame + scene_config["transition_frames"]
        start_focus_frame = end_transition_frame # Focus starts right after transition
        end_focus_frame = start_focus_frame + scene_config["focus_frames"]

        target_cartesian = latlon_to_cartesian(scene_config["target_lat"], scene_config["target_lon"], earth_radius)
        
        # --- Camera Animation ---
        # Keyframe camera at its current position (end of previous scene's focus or initial)
        # This ensures a starting point for the interpolation to the new target
        cam_obj.keyframe_insert(data_path="location", frame=start_transition_frame -1 if start_transition_frame > 1 else 1)
        cam_obj.keyframe_insert(data_path="rotation_euler", frame=start_transition_frame -1 if start_transition_frame > 1 else 1)

        # Set camera at target position for end of transition / start of focus
        target_cam_distance_from_surface = earth_radius * scene_config["camera_distance_factor"]
        look_at(cam_obj, target_cartesian, target_cam_distance_from_surface, earth_radius)
        cam_obj.keyframe_insert(data_path="location", frame=start_focus_frame)
        cam_obj.keyframe_insert(data_path="rotation_euler", frame=start_focus_frame)
        
        # Camera holds position during focus
        cam_obj.keyframe_insert(data_path="location", frame=end_focus_frame)
        cam_obj.keyframe_insert(data_path="rotation_euler", frame=end_focus_frame)

        # --- Country Name Text Animation ---
        text_obj = create_country_name_text(scene_config["text_to_display"], earth_radius, target_cartesian, font_path_from_arg)
        
        text_appear_start_frame = start_focus_frame + int(scene_config["focus_frames"] * 0.3) # Text appears a bit into focus
        text_appear_duration_frames = int(animation_plan["fps"] * 0.8) # Fade/scale in over 0.8 sec
        text_appear_end_frame = text_appear_start_frame + text_appear_duration_frames
        text_disappear_start_frame = end_focus_frame - text_appear_duration_frames # Start fade out
        
        # Initial state (invisible, small)
        text_obj.scale = (0.01, 0.01, 0.01)
        text_obj.hide_render = True # Use hide_render for on/off
        text_obj.keyframe_insert(data_path="scale", frame=start_focus_frame)
        text_obj.keyframe_insert(data_path="hide_render", frame=start_focus_frame)

        # Appear (become visible, scale up)
        text_obj.hide_render = False
        text_obj.keyframe_insert(data_path="hide_render", frame=text_appear_start_frame)
        text_obj.scale = (1.0, 1.0, 1.0) # Target scale
        text_obj.keyframe_insert(data_path="scale", frame=text_appear_end_frame)

        # Hold
        text_obj.keyframe_insert(data_path="scale", frame=text_disappear_start_frame)
        text_obj.keyframe_insert(data_path="hide_render", frame=text_disappear_start_frame)

        # Disappear (scale down, become invisible)
        text_obj.scale = (0.01, 0.01, 0.01)
        text_obj.hide_render = True
        text_obj.keyframe_insert(data_path="scale", frame=end_focus_frame)
        text_obj.keyframe_insert(data_path="hide_render", frame=end_focus_frame)


        # --- "Color Splash" (Spotlight) Animation ---
        splash_obj = create_splash_effect_object(scene_config["splash_color_rgb"], earth_radius, target_cartesian, country_name_safe)
        
        splash_start_frame = start_focus_frame + int(scene_config["focus_frames"] * 0.1) # Splash starts early in focus
        splash_peak_intensity_frame = splash_start_frame + int(animation_plan["fps"] * 0.6) # Time to full intensity
        splash_hold_intensity_frame = end_focus_frame - int(animation_plan["fps"] * 0.8) # Hold before fading
        splash_fade_out_end_frame = end_focus_frame - int(animation_plan["fps"] * 0.2)

        splash_obj.data.energy = 0
        splash_obj.keyframe_insert(data_path="data.energy", frame=start_focus_frame) # Ensure off before start
        splash_obj.keyframe_insert(data_path="data.energy", frame=splash_start_frame) # Still 0 right at start

        splash_obj.data.energy = 3000 # Bright splash (adjust based on Eevee settings and desired effect)
        splash_obj.keyframe_insert(data_path="data.energy", frame=splash_peak_intensity_frame)

        splash_obj.keyframe_insert(data_path="data.energy", frame=splash_hold_intensity_frame) # Hold bright
        
        splash_obj.data.energy = 0 # Fade out
        splash_obj.keyframe_insert(data_path="data.energy", frame=splash_fade_out_end_frame)
        # Ensure it's off by the very end of the focus period
        splash_obj.keyframe_insert(data_path="data.energy", frame=end_focus_frame)


        # --- Glowing Border (Conceptual - Very Simplified) ---
        if scene_config.get('glow_effect'):
            glow_details = scene_config['glow_effect']
            partner_cartesian = latlon_to_cartesian(glow_details["partner_lat"], glow_details["partner_lon"], earth_radius)
            
            # Create a simple emissive object (e.g., a thin cylinder or curve)
            # between target_cartesian and partner_cartesian.
            # This is a placeholder for a more complex border visualization.
            mid_point = (target_cartesian + partner_cartesian) / 2
            bpy.ops.mesh.primitive_cylinder_add(
                radius=0.01 * earth_radius, 
                depth=(target_cartesian - partner_cartesian).length, 
                location=mid_point
            )
            glow_obj = bpy.context.active_object
            glow_obj.name = f"GlowBorder_{country_name_safe}"
            
            # Align cylinder between the two points
            direction = partner_cartesian - target_cartesian
            glow_obj.rotation_euler = direction.to_track_quat('Z','Y').to_euler() # Align Z along direction

            mat_glow = bpy.data.materials.new(name=f"GlowMat_{country_name_safe}")
            glow_obj.data.materials.append(mat_glow)
            mat_glow.use_nodes = True
            bsdf_glow = mat_glow.node_tree.nodes.get('Principled BSDF')
            if bsdf_glow: # Should exist
                mat_glow.node_tree.nodes.remove(bsdf_glow) # Remove BSDF, use Emission directly
            
            emission_node = mat_glow.node_tree.nodes.new('ShaderNodeEmission')
            output_node_mat = mat_glow.node_tree.nodes.get('Material Output')
            if not output_node_mat: output_node_mat = mat_glow.node_tree.nodes.new('ShaderNodeOutputMaterial')
            mat_glow.node_tree.links.new(output_node_mat.inputs['Surface'], emission_node.outputs['Emission'])
            
            glow_color_rgba = glow_details.get('color', [1.0, 0.0, 1.0, 1.0])
            emission_node.inputs['Color'].default_value = glow_color_rgba
            emission_node.inputs['Strength'].default_value = 0 # Initially off

            # Keyframe glow visibility and strength
            glow_start_frame = start_focus_frame + int(scene_config["focus_frames"] * 0.2)
            glow_peak_frame = glow_start_frame + int(animation_plan["fps"] * 0.5)
            glow_end_frame = end_focus_frame - int(scene_config["focus_frames"] * 0.2)

            glow_obj.hide_render = True
            glow_obj.keyframe_insert(data_path="hide_render", frame=start_focus_frame)
            emission_node.inputs['Strength'].default_value = 0
            emission_node.inputs['Strength'].keyframe_insert(data_path="default_value", frame=start_focus_frame)


            glow_obj.hide_render = False
            glow_obj.keyframe_insert(data_path="hide_render", frame=glow_start_frame)
            emission_node.inputs['Strength'].default_value = 15 # Adjust for desired glow intensity with Bloom
            emission_node.inputs['Strength'].keyframe_insert(data_path="default_value", frame=glow_peak_frame)
            
            emission_node.inputs['Strength'].keyframe_insert(data_path="default_value", frame=glow_end_frame) # Hold peak
            
            glow_obj.hide_render = True
            emission_node.inputs['Strength'].default_value = 0
            glow_obj.keyframe_insert(data_path="hide_render", frame=end_focus_frame)
            emission_node.inputs['Strength'].keyframe_insert(data_path="default_value", frame=end_focus_frame)


    # --- Set total animation length for rendering ---
    scene.frame_start = 1
    scene.frame_end = animation_plan["total_animation_frames"]

    # --- Render Animation ---
    output_frame_pattern = os.path.join(output_dir_base, "frame_#####") # Blender uses # for padding
    scene.render.filepath = output_frame_pattern
    print(f"Rendering animation frames to: {output_dir_base}. Total frames: {scene.frame_end}")
    bpy.ops.render.render(animation=True)

    print("Blender rendering process finished.")


if __name__ == "__main__":
    args = sys.argv
    argv = args[args.index("--") + 1:] if "--" in args else []
    
    print(f"Received arguments for Blender script: {argv}")

    plan_file = argv[0] if len(argv) > 0 else "animation_plan.json"
    earth_tex = argv[1] if len(argv) > 1 else "textures/earth_texture.jpg"
    space_tex_arg = argv[2] if len(argv) > 2 else "textures/space_background.jpg"
    font_file_arg = argv[3] if len(argv) > 3 else "fonts/Montserrat-Bold.ttf"
    output_frames_dir = argv[4] if len(argv) > 4 else "/tmp/blender_frames/"

    if not os.path.exists(output_frames_dir):
        os.makedirs(output_frames_dir, exist_ok=True)

    # Validate essential files passed as arguments
    if not os.path.isabs(plan_file) and not os.path.exists(plan_file): plan_file = os.path.join(os.getcwd(), plan_file)
    if not os.path.isabs(earth_tex) and not os.path.exists(earth_tex): earth_tex = os.path.join(os.getcwd(), earth_tex)
    
    if not os.path.exists(plan_file): print(f"FATAL: Plan file missing: {plan_file}"); sys.exit(1)
    if not os.path.exists(earth_tex): print(f"FATAL: Earth texture missing: {earth_tex}"); sys.exit(1)
    
    final_font_file = None
    if font_file_arg:
        if not os.path.isabs(font_file_arg) and not os.path.exists(font_file_arg): font_file_arg = os.path.join(os.getcwd(), font_file_arg)
        if os.path.exists(font_file_arg): final_font_file = font_file_arg
        else: print(f"WARNING: Custom font file not found at '{font_file_arg}'. Using Blender default.")
            
    final_space_tex = ""
    if space_tex_arg:
        if not os.path.isabs(space_tex_arg) and not os.path.exists(space_tex_arg): space_tex_arg = os.path.join(os.getcwd(), space_tex_arg)
        if os.path.exists(space_tex_arg): final_space_tex = space_tex_arg
        else: print(f"WARNING: Space texture file '{space_tex_arg}' not found. Will use black background.")
        
    run_blender_animation(plan_file, earth_tex, final_space_tex, final_font_file, output_frames_dir)
