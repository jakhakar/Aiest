import bpy
import json
import sys
import os
import math
from mathutils import Vector, Euler # For 3D vector and rotation math

# --- Script-Level Configuration (Defaults, can be overridden by plan or args) ---
# These are used by Blender for render output.
# GitHub Actions workflow will pass these as environment variables.
VIDEO_WIDTH_PX = int(os.environ.get('BLENDER_VIDEO_WIDTH_ENV', 720))
VIDEO_HEIGHT_PX = int(os.environ.get('BLENDER_VIDEO_HEIGHT_ENV', 1280))
print(f"Blender script using resolution: {VIDEO_WIDTH_PX}x{VIDEO_HEIGHT_PX}")

# --- Helper Functions ---
def latlon_to_cartesian(lat, lon, radius):
    """Converts Lat/Lon (degrees) to Cartesian (X,Y,Z) coordinates on a sphere."""
    lat_rad = math.radians(lat)
    lon_rad = math.radians(lon)
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

    for c in obj_camera.constraints:
        if c.type == 'TRACK_TO':
            obj_camera.constraints.remove(c)
            
    ttc = obj_camera.constraints.new(type='TRACK_TO')
    earth_obj = bpy.data.objects.get("Earth")
    look_at_target_obj = earth_obj # Prefer to track the Earth object itself
    if not earth_obj: # Fallback if Earth somehow not found
        look_at_target_obj = bpy.data.objects.get("LookAtTarget")
        if not look_at_target_obj:
            bpy.ops.object.empty_add(type='PLAIN_AXES', location=(0,0,0))
            look_at_target_obj = bpy.context.active_object
            look_at_target_obj.name = "LookAtTarget"
    ttc.target = look_at_target_obj
    ttc.track_axis = 'TRACK_NEGATIVE_Z'
    ttc.up_axis = 'UP_Y'

def setup_scene(earth_radius, earth_texture_path, space_texture_path):
    """Creates the basic Earth, lighting, camera, and space background."""
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

    bpy.ops.mesh.primitive_uv_sphere_add(radius=earth_radius, location=(0, 0, 0), segments=128, ring_count=64)
    earth_obj = bpy.context.active_object
    earth_obj.name = "Earth"
    bpy.ops.object.shade_smooth()

    mat_earth = bpy.data.materials.new(name="EarthMaterial")
    earth_obj.data.materials.append(mat_earth)
    mat_earth.use_nodes = True
    nodes = mat_earth.node_tree.nodes
    bsdf = nodes.get('Principled BSDF')
    if not bsdf: bsdf = nodes.new('ShaderNodeBsdfPrincipled')

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

    bpy.ops.object.camera_add(location=(0, -(earth_radius + 5), 0))
    cam_obj = bpy.context.active_object
    cam_obj.name = "SceneCamera"
    bpy.context.scene.camera = cam_obj
    cam_obj.data.lens = 35

    bpy.ops.object.light_add(type='SUN', location=(earth_radius * 3, -earth_radius * 3, earth_radius * 2))
    sun_obj = bpy.context.active_object
    sun_obj.name = "SunLight"
    sun_obj.data.energy = 5.0
    sun_obj.rotation_euler = Euler((math.radians(50), math.radians(-30), math.radians(0)), 'XYZ')

    world_bg = bpy.context.scene.world
    if world_bg is None: world_bg = bpy.data.worlds.new("SpaceBackground"); bpy.context.scene.world = world_bg
    world_bg.use_nodes = True
    bg_tree = world_bg.node_tree
    bg_nodes = bg_tree.nodes
    for node in list(bg_nodes):
        if node.type != 'OUTPUT_WORLD': bg_nodes.remove(node)
    output_node = bg_nodes.get('World Output')
    if not output_node: output_node = bg_nodes.new('ShaderNodeOutputWorld')

    if space_texture_path and os.path.exists(space_texture_path):
        env_tex_node = bg_nodes.new('ShaderNodeTexEnvironment')
        try:
            env_tex_node.image = bpy.data.images.load(space_texture_path)
            print(f"Loaded space background texture: {space_texture_path}")
            bg_tree.links.new(output_node.inputs['Surface'], env_tex_node.outputs['Color'])
        except RuntimeError as e:
            print(f"ERROR loading space texture '{space_texture_path}': {e}. Using black background.")
            bg_color_node = bg_nodes.new('ShaderNodeBackground')
            bg_color_node.inputs['Color'].default_value = (0.0, 0.0, 0.0, 1.0)
            bg_color_node.inputs['Strength'].default_value = 1.0
            bg_tree.links.new(output_node.inputs['Surface'], bg_color_node.outputs['Background'])
    else:
        print(f"WARNING: Space background texture not provided or found ('{space_texture_path}'). Using black background.")
        bg_color_node = bg_nodes.new('ShaderNodeBackground')
        bg_color_node.inputs['Color'].default_value = (0.0, 0.0, 0.0, 1.0)
        bg_color_node.inputs['Strength'].default_value = 1.0
        bg_tree.links.new(output_node.inputs['Surface'], bg_color_node.outputs['Background'])
    return earth_obj, cam_obj

def create_country_name_text(text_content, earth_radius, target_cartesian_on_sphere, font_path_from_arg):
    bpy.ops.object.text_add(location=(0,0,0))
    text_obj = bpy.context.active_object
    text_obj.name = f"Text_{text_content.replace(' ','_')}"
    text_obj.data.body = text_content.upper()

    font_successfully_loaded = False
    if font_path_from_arg and os.path.exists(font_path_from_arg) and os.path.isfile(font_path_from_arg):
        try:
            loaded_font = bpy.data.fonts.load(font_path_from_arg)
            text_obj.data.font = loaded_font
            print(f"Text object '{text_obj.name}' using custom font: {font_path_from_arg}")
            font_successfully_loaded = True
        except RuntimeError as e:
            print(f"ERROR: Blender could not load font at '{font_path_from_arg}' for text '{text_content}': {e}")
    elif font_path_from_arg:
         print(f"WARNING: Custom font file not found or is not a file at '{font_path_from_arg}' for text '{text_content}'.")

    if not font_successfully_loaded:
        print(f"Using Blender's default font for text '{text_content}'.")
        text_obj.data.font = None 
    
    text_obj.data.align_x = 'CENTER'
    text_obj.data.align_y = 'CENTER'
    text_obj.data.extrude = 0.015
    text_obj.data.bevel_depth = 0.003
    text_obj.data.resolution_u = 4
    text_obj.data.size = 0.12

    text_offset_factor = 1.03
    text_pos = target_cartesian_on_sphere.normalized() * earth_radius * text_offset_factor
    text_obj.location = text_pos
    
    normal_direction = target_cartesian_on_sphere.normalized()
    text_obj.rotation_euler = normal_direction.to_track_quat('Z', 'Y').to_euler()

    cam_obj = bpy.data.objects.get("SceneCamera")
    if cam_obj:
        for c in text_obj.constraints:
            if c.type == 'TRACK_TO': text_obj.constraints.remove(c)
        ttc = text_obj.constraints.new(type='TRACK_TO')
        ttc.target = cam_obj
        ttc.track_axis = 'TRACK_NEGATIVE_Z'
        ttc.up_axis = 'UP_Y'

    mat_text = bpy.data.materials.new(name=f"TextMat_{text_content}")
    text_obj.data.materials.append(mat_text)
    mat_text.use_nodes = True
    bsdf_text = mat_text.node_tree.nodes.get('Principled BSDF')
    if bsdf_text:
        bsdf_text.inputs['Base Color'].default_value = (1.0, 1.0, 1.0, 1.0)
        bsdf_text.inputs['Emission'].default_value = (0.9, 0.9, 0.9, 1)
        bsdf_text.inputs['Emission Strength'].default_value = 0.8
    
    text_obj.hide_render = True
    text_obj.scale = (0.01, 0.01, 0.01)
    return text_obj

def create_splash_effect_object(splash_color_rgb, earth_radius, target_cartesian_on_sphere, splash_id=""):
    bpy.ops.object.light_add(type='SPOT', radius=0.2, location=(0,0,0))
    splash_light = bpy.context.active_object
    splash_light.name = f"SplashLight_{splash_id}"
    splash_light.data.color = splash_color_rgb
    splash_light.data.energy = 0
    splash_light.data.spot_size = math.radians(45)
    splash_light.data.spot_blend = 0.7
    splash_light.data.use_custom_distance = True
    splash_light.data.distance = earth_radius * 0.6
    splash_light.data.show_cone = False # Don't show cone in render by default

    light_pos = target_cartesian_on_sphere.normalized() * (earth_radius + 0.05)
    splash_light.location = light_pos
    
    direction_vector = target_cartesian_on_sphere - splash_light.location
    splash_light.rotation_euler = direction_vector.to_track_quat('-Z', 'Y').to_euler()
    return splash_light

def run_blender_animation(plan_filepath, earth_texture_path, space_texture_path, font_path_from_arg, output_dir_base):
    if not os.path.exists(plan_filepath):
        print(f"ERROR: Animation plan file not found: {plan_filepath}"); sys.exit(1)
    with open(plan_filepath, 'r') as f:
        animation_plan = json.load(f)

    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    earth_obj, cam_obj = setup_scene(animation_plan["earth_radius"], earth_texture_path, space_texture_path)

    scene.render.engine = 'BLENDER_EEVEE'
    scene.eevee.taa_render_samples = 48
    scene.eevee.use_bloom = True
    scene.eevee.bloom_intensity = 0.035 # Fine-tune bloom
    scene.eevee.bloom_threshold = 0.9
    scene.render.image_settings.file_format = 'PNG'
    scene.render.fps = animation_plan["fps"]
    scene.render.resolution_x = VIDEO_WIDTH_PX
    scene.render.resolution_y = VIDEO_HEIGHT_PX
    scene.render.resolution_percentage = 100

    earth_radius = animation_plan["earth_radius"]
    initial_cam_distance = earth_radius * 3.5
    cam_obj.location = (0, -initial_cam_distance, earth_radius * 0.5)
    look_at(cam_obj, Vector((0,0,0)), initial_cam_distance - earth_radius, earth_radius)
    cam_obj.keyframe_insert(data_path="location", frame=1)
    cam_obj.keyframe_insert(data_path="rotation_euler", frame=1)

    for scene_config in animation_plan["scenes"]:
        country_name_safe = scene_config['target_country_name'].replace(' ','_')
        print(f"Configuring animation for: {scene_config['target_country_name']}")
        
        start_transition_frame = scene_config["start_frame_offset"] + 1
        end_transition_frame = start_transition_frame + scene_config["transition_frames"]
        start_focus_frame = end_transition_frame
        end_focus_frame = start_focus_frame + scene_config["focus_frames"]

        target_cartesian = latlon_to_cartesian(scene_config["target_lat"], scene_config["target_lon"], earth_radius)
        
        cam_obj.keyframe_insert(data_path="location", frame=max(1, start_transition_frame -1)) # Ensure frame >= 1
        cam_obj.keyframe_insert(data_path="rotation_euler", frame=max(1, start_transition_frame -1))
        target_cam_distance_from_surface = earth_radius * scene_config["camera_distance_factor"]
        look_at(cam_obj, target_cartesian, target_cam_distance_from_surface, earth_radius)
        cam_obj.keyframe_insert(data_path="location", frame=start_focus_frame)
        cam_obj.keyframe_insert(data_path="rotation_euler", frame=start_focus_frame)
        cam_obj.keyframe_insert(data_path="location", frame=end_focus_frame)
        cam_obj.keyframe_insert(data_path="rotation_euler", frame=end_focus_frame)

        text_obj = create_country_name_text(scene_config["text_to_display"], earth_radius, target_cartesian, font_path_from_arg)
        text_appear_start_frame = start_focus_frame + int(scene_config["focus_frames"] * 0.3)
        text_appear_duration_frames = int(animation_plan["fps"] * 0.8)
        text_appear_end_frame = text_appear_start_frame + text_appear_duration_frames
        text_disappear_start_frame = end_focus_frame - text_appear_duration_frames
        text_obj.scale = (0.01, 0.01, 0.01); text_obj.hide_render = True
        text_obj.keyframe_insert(data_path="scale", frame=start_focus_frame)
        text_obj.keyframe_insert(data_path="hide_render", frame=start_focus_frame)
        text_obj.hide_render = False
        text_obj.keyframe_insert(data_path="hide_render", frame=text_appear_start_frame)
        text_obj.scale = (1.0, 1.0, 1.0)
        text_obj.keyframe_insert(data_path="scale", frame=text_appear_end_frame)
        text_obj.keyframe_insert(data_path="scale", frame=text_disappear_start_frame)
        text_obj.keyframe_insert(data_path="hide_render", frame=text_disappear_start_frame)
        text_obj.scale = (0.01, 0.01, 0.01); text_obj.hide_render = True
        text_obj.keyframe_insert(data_path="scale", frame=end_focus_frame)
        text_obj.keyframe_insert(data_path="hide_render", frame=end_focus_frame)

        splash_obj = create_splash_effect_object(scene_config["splash_color_rgb"], earth_radius, target_cartesian, country_name_safe)
        light_data_block = splash_obj.data
        splash_start_frame = start_focus_frame + int(scene_config["focus_frames"] * 0.1)
        splash_peak_intensity_frame = splash_start_frame + int(animation_plan["fps"] * 0.6)
        splash_hold_intensity_frame = end_focus_frame - int(animation_plan["fps"] * 0.8)
        splash_fade_out_end_frame = end_focus_frame - int(animation_plan["fps"] * 0.2)
        light_data_block.energy = 0
        light_data_block.keyframe_insert(data_path="energy", frame=max(1,start_focus_frame)) # Ensure frame >= 1
        light_data_block.keyframe_insert(data_path="energy", frame=splash_start_frame)
        light_data_block.energy = 3000
        light_data_block.keyframe_insert(data_path="energy", frame=splash_peak_intensity_frame)
        light_data_block.keyframe_insert(data_path="energy", frame=splash_hold_intensity_frame)
        light_data_block.energy = 0
        light_data_block.keyframe_insert(data_path="energy", frame=splash_fade_out_end_frame)
        light_data_block.keyframe_insert(data_path="energy", frame=end_focus_frame)

        if scene_config.get('glow_effect'):
            glow_details = scene_config['glow_effect']
            partner_cartesian = latlon_to_cartesian(glow_details["partner_lat"], glow_details["partner_lon"], earth_radius)
            mid_point = (target_cartesian + partner_cartesian) / 2
            bpy.ops.mesh.primitive_cylinder_add(radius=0.01 * earth_radius, depth=(target_cartesian - partner_cartesian).length, location=mid_point)
            glow_obj = bpy.context.active_object
            glow_obj.name = f"GlowBorder_{country_name_safe}"
            direction = partner_cartesian - target_cartesian
            if direction.length > 0: glow_obj.rotation_euler = direction.to_track_quat('Z','Y').to_euler()
            mat_glow = bpy.data.materials.new(name=f"GlowMat_{country_name_safe}")
            glow_obj.data.materials.append(mat_glow)
            mat_glow.use_nodes = True
            nodes_glow = mat_glow.node_tree.nodes
            for node in list(nodes_glow):
                if node.type != 'OUTPUT_MATERIAL': nodes_glow.remove(node)
            emission_node = nodes_glow.new('ShaderNodeEmission')
            output_node_mat = nodes_glow.get('Material Output');
            if not output_node_mat: output_node_mat = nodes_glow.new('ShaderNodeOutputMaterial')
            mat_glow.node_tree.links.new(output_node_mat.inputs['Surface'], emission_node.outputs['Emission'])
            glow_color_rgba = glow_details.get('color', [1.0, 0.0, 1.0, 1.0])
            emission_node.inputs['Color'].default_value = glow_color_rgba
            emission_node.inputs['Strength'].default_value = 0 
            glow_start_frame = start_focus_frame + int(scene_config["focus_frames"] * 0.2)
            glow_peak_frame = glow_start_frame + int(animation_plan["fps"] * 0.5)
            glow_end_frame = end_focus_frame - int(scene_config["focus_frames"] * 0.2)
            glow_obj.hide_render = True
            glow_obj.keyframe_insert(data_path="hide_render", frame=max(1,start_focus_frame))
            emission_node.inputs['Strength'].keyframe_insert(data_path="default_value", frame=max(1,start_focus_frame))
            glow_obj.hide_render = False
            glow_obj.keyframe_insert(data_path="hide_render", frame=glow_start_frame)
            emission_node.inputs['Strength'].default_value = 15 
            emission_node.inputs['Strength'].keyframe_insert(data_path="default_value", frame=glow_peak_frame)
            emission_node.inputs['Strength'].keyframe_insert(data_path="default_value", frame=glow_end_frame) 
            glow_obj.hide_render = True; emission_node.inputs['Strength'].default_value = 0
            glow_obj.keyframe_insert(data_path="hide_render", frame=end_focus_frame)
            emission_node.inputs['Strength'].keyframe_insert(data_path="default_value", frame=end_focus_frame)

    scene.frame_start = 1
    scene.frame_end = animation_plan["total_animation_frames"]
    if scene.frame_end < scene.frame_start : # Ensure end frame is not before start
        print(f"Warning: Calculated end frame ({scene.frame_end}) is before start frame ({scene.frame_start}). Setting end frame to start frame + 1.")
        scene.frame_end = scene.frame_start + 1 # Render at least one frame if total_frames was 0 or negative
    if scene.frame_end == 0 : # If total_animation_frames was 0
        print("Warning: Total animation frames is 0. Rendering a single frame (frame 1).")
        scene.frame_end = 1


    output_frame_pattern = os.path.join(output_dir_base, "frame_#####")
    scene.render.filepath = output_frame_pattern
    print(f"Rendering animation frames to: {output_dir_base}. Start: {scene.frame_start}, End: {scene.frame_end}")
    if scene.frame_end >= scene.frame_start:
        bpy.ops.render.render(animation=True)
    else:
        print("Skipping render due to invalid frame range.")
    print("Blender rendering process finished.")

if __name__ == "__main__":
    print(f"--- Blender Python Script Start (render_blender_scene.py) ---")
    print(f"Raw sys.argv: {sys.argv}")
    
    args = sys.argv
    argv = args[args.index("--") + 1:] if "--" in args else []
    
    print(f"Arguments after '--': {argv}")
    print(f"Number of arguments after '--': {len(argv)}")

    # Define default filenames or identify them if script is in a 'scripts' subdirectory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root_guess = os.path.dirname(script_dir) # Assumes script is in 'scripts/'

    def get_default_path(sub_path_in_repo):
        return os.path.join(repo_root_guess, sub_path_in_repo)

    plan_file_default = get_default_path("animation_plan.json")
    earth_tex_default = get_default_path("textures/earth_texture.jpg")
    space_tex_default = get_default_path("textures/space_background.jpg")
    font_file_default = get_default_path("fonts/Montserrat-Bold.ttf") # Example default
    output_frames_dir_default = "/tmp/blender_frames/" # Good for headless runners

    plan_file         = argv[0] if len(argv) > 0 else plan_file_default
    earth_tex         = argv[1] if len(argv) > 1 else earth_tex_default
    space_tex_arg     = argv[2] if len(argv) > 2 else space_tex_default
    font_file_arg     = argv[3] if len(argv) > 3 else font_file_default
    output_frames_dir = argv[4] if len(argv) > 4 else output_frames_dir_default

    print(f"Resolved plan_file: {plan_file}")
    print(f"Resolved earth_tex: {earth_tex}")
    print(f"Resolved space_tex_arg: {space_tex_arg}")
    print(f"Resolved font_file_arg: {font_file_arg}")
    print(f"Resolved output_frames_dir: {output_frames_dir}")

    if not os.path.exists(output_frames_dir):
        os.makedirs(output_frames_dir, exist_ok=True)

    if not os.path.exists(plan_file): print(f"FATAL: Plan file missing: {plan_file}"); sys.exit(1)
    if not os.path.exists(earth_tex): print(f"FATAL: Earth texture missing: {earth_tex}"); sys.exit(1)
    
    final_font_file = None
    if font_file_arg and os.path.exists(font_file_arg) and os.path.isfile(font_file_arg):
        final_font_file = font_file_arg
    elif font_file_arg: # Path was given but not found/not a file
        print(f"WARNING: Custom font file not found or is not a file at '{font_file_arg}'. Using Blender default.")
            
    final_space_tex = ""
    if space_tex_arg and os.path.exists(space_tex_arg) and os.path.isfile(space_tex_arg):
        final_space_tex = space_tex_arg
    elif space_tex_arg: # Path was given but not found/not a file
        print(f"WARNING: Space texture file '{space_tex_arg}' not found. Will use black background.")
        
    run_blender_animation(plan_file, earth_tex, final_space_tex, final_font_file, output_frames_dir)
