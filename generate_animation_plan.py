import geopandas
import pandas as pd
import json
import os

# --- Configuration for the Animation Plan ---
# This should match the countries and effects you want in your video
# 'splash_color' should be a list [R, G, B] with values 0.0 to 1.0
# 'glow_details': {'partner_country': "Name", 'color': [R,G,B,A]} (A for alpha/intensity)
ANIMATION_STEPS_CONFIG = [
    {
        'target_country_name': "France",
        'transition_sec': 2.0, 'focus_sec': 3.5, 'text': "FRANCE",
        'splash_color': [1.0, 0.6, 0.0], # Orange
        'camera_distance_factor': 1.5, # Smaller factor = closer zoom
    },
    {
        'target_country_name': "Russia",
        'transition_sec': 3.0, 'focus_sec': 4.0, 'text': "RUSSIA",
        'splash_color': [0.0, 0.2, 0.8], # Blue
        'camera_distance_factor': 1.8,
    },
    {
        'target_country_name': "Ukraine",
        'transition_sec': 2.5, 'focus_sec': 4.0, 'text': "UKRAINE",
        'splash_color': [1.0, 0.84, 0.0], # Yellow
        'camera_distance_factor': 1.6,
        'glow_details': {'partner_country': "Russia", 'color': [1.0, 0.27, 0.0, 1.0]} # Orange-Red glow
    },
    # Add more steps as needed
]

OUTPUT_JSON_PATH = "animation_plan.json"
FPS = 30

# Path to your downloaded Natural Earth shapefile (must exist when this script runs)
# In GitHub Actions, this path would be relative to the runner's workspace
# after the download step for the shapefile.
# For local testing, ensure it points to your local shapefile.
# If running this script *after* the shapefile download script in Colab/Actions, this path is correct.
NATURAL_EARTH_SHP_PATH = "/content/natural_earth_data/ne_110m_admin_0_countries/ne_110m_admin_0_countries.shp"


def generate_plan():
    print(f"Loading shapefile from: {NATURAL_EARTH_SHP_PATH}")
    if not os.path.exists(NATURAL_EARTH_SHP_PATH):
        print(f"ERROR: Shapefile not found at {NATURAL_EARTH_SHP_PATH}. Cannot generate plan.")
        print("Please ensure the shapefile download script has run successfully first, or adjust the path.")
        return

    world_gdf = geopandas.read_file(NATURAL_EARTH_SHP_PATH)
    # Ensure we are using geographic coordinates (lat/lon) for centroids
    world_gdf = world_gdf.to_crs("EPSG:4326")

    animation_plan = {
        "fps": FPS,
        "earth_radius": 1.0, # Define a unit radius for Blender scene
        "scenes": []
    }

    total_frames_offset = 0

    for i, step_config in enumerate(ANIMATION_STEPS_CONFIG):
        country_name = step_config['target_country_name']
        country_data = world_gdf[world_gdf['NAME'].astype(str).str.lower() == country_name.lower()]

        if country_data.empty:
            # Try ADMIN column as a fallback
            country_data = world_gdf[world_gdf['ADMIN'].astype(str).str.lower() == country_name.lower()]
        if country_data.empty:
            # Try SOVEREIGNT column
            country_data = world_gdf[world_gdf['SOVEREIGNT'].astype(str).str.lower() == country_name.lower()]


        if country_data.empty:
            print(f"Warning: Could not find coordinates for '{country_name}'. Skipping this step.")
            continue

        # Use representative point for better label placement than centroid for complex shapes
        # representative_point() is in the geometry's original CRS (EPSG:4326 here)
        # Blender typically uses Y-up or Z-up. We'll need to convert lat/lon.
        # For Blender: Latitude -> rotation around X, Longitude -> rotation around Z (if Z is up)
        # Or convert to Cartesian and place camera to look at that point.
        # Let's pass lat/lon and Blender script will handle conversion.
        target_point = country_data.geometry.iloc[0].representative_point()
        target_lon, target_lat = target_point.x, target_point.y # GeoPandas gives lon, lat

        transition_frames = int(step_config['transition_sec'] * FPS)
        focus_frames = int(step_config['focus_sec'] * FPS)

        scene_data = {
            "id": f"scene_{i+1}_{country_name.replace(' ', '_')}",
            "target_country_name": country_name,
            "target_lat": target_lat,
            "target_lon": target_lon,
            "camera_distance_factor": step_config.get('camera_distance_factor', 2.0),
            "start_frame_offset": total_frames_offset, # When this scene's transition begins
            "transition_frames": transition_frames,
            "focus_frames": focus_frames,
            "text_to_display": step_config.get('text', country_name.upper()),
            "splash_color_rgb": step_config.get('splash_color', [0.7, 0.7, 0.7]), # Default gray
        }

        if 'glow_details' in step_config:
            partner_name = step_config['glow_details']['partner_country']
            partner_data = world_gdf[world_gdf['NAME'].astype(str).str.lower() == partner_name.lower()]
            if partner_data.empty:
                 partner_data = world_gdf[world_gdf['ADMIN'].astype(str).str.lower() == partner_name.lower()]
            if partner_data.empty:
                 partner_data = world_gdf[world_gdf['SOVEREIGNT'].astype(str).str.lower() == partner_name.lower()]


            if not partner_data.empty:
                partner_point = partner_data.geometry.iloc[0].representative_point()
                scene_data['glow_effect'] = {
                    "partner_lat": partner_point.y,
                    "partner_lon": partner_point.x,
                    "color_rgba": step_config['glow_details'].get('color', [1.0, 0.0, 1.0, 1.0]) # Default magenta
                }
            else:
                print(f"Warning: Could not find partner country '{partner_name}' for glow effect.")
        
        animation_plan["scenes"].append(scene_data)
        total_frames_offset += (transition_frames + focus_frames)

    animation_plan["total_animation_frames"] = total_frames_offset

    with open(OUTPUT_JSON_PATH, 'w') as f:
        json.dump(animation_plan, f, indent=4)

    print(f"Animation plan generated: {OUTPUT_JSON_PATH}")
    print(f"Total estimated frames for Blender: {total_frames_offset}")

if __name__ == "__main__":
    # First, ensure the Natural Earth shapefile is available.
    # This script assumes the shapefile download script (or manual upload) has already happened.
    if not os.path.exists(NATURAL_EARTH_SHP_PATH):
        print(f"ERROR: Prerequisite shapefile not found at {NATURAL_EARTH_SHP_PATH}")
        print("Please run the shapefile download script first or ensure the file is correctly placed.")
    else:
        generate_plan()
