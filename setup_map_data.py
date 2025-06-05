import os
import requests
import zipfile
import shutil # For removing the temporary download directory

# --- Configuration ---
# CORRECTED AND MORE RELIABLE URL for the 110m Admin 0 - Countries shapefile
DATA_URL = "https://naturalearth.s3.amazonaws.com/110m_cultural/ne_110m_admin_0_countries.zip"

# Target directory structure in Colab's /content/
BASE_TARGET_DIR = "/content/natural_earth_data"
SHAPEFILE_SUBDIR = "ne_110m_admin_0_countries"
FULL_TARGET_SHAPEFILE_DIR = os.path.join(BASE_TARGET_DIR, SHAPEFILE_SUBDIR)

# Expected final shapefile path to check for successful setup
EXPECTED_SHP_FILE = os.path.join(FULL_TARGET_SHAPEFILE_DIR, "ne_110m_admin_0_countries.shp")

# Temporary download location for the zip file
TEMP_DOWNLOAD_DIR = "/content/temp_natural_earth_download"
ZIP_FILENAME = "ne_110m_admin_0_countries.zip"
ZIP_FILEPATH = os.path.join(TEMP_DOWNLOAD_DIR, ZIP_FILENAME)

def download_and_setup_natural_earth_data():
    """
    Downloads the Natural Earth 110m Admin 0 - Countries shapefile,
    unzips it, and places it into the expected /content/ directory structure.
    """
    print("--- Natural Earth Data Setup Script (S3 Link & User-Agent) ---")

    # 1. Check if data is already set up
    if os.path.exists(EXPECTED_SHP_FILE):
        print(f"Shapefile already found at: {EXPECTED_SHP_FILE}")
        print("Skipping download and setup.")
        print("Data should be ready for the main animation script.")
        return True

    print(f"Target shapefile not found. Proceeding with download and setup.")

    # 2. Create necessary directories
    try:
        os.makedirs(TEMP_DOWNLOAD_DIR, exist_ok=True)
        # Important: Create the final destination for unzipped files *before* unzipping
        os.makedirs(FULL_TARGET_SHAPEFILE_DIR, exist_ok=True)
        print(f"Created temporary download directory: {TEMP_DOWNLOAD_DIR}")
        print(f"Created target data directory: {FULL_TARGET_SHAPEFILE_DIR}")
    except OSError as e:
        print(f"Error creating directories: {e}")
        return False

    # 3. Download the ZIP file
    print(f"Downloading Natural Earth data from: {DATA_URL}")
    print(f"This may take a moment...")
    try:
        # Add a common User-Agent header
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(DATA_URL, stream=True, headers=headers, timeout=180) # Increased timeout
        response.raise_for_status()  # Will raise an HTTPError for bad responses (4XX or 5XX)

        with open(ZIP_FILEPATH, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Successfully downloaded: {ZIP_FILEPATH}")
    except requests.exceptions.RequestException as e:
        print(f"Error downloading file: {e}")
        print(f"Please check the URL or your internet connection.")
        # Attempt to clean up temporary download directory if download fails
        if os.path.exists(TEMP_DOWNLOAD_DIR):
            try:
                shutil.rmtree(TEMP_DOWNLOAD_DIR)
                print(f"Cleaned up temporary directory: {TEMP_DOWNLOAD_DIR}")
            except OSError as e_clean:
                print(f"Warning: Could not remove temporary directory {TEMP_DOWNLOAD_DIR} after download error: {e_clean}")
        return False

    # 4. Unzip the file into the final target directory
    print(f"Unzipping {ZIP_FILEPATH} to {FULL_TARGET_SHAPEFILE_DIR}...")
    try:
        with zipfile.ZipFile(ZIP_FILEPATH, 'r') as zip_ref:
            zip_ref.extractall(FULL_TARGET_SHAPEFILE_DIR)
        print(f"Successfully unzipped files to: {FULL_TARGET_SHAPEFILE_DIR}")
    except zipfile.BadZipFile:
        print(f"Error: Downloaded file is not a valid ZIP archive or is corrupted.")
        if os.path.exists(TEMP_DOWNLOAD_DIR):
            try:
                shutil.rmtree(TEMP_DOWNLOAD_DIR)
                print(f"Cleaned up temporary directory: {TEMP_DOWNLOAD_DIR}")
            except OSError as e_clean:
                print(f"Warning: Could not remove temporary directory {TEMP_DOWNLOAD_DIR} after unzip error: {e_clean}")
        return False
    except Exception as e:
        print(f"An error occurred during unzipping: {e}")
        if os.path.exists(TEMP_DOWNLOAD_DIR):
            try:
                shutil.rmtree(TEMP_DOWNLOAD_DIR)
                print(f"Cleaned up temporary directory: {TEMP_DOWNLOAD_DIR}")
            except OSError as e_clean:
                print(f"Warning: Could not remove temporary directory {TEMP_DOWNLOAD_DIR} after unzip error: {e_clean}")
        return False


    # 5. Clean up the temporary download directory and the ZIP file
    print(f"Cleaning up temporary download files...")
    try:
        if os.path.exists(TEMP_DOWNLOAD_DIR):
            shutil.rmtree(TEMP_DOWNLOAD_DIR) # Removes the directory and its contents (the zip file)
            print(f"Removed temporary download directory: {TEMP_DOWNLOAD_DIR}")
    except OSError as e:
        print(f"Warning: Could not remove temporary download directory {TEMP_DOWNLOAD_DIR}: {e}")

    # 6. Verify successful setup
    if os.path.exists(EXPECTED_SHP_FILE):
        print("--- Setup Successful! ---")
        print(f"Natural Earth data is now available at: {FULL_TARGET_SHAPEFILE_DIR}")
        print(f"Contents: {os.listdir(FULL_TARGET_SHAPEFILE_DIR)}")
        print("You should now be able to run the main map animation script.")
        return True
    else:
        print("--- Setup Potentially Incomplete ---")
        print(f"The expected shapefile was not found at: {EXPECTED_SHP_FILE}")
        print(f"Please check the contents of: {FULL_TARGET_SHAPEFILE_DIR} (if it exists).")
        if os.path.exists(FULL_TARGET_SHAPEFILE_DIR):
            print(f"Actual contents of {FULL_TARGET_SHAPEFILE_DIR}: {os.listdir(FULL_TARGET_SHAPEFILE_DIR)}")
        else:
            print(f"Target directory {FULL_TARGET_SHAPEFILE_DIR} does not exist.")
        return False

if __name__ == "__main__":
    download_and_setup_natural_earth_data()
