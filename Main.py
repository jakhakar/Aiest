#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import json
import random
import requests
import subprocess
import numpy as np
import moviepy.editor as mp
from moviepy.audio.fx.all import audio_normalize, volumex 
from moviepy.video.fx.all import fadein, fadeout

_whisper_available = False
try:
    import whisper_timestamped
    _whisper_available = True
except ImportError:
    print("ERROR: whisper-timestamped library not found. Please install: pip install -U whisper-timestamped")
    print(f"Python executable: {sys.executable}"); sys.exit(1)

_deepgram_available = False
try:
    from deepgram import DeepgramClient, SpeakOptions
    try: import dotenv; dotenv.load_dotenv()
    except ImportError: pass
    _deepgram_available = True
except ImportError:
    print("ERROR: deepgram-sdk library not found. Please install: pip install deepgram-sdk"); sys.exit(1)

_gemini_available = False
try:
    import google.generativeai as genai
    _gemini_available = True
except ImportError:
    print("WARNING: google-generativeai library not found. Gemini features will be unavailable. Install with: pip install -q google-generativeai")

_manim_available_for_script = False 
try:
    from manim import Scene 
    _manim_available_for_script = True
except ImportError:
    print("WARNING: manim library not found. Manim overlay generation will be unavailable. Install with: pip install manim")


import logging
import coloredlogs
import math
import time
from typing import List, Dict, Tuple, Optional, Any, Set
import re
import shutil
import concurrent.futures

log_handler = logging.StreamHandler(sys.stdout)
log_formatter = coloredlogs.ColoredFormatter('%(asctime)s %(levelname)s [%(name)s:%(funcName)s:%(lineno)d] %(message)s')
log_handler.setFormatter(log_formatter)
logger = logging.getLogger(__name__)
if not logger.hasHandlers(): logger.addHandler(log_handler); logger.setLevel(logging.INFO)
coloredlogs.install(level=logger.getEffectiveLevel(), logger=logger, fmt='%(asctime)s %(levelname)s %(message)s')
for lib_logger_name in ["moviepy", "PIL", "urllib3", "requests", "matplotlib", "httpx", "google", "manim"]: logging.getLogger(lib_logger_name).setLevel(logging.WARNING)
for lib_logger_name_info in ["whisper_timestamped", "ffmpeg", "deepgram"]: logging.getLogger(lib_logger_name_info).setLevel(logging.INFO)

class Config:
    # --- Deepgram Configuration ---
    DEEPGRAM_API_KEY_PLACEHOLDER_VALUE = "032789a7499ff620e34e634bd574efa7ef07cyba" 
    DEEPGRAM_API_KEY: Optional[str] = os.environ.get("DEEPGRAM_API_KEY", DEEPGRAM_API_KEY_PLACEHOLDER_VALUE)
    _raw_dg_key = DEEPGRAM_API_KEY or ""
    if not _raw_dg_key or "YOUR_DEEPGRAM_API_KEY" in _raw_dg_key.upper() or len(_raw_dg_key) < 20 :
        if _raw_dg_key != DEEPGRAM_API_KEY_PLACEHOLDER_VALUE:
             logger.warning(f"Deepgram API Key from ENV 'DEEPGRAM_API_KEY' ('{str(_raw_dg_key)[:10]}...') appears invalid. Using script default placeholder.")
        else: 
            logger.warning(f"Deepgram API Key not found in ENV or is placeholder. Using script default placeholder. TTS may fail.")
        DEEPGRAM_API_KEY = str(DEEPGRAM_API_KEY_PLACEHOLDER_VALUE)
    elif _raw_dg_key == DEEPGRAM_API_KEY_PLACEHOLDER_VALUE : 
         logger.warning(f"Deepgram API Key is the script's default placeholder. TTS will likely fail unless this is your actual (unlikely) key.")
    else: 
        logger.info("Deepgram API key loaded from ENV.")
    DEEPGRAM_AURA_MODEL: str = "aura-asteria-en"

    @classmethod
    def deepgram_key_configured(cls) -> bool:
        key = cls.DEEPGRAM_API_KEY
        is_valid = bool(
            key and isinstance(key, str) and len(key) >= 20 and
            "YOUR_DEEPGRAM_API_KEY" not in key.upper() and
            key != cls.DEEPGRAM_API_KEY_PLACEHOLDER_VALUE
        )
        if not is_valid:
             logger.warning(f"Deepgram key ('{str(key)[:10]}...') appears to be a placeholder or invalid. TTS may fail.")
        return is_valid

    # --- AI Video API Keys Configuration ---
    _default_ai_video_placeholders: List[str] = ["YOUR_AIVIDEO_KEY_1_PLACEHOLDER", "YOUR_AIVIDEO_KEY_2_PLACEHOLDER"]
    AIVIDEO_API_KEYS_STR: Optional[str] = os.environ.get("AIVIDEO_API_KEYS")
    AIVIDEO_API_KEYS: List[str] = []
    if AIVIDEO_API_KEYS_STR:
        AIVIDEO_API_KEYS = [key.strip() for key in AIVIDEO_API_KEYS_STR.split(',') if key.strip()]
        if AIVIDEO_API_KEYS: logger.info(f"Loaded {len(AIVIDEO_API_KEYS)} AI Video keys from ENV.")
        else: logger.warning("ENV AIVIDEO_API_KEYS empty. Falling back to placeholders."); AIVIDEO_API_KEYS = list(_default_ai_video_placeholders)
    else: logger.warning("ENV AIVIDEO_API_KEYS not found. Using placeholders."); AIVIDEO_API_KEYS = list(_default_ai_video_placeholders)

    _current_api_key_index: int = 0
    @classmethod
    def get_api_key(cls, key_index: int) -> Optional[str]:
        if not cls.AIVIDEO_API_KEYS: return None
        try: return cls.AIVIDEO_API_KEYS[key_index % len(cls.AIVIDEO_API_KEYS)]
        except (ZeroDivisionError, IndexError): return None
    @classmethod
    def get_current_key_index(cls) -> int:
        if not cls.AIVIDEO_API_KEYS: return 0
        try: return cls._current_api_key_index % len(cls.AIVIDEO_API_KEYS)
        except ZeroDivisionError: return 0
    @classmethod
    def set_current_key_index(cls, index: int):
        if cls.AIVIDEO_API_KEYS:
            try: cls._current_api_key_index = index % len(cls.AIVIDEO_API_KEYS)
            except ZeroDivisionError: cls._current_api_key_index = 0
        else: cls._current_api_key_index = 0
    @classmethod
    def get_total_keys(cls) -> int: return len(cls.AIVIDEO_API_KEYS)
    @classmethod
    def keys_configured(cls, check_validity=True) -> bool:
        if not cls.AIVIDEO_API_KEYS: return False
        if check_validity:
            return any(
                isinstance(k, str) and len(k) > 10 and "PLACEHOLDER" not in k.upper() and "YOUR_" not in k.upper() and k not in cls._default_ai_video_placeholders
                for k in cls.AIVIDEO_API_KEYS
            )
        return bool(cls.AIVIDEO_API_KEYS)

    AIVIDEO_BASE_URL: str = "https://api.example-aivideo.com" 
    AIVIDEO_GENERATE_ENDPOINT: str = "/v1/generate"; AIVIDEO_STATUS_ENDPOINT: str = "/v1/status"
    AIVIDEO_GENERATE_URL: str = AIVIDEO_BASE_URL.rstrip('/') + '/' + AIVIDEO_GENERATE_ENDPOINT.lstrip('/')
    AIVIDEO_STATUS_URL: str = AIVIDEO_BASE_URL.rstrip('/') + '/' + AIVIDEO_STATUS_ENDPOINT.lstrip('/')
    AIVIDEO_MODEL: str = "stable-video-diffusion-xt"

    # --- Gemini AI Configuration ---
    GEMINI_API_KEY_PLACEHOLDER_VALUE = "YOUR_GEMINI_API_KEY_PLACEHOLDER"
    GEMINI_API_KEY: Optional[str] = os.environ.get("GEMINI_API_KEY", GEMINI_API_KEY_PLACEHOLDER_VALUE)
    if not GEMINI_API_KEY or "YOUR_GEMINI_API_KEY" in GEMINI_API_KEY.upper() or len(GEMINI_API_KEY) < 20:
        if GEMINI_API_KEY != GEMINI_API_KEY_PLACEHOLDER_VALUE: 
            logger.warning(f"Gemini API Key from ENV 'GEMINI_API_KEY' ('{str(GEMINI_API_KEY)[:10]}...') appears invalid. Using script default placeholder.")
        else: 
            logger.warning(f"Gemini API Key not found in ENV 'GEMINI_API_KEY' or is placeholder. Script generation with Gemini will fail.")
        GEMINI_API_KEY = GEMINI_API_KEY_PLACEHOLDER_VALUE
    else:
        logger.info("Gemini API key loaded from ENV.")
    GEMINI_MODEL_NAME: str = "gemini-1.5-flash-latest" 
    GEMINI_MAX_RETRIES: int = 3
    GEMINI_RETRY_DELAY_SECONDS: int = 5

    @classmethod
    def gemini_key_configured(cls) -> bool:
        key = cls.GEMINI_API_KEY
        is_valid = bool(
            key and isinstance(key, str) and len(key) >= 20 and
            "YOUR_GEMINI_API_KEY" not in key.upper() and
            key != cls.GEMINI_API_KEY_PLACEHOLDER_VALUE
        )
        if not is_valid and _gemini_available: 
             logger.warning(f"Gemini API key ('{str(key)[:10]}...') appears to be a placeholder or invalid. Script generation with Gemini may fail.")
        return is_valid

    # --- Video Settings ---
    VIDEO_WIDTH: int = 1080; VIDEO_HEIGHT: int = 1920; VIDEO_FPS: int = 30
    VIDEO_BITRATE: str = '8000k'; TRANSITION_DURATION: float = 0.1 # Used for MoviePy concatenate if multiple clips per segment
    SEGMENT_FADE_DURATION: float = 0.3; API_EXPECTED_CLIP_DURATION: float = 4.0
    API_MOTION_LEVEL: int = 127; API_SEED: int = 0 # 0 for random seed
    POLLING_INTERVAL_SECONDS: int = 15; MAX_POLLING_ATTEMPTS: int = 100
    VIDEO_WRITE_PRESET_SEGMENT: str = 'slow'; VIDEO_WRITE_PRESET_FINAL: str = 'medium'
    VIDEO_WRITE_CRF_FINAL: int = 20

    # --- Font & Style Settings ---
    FONT_NAME: str = "Montserrat Bold" 
    FONT_DIR: str = "./fonts" 
    FONT_SIZE_BASE: int = 70
    LETTER_SPACING: float = -1.5
    PRIMARY_COLOR: str = "&H00FFFFFF&"; 
    OUTLINE_THICKNESS: float = 9 
    OUTLINE_COLOR: str = "&H00000000&"; 
    SHADOW_COLOR: str = "&H80000000&"; 
    SHADOW_DISTANCE: float = 3 
    HIGHLIGHT_COLORS_LIST: List[str] = ["&H0000FF00&", "&H000000FF&", "&H00FFFF00&"] # Green, Red, Yellow
    HIGHLIGHT_FONT_SIZE_MODE: str = "factor" 
    FONT_SIZE_HIGHLIGHT_BASE_FACTOR: float = 1.15  
    FONT_SIZE_HIGHLIGHT_ABSOLUTE: Optional[int] = 80 
    FONT_SIZE_HIGHLIGHT_LIST: List[int] = [76, 80, 84] 
    FONT_HIGHLIGHT_BLUR_AMOUNT: float = 0 
    LINE_FADE_DURATION_MS: int = 150
    RANDOM_TILT_CHANCE: float = 0.3 
    RANDOM_TILT_MAX_ANGLE_DEGREES: float = 6.0 
    TARGET_WORDS_PER_DISPLAY_SEGMENT_MIN: int = 1; TARGET_WORDS_PER_DISPLAY_SEGMENT_MAX: int = 4
    MAX_CHARS_PER_LINE: int = 28 
    MAX_LINES_PER_SUBTITLE: int = 3 
    SUBTITLE_ALIGNMENT: int = 2; # Bottom Center. Change to 5 for true Middle Center if Manim overlays allow.
    SUBTITLE_MARGIN_L: int = 30
    SUBTITLE_MARGIN_R: int = 30
    SUBTITLE_MARGIN_V: int = int(VIDEO_HEIGHT * 0.35) # Pushes text up from bottom for align 2.
                                                    # If align 5, use smaller values like 10 or 0.
    WHISPER_MODEL_SIZE: str = "base.en"; WHISPER_DEVICE: Optional[str] = None
    VIDEO_BACKGROUND_COLOR_RGB: Tuple[int, int, int] = (255, 255, 255)

    # --- Background Music (BGM) Configuration ---
    BGM_FILE_PATH: Optional[str] = os.environ.get("BGM_PATH", "./bgm/epic_background_music.mp3") 
    BGM_VOLUME: float = 0.08  
    BGM_FADEIN_DURATION: float = 1.5 
    BGM_FADEOUT_DURATION: float = 3.0 

    # --- Manim Configuration ---
    ENABLE_MANIM_OVERLAYS: bool = True 
    MANIM_RENDER_QUALITY: str = "-ql" # -ql (low), -qm (medium), -qh (high), -qk (4k)
    MANIM_OUTPUT_FORMAT: str = "mov" # "mov" (ProRes 4444 for alpha), "webm" (VP9 for alpha)


config = Config() 

def fmt_time(seconds: Optional[float]) -> str:
    if seconds is None: return "0:00:00.00"
    s = max(0.0, float(seconds)); h = int(s // 3600); m = int((s % 3600) // 60); sec = int(s % 60); cs = int((s % 1) * 100)
    return f"{h}:{m:02}:{sec:02}.{cs:02}"

def clean_filename(name: str, max_length: int = 100) -> str:
    if not name or not isinstance(name, str): return "unnamed_video"
    cl = name.strip().replace(" ", "_"); cl = "".join(c if c.isalnum() or c in ('-', '_') else '' for c in cl)
    cl = '_'.join(filter(None, cl.split('_'))); cl = cl.strip('_-')
    if cl.startswith('-'): cl = '_' + cl[1:]
    if not cl: return "cleaned_unnamed_video"
    return cl[:max_length]

def check_ffmpeg() -> bool:
    try:
        p = subprocess.run(["ffmpeg", "-version"], check=True, capture_output=True, timeout=10, text=True, errors='replace')
        logger.info(f"FFmpeg found. Version: {p.stdout.splitlines(False)[0] if p.stdout else 'Unknown'}")
        return True
    except FileNotFoundError: logger.critical("FFmpeg not found."); return False
    except subprocess.CalledProcessError as e: logger.error(f"FFmpeg version check failed. Stderr: {e.stderr}"); return False
    except subprocess.TimeoutExpired: logger.error("FFmpeg version check timed out."); return False
    except Exception as e: logger.critical(f"FFmpeg check failed: {e}", exc_info=True); return False

def check_manim() -> bool:
    global _manim_available_for_script
    if not _manim_available_for_script: 
        logger.warning("Manim Python library not imported during initial checks. Manim features will be disabled.")
        return False
    try:
        process = subprocess.run(["manim", "--version"], check=True, capture_output=True, text=True, timeout=15)
        logger.info(f"Manim CLI found. Version: {process.stdout.strip()}")
        return True
    except FileNotFoundError:
        logger.warning("Manim command-line tool not found in PATH. Manim features disabled.")
        _manim_available_for_script = False
        return False
    except subprocess.CalledProcessError as e:
        logger.warning(f"Manim --version check failed (stderr: {e.stderr.strip()}). Manim features may be unreliable, but library imported.")
        return True 
    except subprocess.TimeoutExpired:
        logger.warning("Manim --version check timed out. Manim features may be unreliable if CLI is needed for certain operations.")
        return True 
    except Exception as e:
        logger.warning(f"Unexpected error during Manim CLI check: {e}. Manim features disabled.")
        _manim_available_for_script = False
        return False

def call_aivideo_api(prompt: str, output_path: str, config_obj: Config, logger_obj: logging.Logger) -> bool:
    if not config_obj.keys_configured(): logger_obj.error("AI Video API: No valid API keys configured."); return False
    num_total_keys = config_obj.get_total_keys(); current_key_start_index = config_obj.get_current_key_index()
    generated_job_id: Optional[str] = None; successful_key_index: Optional[int] = None
    for i in range(num_total_keys):
        key_index_to_try = (current_key_start_index + i) % num_total_keys
        api_key_to_use = config_obj.get_api_key(key_index_to_try); is_placeholder_key = False
        if api_key_to_use:
            for pk in config_obj._default_ai_video_placeholders:
                if api_key_to_use == pk: is_placeholder_key = True; break
            if "YOUR_AI_VIDEO_KEY" in (api_key_to_use or "").upper() or "PLACEHOLDER" in (api_key_to_use or "").upper(): is_placeholder_key = True
        if not api_key_to_use or is_placeholder_key: logger_obj.warning(f"Skipping placeholder/invalid AI Video API key at index {key_index_to_try}."); continue
        masked_key = api_key_to_use[:4] + "****" + api_key_to_use[-4:]
        logger_obj.info(f"Attempting AI Video POST with key index {key_index_to_try} ({masked_key})...")
        headers = {"accept": "application/json", "content-type": "application/json", "Authorization": f"Bearer {api_key_to_use}"}
        payload = { "text_prompt": prompt, "model": config_obj.AIVIDEO_MODEL, "width": config_obj.VIDEO_WIDTH, "height": config_obj.VIDEO_HEIGHT, "motion_score": config_obj.API_MOTION_LEVEL }
        if config_obj.API_SEED > 0: payload["seed"] = config_obj.API_SEED
        logger_obj.info(f"  Requesting video: {config_obj.VIDEO_WIDTH}x{config_obj.VIDEO_HEIGHT}")
        try:
            response = requests.post(config_obj.AIVIDEO_GENERATE_URL, headers=headers, json=payload, timeout=90); response.raise_for_status()
            try: response_data = response.json()
            except json.JSONDecodeError: logger_obj.error(f"AI Video API: Invalid JSON (Status {response.status_code}), key index {key_index_to_try}. Response: {response.text[:100]}. Rotating."); continue
            id_keys_to_check = ['job_id', 'uuid', 'task_id', 'id', 'request_id']; temp_job_id = None; job_id_found_in_outer_loop = False
            for key_name_outer_id_check in id_keys_to_check:
                temp_job_id = response_data.get(key_name_outer_id_check)
                if temp_job_id: logger_obj.debug(f"Found job ID using key: '{key_name_outer_id_check}'"); job_id_found_in_outer_loop = True; break
            if not job_id_found_in_outer_loop and isinstance(response_data.get('data'), dict):
                for key_name_inner_id_check in id_keys_to_check:
                    temp_job_id = response_data['data'].get(key_name_inner_id_check)
                    if temp_job_id: logger_obj.debug(f"Found nested job ID: 'data.{key_name_inner_id_check}'"); break
            if not temp_job_id: logger_obj.warning(f"AI Video API: Success (Status {response.status_code}) but no job ID, key index {key_index_to_try}. Response: {str(response_data)[:200]}. Rotating."); continue
            generated_job_id = str(temp_job_id); logger_obj.info(f"AI Video API: Request successful, key index {key_index_to_try}. Job ID: {generated_job_id}")
            successful_key_index = key_index_to_try; config_obj.set_current_key_index(successful_key_index); break
        except requests.exceptions.HTTPError as http_err:
            status_code = http_err.response.status_code if http_err.response is not None else None; response_text = http_err.response.text if http_err.response is not None else "<No Response Body>"
            error_message_from_json = "";
            if http_err.response is not None:
                try: err_json = http_err.response.json(); error_message_from_json = (str(err_json.get('error','')).lower() or str(err_json.get('detail','')).lower() or str(err_json.get('message','')).lower())
                except json.JSONDecodeError: pass
            full_error_text_lower = (error_message_from_json or response_text).lower(); is_rate_limit = (status_code == 429)
            is_quota_issue = ("quota" in full_error_text_lower or "limit exceeded" in full_error_text_lower or "insufficient credits" in full_error_text_lower or "balance is too low" in full_error_text_lower)
            if is_rate_limit or is_quota_issue: reason = "Rate Limit/Quota" if is_rate_limit else "Quota Keyword"; logger_obj.warning(f"AI Video API: {reason} (Status {status_code}), key index {key_index_to_try}. Rotating. Error: {response_text[:200]}")
            else: logger_obj.error(f"AI Video API: Request failed (HTTP {status_code}), key index {key_index_to_try}. Aborting segment. Error: {response_text[:500]}"); return False
        except requests.exceptions.Timeout: logger_obj.error(f"AI Video API: Request timed out, key index {key_index_to_try}. Aborting segment."); return False
        except requests.exceptions.RequestException as req_err: logger_obj.error(f"AI Video API: Network error, key index {key_index_to_try}: {req_err}. Aborting segment."); return False
        except Exception as general_err: logger_obj.error(f"Unexpected Python error during AI Video POST, key index {key_index_to_try}: {general_err}", exc_info=True); return False
    if successful_key_index is None or generated_job_id is None: logger_obj.error("AI Video API: Failed to init job with ANY key."); return False
    polling_api_key = config_obj.get_api_key(successful_key_index)
    if not polling_api_key: logger_obj.error(f"Internal Error: Could not get API key for polling (index {successful_key_index})."); return False
    logger_obj.info(f"Polling AI Video API for job status (Job ID: {generated_job_id}), key index {successful_key_index}")
    polling_headers = {"accept": "application/json", "Authorization": f"Bearer {polling_api_key}"}
    status_url_with_job_id = f"{config_obj.AIVIDEO_STATUS_URL}?job_id={generated_job_id}"; logger_obj.debug(f"Polling URL: {status_url_with_job_id}")
    for attempt_num in range(config_obj.MAX_POLLING_ATTEMPTS):
        logger_obj.info(f"Polling attempt {attempt_num + 1}/{config_obj.MAX_POLLING_ATTEMPTS} for Job ID: {generated_job_id}...")
        try:
            status_response = requests.get(status_url_with_job_id, headers=polling_headers, timeout=30); status_response.raise_for_status()
            try: status_data = status_response.json()
            except json.JSONDecodeError: logger_obj.error(f"Polling: Invalid JSON (Status {status_response.status_code}). Text: {status_response.text[:100]}"); time.sleep(config_obj.POLLING_INTERVAL_SECONDS); continue
            logger_obj.debug(f"Polling Status (HTTP {status_response.status_code}): {str(status_data)[:300]}")
            api_status = str(status_data.get('status', status_data.get('state', 'unknown'))).lower(); logger_obj.debug(f"Interpreted API status: '{api_status}'")
            if api_status in ('completed', 'success', 'succeeded', 'finished', 'done'):
                logger_obj.info(f"Video generation successful (API status: '{api_status}')!"); video_url: Optional[str] = None
                url_keys_to_check = ['url', 'output_url', 'video_url', 'result_url', 'download_url', 'file_url', 'asset_url']
                url_found_in_outer_polling_loop = False
                for key_name_outer_url in url_keys_to_check:
                    video_url = status_data.get(key_name_outer_url)
                    if isinstance(video_url, str) and video_url.startswith('http'): url_found_in_outer_polling_loop = True; break
                if not url_found_in_outer_polling_loop and isinstance(status_data.get('data'), dict):
                    for key_name_inner_url in url_keys_to_check:
                        video_url = status_data['data'].get(key_name_inner_url)
                        if isinstance(video_url, str) and video_url.startswith('http'): break
                if video_url:
                    logger_obj.info(f"Video URL found: {video_url}. Downloading to: {output_path}...")
                    try:
                        download_response = requests.get(video_url, stream=True, timeout=600); download_response.raise_for_status()
                        output_dir_dl = os.path.dirname(output_path);
                        if output_dir_dl: os.makedirs(output_dir_dl, exist_ok=True)
                        with open(output_path, 'wb') as f_out:
                            for chunk in download_response.iter_content(chunk_size=1024*1024):
                                if chunk: f_out.write(chunk)
                        if os.path.exists(output_path) and os.path.getsize(output_path) >= 1024: logger_obj.info(f"Download successful: {output_path} (Size: {os.path.getsize(output_path)/1024/1024:.2f}MB)"); return True
                        else: logger_obj.error(f"Download error: Output file issue at {output_path}.");
                        if os.path.exists(output_path): os.remove(output_path); return False
                    except requests.exceptions.RequestException as dl_err:
                        logger_obj.error(f"Download failed: {dl_err}")
                        if os.path.exists(output_path): os.remove(output_path)
                        return False
                    except IOError as io_err:
                        logger_obj.error(f"File write error during download: {io_err}")
                        return False
                    except Exception as general_dl_err:
                        logger_obj.error(f"Unexpected error during video download or write: {general_dl_err}", exc_info=True)
                        if os.path.exists(output_path): os.remove(output_path)
                        return False
                else: logger_obj.error(f"API status '{api_status}' but no video URL found: {str(status_data)[:300]}"); return False
            elif api_status in ('failed', 'error', 'cancelled', 'canceled', 'aborted', 'rejected'):
                error_reason = status_data.get('error', status_data.get('message', status_data.get('detail', 'No reason.')))
                logger_obj.error(f"Video generation failed (API status: '{api_status}'). Reason: {error_reason}"); return False
            elif api_status in ('processing', 'pending', 'queued', 'running', 'starting', 'generating', 'submitted', 'accepted', 'in_progress'):
                progress_val = status_data.get('progress'); progress_str = f"(Progress: {progress_val})" if progress_val is not None else ""
                logger_obj.info(f"Video generation in progress (API status: '{api_status}'){progress_str}. Waiting {config_obj.POLLING_INTERVAL_SECONDS}s...")
                time.sleep(config_obj.POLLING_INTERVAL_SECONDS)
            else: logger_obj.warning(f"Unknown API status: '{api_status}'. Treating as 'in progress'. Response: {str(status_data)[:200]}"); time.sleep(config_obj.POLLING_INTERVAL_SECONDS)
        except requests.exceptions.HTTPError as poll_http_err:
            poll_status_code = poll_http_err.response.status_code if poll_http_err.response is not None else None
            logger_obj.warning(f"Polling HTTP Error (Status {poll_status_code}) for Job {generated_job_id}: {poll_http_err}")
            if poll_status_code == 404: logger_obj.error(f"Polling: Job {generated_job_id} not found (404). Expired? Stopping."); return False
            elif poll_status_code in [401, 403]:
                logger_obj.error(f"Polling Auth Error (Status {poll_status_code}). Key index {successful_key_index} invalid? Stopping.")
                return False
            elif poll_status_code == 429: wait_time = config_obj.POLLING_INTERVAL_SECONDS*3; logger_obj.warning(f"Rate limited on polling. Waiting {wait_time}s..."); time.sleep(wait_time)
            else: time.sleep(config_obj.POLLING_INTERVAL_SECONDS)
        except requests.exceptions.Timeout: logger_obj.warning(f"Polling timed out for Job {generated_job_id}. Waiting..."); time.sleep(config_obj.POLLING_INTERVAL_SECONDS)
        except requests.exceptions.RequestException as poll_req_err: logger_obj.warning(f"Polling network error for Job {generated_job_id}: {poll_req_err}. Waiting..."); time.sleep(config_obj.POLLING_INTERVAL_SECONDS)
        except Exception as poll_general_err: logger_obj.error(f"Unexpected Python error during polling for Job {generated_job_id}: {poll_general_err}", exc_info=True); return False
    logger_obj.error(f"Polling timed out for Job ID {generated_job_id} after {config_obj.MAX_POLLING_ATTEMPTS} attempts."); return False

def generate_random_topic_for_shorts(config_obj: Config, logger_obj: logging.Logger) -> str:
    default_topics = [
        "Unbelievable historical coincidences you won't believe.",
        "The craziest animal survival tactics in nature.",
        "Mind-blowing facts about the human brain.",
        "Forgotten technologies that were ahead of their time.",
        "The bizarre story of the most expensive mistake in history.",
        "Incredible ancient inventions that still baffle scientists.",
        "The weirdest deep-sea creatures ever discovered.",
        "Strange historical laws that actually existed.",
        "Optical illusions that will trick your brain.",
        "The most mysterious unsolved disappearances."
    ]
    
    if _gemini_available and config_obj.gemini_key_configured():
        logger_obj.info("Attempting to generate a random video topic with Gemini AI...")
        try:
            model = genai.GenerativeModel(config_obj.GEMINI_MODEL_NAME) 
            
            topic_prompt = (
                "Generate a single, highly engaging, and slightly sensational topic suitable for a 30-60 second vertical short video "
                "in the style of 'crazy facts', 'unbelievable history', or 'bizarre science'. "
                "The topic should be a compelling statement or question, ideally 10-20 words long. "
                "Output only the topic string itself, with no extra text, labels, or quotation marks. "
                "Example: 'The ancient Roman emperor who declared war on the sea itself!'"
            )
            
            for attempt in range(config_obj.GEMINI_MAX_RETRIES):
                try:
                    response = model.generate_content(
                        topic_prompt,
                        generation_config=genai.types.GenerationConfig(temperature=0.95)
                    )
                    if response.candidates and response.candidates[0].content.parts:
                        generated_topic = response.candidates[0].content.parts[0].text.strip().replace('"', '') 
                        if generated_topic and 10 < len(generated_topic) < 200 and '\n' not in generated_topic:
                            logger_obj.info(f"Gemini generated video topic: '{generated_topic}'")
                            return generated_topic
                        else:
                            logger_obj.warning(f"Gemini generated an invalid or poorly formatted topic on attempt {attempt+1}: '{generated_topic}'. Retrying if possible.")
                    elif hasattr(response, 'prompt_feedback') and response.prompt_feedback.block_reason:
                        logger_obj.error(f"Gemini topic generation blocked: {response.prompt_feedback.block_reason_message or response.prompt_feedback.block_reason}. Cannot retry.")
                        break 
                    else:
                        logger_obj.warning(f"Gemini topic generation: No content in response on attempt {attempt+1}. Retrying if possible.")

                except Exception as e_inner:
                    logger_obj.warning(f"Error during Gemini topic generation attempt {attempt+1}: {e_inner}")
                
                if attempt < config_obj.GEMINI_MAX_RETRIES - 1:
                    logger_obj.info(f"Waiting {config_obj.GEMINI_RETRY_DELAY_SECONDS}s before next topic generation retry.")
                    time.sleep(config_obj.GEMINI_RETRY_DELAY_SECONDS)
            
            logger_obj.warning("Max retries reached or unrecoverable error for Gemini topic generation. Falling back to default topics.")

        except Exception as e:
            logger_obj.warning(f"Outer error setting up Gemini for topic generation: {e}. Falling back to default topics.")

    chosen_topic = random.choice(default_topics)
    logger_obj.info(f"Using fallback random video topic: '{chosen_topic}'")
    return chosen_topic

def generate_detailed_video_plan_with_gemini(user_topic_prompt: str, config_obj: Config, logger_obj: logging.Logger) -> Optional[Dict[str, Any]]:
    if not _gemini_available:
        logger_obj.error("Gemini library (google-generativeai) is not available. Cannot generate script.")
        return None
    if not config_obj.gemini_key_configured():
        logger_obj.error("Gemini API key is not configured correctly. Cannot generate script.")
        return None

    try:
        model = genai.GenerativeModel(config_obj.GEMINI_MODEL_NAME)
        logger_obj.info(f"Gemini AI: Configured with model '{config_obj.GEMINI_MODEL_NAME}'.")
    except Exception as e:
        logger_obj.error(f"Gemini AI: Failed to configure or initialize model: {e}", exc_info=True)
        return None

    system_instruction = f"""
You are an expert scriptwriter and creative director for viral short-form vertical videos (like TikTok, YouTube Shorts).
Your task is to generate a complete video plan in JSON format based on the provided topic.
The video should be fast-paced, visually dynamic, engaging, and surprising, typically 30-60 seconds long.

The JSON output MUST adhere to the following structure:
{{
  "video_title": "A catchy and concise title for the video (max 60 chars, related to the topic)",
  "overall_mood": "A few keywords describing the overall video mood (e.g., 'Mysterious & Epic', 'Quirky & Astonishing', 'Dark & Intriguing', 'Fast-paced & Informative')",
  "target_duration_seconds": "An estimated target duration for the whole video in seconds (e.g., 45)",
  "suggested_bgm_style": "A brief suggestion for the background music style and tempo (e.g., 'Tense orchestral underscore, rising intensity', 'Upbeat quirky electronic, medium tempo', 'Somber solo piano, slow tempo', 'Epic cinematic trailer music, powerful drums')",
  "voiceover": [ // A list of 5-8 segment objects
    // Each segment object MUST have these keys:
    {{
      "segment_id": "A unique short ID for the segment (e.g., 'scene_01_hook', 'fact_02_reveal')",
      "duration_hint_seconds": "Estimated duration for this segment's voiceover in seconds (e.g., '3-7' or '5')",
      "text": "Voiceover text for this segment. VERY concise: 1, or at most 2, short sentences (10-25 words). Make it punchy and easy to read as an on-screen caption.",
      "image_prompt": "A vivid, dynamic visual prompt for an AI image/video generator. Describe an exciting scene, action, or symbolic imagery. Suggest camera angles or motion if it enhances the story (e.g., 'dramatic low-angle shot of [subject]', 'fast zoom out revealing [larger scene]', 'slow-motion shot of [action]', 'mysterious figure silhouetted against a bright light', 'time-lapse of a flower blooming'). Avoid generic prompts; be specific and creative.",
      "highlight_words": [ // List of 1-3 short, impactful phrases (1-4 words each) from THIS segment's "text".
                           // These exact phrases MUST appear verbatim in THIS segment's "text".
                           // Choose words that would look good emphasized as large animated text.
                           // If no words are suitable for highlighting, provide an empty list [].
                         ],
      "sfx_suggestions": [ // List of 1-3 brief sound effect suggestions relevant to the visuals or text.
                           // (e.g., ["dramatic whoosh", "stone grinding", "ethereal hum", "sword clash", "clock ticking rapidly"])
                           // Provide an empty list [] if no SFX are strongly indicated.
                         ],
      "manim_animation_concept": "Optional: A very simple, clear concept for a ManimCE animation overlay (max 15 words). Describe WHAT to animate (e.g., 'Text 'WARNING!' flashes red', 'Date '1066 AD' appears and scales up', 'Simple arrow animates pointing left then right', 'A circle grows from center'). Keep it EXTREMELY simple for programmatic generation. If no good concept, use an empty string ''.",
      "manim_position_hint": "Optional: Suggests Manim animation position (e.g., 'top_right', 'center_bottom', 'middle_left', 'UR', 'DL'). Default is 'center'. Use empty string '' if default.",
      "manim_color_hint": "Optional: Suggests a Manim color for text/shape (e.g., 'YELLOW', 'RED_C', 'BLUE', '#FF8C00'). Default is Manim's default. Use empty string '' if default."
    }}
  ]
}}

CRITICAL INSTRUCTIONS:
- Hook: The first segment's "text" and "image_prompt" MUST be a strong hook.
- Flow: Ensure a logical and engaging flow between segments.
- Visuals: "image_prompt"s should be exciting and descriptive.
- Highlights: "highlight_words" MUST be exact substrings of their segment's "text".
- Conciseness: "text" segments must be brief.
- Manim Concepts: Keep "manim_animation_concept" VERY simple (basic text, shapes, fades, simple transforms/movements, scaling). Avoid complex multi-object interactions or physics.
- Conclusion: The final segment should provide a memorable takeaway.

Generate a script for the topic: "{user_topic_prompt}"

Output ONLY the valid JSON object. No explanations, no markdown, just the JSON.
"""
    
    logger_obj.info(f"Gemini AI: Sending detailed video plan generation prompt for topic: '{user_topic_prompt}'")
    
    for attempt in range(config_obj.GEMINI_MAX_RETRIES):
        logger_obj.info(f"Gemini AI: Video plan generation attempt {attempt + 1}/{config_obj.GEMINI_MAX_RETRIES}...")
        try:
            response = model.generate_content(
                system_instruction, 
                generation_config=genai.types.GenerationConfig(temperature=0.8) 
            )
            
            if hasattr(response, 'prompt_feedback') and response.prompt_feedback.block_reason:
                logger_obj.error(f"Gemini AI: Prompt blocked on attempt {attempt + 1}. Reason: {response.prompt_feedback.block_reason_message or response.prompt_feedback.block_reason}. Cannot retry this prompt.")
                return None 

            if not response.candidates or not response.candidates[0].content.parts:
                logger_obj.warning(f"Gemini AI: No content in response on attempt {attempt + 1}.")
                if attempt < config_obj.GEMINI_MAX_RETRIES - 1:
                    logger_obj.info(f"Waiting {config_obj.GEMINI_RETRY_DELAY_SECONDS}s before next retry.")
                    time.sleep(config_obj.GEMINI_RETRY_DELAY_SECONDS)
                    continue
                else:
                    logger_obj.error("Gemini AI: Max retries reached, no content returned.")
                    return None

            raw_json_text = response.candidates[0].content.parts[0].text
            logger_obj.debug(f"Gemini AI: Raw response text (attempt {attempt + 1}):\n{raw_json_text[:1000]}...")

            json_match = re.search(r"```json\s*([\s\S]*?)\s*```", raw_json_text, re.IGNORECASE)
            if json_match:
                cleaned_json_text = json_match.group(1).strip()
            else:
                cleaned_json_text = raw_json_text.strip()
            
            if not (cleaned_json_text.startswith('{') and cleaned_json_text.endswith('}')):
                logger_obj.warning(f"Gemini AI: Response (attempt {attempt+1}) does not appear to be valid JSON structure. Content: {cleaned_json_text[:500]}")
                if attempt < config_obj.GEMINI_MAX_RETRIES - 1:
                    time.sleep(config_obj.GEMINI_RETRY_DELAY_SECONDS); continue
                else: return None

            try:
                script_dict = json.loads(cleaned_json_text)
                logger_obj.info("Gemini AI: Successfully parsed JSON response for video plan.")
                return script_dict
            except json.JSONDecodeError as e:
                logger_obj.error(f"Gemini AI: Failed to decode JSON response on attempt {attempt + 1}: {e}")
                logger_obj.error(f"Gemini AI: Offending text snippet: {cleaned_json_text[max(0,e.pos-30):e.pos+30]}")
                if attempt < config_obj.GEMINI_MAX_RETRIES - 1:
                    time.sleep(config_obj.GEMINI_RETRY_DELAY_SECONDS); continue
                else: return None
        except Exception as e: 
            logger_obj.error(f"Gemini AI: Error during API call/processing on attempt {attempt + 1}: {e}", exc_info=False) 
            is_retriable_api_error = any(err_str in str(e).lower() for err_str in [
                "resource exhausted", "unavailable", "internal error", "deadline exceeded", "service currently unavailable"
            ])
            
            if is_retriable_api_error and attempt < config_obj.GEMINI_MAX_RETRIES - 1:
                logger_obj.info(f"Retriable API error. Waiting {config_obj.GEMINI_RETRY_DELAY_SECONDS}s before next retry.")
                time.sleep(config_obj.GEMINI_RETRY_DELAY_SECONDS)
            else:
                logger_obj.error(f"Gemini AI: Max retries reached or non-retriable/unknown API error: {e}", exc_info=True)
                return None
    
    logger_obj.error("Gemini AI: All video plan generation attempts failed.")
    return None

def get_user_script() -> Optional[Dict[str, Any]]:
    script_to_validate: Optional[Dict[str, Any]] = None
    use_gemini = _gemini_available and config.gemini_key_configured()

    if use_gemini:
        logger.info("--- Attempting Script Generation with Gemini AI ---")
        random_topic = generate_random_topic_for_shorts(config, logger)
        if random_topic:
            script_to_validate = generate_detailed_video_plan_with_gemini(random_topic, config, logger) 
            if script_to_validate:
                logger.info("--- Gemini AI Script (Video Plan) Received ---")
                logger.debug(json.dumps(script_to_validate, indent=2))
            else:
                logger.error("Failed to generate script with Gemini AI. Falling back to example script.")
        else:
            logger.error("Failed to generate a random topic for Gemini. Falling back to example script.")
    else:
        if not _gemini_available:
            logger.info("Gemini library not available. Skipping Gemini script generation.")
        elif not config.gemini_key_configured():
            logger.info("Gemini API key not configured. Skipping Gemini script generation.")
    
    if not script_to_validate: 
        logger.info("--- Using Example Script JSON as Fallback ---")
        example_script = {
            "video_title": "Alexander's Bizarre Truths (Fallback)",
            "overall_mood": "Epic, Mysterious, Historical",
            "target_duration_seconds": "45",
            "suggested_bgm_style": "Epic orchestral underscore with moments of tension and wonder",
            "voiceover": [
                {
                "segment_id": "scene_01_hook",
                "duration_hint_seconds": "5",
                "text": "Crazy and bizarre facts about Alexander you won't learn anywhere else!",
                "image_prompt": "Ultra wide shot of Alexander the Great on a horse, silhouetted against a dramatic sunset over a battlefield. Dust and embers float in the air. Cinematic, god rays.",
                "highlight_words": ["Crazy and bizarre", "Alexander", "anywhere else"],
                "sfx_suggestions": ["distant battle horn", "horse neigh"],
                "manim_animation_concept": "Text 'SHOCKING FACTS!' flashes quickly in center",
                "manim_position_hint": "center",
                "manim_color_hint": "YELLOW"
                },
                {
                "segment_id": "scene_02_mommys_boy",
                "duration_hint_seconds": "7",
                "text": "Alexander was a mommy's boy! His mother, Olympias, brainwashed him into believing his real father was the Greek God Zeus.",
                "image_prompt": "Split screen effect. Left: Close up on young Alexander's determined face as his mother Olympias (regal, intense eyes) whispers intensely in his ear, ancient Greek palace interior. Right: A swirling, ethereal depiction of Zeus forming in the clouds, lightning crackling.",
                "highlight_words": ["mommy's boy", "brainwashed him", "God Zeus"],
                "sfx_suggestions": ["soft whisper", "distant thunder clap"],
                "manim_animation_concept": "", "manim_position_hint": "", "manim_color_hint": ""
                },
                {
                "segment_id": "scene_08_buried_alive",
                "duration_hint_seconds": "5",
                "text": "So, history's greatest conqueror was likely buried alive. Imagine that!",
                "image_prompt": "Slow zoom out from a close-up of Alexander's face (eyes closed but looking peaceful) inside a grand, ornate sarcophagus as it's being sealed in a dark, echoing tomb.",
                "highlight_words": ["buried alive", "Imagine that"],
                "sfx_suggestions": ["heavy stone grinding shut", "final echoing thud", "wind howling faintly"],
                "manim_animation_concept": "Text 'BURIED ALIVE?' fades in slowly and trembles", 
                "manim_position_hint": "bottom_center", 
                "manim_color_hint": "DARKRED"
                }
            ]
        }
        logger.debug(json.dumps(example_script, indent=2))
        script_to_validate = example_script
        
    if not script_to_validate:
        logger.critical("No script data available for validation (Gemini failed and no fallback).") 
        return None
        
    try: # Script Validation
        if not isinstance(script_to_validate, dict): raise ValueError("Script must be a JSON object.")
        
        required_top_level_keys = ["video_title", "overall_mood", "target_duration_seconds", "suggested_bgm_style", "voiceover"]
        for key in required_top_level_keys:
            if key not in script_to_validate:
                logger.warning(f"Script missing top-level key: '{key}'. Using default or ignoring.")
                if key == "video_title": script_to_validate[key] = "AI Generated Video"
                elif key == "overall_mood": script_to_validate[key] = "Neutral"
                elif key == "target_duration_seconds": script_to_validate[key] = "45" # Default string
                elif key == "suggested_bgm_style": script_to_validate[key] = "None"


        if not isinstance(script_to_validate.get("video_title"), str) or not script_to_validate["video_title"].strip():
            script_to_validate["video_title"] = "AI Generated Video" 
        else: 
            script_to_validate["video_title"] = script_to_validate["video_title"][:100]

        if not isinstance(script_to_validate.get("voiceover"), list) or not script_to_validate["voiceover"]:
            raise ValueError("Script 'voiceover' list is invalid or empty.")
        
        for i, item in enumerate(script_to_validate["voiceover"]):
            if not isinstance(item, dict): raise ValueError(f"Voiceover item {i+1} is not a dictionary.")
            
            required_segment_keys = ["segment_id", "duration_hint_seconds", "text", "image_prompt", 
                                     "highlight_words", "sfx_suggestions", "manim_animation_concept", 
                                     "manim_position_hint", "manim_color_hint"]
            for req_key in required_segment_keys:
                if req_key not in item:
                    if req_key in ["highlight_words", "sfx_suggestions"]: item[req_key] = []
                    elif req_key in ["manim_animation_concept", "manim_position_hint", "manim_color_hint"]: item[req_key] = ""
                    elif req_key == "segment_id": item[req_key] = f"segment_{i+1}_auto_id"
                    elif req_key == "duration_hint_seconds": item[req_key] = "5" 
                    else: raise ValueError(f"Voiceover item {i+1} missing required key '{req_key}'.")
            
            if not isinstance(item.get("text"), str) or not item["text"].strip():
                raise ValueError(f"Voiceover item {i+1} has invalid/empty 'text'.")
            if not isinstance(item.get("image_prompt"), str) or not item["image_prompt"].strip():
                 logger.warning(f"Voiceover item {i+1} has missing/empty 'image_prompt'. Will use 'text' as fallback.")
                 item["image_prompt"] = item["text"][:100] 
            
            hw = item.get("highlight_words")
            if not (isinstance(hw, list) and all(isinstance(w, str) for w in hw)):
                logger.warning(f"Voiceover item {i+1} 'highlight_words' is not a list of strings. Resetting to empty list.")
                item["highlight_words"] = []
            else: 
                text_for_check = item["text"].lower() 
                valid_hw = []
                for phrase_candidate in hw:
                    if phrase_candidate.strip().lower() in text_for_check: # Ensure phrase exists
                        valid_hw.append(phrase_candidate) 
                    else:
                        logger.warning(f"Voiceover item {i+1}: Highlight phrase '{phrase_candidate}' not found exactly in text '{item['text']}'. Ignoring.")
                item["highlight_words"] = valid_hw
        logger.info("Script validation successful.")
        return script_to_validate
    except ValueError as e:
        logger.critical(f"Script validation failed: {e}")
        return None
    except Exception as e_val:
        logger.critical(f"Unexpected error during script validation: {e_val}", exc_info=True)
        return None

def make_silent_stereo_frame_robust(t_val):
    if isinstance(t_val, (int, float)): return np.zeros((1, 2), dtype=np.float32)
    elif isinstance(t_val, np.ndarray): return np.zeros((len(t_val), 2), dtype=np.float32)
    else: logger.warning(f"make_silent_stereo_frame_robust received type: {type(t_val)}."); return np.zeros((1, 2), dtype=np.float32)

def generate_voiceover(script_segments: List[Dict[str, Any]], output_audio_path: str) -> Tuple[float, List[Dict[str, Any]]]:
    logger.info(f"Generating voiceover for {len(script_segments)} segments using Deepgram.")
    if not _deepgram_available: logger.critical("Deepgram SDK is not available."); return 0.0, []
    if not config.deepgram_key_configured(): logger.critical("Deepgram API Key is not configured correctly."); return 0.0, []
    try:
        deepgram_client = DeepgramClient(api_key=config.DEEPGRAM_API_KEY)
        speak_options = SpeakOptions(model=config.DEEPGRAM_AURA_MODEL)
        logger.info(f"Deepgram client initialized. Using model: '{config.DEEPGRAM_AURA_MODEL}'")
    except Exception as e: logger.critical(f"Failed to initialize Deepgram client: {e}", exc_info=True); return 0.0, []
    segment_audio_files: List[Optional[str]] = []; temp_audio_files_to_clean: Set[str] = set()
    segment_durations: List[float] = []; timed_script_segments: List[Dict[str, Any]] = []
    estimated_total_duration = 0.0; default_silence_duration = 0.05
    temp_audio_segment_dir = "temp_audio_segments"; main_output_audio_dir = os.path.dirname(output_audio_path)
    try:
        if main_output_audio_dir: os.makedirs(main_output_audio_dir, exist_ok=True)
        os.makedirs(temp_audio_segment_dir, exist_ok=True)
    except OSError as e: logger.error(f"Failed to create directories for audio generation: {e}"); return 0.0, []
    for i, segment_data in enumerate(script_segments): # segment_data here is from Gemini's detailed plan
        text_to_speak = str(segment_data.get("text", "")).strip()
        clean_text_prefix = clean_filename(text_to_speak[:20], 20)
        temp_segment_audio_path = os.path.join(temp_audio_segment_dir, f"segment_{i+1}_{clean_text_prefix}_audio.mp3")
        temp_audio_files_to_clean.add(temp_segment_audio_path)
        current_segment_duration = 0.0; generated_audio_file_path: Optional[str] = None
        
        # We need to pass through the original segment data from Gemini's plan
        current_segment_info_with_timing = segment_data.copy() # Start with all Gemini plan data

        if not text_to_speak:
            logger.warning(f"  Segment {i+1}: Text is empty. Using {default_silence_duration}s of silence.")
            current_segment_duration = default_silence_duration
        else:
            logger.info(f"  Segment {i+1}: Requesting Deepgram TTS for: '{text_to_speak[:60]}...'")
            try:
                source_data = {"text": text_to_speak}
                start_time_tts = time.time()
                response_from_save = deepgram_client.speak.rest.v("1").save(temp_segment_audio_path, source_data, options=speak_options)
                tts_request_time = time.time() - start_time_tts
                dg_request_id = response_from_save.headers.get('dg-request-id', 'N/A') if hasattr(response_from_save, 'headers') else 'N/A'
                logger.debug(f"  Segment {i+1}: Deepgram API req sent in {tts_request_time:.2f}s. ID: {dg_request_id}. Saved to: {temp_segment_audio_path}")
                if os.path.exists(temp_segment_audio_path) and os.path.getsize(temp_segment_audio_path) > 100:
                    audio_clip_for_duration: Optional[mp.AudioFileClip] = None
                    try:
                        audio_clip_for_duration = mp.AudioFileClip(temp_segment_audio_path)
                        current_segment_duration = audio_clip_for_duration.duration if audio_clip_for_duration.duration else 0.0
                        if current_segment_duration < 0.01:
                            logger.warning(f"Segment {i+1}: Deepgram audio near-zero duration. Estimating."); current_segment_duration = max(default_silence_duration, len(text_to_speak) / 15.0)
                            if os.path.exists(temp_segment_audio_path): os.remove(temp_segment_audio_path)
                        else: logger.info(f"  Segment {i+1}: Audio generated. Duration: {current_segment_duration:.2f}s"); generated_audio_file_path = temp_segment_audio_path
                    finally:
                        if audio_clip_for_duration:
                            try: audio_clip_for_duration.close()
                            except Exception as close_err: logger.debug(f"Error closing audio clip for duration check: {close_err}")
                else:
                    file_status_message = "missing" if not os.path.exists(temp_segment_audio_path) else f"too small ({os.path.getsize(temp_segment_audio_path)} bytes)"
                    logger.error(f"Segment {i+1}: Deepgram TTS failed/invalid file ({file_status_message}). Estimating duration."); current_segment_duration = max(default_silence_duration, len(text_to_speak) / 15.0)
                    if os.path.exists(temp_segment_audio_path): os.remove(temp_segment_audio_path)
            except Exception as e:
                logger.error(f"  Segment {i+1}: Error during Deepgram TTS: {e}", exc_info=False);
                current_segment_duration = max(default_silence_duration, len(text_to_speak) / 15.0)
                if os.path.exists(temp_segment_audio_path): os.remove(temp_segment_audio_path)
        
        current_segment_info_with_timing['prelim_start_time'] = estimated_total_duration
        current_segment_info_with_timing['prelim_duration'] = current_segment_duration
        current_segment_info_with_timing['audio_file'] = generated_audio_file_path
        timed_script_segments.append(current_segment_info_with_timing) # This now holds the full Gemini plan + timing
        
        segment_durations.append(current_segment_duration); segment_audio_files.append(generated_audio_file_path)
        estimated_total_duration += current_segment_duration

    if estimated_total_duration <= 0.01:
        logger.error("Total estimated voiceover duration is zero. Aborting.");
        if os.path.exists(temp_audio_segment_dir) and not os.listdir(temp_audio_segment_dir):
            try: os.rmdir(temp_audio_segment_dir)
            except OSError: pass
        return 0.0, []
    final_audio_clips_for_concat: List[mp.AudioClip] = []; concatenated_audio_clip: Optional[mp.AudioClip] = None
    normalized_final_audio_clip: Optional[mp.AudioClip] = None; actual_total_duration_before_norm = 0.0
    try:
        for i, audio_file_path_for_concat in enumerate(segment_audio_files):
            duration_for_this_clip = segment_durations[i]; clip_to_add_for_concat: Optional[mp.AudioClip] = None
            try:
                if audio_file_path_for_concat and os.path.exists(audio_file_path_for_concat): clip_to_add_for_concat = mp.AudioFileClip(audio_file_path_for_concat)
                elif duration_for_this_clip > 0.0: logger.debug(f"Creating silent audio for segment {i+1}."); clip_to_add_for_concat = mp.AudioClip(make_silent_stereo_frame_robust, duration=duration_for_this_clip, fps=44100)
                else: continue
                if clip_to_add_for_concat: final_audio_clips_for_concat.append(clip_to_add_for_concat); actual_total_duration_before_norm += duration_for_this_clip
            except Exception as load_err_concat:
                logger.error(f"Error loading/creating audio clip for segment {i+1}: {load_err_concat}. Adding silence.", exc_info=True)
                if duration_for_this_clip > 0.0:
                    try: fallback_silent_clip = mp.AudioClip(make_silent_stereo_frame_robust, duration=duration_for_this_clip, fps=44100); final_audio_clips_for_concat.append(fallback_silent_clip); actual_total_duration_before_norm += duration_for_this_clip
                    except Exception as silent_clip_create_err: logger.error(f"Failed to create fallback silent clip for segment {i+1}: {silent_clip_create_err}")
        if not final_audio_clips_for_concat: logger.error("No audio clips for concatenation."); return 0.0, []
        logger.info(f"Concatenating {len(final_audio_clips_for_concat)} audio segments. Summed duration before norm: {actual_total_duration_before_norm:.3f}s")
        concatenated_audio_clip = mp.concatenate_audioclips(final_audio_clips_for_concat)
        duration_after_concatenation = concatenated_audio_clip.duration if concatenated_audio_clip else 0.0
        if not concatenated_audio_clip or duration_after_concatenation <= 0.01:
            logger.error("Audio concatenation resulted in zero duration.");
            for clip in final_audio_clips_for_concat:
                if clip and hasattr(clip, 'close'): clip.close()
            return 0.0, []
        logger.info(f"Audio concatenated. Duration before norm: {duration_after_concatenation:.3f}s. Normalizing...")
        normalized_final_audio_clip = audio_normalize(concatenated_audio_clip)
        final_normalized_voiceover_duration = normalized_final_audio_clip.duration if normalized_final_audio_clip else 0.0
        if final_normalized_voiceover_duration <= 0.01:
            logger.error("Audio normalization resulted in zero duration.");
            for clip in final_audio_clips_for_concat:
                if clip and hasattr(clip, 'close'): clip.close()
            if concatenated_audio_clip and hasattr(concatenated_audio_clip, 'close'): concatenated_audio_clip.close()
            return 0.0, []
        logger.info(f"Audio normalization successful. Final duration: {final_normalized_voiceover_duration:.3f}s")
        if actual_total_duration_before_norm > 0.01:
            rescale_factor = final_normalized_voiceover_duration / actual_total_duration_before_norm
            logger.info(f"Rescaling segment timings by factor: {rescale_factor:.6f}.")
            current_timeline_marker = 0.0
            for i, segment_info_item in enumerate(timed_script_segments): # This is timed_script_segments from above
                 rescaled_segment_duration = segment_durations[i] * rescale_factor
                 segment_info_item['start_time'] = current_timeline_marker # Add final start_time
                 segment_info_item['duration'] = rescaled_segment_duration # Add final duration
                 segment_info_item['end_time'] = current_timeline_marker + rescaled_segment_duration; current_timeline_marker += rescaled_segment_duration
                 segment_info_item.pop('prelim_start_time', None); segment_info_item.pop('prelim_duration', None)
            if timed_script_segments:
                timed_script_segments[-1]['end_time'] = final_normalized_voiceover_duration
                timed_script_segments[-1]['duration'] = max(0.0, final_normalized_voiceover_duration - timed_script_segments[-1].get('start_time', 0.0))
        else:
            logger.warning("Cannot rescale segment timings. Using preliminary estimates.")
            current_timeline_marker = 0.0
            for segment_info_item in timed_script_segments:
                duration_val = segment_info_item.get('prelim_duration', 0.0)
                segment_info_item['start_time'] = current_timeline_marker; segment_info_item['duration'] = duration_val
                segment_info_item['end_time'] = current_timeline_marker + duration_val; current_timeline_marker += duration_val
                segment_info_item.pop('prelim_start_time', None); segment_info_item.pop('prelim_duration', None)
        logger.info(f"Writing final voiceover audio to: {output_audio_path}")
        normalized_final_audio_clip.write_audiofile(output_audio_path, codec='mp3', logger=None, bitrate='192k')
        if not os.path.exists(output_audio_path) or os.path.getsize(output_audio_path) < 100:
            logger.error(f"Failed to write final voiceover audio or file too small.");
            for clip_obj in final_audio_clips_for_concat + ([concatenated_audio_clip, normalized_final_audio_clip] if concatenated_audio_clip else []):
                if clip_obj and hasattr(clip_obj, 'close'):
                    try: clip_obj.close()
                    except: pass
            return 0.0, []
        logger.info(f"Voiceover generation successful. File: {output_audio_path} (Duration: {final_normalized_voiceover_duration:.3f}s)")
        return final_normalized_voiceover_duration, timed_script_segments # Return the list with full Gemini plan + new timings
    except Exception as e: logger.error(f"Unexpected error during audio finalization: {e}", exc_info=True); return 0.0, []
    finally:
        logger.debug("Closing MoviePy audio objects from voiceover generation...")
        all_clips_to_close = []
        if 'final_audio_clips_for_concat' in locals(): all_clips_to_close.extend(final_audio_clips_for_concat)
        if 'concatenated_audio_clip' in locals() and concatenated_audio_clip: all_clips_to_close.append(concatenated_audio_clip)
        if 'normalized_final_audio_clip' in locals() and normalized_final_audio_clip: all_clips_to_close.append(normalized_final_audio_clip)
        closed_count = 0
        for clip_obj in all_clips_to_close:
            if clip_obj and hasattr(clip_obj, 'close') and callable(clip_obj.close):
                try: clip_obj.close(); closed_count +=1
                except Exception as e_close: logger.debug(f"Error closing an audio clip: {e_close}")
        logger.debug(f"Closed {closed_count} audio objects in voiceover generation.")
        logger.debug(f"Cleaning up {len(temp_audio_files_to_clean)} temporary audio segment files from '{temp_audio_segment_dir}'...")
        cleaned_temp_files_count = 0
        for temp_file_path_to_remove in temp_audio_files_to_clean:
            if temp_file_path_to_remove and os.path.exists(temp_file_path_to_remove):
                try: os.remove(temp_file_path_to_remove); cleaned_temp_files_count +=1
                except OSError as e_remove_temp: logger.warning(f"Could not remove temp audio file '{temp_file_path_to_remove}': {e_remove_temp}")
        logger.debug(f"Removed {cleaned_temp_files_count} temporary audio files.")
        if os.path.exists(temp_audio_segment_dir) and not os.listdir(temp_audio_segment_dir):
            try: os.rmdir(temp_audio_segment_dir); logger.debug(f"Removed empty temp audio directory: {temp_audio_segment_dir}")
            except OSError as e_rmdir_temp: logger.warning(f"Could not remove temp audio directory '{temp_audio_segment_dir}': {e_rmdir_temp}")
        elif os.path.exists(temp_audio_segment_dir):
             try:
                  if os.listdir(temp_audio_segment_dir): logger.warning(f"Temp audio directory '{temp_audio_segment_dir}' not empty after cleanup.")
             except OSError: logger.warning(f"Could not check contents of temp audio directory '{temp_audio_segment_dir}'.")

def generate_video_segment(segment_plan: Dict[str, Any], segment_index: int, base_filename: str) -> Optional[str]: # segment_plan is now the full detail from Gemini
    # Use image_prompt from segment_plan
    prompt_text = str(segment_plan.get("image_prompt") or segment_plan.get("text") or f"Abstract visual for segment {segment_index + 1}").strip()
    # Use actual segment_duration after voiceover generation
    segment_duration = segment_plan.get('duration', 0.0) # This 'duration' is added by generate_voiceover
    
    clean_prompt_part = clean_filename(prompt_text[:30], 30)
    segment_id_from_plan = segment_plan.get("segment_id", f"auto_id_{segment_index + 1}")
    segment_identifier = f"{segment_id_from_plan}_{clean_prompt_part}" # Use segment_id for uniqueness
    
    final_segment_video_path = f"{base_filename}_{segment_identifier}_final.mp4"
    temp_raw_clips_dir = f"temp_{base_filename}_{segment_identifier}_raw_clips"
    
    try:
        parent_dir_for_final_segment = os.path.dirname(final_segment_video_path)
        if parent_dir_for_final_segment: os.makedirs(parent_dir_for_final_segment, exist_ok=True)
        os.makedirs(temp_raw_clips_dir, exist_ok=True)
    except OSError as e:
        logger.error(f"Failed to create dirs for video segment {segment_id_from_plan}: {e}"); return None
    
    raw_clip_files_generated_this_segment: Set[str] = set()
    
    if segment_duration <= 0.05:
        logger.warning(f"Video segment {segment_id_from_plan}: Duration ({segment_duration:.3f}s) too short. Skipping.")
        if os.path.exists(temp_raw_clips_dir) and not os.listdir(temp_raw_clips_dir):
            try: os.rmdir(temp_raw_clips_dir)
            except OSError: pass
        return None

    logger.info(f"--- Generating Video Segment '{segment_id_from_plan}' (Target Duration: {segment_duration:.3f}s) ---")
    logger.info(f"  Visual Prompt: '{prompt_text}'")
    
    api_clip_expected_duration = config.API_EXPECTED_CLIP_DURATION
    if api_clip_expected_duration <= 0.1: api_clip_expected_duration = 4.0
    
    num_api_clips_needed_float = max(0, segment_duration - 0.001) / api_clip_expected_duration
    num_api_clips_to_request = max(1, math.ceil(num_api_clips_needed_float))
    logger.info(f"  Will request {num_api_clips_to_request} clip(s) from API (API expected duration ~{api_clip_expected_duration:.1f}s each).")

    def process_single_api_clip(task_args: Tuple[int, str]) -> Optional[str]:
        clip_idx, raw_clip_path_for_api = task_args
        thread_logger = logging.getLogger(f"{logger.name}.vidgen_thread")
        thread_logger.info(f"Segment '{segment_id_from_plan}', API Clip {clip_idx + 1}/{num_api_clips_to_request}: Starting generation.")
        
        api_call_successful = call_aivideo_api(prompt_text, raw_clip_path_for_api, config, thread_logger)
        
        if api_call_successful and os.path.exists(raw_clip_path_for_api) and os.path.getsize(raw_clip_path_for_api) >= 1024:
            thread_logger.info(f"Segment '{segment_id_from_plan}', API Clip {clip_idx + 1}: Successfully downloaded to {raw_clip_path_for_api}.")
            return raw_clip_path_for_api
        else:
            thread_logger.error(f"Segment '{segment_id_from_plan}', API Clip {clip_idx + 1}: Failed or produced invalid file. Path: {raw_clip_path_for_api}")
            if os.path.exists(raw_clip_path_for_api):
                try: os.remove(raw_clip_path_for_api)
                except OSError as e_rem: thread_logger.warning(f"Could not remove failed raw clip '{raw_clip_path_for_api}': {e_rem}")
            return None

    tasks_for_api_calls: List[Tuple[int, str]] = []
    for i in range(num_api_clips_to_request):
        raw_clip_output_path = os.path.join(temp_raw_clips_dir, f"raw_clip_{i + 1}.mp4")
        raw_clip_files_generated_this_segment.add(raw_clip_output_path) 
        tasks_for_api_calls.append((i, raw_clip_output_path))

    validated_api_clip_paths: List[str] = []
    if num_api_clips_to_request > 0:
        num_total_keys = Config.get_total_keys()
        max_workers_by_keys = max(1, num_total_keys if Config.keys_configured(check_validity=False) else 1)
        max_workers_by_cpu = (os.cpu_count() or 1) * 2 
        
        actual_max_workers = min(num_api_clips_to_request, max_workers_by_keys, max_workers_by_cpu)
        actual_max_workers = max(1, actual_max_workers) 

        logger.info(f"  Segment '{segment_id_from_plan}': Initializing ThreadPoolExecutor with max_workers={actual_max_workers} for {num_api_clips_to_request} API calls.")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=actual_max_workers) as executor:
            future_to_task_map = {executor.submit(process_single_api_clip, task_args): task_args for task_args in tasks_for_api_calls}
            for future in concurrent.futures.as_completed(future_to_task_map):
                task_idx_completed, _ = future_to_task_map[future]
                try:
                    result_path = future.result()
                    if result_path:
                        temp_clip_check: Optional[mp.VideoFileClip] = None
                        try:
                            temp_clip_check = mp.VideoFileClip(result_path)
                            clip_actual_duration = temp_clip_check.duration or 0.0
                            if clip_actual_duration < 0.1:
                                logger.warning(f"Segment '{segment_id_from_plan}', API Clip {os.path.basename(result_path)} (task {task_idx_completed+1}) too short ({clip_actual_duration:.2f}s). Discarding.")
                                if os.path.exists(result_path): os.remove(result_path)
                            else:
                                logger.info(f"Segment '{segment_id_from_plan}', API Clip {os.path.basename(result_path)} (task {task_idx_completed+1}) validated. Duration: {clip_actual_duration:.2f}s")
                                validated_api_clip_paths.append(result_path)
                        except Exception as clip_validation_error:
                            logger.warning(f"Segment '{segment_id_from_plan}', API Clip {os.path.basename(result_path)} (task {task_idx_completed+1}) failed MoviePy validation: {clip_validation_error}. Discarding.")
                            if os.path.exists(result_path): os.remove(result_path)
                        finally:
                            if temp_clip_check:
                                try: temp_clip_check.close()
                                except Exception: pass
                except Exception as exc:
                    logger.error(f"Segment '{segment_id_from_plan}', Task for API clip {task_idx_completed+1} raised an exception: {exc}", exc_info=True)
    
    generated_raw_clip_paths = sorted(validated_api_clip_paths) 
    successfully_generated_clips_count = len(generated_raw_clip_paths)

    moviepy_clip_objects_to_close: List[Any] = []
    final_segment_video_object: Optional[mp.VideoClip] = None
    try:
        if successfully_generated_clips_count == 0:
            logger.error(f"Video Segment '{segment_id_from_plan}': No valid raw clips from API (or all failed validation).")
            try:
                logger.warning(f"Creating fallback video for segment '{segment_id_from_plan}' (BG: {config.VIDEO_BACKGROUND_COLOR_RGB}).")
                fallback_clip = mp.ColorClip(size=(config.VIDEO_WIDTH, config.VIDEO_HEIGHT), color=config.VIDEO_BACKGROUND_COLOR_RGB, duration=segment_duration).set_fps(config.VIDEO_FPS)
                moviepy_clip_objects_to_close.append(fallback_clip)
                fallback_clip.write_videofile(final_segment_video_path, fps=config.VIDEO_FPS, codec='libx264', logger=None, audio=False, preset=config.VIDEO_WRITE_PRESET_SEGMENT)
                if not os.path.exists(final_segment_video_path) or os.path.getsize(final_segment_video_path) < 100:
                    logger.error(f"Fallback video write failed for segment '{segment_id_from_plan}'."); return None
                logger.info(f"Fallback video created: {final_segment_video_path}"); return final_segment_video_path
            except Exception as fallback_err:
                logger.error(f"FATAL: Fallback video creation failed for segment '{segment_id_from_plan}': {fallback_err}", exc_info=True); return None

        logger.info(f"  Processing {successfully_generated_clips_count} raw API clips for segment '{segment_id_from_plan}'...")
        processed_clips_for_concatenation: List[mp.VideoClip] = []
        total_duration_of_raw_processed_clips = 0.0
        target_width = config.VIDEO_WIDTH; target_height = config.VIDEO_HEIGHT
        
        for raw_clip_path in generated_raw_clip_paths:
            loaded_raw_clip: Optional[mp.VideoFileClip] = None; current_processed_clip: Optional[mp.VideoClip] = None
            try:
                loaded_raw_clip = mp.VideoFileClip(raw_clip_path); moviepy_clip_objects_to_close.append(loaded_raw_clip); current_processed_clip = loaded_raw_clip
                logger.info(f"    Processing raw clip: {os.path.basename(raw_clip_path)} (Original: {current_processed_clip.w}x{current_processed_clip.h} @ {current_processed_clip.duration:.2f}s)")
                
                resized_clip_w = current_processed_clip.resize(width=target_width)
                if resized_clip_w is not current_processed_clip: moviepy_clip_objects_to_close.append(resized_clip_w)
                current_processed_clip = resized_clip_w
                logger.debug(f"      Resized to width {target_width}: {current_processed_clip.w}x{current_processed_clip.h}")
                
                current_clip_height_after_w_resize = current_processed_clip.h
                if current_clip_height_after_w_resize > target_height:
                    y_center = current_clip_height_after_w_resize / 2
                    cropped_clip_h = current_processed_clip.crop(y_center=y_center, height=target_height)
                    if cropped_clip_h is not current_processed_clip: moviepy_clip_objects_to_close.append(cropped_clip_h)
                    current_processed_clip = cropped_clip_h
                    logger.debug(f"      Cropped to height {target_height}: {current_processed_clip.w}x{current_processed_clip.h}")
                elif current_clip_height_after_w_resize < target_height:
                    padding_top = int((target_height - current_clip_height_after_w_resize) / 2)
                    padding_bottom = target_height - current_clip_height_after_w_resize - padding_top
                    padded_clip_h = current_processed_clip.margin(top=padding_top, bottom=padding_bottom, color=config.VIDEO_BACKGROUND_COLOR_RGB)
                    if padded_clip_h is not current_processed_clip: moviepy_clip_objects_to_close.append(padded_clip_h)
                    current_processed_clip = padded_clip_h
                    logger.debug(f"      Padded to height {target_height}: {current_processed_clip.w}x{current_processed_clip.h}")
                
                if current_processed_clip.w != target_width or current_processed_clip.h != target_height:
                     logger.debug(f"      Performing final corrective resize to {target_width}x{target_height}")
                     force_resized_clip = current_processed_clip.resize(width=target_width, height=target_height)
                     if force_resized_clip is not current_processed_clip: moviepy_clip_objects_to_close.append(force_resized_clip)
                     current_processed_clip = force_resized_clip
                
                final_processed_sub_clip = current_processed_clip.set_fps(config.VIDEO_FPS)
                if final_processed_sub_clip is not current_processed_clip: moviepy_clip_objects_to_close.append(final_processed_sub_clip)
                
                processed_clips_for_concatenation.append(final_processed_sub_clip)
                total_duration_of_raw_processed_clips += (final_processed_sub_clip.duration or 0.0)
                logger.debug(f"    Finished processing clip: {final_processed_sub_clip.w}x{final_processed_sub_clip.h} @ {final_processed_sub_clip.duration:.2f}s")
            except Exception as proc_err:
                logger.error(f"Error processing raw clip '{os.path.basename(raw_clip_path)}': {proc_err}. Skipping.", exc_info=True)

        if not processed_clips_for_concatenation:
            logger.error(f"FATAL: No clips survived processing for segment '{segment_id_from_plan}'. Using fallback.");
            try:
                fallback_clip = mp.ColorClip(size=(target_width, target_height), color=config.VIDEO_BACKGROUND_COLOR_RGB, duration=segment_duration).set_fps(config.VIDEO_FPS); moviepy_clip_objects_to_close.append(fallback_clip)
                fallback_clip.write_videofile(final_segment_video_path, fps=config.VIDEO_FPS, codec='libx264', logger=None, audio=False, preset=config.VIDEO_WRITE_PRESET_SEGMENT)
                return final_segment_video_path if os.path.exists(final_segment_video_path) and os.path.getsize(final_segment_video_path) > 100 else None
            except Exception as fb_err_2: logger.error(f"Fallback creation failed during clip survival check: {fb_err_2}"); return None

        concatenated_clip_object: Optional[mp.VideoClip] = None; concatenated_duration = 0.0
        if len(processed_clips_for_concatenation) == 1:
            concatenated_clip_object = processed_clips_for_concatenation[0]
            concatenated_duration = concatenated_clip_object.duration or 0.0
            logger.info(f"Using single processed clip for segment '{segment_id_from_plan}' (Duration: {concatenated_duration:.3f}s).")
        else:
            logger.info(f"Concatenating {len(processed_clips_for_concatenation)} clips for segment '{segment_id_from_plan}' (Total raw: {total_duration_of_raw_processed_clips:.3f}s).")
            video_only_clips = [clip.without_audio() for clip in processed_clips_for_concatenation]
            transition_padding = -config.TRANSITION_DURATION if config.TRANSITION_DURATION > 0.01 else 0
            concatenated_clip_object = mp.concatenate_videoclips(video_only_clips, padding=transition_padding, method="compose")
            concatenated_duration = concatenated_clip_object.duration if concatenated_clip_object else 0.0
            logger.info(f"Concatenated clip duration for segment '{segment_id_from_plan}': {concatenated_duration:.3f}s")
            if concatenated_clip_object and concatenated_clip_object not in moviepy_clip_objects_to_close:
                moviepy_clip_objects_to_close.append(concatenated_clip_object)

        if concatenated_clip_object is None or concatenated_duration <= 0:
            logger.error(f"FATAL: Failed to combine clips for segment '{segment_id_from_plan}'."); return None
        
        adjusted_duration_clip: Optional[mp.VideoClip] = None
        if concatenated_duration >= segment_duration:
            logger.debug(f"Trimming combined clip from {concatenated_duration:.3f}s to target {segment_duration:.3f}s.")
            adjusted_duration_clip = concatenated_clip_object.subclip(0, segment_duration)
        else:
            logger.warning(f"Combined clip duration {concatenated_duration:.3f}s too short for target {segment_duration:.3f}s. Looping.")
            adjusted_duration_clip = concatenated_clip_object.loop(duration=segment_duration)
        
        if adjusted_duration_clip is not concatenated_clip_object and adjusted_duration_clip not in moviepy_clip_objects_to_close:
            moviepy_clip_objects_to_close.append(adjusted_duration_clip)

        final_segment_video_object = adjusted_duration_clip
        if final_segment_video_object is None:
            logger.error(f"FATAL: Failed to adjust duration for segment '{segment_id_from_plan}'."); return None
        
        final_segment_video_object = final_segment_video_object.set_duration(segment_duration).set_fps(config.VIDEO_FPS)
        logger.info(f"Writing final video for segment '{segment_id_from_plan}' to: {final_segment_video_path}")
        final_segment_video_object.write_videofile(
            final_segment_video_path, fps=config.VIDEO_FPS, codec='libx264', audio=False, logger=None,
            bitrate=config.VIDEO_BITRATE, threads=os.cpu_count() or 4, preset=config.VIDEO_WRITE_PRESET_SEGMENT
        )

        if not os.path.exists(final_segment_video_path) or os.path.getsize(final_segment_video_path) < 1024:
            logger.error(f"ERROR: Final video for segment '{segment_id_from_plan}' write failed or file too small."); return None
        
        logger.info(f"--- Finished Video Segment '{segment_id_from_plan}' -> {os.path.basename(final_segment_video_path)} ---"); return final_segment_video_path
    
    except Exception as e:
        logger.error(f"Unexpected error during video segment '{segment_id_from_plan}' processing: {e}", exc_info=True); return None
    finally:
        logger.debug(f"Closing {len(moviepy_clip_objects_to_close)} MoviePy objects for segment '{segment_id_from_plan}'.")
        closed_count = 0
        for clip_obj_to_close in reversed(moviepy_clip_objects_to_close): 
             if clip_obj_to_close and hasattr(clip_obj_to_close, 'close') and callable(clip_obj_to_close.close):
                 is_fileclip = isinstance(clip_obj_to_close, (mp.VideoFileClip, mp.AudioFileClip))
                 should_close = True
                 if is_fileclip: 
                     has_reader = hasattr(clip_obj_to_close, 'reader')
                     is_reader_open = has_reader and clip_obj_to_close.reader is not None
                     should_close = is_reader_open
                 if should_close:
                    try: clip_obj_to_close.close(); closed_count += 1
                    except Exception as close_ex: logger.debug(f"Error closing MoviePy clip object for segment '{segment_id_from_plan}': {close_ex}")
        logger.debug(f"Closed {closed_count} MoviePy clip objects for segment '{segment_id_from_plan}'.")
        
        logger.debug(f"Cleaning up temporary raw clip files for segment '{segment_id_from_plan}' from: {temp_raw_clips_dir}")
        cleaned_raw_files_count = 0
        for raw_file_path in raw_clip_files_generated_this_segment: 
            if raw_file_path and os.path.exists(raw_file_path): 
                try: os.remove(raw_file_path); cleaned_raw_files_count += 1
                except OSError as e_remove: logger.warning(f"Could not remove temp raw clip '{raw_file_path}': {e_remove}")
        if os.path.exists(temp_raw_clips_dir):
            try:
                if not os.listdir(temp_raw_clips_dir): 
                    os.rmdir(temp_raw_clips_dir)
                    logger.debug(f"Removed empty temp raw clips directory: {temp_raw_clips_dir}")
                else:
                    logger.warning(f"Temp raw clips directory '{temp_raw_clips_dir}' not empty after processing segment '{segment_id_from_plan}', not removing. (Contains: {os.listdir(temp_raw_clips_dir)[:5]})")
            except OSError as e_rmdir:
                logger.warning(f"Could not remove/check temp raw clips directory '{temp_raw_clips_dir}': {e_rmdir}")
        logger.debug(f"Attempted removal of {cleaned_raw_files_count} tracked temporary raw clip files for segment '{segment_id_from_plan}'.")

def create_final_video(
    segment_video_paths: List[Optional[str]], 
    voiceover_audio_path: str, 
    output_video_path: str, 
    timed_script_segments: List[Dict[str, Any]], # This has correct timings & original Gemini plan items
    total_audio_duration: float,
    generated_manim_overlays: Dict[str, str], # Key: segment_id, Value: path to Manim MOV/WEBM
    original_script_plan_segments: List[Dict[str, Any]] # Original Gemini plan, used for segment_id
) -> bool:
    logger.info("Assembling final video from segments, voiceover, Manim overlays (if any), and BGM...")
    video_clips_for_composition: List[mp.VideoClip] = []
    moviepy_objects_to_close_in_final: List[Any] = []
    final_video_target_duration = total_audio_duration
    if final_video_target_duration <= 0.01: 
        logger.error("Cannot compose final video: Target duration zero."); return False
    logger.info(f"Target duration for final composed video: {final_video_target_duration:.3f}s")
    
    target_comp_width = config.VIDEO_WIDTH
    target_comp_height = config.VIDEO_HEIGHT

    try:
        # Assemble video track from segments and Manim overlays
        for i, segment_timing_info in enumerate(timed_script_segments):
            segment_video_file_path = segment_video_paths[i] if i < len(segment_video_paths) else None
            segment_start_time = segment_timing_info.get('start_time')
            segment_duration = segment_timing_info.get('duration')
            segment_id = original_script_plan_segments[i].get("segment_id", f"auto_id_{i+1}") # Get segment_id from original plan

            if segment_start_time is None or segment_duration is None or segment_duration < 0.01:
                logger.warning(f"Segment '{segment_id}': Invalid timing. Skipping."); continue

            base_segment_clip_obj: Optional[mp.VideoClip] = None
            if segment_video_file_path and os.path.exists(segment_video_file_path) and os.path.getsize(segment_video_file_path) >= 1024:
                logger.info(f"  Loading video for segment '{segment_id}': {os.path.basename(segment_video_file_path)}")
                try:
                    loaded_segment_clip = mp.VideoFileClip(segment_video_file_path,চু চু চু চু চু).resize((target_comp_width, target_comp_height)) # Ensure size
                    moviepy_objects_to_close_in_final.append(loaded_segment_clip)
                    base_segment_clip_obj = loaded_segment_clip.set_duration(segment_duration)
                    if base_segment_clip_obj is not loaded_segment_clip and base_segment_clip_obj not in moviepy_objects_to_close_in_final:
                        moviepy_objects_to_close_in_final.append(base_segment_clip_obj)
                except Exception as e:
                    logger.error(f"Error loading video for segment '{segment_id}': {e}. Using fallback.", exc_info=True)
            
            if base_segment_clip_obj is None: # Fallback if loading failed or no path
                try:
                    logger.debug(f"Creating fallback ColorClip for segment '{segment_id}'.")
                    fallback_seg_clip = mp.ColorClip(size=(target_comp_width, target_comp_height), color=config.VIDEO_BACKGROUND_COLOR_RGB, duration=segment_duration).set_fps(config.VIDEO_FPS)
                    base_segment_clip_obj = fallback_seg_clip
                    moviepy_objects_to_close_in_final.append(fallback_seg_clip)
                except Exception as fallback_clip_err:
                    logger.error(f"Failed to create fallback ColorClip for segment '{segment_id}': {fallback_clip_err}. Skipping.", exc_info=True)
                    continue
            
            # Apply Manim overlay if available for this segment_id
            manim_overlay_path = generated_manim_overlays.get(segment_id)
            final_clip_for_this_segment_timeline = base_segment_clip_obj

            if manim_overlay_path and os.path.exists(manim_overlay_path):
                logger.info(f"  Found Manim overlay for segment '{segment_id}': {manim_overlay_path}")
                try:
                    manim_clip = mp.VideoFileClip(manim_overlay_path, has_mask=True, transparent=True) # Assume alpha
                    moviepy_objects_to_close_in_final.append(manim_clip)
                    
                    # Ensure manim_clip duration matches base_segment_clip_obj or handle as needed
                    # For simplicity, let's set its duration to the segment's duration.
                    # More complex logic could try to match `duration_hint_seconds` from Manim concept.
                    manim_clip = manim_clip.set_duration(segment_duration).set_position('center')

                    # Composite Manim overlay on top of the base segment video
                    final_clip_for_this_segment_timeline = mp.CompositeVideoClip(
                        [base_segment_clip_obj, manim_clip], 
                        size=(target_comp_width, target_comp_height)
                    ).set_duration(segment_duration)
                    if final_clip_for_this_segment_timeline not in moviepy_objects_to_close_in_final:
                         moviepy_objects_to_close_in_final.append(final_clip_for_this_segment_timeline)
                    logger.info(f"    Successfully composited Manim overlay for segment '{segment_id}'.")
                except Exception as manim_comp_err:
                    logger.error(f"    Error loading or compositing Manim overlay for segment '{segment_id}': {manim_comp_err}. Using base video only.", exc_info=True)
            
            # Apply segment-level fade effects to the (potentially Manim-overlaid) clip
            processed_clip_obj = final_clip_for_this_segment_timeline
            fade_time = config.SEGMENT_FADE_DURATION
            if fade_time > 0.01:
                if segment_duration > fade_time * 2.01: processed_clip_obj = fadeout(fadein(processed_clip_obj, fade_time), fade_time)
                elif segment_duration > fade_time * 1.01: processed_clip_obj = fadein(processed_clip_obj, fade_time)
            
            if processed_clip_obj is not final_clip_for_this_segment_timeline and processed_clip_obj not in moviepy_objects_to_close_in_final:
                moviepy_objects_to_close_in_final.append(processed_clip_obj)
            
            final_segment_clip_for_comp = processed_clip_obj.set_start(segment_start_time).set_position('center').set_fps(config.VIDEO_FPS)
            if final_segment_clip_for_comp is not processed_clip_obj and final_segment_clip_for_comp not in moviepy_objects_to_close_in_final:
                 moviepy_objects_to_close_in_final.append(final_segment_clip_for_comp)
            video_clips_for_composition.append(final_segment_clip_for_comp)

        if not video_clips_for_composition: logger.error("No video clips for final composition."); return False
        
        final_video_composition = mp.CompositeVideoClip(video_clips_for_composition, size=(target_comp_width, target_comp_height), bg_color=config.VIDEO_BACKGROUND_COLOR_RGB)
        final_video_composition = final_video_composition.set_duration(final_video_target_duration).set_fps(config.VIDEO_FPS)
        moviepy_objects_to_close_in_final.append(final_video_composition)

        final_audio_for_video: Optional[mp.AudioClip] = None
        voiceover_clip_processed: Optional[mp.AudioClip] = None
        
        if os.path.exists(voiceover_audio_path) and os.path.getsize(voiceover_audio_path) >= 100:
            try:
                logger.info(f"Loading voiceover audio from: {os.path.basename(voiceover_audio_path)}")
                main_audio_clip_temp = mp.AudioFileClip(voiceover_audio_path)
                moviepy_objects_to_close_in_final.append(main_audio_clip_temp)
                
                if abs(main_audio_clip_temp.duration - final_video_target_duration) > 0.05:
                    logger.warning(f"Voiceover duration ({main_audio_clip_temp.duration:.2f}s) differs from video ({final_video_target_duration:.2f}s). Adjusting voiceover.")
                    voiceover_clip_processed = main_audio_clip_temp.set_duration(final_video_target_duration)
                    if voiceover_clip_processed is not main_audio_clip_temp and voiceover_clip_processed not in moviepy_objects_to_close_in_final:
                        moviepy_objects_to_close_in_final.append(voiceover_clip_processed)
                else:
                    voiceover_clip_processed = main_audio_clip_temp
                logger.info(f"Voiceover audio loaded (Duration: {voiceover_clip_processed.duration:.2f}s).")
            except Exception as vo_err:
                logger.error(f"Error loading or processing voiceover audio: {vo_err}", exc_info=True)
        else:
            logger.warning(f"Voiceover audio file '{voiceover_audio_path}' missing or empty.")

        bgm_clip_processed: Optional[mp.AudioClip] = None
        if config.BGM_FILE_PATH and os.path.exists(config.BGM_FILE_PATH):
            try:
                logger.info(f"Loading BGM from: {os.path.basename(config.BGM_FILE_PATH)}")
                bgm_clip_temp = mp.AudioFileClip(config.BGM_FILE_PATH)
                moviepy_objects_to_close_in_final.append(bgm_clip_temp)

                bgm_with_volume = bgm_clip_temp.fx(mp.afx.volumex, config.BGM_VOLUME) # Use mp.afx.volumex
                if bgm_with_volume is not bgm_clip_temp and bgm_with_volume not in moviepy_objects_to_close_in_final:
                     moviepy_objects_to_close_in_final.append(bgm_with_volume)
                
                current_bgm_duration = bgm_with_volume.duration
                if current_bgm_duration < final_video_target_duration:
                    logger.info(f"BGM duration ({current_bgm_duration:.2f}s) shorter than video ({final_video_target_duration:.2f}s). Looping BGM.")
                    num_loops = math.ceil(final_video_target_duration / current_bgm_duration)
                    looped_clips = [bgm_with_volume] * num_loops
                    bgm_looped_temp = mp.concatenate_audioclips(looped_clips)
                    if bgm_looped_temp not in moviepy_objects_to_close_in_final: moviepy_objects_to_close_in_final.append(bgm_looped_temp)
                    bgm_clip_processed = bgm_looped_temp.subclip(0, final_video_target_duration)
                else: 
                    logger.info(f"BGM duration ({current_bgm_duration:.2f}s) >= video ({final_video_target_duration:.2f}s). Trimming BGM.")
                    bgm_clip_processed = bgm_with_volume.subclip(0, final_video_target_duration)
                
                if bgm_clip_processed is not bgm_with_volume and bgm_clip_processed not in moviepy_objects_to_close_in_final:
                     moviepy_objects_to_close_in_final.append(bgm_clip_processed)

                if bgm_clip_processed:
                    bgm_final_duration = bgm_clip_processed.duration
                    if config.BGM_FADEIN_DURATION > 0 and bgm_final_duration > config.BGM_FADEIN_DURATION:
                        bgm_clip_processed = bgm_clip_processed.audio_fadein(config.BGM_FADEIN_DURATION)
                    if config.BGM_FADEOUT_DURATION > 0 and bgm_final_duration > config.BGM_FADEOUT_DURATION:
                         bgm_clip_processed = bgm_clip_processed.audio_fadeout(config.BGM_FADEOUT_DURATION)
                
                logger.info(f"BGM processed (Volume: {config.BGM_VOLUME*100:.0f}%, Duration: {bgm_clip_processed.duration:.2f}s).")
            except Exception as bgm_err:
                logger.error(f"Error loading or processing BGM: {bgm_err}", exc_info=True)
                bgm_clip_processed = None
        elif config.BGM_FILE_PATH: 
            logger.warning(f"BGM file specified ('{config.BGM_FILE_PATH}') but not found.")

        if voiceover_clip_processed and bgm_clip_processed:
            logger.info("Compositing voiceover and BGM.")
            final_audio_for_video = mp.CompositeAudioClip([voiceover_clip_processed, bgm_clip_processed])
            if final_audio_for_video not in moviepy_objects_to_close_in_final:
                 moviepy_objects_to_close_in_final.append(final_audio_for_video)
        elif voiceover_clip_processed:
            logger.info("Using voiceover audio only.")
            final_audio_for_video = voiceover_clip_processed
        elif bgm_clip_processed:
            logger.info("Using BGM audio only.")
            final_audio_for_video = bgm_clip_processed
        else:
            logger.warning("No voiceover or BGM available. Video will have NO SOUND.")
            final_audio_for_video = None
            
        video_with_audio = final_video_composition.set_audio(final_audio_for_video)
        if video_with_audio is not final_video_composition and video_with_audio not in moviepy_objects_to_close_in_final:
            moviepy_objects_to_close_in_final.append(video_with_audio)

        if video_with_audio is None: logger.error("Critical error: Final video object for writing not created."); return False
        
        logger.info(f"Writing composed video (pre-subtitles) to: {output_video_path}...")
        video_with_audio.write_videofile(
            output_video_path, fps=config.VIDEO_FPS, codec='libx264', 
            audio_codec='aac', bitrate=config.VIDEO_BITRATE, audio_bitrate='192k', 
            threads=os.cpu_count() or 4, preset='medium', logger=None
        )

        if not os.path.exists(output_video_path) or os.path.getsize(output_video_path) < 1024: 
            logger.error(f"ERROR: Failed to write composed video or file too small."); return False
        logger.info(f"Composed video (pre-subtitles) successfully created: {os.path.basename(output_video_path)}"); return True
    
    except Exception as e: 
        logger.error(f"UNEXPECTED ERROR during final video composition: {e}", exc_info=True); return False
    finally: 
        logger.debug(f"Closing {len(moviepy_objects_to_close_in_final)} MoviePy objects from final composition.")
        closed_count = 0
        for clip_obj_to_close in reversed(moviepy_objects_to_close_in_final):
            if clip_obj_to_close and hasattr(clip_obj_to_close, 'close') and callable(clip_obj_to_close.close):
                try: clip_obj_to_close.close(); closed_count += 1
                except Exception as close_ex: logger.debug(f"Error closing MoviePy object in final composition: {close_ex}")
        logger.debug(f"Closed {closed_count} MoviePy objects from final composition.")

def transcribe(audio_path: str) -> Optional[List[Dict[str, Any]]]:
    if not _whisper_available: logger.error("Whisper-timestamped not available."); return None
    logger.info(f"Transcribing audio: {os.path.basename(audio_path)} (Model: '{config.WHISPER_MODEL_SIZE}')")
    if not os.path.exists(audio_path): logger.error(f"Audio for transcription not found: {audio_path}"); return None
    if os.path.getsize(audio_path) < 100: logger.error(f"Audio for transcription too small: {audio_path}"); return None
    whisper_model = None; compute_device = config.WHISPER_DEVICE
    if compute_device is None:
        try:
            import torch
            if torch.cuda.is_available(): compute_device = "cuda"; logger.info("Whisper using 'cuda'.")
            else: compute_device = "cpu"; logger.info("Whisper using 'cpu'.")
        except ImportError: compute_device = "cpu"; logger.info("PyTorch not found. Whisper using 'cpu'.")
        except Exception as e_torch: compute_device = "cpu"; logger.warning(f"Error during PyTorch/CUDA check: {e_torch}. Defaulting to 'cpu'.")
    else: logger.info(f"Using specified Whisper device: '{compute_device}'.")
    try:
        logger.info(f"Loading Whisper model '{config.WHISPER_MODEL_SIZE}' onto '{compute_device}'..."); load_start_time = time.time()
        whisper_model = whisper_timestamped.load_model(config.WHISPER_MODEL_SIZE, device=compute_device)
        logger.info(f"Whisper model loaded in {time.time() - load_start_time:.2f}s.")
        logger.info("Starting audio transcription..."); transcribe_start_time = time.time()
        transcription_result = whisper_timestamped.transcribe(model=whisper_model, audio=audio_path, language="en", beam_size=5, best_of=5, temperature=(0.0, 0.2, 0.4, 0.6, 0.8), detect_disfluencies=False, vad=True, fp16=(compute_device == 'cuda'))
        logger.info(f"Transcription API call completed in {time.time() - transcribe_start_time:.2f}s.")
        if not transcription_result or not isinstance(transcription_result.get("segments"), list): logger.warning("Transcription result invalid/no segments."); return []
        processed_segments: List[Dict[str, Any]] = []; total_word_count = 0
        logger.info(f"Received {len(transcription_result['segments'])} raw segments. Processing words...")
        for i, raw_segment in enumerate(transcription_result["segments"]):
            if not isinstance(raw_segment, dict) or not isinstance(raw_segment.get("words"), list) or not raw_segment["words"]: logger.debug(f"Skipping raw segment {i} invalid/no words."); continue
            word_data_for_segment: List[Dict[str, Any]] = []
            for word_info in raw_segment["words"]:
                if not isinstance(word_info, dict): continue
                text = word_info.get("text"); start_time = word_info.get("start"); end_time = word_info.get("end")
                if not isinstance(text, str) or start_time is None or end_time is None or not isinstance(start_time, (float, int)) or not isinstance(end_time, (float, int)): logger.debug(f"Skipping invalid word data: {word_info}"); continue
                word_cleaned_key = text.strip(".,!?;:\"'()[]{} ").upper() 
                word_original_display = text.strip().upper() 
                if not word_cleaned_key: continue 
                start_time_float = float(start_time); end_time_float = float(end_time); min_word_duration = 0.02
                if end_time_float <= start_time_float: end_time_float = start_time_float + min_word_duration
                elif (end_time_float - start_time_float) < min_word_duration: end_time_float = start_time_float + min_word_duration
                word_entry = {
                    "word_key": word_cleaned_key, 
                    "original_text_unmodified": text, 
                    "original_text": word_original_display, 
                    "start": start_time_float, "end": end_time_float
                }
                confidence = word_info.get("confidence")
                if confidence is not None:
                    try: word_entry["confidence"] = float(confidence)
                    except ValueError: pass
                word_data_for_segment.append(word_entry); total_word_count += 1
            if word_data_for_segment:
                segment_start = float(raw_segment.get("start", word_data_for_segment[0]["start"])); segment_end = float(raw_segment.get("end", word_data_for_segment[-1]["end"]))
                if segment_end < segment_start: segment_end = word_data_for_segment[-1]["end"]
                processed_segments.append({"text": raw_segment.get("text", "").strip(), "start": segment_start, "end": segment_end, "words": word_data_for_segment})
        logger.info(f"Transcription complete. Produced {len(processed_segments)} segments, {total_word_count} words."); return processed_segments
    except Exception as e: logger.error(f"UNEXPECTED ERROR during transcription: {e}", exc_info=True); return None
    finally:
        if whisper_model:
            logger.info("Unloading Whisper model..."); del whisper_model; import gc; gc.collect()
            if compute_device == "cuda":
                try: import torch; torch.cuda.empty_cache(); logger.info("PyTorch CUDA cache cleared.")
                except ImportError: pass
                except Exception as cuda_ex: logger.warning(f"Error clearing CUDA cache: {cuda_ex}")
            logger.info("Whisper model unloaded.")

def format_text_into_lines_by_char_and_line_limit(word_pieces: List[str],char_limit_per_line: int,max_lines: int) -> str:
    if not word_pieces: return ""
    def get_visible_length(piece: str) -> int: return len(re.sub(r"\{[^}]*\}", "", piece))
    lines_output: List[str] = []
    current_line_word_pieces: List[str] = []
    current_line_visible_char_count = 0
    for piece_idx, piece in enumerate(word_pieces):
        piece_visible_len = get_visible_length(piece)
        space_len = 1 if current_line_word_pieces and get_visible_length(current_line_word_pieces[-1]) > 0 else 0
        if current_line_word_pieces and (current_line_visible_char_count + space_len + piece_visible_len > char_limit_per_line):
            if len(lines_output) < max_lines - 1: 
                lines_output.append(" ".join(current_line_word_pieces))
                current_line_word_pieces = [piece] 
                current_line_visible_char_count = piece_visible_len
            elif len(lines_output) == max_lines - 1: 
                if current_line_word_pieces: current_line_visible_char_count += space_len
                current_line_word_pieces.append(piece)
                current_line_visible_char_count += piece_visible_len
            else: 
                logger.warning(f"Max lines ({max_lines}) reached. Could not place piece '{piece[:20]}...' due to char limit on new line. Skipping.")
                continue 
        else: 
            if current_line_word_pieces: current_line_visible_char_count += space_len 
            current_line_word_pieces.append(piece)
            current_line_visible_char_count += piece_visible_len
    if current_line_word_pieces:
        if len(lines_output) < max_lines:
            lines_output.append(" ".join(current_line_word_pieces))
        elif len(lines_output) == max_lines:
             current_final_line_str = " ".join(current_line_word_pieces)
             if not lines_output[-1].endswith(current_final_line_str): 
                 logger.debug(f"Appending remaining words to the last line, potentially exceeding its char limit: '{current_final_line_str}'")
                 lines_output[-1] = current_final_line_str 
    return "\\N".join(lines_output)

def split_transcript_into_display_segments(transcript_segments: List[Dict[str, Any]], timed_script_segments: List[Dict[str, Any]] ) -> List[Dict[str, Any]]:
    # timed_script_segments here is the list that includes original Gemini plan details PLUS 'start_time', 'duration', 'end_time' for voiceover.
    # It's used to map Whisper's transcribed words back to the original script segment for `highlight_words`.
    display_segments = []
    if not transcript_segments: return []
    min_words = config.TARGET_WORDS_PER_DISPLAY_SEGMENT_MIN
    max_words = config.TARGET_WORDS_PER_DISPLAY_SEGMENT_MAX
    for ts_segment_idx, ts_segment in enumerate(transcript_segments):
        words_in_ts_segment: List[Dict[str, Any]] = ts_segment.get('words', [])
        if not words_in_ts_segment: continue
        current_original_script_idx = -1
        original_highlight_phrases_for_this_ts_segment_group = set()
        first_word_start_time = words_in_ts_segment[0]['start']
        
        # Find which original script segment this Whisper segment belongs to
        if timed_script_segments: 
            for idx, original_seg_data in enumerate(timed_script_segments):
                s_start = original_seg_data.get('start_time', -1.0)
                s_end = original_seg_data.get('end_time', float('inf'))
                if s_start <= first_word_start_time < s_end:
                    current_original_script_idx = idx
                    # Get highlight_words from the corresponding original script segment plan
                    hw_list = original_seg_data.get("highlight_words", []) 
                    if hw_list and isinstance(hw_list, list):
                        original_highlight_phrases_for_this_ts_segment_group = {str(p).strip().upper() for p in hw_list if isinstance(p, str) and p.strip()}
                    break
        
        current_chunk_words: List[Dict[str, Any]] = []
        word_idx_in_ts_segment = 0
        while word_idx_in_ts_segment < len(words_in_ts_segment):
            word_info = words_in_ts_segment[word_idx_in_ts_segment]
            word_original_unmodified = word_info['original_text_unmodified'] 
            ends_with_strong_punctuation = any(word_original_unmodified.endswith(punc) for punc in ['.', '!', '?'])
            ends_with_comma = word_original_unmodified.endswith(',')
            current_chunk_words.append(word_info.copy()) 
            word_idx_in_ts_segment += 1
            should_split_now = False
            current_chunk_len = len(current_chunk_words)
            if current_chunk_len >= max_words:
                should_split_now = True
            elif ends_with_strong_punctuation and (current_chunk_len >= min_words or current_chunk_len == 1):
                should_split_now = True
            elif ends_with_comma and (current_chunk_len >= min_words or current_chunk_len == 1):
                if current_chunk_len == max_words or \
                   (word_idx_in_ts_segment < len(words_in_ts_segment) and current_chunk_len + 1 > max_words) or \
                   word_idx_in_ts_segment == len(words_in_ts_segment):
                    should_split_now = True
            if word_idx_in_ts_segment == len(words_in_ts_segment): 
                should_split_now = True
            if should_split_now and current_chunk_words:
                last_word_in_chunk_obj = current_chunk_words[-1]
                if last_word_in_chunk_obj['original_text_unmodified'].endswith(','):
                     current_display_text = last_word_in_chunk_obj['original_text']
                     if current_display_text.endswith(','): 
                        last_word_in_chunk_obj['original_text'] = current_display_text.rstrip(',')
                display_segments.append({
                    "text": " ".join(w['original_text'] for w in current_chunk_words), 
                    "start": current_chunk_words[0]['start'],
                    "end": current_chunk_words[-1]['end'],
                    "words": list(current_chunk_words), 
                    "original_script_segment_index": current_original_script_idx, # This index links to timed_script_segments
                    "original_highlight_phrases": list(original_highlight_phrases_for_this_ts_segment_group) 
                })
                current_chunk_words = []
        if current_chunk_words: 
            last_word_in_chunk_obj_final = current_chunk_words[-1]
            if last_word_in_chunk_obj_final['original_text_unmodified'].endswith(','):
                current_display_text_final = last_word_in_chunk_obj_final['original_text']
                if current_display_text_final.endswith(','):
                    last_word_in_chunk_obj_final['original_text'] = current_display_text_final.rstrip(',')
            display_segments.append({
                "text": " ".join(w['original_text'] for w in current_chunk_words),
                "start": current_chunk_words[0]['start'], "end": current_chunk_words[-1]['end'],
                "words": list(current_chunk_words),
                "original_script_segment_index": current_original_script_idx,
                "original_highlight_phrases": list(original_highlight_phrases_for_this_ts_segment_group)
            })
    if display_segments: logger.info(f"Split into {len(display_segments)} display segments based on word count/punctuation rules.")
    else: logger.warning("No display segments created after splitting.")
    return display_segments

def generate_ass(
    processed_display_segments: List[Dict[str, Any]],
    original_script_segments_from_user: List[Dict[str, Any]], 
    output_ass_path: str,
    video_title: str
) -> bool:
    logger.info("Generating ASS subtitles with robust phrase highlighting...")

    def _normalize_word_for_matching(word_text: str) -> str:
        normalized = word_text.strip().upper()
        normalized = re.sub(r"[.,!?;:]+$", "", normalized)
        return normalized

    font_name_input = config.FONT_NAME.strip()
    is_bold_specified_in_name = "bold" in font_name_input.lower()
    font_name_for_ass_style = re.sub(r'\s+bold\b', '', font_name_input, flags=re.IGNORECASE).strip()
    if not font_name_for_ass_style: font_name_for_ass_style = font_name_input
    ass_bold_flag = -1 if is_bold_specified_in_name else 0

    style_format_string = ("Style: HormoziBase,{font_name},{font_size},{primary_colour},{secondary_colour},{outline_colour},{shadow_colour},"
                           "{bold},{italic},{underline},{strikeout},{scale_x},{scale_y},{spacing},{angle},"
                           "{border_style},{outline_thickness},{shadow_distance},{alignment},"
                           "{margin_l},{margin_r},{margin_v},{encoding}")

    hormozi_style_line = style_format_string.format(
        font_name=font_name_for_ass_style, font_size=config.FONT_SIZE_BASE,
        primary_colour=config.PRIMARY_COLOR, secondary_colour=config.PRIMARY_COLOR,
        outline_colour=config.OUTLINE_COLOR, shadow_colour=config.SHADOW_COLOR,
        bold=ass_bold_flag, italic=0, underline=0, strikeout=0,
        scale_x=100, scale_y=100, spacing=config.LETTER_SPACING, angle=0,
        border_style=1, outline_thickness=config.OUTLINE_THICKNESS, shadow_distance=config.SHADOW_DISTANCE,
        alignment=config.SUBTITLE_ALIGNMENT,
        margin_l=config.SUBTITLE_MARGIN_L, margin_r=config.SUBTITLE_MARGIN_R, margin_v=config.SUBTITLE_MARGIN_V,
        encoding=1
    )
    ass_header = (
        f"[Script Info]\nTitle: {video_title}\nScriptType: v4.00+\nPlayResX: {config.VIDEO_WIDTH}\nPlayResY: {config.VIDEO_HEIGHT}\nScaledBorderAndShadow: yes\nWrapStyle: 0\n\n"
        f"[V4+ Styles]\nFormat: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,OutlineColour,BackColour,Bold,Italic,Underline,StrikeOut,ScaleX,ScaleY,Spacing,Angle,BorderStyle,Outline,Shadow,Alignment,MarginL,MarginR,MarginV,Encoding\n"
        f"{hormozi_style_line}\n\n"
        f"[Events]\nFormat: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text\n"
    )
    ass_dialogue_lines: List[str] = [ass_header]

    if not processed_display_segments:
        logger.warning("No display segments. Writing empty ASS file.");
        try:
            with open(output_ass_path, "w", encoding="utf-8") as f_ass: f_ass.writelines(ass_dialogue_lines)
            return True
        except IOError as e_io: logger.error(f"Error writing empty ASS file: {e_io}"); return False

    highlight_color_palette = config.HIGHLIGHT_COLORS_LIST
    if not highlight_color_palette:
        logger.warning("No highlight colors in HIGHLIGHT_COLORS_LIST. Using primary color for highlights.")
        highlight_color_palette = [config.PRIMARY_COLOR]
        
    current_highlight_color_idx = 0 
    
    highlight_phrase_instances: Dict[Tuple[int, str], int] = {}
    next_instance_id = 0
    
    word_to_instance_id_map: List[List[Optional[int]]] = [
        [None] * len(ds.get('words', [])) for ds in processed_display_segments
    ]

    for original_script_idx, user_script_segment_info in enumerate(original_script_segments_from_user):
        user_highlight_phrases = user_script_segment_info.get("highlight_words", [])
        if not isinstance(user_highlight_phrases, list): user_highlight_phrases = []

        sorted_user_phrases = sorted(
            [str(p).strip().upper() for p in user_highlight_phrases if isinstance(p, str) and p.strip()],
            key=len,
            reverse=True
        )

        for phrase_text_upper in sorted_user_phrases: 
            if not phrase_text_upper: continue
            instance_key = (original_script_idx, phrase_text_upper)
            if instance_key not in highlight_phrase_instances:
                highlight_phrase_instances[instance_key] = next_instance_id
                next_instance_id += 1
            current_phrase_instance_id = highlight_phrase_instances[instance_key]
            
            user_phrase_words_normalized = [_normalize_word_for_matching(p_word) for p_word in phrase_text_upper.split(' ')]
            len_user_phrase = len(user_phrase_words_normalized)
            if len_user_phrase == 0: continue
            
            for ds_idx, display_segment in enumerate(processed_display_segments):
                if display_segment.get("original_script_segment_index") != original_script_idx:
                    continue

                words_in_ds = display_segment.get('words', [])
                normalized_words_in_ds = [_normalize_word_for_matching(w['original_text']) for w in words_in_ds]
                
                for i in range(len(normalized_words_in_ds) - len_user_phrase + 1): 
                    window_from_ds_normalized = normalized_words_in_ds[i : i + len_user_phrase]
                    
                    if window_from_ds_normalized == user_phrase_words_normalized:
                        can_tag_this_window = True
                        for k_check in range(len_user_phrase):
                            if word_to_instance_id_map[ds_idx][i+k_check] is not None:
                                can_tag_this_window = False
                                break
                        
                        if can_tag_this_window:
                            for k_apply in range(len_user_phrase):
                                word_to_instance_id_map[ds_idx][i+k_apply] = current_phrase_instance_id
    
    phrase_style_cache: Dict[int, Dict[str, Any]] = {} 
    dialogue_events_count = 0

    for display_segment_idx, display_segment in enumerate(processed_display_segments):
        words_in_this_display_segment = display_segment.get('words', [])
        if not words_in_this_display_segment: continue

        instance_ids_for_words_in_this_segment = word_to_instance_id_map[display_segment_idx]

        line_start_time = display_segment['start']
        line_end_time = display_segment['end']
        min_display_duration = 0.1 + (0.05 * len(words_in_this_display_segment))
        if (line_end_time - line_start_time) < min_display_duration:
             line_end_time = line_start_time + min_display_duration
        
        line_fade_tag_str = f"\\fad({config.LINE_FADE_DURATION_MS},{config.LINE_FADE_DURATION_MS})" if config.LINE_FADE_DURATION_MS > 0 else ""
        tilt_tag_str = ""
        if config.RANDOM_TILT_CHANCE > 0 and random.random() < config.RANDOM_TILT_CHANCE:
            angle_val = random.uniform(-config.RANDOM_TILT_MAX_ANGLE_DEGREES, config.RANDOM_TILT_MAX_ANGLE_DEGREES)
            if abs(angle_val) > 0.1: tilt_tag_str = f"\\frz{angle_val:.2f}"
        line_level_tags = f"{{{line_fade_tag_str}{tilt_tag_str}}}"

        text_pieces_for_line: List[str] = []
        for word_idx, word_info in enumerate(words_in_this_display_segment):
            word_text_for_display = word_info['original_text'] 
            instance_id_for_this_word = instance_ids_for_words_in_this_segment[word_idx]
            
            word_specific_tags = ""
            if instance_id_for_this_word is not None: 
                if instance_id_for_this_word not in phrase_style_cache:
                    chosen_color = highlight_color_palette[current_highlight_color_idx % len(highlight_color_palette)]
                    current_highlight_color_idx += 1 

                    chosen_size = config.FONT_SIZE_BASE 
                    if config.HIGHLIGHT_FONT_SIZE_MODE == "factor":
                        chosen_size = int(config.FONT_SIZE_BASE * config.FONT_SIZE_HIGHLIGHT_BASE_FACTOR)
                    elif config.HIGHLIGHT_FONT_SIZE_MODE == "random_from_list" and config.FONT_SIZE_HIGHLIGHT_LIST:
                        chosen_size = random.choice(config.FONT_SIZE_HIGHLIGHT_LIST)
                    elif config.HIGHLIGHT_FONT_SIZE_MODE == "fixed" and config.FONT_SIZE_HIGHLIGHT_ABSOLUTE is not None:
                        chosen_size = config.FONT_SIZE_HIGHLIGHT_ABSOLUTE
                    
                    phrase_style_cache[instance_id_for_this_word] = {'color': chosen_color, 'size': chosen_size}

                style_to_apply = phrase_style_cache[instance_id_for_this_word]
                blur_tag = f"\\blur{config.FONT_HIGHLIGHT_BLUR_AMOUNT:.1f}" if config.FONT_HIGHLIGHT_BLUR_AMOUNT > 0.05 else ""
                word_specific_tags = (f"{{\\c{style_to_apply['color']}\\fs{style_to_apply['size']}"
                                      f"{blur_tag}}}")
            else: 
                word_specific_tags = f"{{\\c{config.PRIMARY_COLOR}\\fs{config.FONT_SIZE_BASE}}}"
            
            text_pieces_for_line.append(f"{word_specific_tags}{word_text_for_display}")

        formatted_line_text_content = format_text_into_lines_by_char_and_line_limit(
            text_pieces_for_line, config.MAX_CHARS_PER_LINE, config.MAX_LINES_PER_SUBTITLE
        )
        final_text_for_dialogue = f"{line_level_tags}{formatted_line_text_content}"

        dialogue_line = f"Dialogue: 0,{fmt_time(line_start_time)},{fmt_time(line_end_time)},HormoziBase,,0,0,0,,{final_text_for_dialogue}\n"
        ass_dialogue_lines.append(dialogue_line)
        dialogue_events_count += 1

    try:
        ass_output_dir = os.path.dirname(output_ass_path)
        if ass_output_dir: os.makedirs(ass_output_dir, exist_ok=True)
        with open(output_ass_path, "w", encoding="utf-8") as f_ass: f_ass.writelines(ass_dialogue_lines)
        logger.info(f"ASS subtitle file generated: {output_ass_path} ({dialogue_events_count} dialogue events)")
        return True
    except IOError as e_io:
        logger.error(f"Error writing ASS subtitle file '{output_ass_path}': {e_io}", exc_info=True); return False

def burn_subtitles(video_in_path: str, ass_file_path: str, video_out_path: str) -> bool:
    logger.info(f"Attempting to burn subtitles from '{os.path.basename(ass_file_path)}' into '{os.path.basename(video_in_path)}'...")
    if not os.path.exists(video_in_path) or os.path.getsize(video_in_path) < 1024: logger.error(f"Input video for burn missing/small: {video_in_path}"); return False
    if not os.path.exists(ass_file_path) or os.path.getsize(ass_file_path) < 150: logger.error(f"ASS file for burn missing/small: {ass_file_path}"); return False
    absolute_ass_path = os.path.abspath(ass_file_path)
    escaped_ass_path = absolute_ass_path.replace("\\", "/") 
    if sys.platform == "win32": 
        escaped_ass_path = re.sub(r"^(?P<drive>[a-zA-Z]):", r"\g<drive>\\:", escaped_ass_path) 
    escaped_ass_path_for_filter = escaped_ass_path.replace("'", "'\\''")
    fonts_dir_option_for_ffmpeg = ""
    font_name_to_search_base = re.sub(r'\s+bold\b', '', config.FONT_NAME, flags=re.IGNORECASE).strip()
    common_system_fonts = {"impact", "arial", "verdana", "tahoma", "times new roman", "helvetica", "sans-serif", "arial black", "roboto", "opensans", "montserrat", "comic sans ms"}
    if font_name_to_search_base.lower() not in common_system_fonts and config.FONT_DIR:
        font_directories_to_check = [config.FONT_DIR, "."] 
        if sys.platform == "win32":
            font_directories_to_check.append("C:/Windows/Fonts")
            if os.environ.get("LOCALAPPDATA"):
                font_directories_to_check.append(os.path.join(os.environ["LOCALAPPDATA"], "Microsoft", "Windows", "Fonts"))
        elif sys.platform == "darwin": 
            font_directories_to_check.append(os.path.expanduser("~/Library/Fonts"))
            font_directories_to_check.append("/Library/Fonts")
            font_directories_to_check.append("/System/Library/Fonts")
        else: 
            font_directories_to_check.append(os.path.expanduser("~/.fonts"))
            font_directories_to_check.append(os.path.expanduser("~/.local/share/fonts"))
            font_directories_to_check.append("/usr/share/fonts")
            font_directories_to_check.append("/usr/local/share/fonts")
        font_file_found_in_dir_path: Optional[str] = None
        font_extensions = ['.ttf', '.otf', '.TTF', '.OTF', '.ttc', '.TTC']
        potential_filenames_to_try = [config.FONT_NAME.replace(" ", "") + ext for ext in font_extensions] 
        potential_filenames_to_try.extend([config.FONT_NAME + ext for ext in font_extensions]) 
        if "bold" in config.FONT_NAME.lower():
             potential_filenames_to_try.extend([font_name_to_search_base + "-Bold" + ext for ext in font_extensions]) 
             potential_filenames_to_try.extend([font_name_to_search_base + " Bold" + ext for ext in font_extensions])
        potential_filenames_to_try.extend([font_name_to_search_base + ext for ext in font_extensions]) 
        unique_potential_filenames = list(dict.fromkeys(potential_filenames_to_try)) 
        for directory_path_str in font_directories_to_check:
            absolute_dir_path_str = os.path.abspath(directory_path_str)
            if not os.path.isdir(absolute_dir_path_str): continue
            for fname_to_check in unique_potential_filenames:
                potential_font_file_full_path = os.path.join(absolute_dir_path_str, fname_to_check)
                if os.path.exists(potential_font_file_full_path):
                    font_file_found_in_dir_path = absolute_dir_path_str 
                    logger.info(f"Font '{config.FONT_NAME}' likely present in directory: '{font_file_found_in_dir_path}' (found as '{fname_to_check}')")
                    break 
            if font_file_found_in_dir_path: break
        if font_file_found_in_dir_path:
            escaped_font_dir_path = font_file_found_in_dir_path.replace("\\", "/")
            if sys.platform == "win32":
                escaped_font_dir_path = re.sub(r"^(?P<drive>[a-zA-Z]):", r"\g<drive>\\:", escaped_font_dir_path)
            escaped_font_dir_for_filter = escaped_font_dir_path.replace("'", "'\\''")
            fonts_dir_option_for_ffmpeg = f":fontsdir='{escaped_font_dir_for_filter}'"
            logger.info(f"FFmpeg will use fontsdir hint: '{fonts_dir_option_for_ffmpeg}'")
        else:
            logger.warning(f"Custom font '{config.FONT_NAME}' not found in specified FONT_DIR ('{config.FONT_DIR}') or common system paths. FFmpeg will rely on system font discovery. Subtitles might not render as intended if the font is not globally available to FFmpeg/libass.")
    video_filter_string = f"ass=filename='{escaped_ass_path_for_filter}'{fonts_dir_option_for_ffmpeg}"
    ffmpeg_command = ["ffmpeg", "-y", "-i", video_in_path, "-vf", video_filter_string, "-map", "0:v:0", "-map", "0:a:0?", "-c:v", "libx264", "-preset", config.VIDEO_WRITE_PRESET_FINAL, "-crf", str(config.VIDEO_WRITE_CRF_FINAL), "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", video_out_path]
    logger.debug(f"FFmpeg command for subtitle burn: {' '.join(ffmpeg_command)}")
    logger.debug(f"FFmpeg -vf filter details: {video_filter_string}")
    try:
        output_video_parent_dir = os.path.dirname(video_out_path)
        if output_video_parent_dir: os.makedirs(output_video_parent_dir, exist_ok=True)
        ffmpeg_process = subprocess.run(ffmpeg_command, check=True, capture_output=True, text=True, errors='replace', timeout=600)
        if ffmpeg_process.stderr:
            ffmpeg_stderr_lines = ffmpeg_process.stderr.strip().split('\n')
            if logger.getEffectiveLevel() > logging.DEBUG: 
                non_progress_lines = [line for line in ffmpeg_stderr_lines if not line.strip().lower().startswith(('frame=', 'size=', 'time=', 'bitrate=', 'speed=', 'libass:'))]
            else: 
                non_progress_lines = ffmpeg_stderr_lines
            if non_progress_lines:
                joined_non_progress_stderr = '\n'.join(non_progress_lines)
                log_level_for_stderr = logging.WARNING if any(w in joined_non_progress_stderr.lower() for w in ["error", "warning", "fail", "not found", "fontselect"]) else logging.DEBUG
                logger.log(log_level_for_stderr, f"FFmpeg stderr output (filtered):\n{joined_non_progress_stderr}")
            if any("libass" in line.lower() and any(w in line.lower() for w in ["error", "warning", "fail", "not found", "fontselect", "fontconfig", "cannot find font"]) for line in ffmpeg_stderr_lines):
                 logger.warning(f"Potential libass/font messages detected in FFmpeg stderr. Please check subtitle appearance in the final video, especially if a custom font ('{config.FONT_NAME}') was intended.")
        if not os.path.exists(video_out_path) or os.path.getsize(video_out_path) < 1024: 
            logger.error(f"FFmpeg command seemed to succeed, but output video is missing or too small: {video_out_path}"); 
            logger.error(f"Please check full FFmpeg logs for critical errors not caught by parsing."); return False
        logger.info(f"Subtitles successfully burned into video: {os.path.basename(video_out_path)}"); return True
    except subprocess.CalledProcessError as e: 
        logger.error(f"FFmpeg subtitle burn FAILED! Return Code: {e.returncode}")
        logger.error(f"Command: {' '.join(e.cmd)}")
        logger.error(f"Stderr (last 15 lines):\n{chr(10).join(e.stderr.splitlines()[-15:]) if e.stderr else 'No Stderr available'}")
        return False
    except subprocess.TimeoutExpired: 
        logger.error(f"FFmpeg subtitle burn TIMED OUT.\nCommand: {' '.join(ffmpeg_command)}"); return False
    except FileNotFoundError: 
        logger.critical("FFmpeg executable not found. Ensure FFmpeg is installed and in your system's PATH."); return False
    except Exception as e: 
        logger.error(f"An unexpected error occurred during subtitle burning: {e}", exc_info=True); return False

def main():
    start_tm = time.time()
    logger.info(f"--- Starting AI Video Pipeline ({time.strftime('%Y-%m-%d %H:%M:%S')}) ---")
    logger.info(f"Target video dimensions: {config.VIDEO_WIDTH}x{config.VIDEO_HEIGHT} (W x H)")
    logger.info(f"Video background color for fallbacks/composition: {config.VIDEO_BACKGROUND_COLOR_RGB}")
    logger.info(f"\n{'='*15} STEP 1: PRE-CHECKS {'='*15}")
    critical_pre_check_failed = False
    if not config.deepgram_key_configured(): logger.critical("Deepgram API Key not configured correctly."); critical_pre_check_failed = True
    if not config.keys_configured(check_validity=True): logger.warning("AI Video API keys not configured with valid-looking keys. Video generation will be skipped/fail.")
    if _gemini_available and not config.gemini_key_configured():
        logger.warning("Gemini API key not configured. Script generation with Gemini will default to example script.")
    if not check_ffmpeg(): critical_pre_check_failed = True
    if not _whisper_available: logger.critical("Whisper-timestamped not available."); critical_pre_check_failed = True
    if not _deepgram_available: logger.critical("Deepgram SDK not available."); critical_pre_check_failed = True
    
    manim_ready = check_manim()
    if Config.ENABLE_MANIM_OVERLAYS and not manim_ready:
        logger.warning("Manim overlays are enabled in config, but Manim is not fully functional. Overlays will be skipped.")
    
    if critical_pre_check_failed: logger.critical("Critical pre-checks failed. Exiting."); sys.exit(1)
    logger.info("All critical pre-checks passed.")

    logger.info(f"\n{'='*15} STEP 2: LOAD/GENERATE SCRIPT (VIDEO PLAN) {'='*15}")
    script_data = get_user_script() 
    if not script_data: logger.critical("Failed to load or generate script. Exiting."); sys.exit(1)
    
    script_segments_from_user_plan = script_data.get("voiceover", []) 
    video_title_from_script = str(script_data.get("video_title", "AI_Video")).strip()
    if not script_segments_from_user_plan: logger.critical("Script 'voiceover' (segments plan) list empty. Exiting."); sys.exit(1)
    logger.info(f"Script (Video Plan) loaded/generated: '{video_title_from_script}' with {len(script_segments_from_user_plan)} segments.")
    
    overall_mood = script_data.get("overall_mood", "N/A")
    target_duration_est = script_data.get("target_duration_seconds", "N/A")
    suggested_bgm_style = script_data.get("suggested_bgm_style", "N/A")
    logger.info(f"Gemini Plan - Overall Mood: {overall_mood}, Target Duration: {target_duration_est}s, Suggested BGM: {suggested_bgm_style}")

    base_filename_prefix = clean_filename(video_title_from_script, 50); run_timestamp_id = int(start_tm)
    base_run_filename = f"{base_filename_prefix}_{run_timestamp_id}"; logger.info(f"Base filename: {base_run_filename}")
    output_dir = "outputs"; os.makedirs(output_dir, exist_ok=True)
    manim_output_dir = os.path.join(output_dir, f"manim_renders_{base_run_filename}") 

    final_voiceover_audio_file = os.path.join(output_dir, f"{base_run_filename}_full_audio.mp3")
    composed_video_no_subs_file = os.path.join(output_dir, f"{base_run_filename}_composed_no_subs.mp4")
    subtitles_ass_file = os.path.join(output_dir, f"{base_run_filename}_subtitles.ass")
    final_video_with_subs_file = os.path.join(output_dir, f"{base_run_filename}_final_video.mp4")
    segment_output_base_dir = os.path.join(output_dir, f"segments_{base_run_filename}")
    
    files_tracker = {
        "voiceover_audio": {os.path.abspath(final_voiceover_audio_file)}, 
        "individual_segment_videos": set(),
        "composed_video_no_subs": {os.path.abspath(composed_video_no_subs_file)},
        "subtitles_ass": {os.path.abspath(subtitles_ass_file)},
        "final_video_with_subs": {os.path.abspath(final_video_with_subs_file)},
        "manim_overlays": set() 
    }
    current_pipeline_status = "START"; is_pipeline_successful = False; total_voiceover_duration = 0.0
    timed_script_segments_with_audio: List[Dict[str, Any]] = []; generated_segment_video_paths: List[Optional[str]] = []
    transcribed_subtitle_segments: Optional[List[Dict[str, Any]]] = None
    generated_manim_overlays: Dict[str, str] = {} 

    try:
        current_pipeline_status = "VOICEOVER_GENERATION"
        logger.info(f"\n{'='*15} STEP 3: GENERATING VOICEOVER {'='*15}")
        total_voiceover_duration, timed_script_segments_with_audio = generate_voiceover(script_segments_from_user_plan, final_voiceover_audio_file)
        if total_voiceover_duration <= 0.01 or not timed_script_segments_with_audio: raise RuntimeError("Voiceover generation failed.")
        logger.info(f"Voiceover generation successful. Duration: {total_voiceover_duration:.3f}s")

        logger.info(f"\n{'='*15} GEMINI CREATIVE SUGGESTIONS (FOR MANUAL POST-PRODUCTION OR LOGGING) {'='*15}")
        for i, original_plan_segment in enumerate(script_segments_from_user_plan): 
            logger.info(f"--- Segment {original_plan_segment.get('segment_id', f'auto_id_{i+1}')} (VO text: \"{original_plan_segment.get('text', '')[:50]}...\") ---")
            if original_plan_segment.get("sfx_suggestions"): logger.info(f"  Suggested SFX: {', '.join(original_plan_segment['sfx_suggestions'])}")
            if original_plan_segment.get("manim_animation_concept"): logger.info(f"  Manim Concept: {original_plan_segment['manim_animation_concept']}")
        logger.info(f"{'='*60}\n")

        current_pipeline_status = "VIDEO_SEGMENT_GENERATION"
        logger.info(f"\n{'='*15} STEP 4: GENERATING VIDEO SEGMENTS {'='*15}")
        if not config.keys_configured(check_validity=True):
             logger.warning("Skipping video segment generation: AI Video API keys not validly configured.")
             generated_segment_video_paths = [None] * len(timed_script_segments_with_audio); any_segment_generation_failed = True
        else:
            any_segment_generation_failed = False; os.makedirs(segment_output_base_dir, exist_ok=True)
            for i, segment_data_with_timing in enumerate(timed_script_segments_with_audio): 
                original_plan_segment = script_segments_from_user_plan[i] 
                
                logger.info(f"---> Processing video generation for segment {original_plan_segment.get('segment_id', f'auto_id_{i+1}')} "
                            f"(Audio Duration: {segment_data_with_timing.get('duration', 0):.3f}s) <---")
                logger.info(f"     Visual Concept Prompt: {original_plan_segment.get('image_prompt')}")
                
                segment_relative_path_base = os.path.join(segment_output_base_dir, base_run_filename)
                # Pass original_plan_segment to generate_video_segment as it contains the image_prompt and segment_id
                # but also merge in the 'duration' from segment_data_with_timing for accuracy
                current_segment_plan_for_video_gen = original_plan_segment.copy()
                current_segment_plan_for_video_gen['duration'] = segment_data_with_timing.get('duration', 0.0)

                video_path_for_segment = generate_video_segment(current_segment_plan_for_video_gen, i, segment_relative_path_base)
                generated_segment_video_paths.append(video_path_for_segment)
                if video_path_for_segment: files_tracker["individual_segment_videos"].add(os.path.abspath(video_path_for_segment)); logger.info(f"Video segment {i + 1} processing finished: {os.path.basename(video_path_for_segment) if video_path_for_segment else 'FAILED/FALLBACK'}")
                else: any_segment_generation_failed = True; logger.warning(f"Video generation failed or resulted in fallback for segment {i + 1}.")
            if any_segment_generation_failed: logger.warning("One or more video segments failed or used fallbacks. Overall video might be affected.")

        # --- EXPERIMENTAL MANIM OVERLAY GENERATION ---
        if Config.ENABLE_MANIM_OVERLAYS and _manim_available_for_script and _gemini_available and Config.gemini_key_configured():
            logger.info(f"\n{'='*15} STEP 4.5: GENERATING & RENDERING MANIM OVERLAYS (EXPERIMENTAL) {'='*15}")
            os.makedirs(manim_output_dir, exist_ok=True)
            
            for seg_idx, original_segment_detail in enumerate(script_segments_from_user_plan):
                segment_id = original_segment_detail.get("segment_id")
                manim_concept = original_segment_detail.get("manim_animation_concept")
                
                if not segment_id: logger.warning(f"Segment {seg_idx+1} in plan has no segment_id, cannot process for Manim."); continue
                if not manim_concept or not isinstance(manim_concept, str) or manim_concept.strip() == "":
                    logger.debug(f"No valid Manim concept for segment '{segment_id}'.")
                    continue

                logger.info(f"  Attempting Manim for segment '{segment_id}', concept: '{manim_concept}'")
                
                try: 
                    duration_str = original_segment_detail.get("duration_hint_seconds", "3")
                    if isinstance(duration_str, str) and '-' in duration_str:
                        parts = duration_str.split('-')
                        anim_duration = (float(parts[0].strip()) + float(parts[1].strip())) / 2
                    else: anim_duration = float(duration_str)
                    anim_duration = max(1.0, min(anim_duration, 7.0)) 
                except ValueError: anim_duration = 3.0 

                position_hint = original_segment_detail.get("manim_position_hint")
                color_hint = original_segment_detail.get("manim_color_hint")

                manim_python_code = generate_manim_code_from_concept_with_gemini(
                    manim_concept, anim_duration, position_hint, color_hint,
                    Config.VIDEO_WIDTH, Config.VIDEO_HEIGHT, config, logger
                )
                
                if manim_python_code:
                    manim_overlay_video_path = execute_manim_script(
                        manim_python_code, "GeneratedManimOverlayScene", 
                        manim_output_dir, 
                        f"{base_run_filename}_manim_{segment_id.replace(' ','_')}_{seg_idx}",
                        logger
                    )
                    if manim_overlay_video_path:
                        generated_manim_overlays[segment_id] = os.path.abspath(manim_overlay_video_path)
                        files_tracker.setdefault("manim_overlays", set()).add(os.path.abspath(manim_overlay_video_path))
                        logger.info(f"    Successfully rendered Manim overlay for '{segment_id}' to: {manim_overlay_video_path}")
                    else: logger.error(f"    Failed to render Manim overlay for '{segment_id}'.")
                else: logger.warning(f"    Gemini failed to generate Manim code for '{segment_id}'.")
        else:
            logger.info("Manim overlay generation skipped (not enabled, or Gemini/Manim not available/configured).")

        current_pipeline_status = "VIDEO_COMPOSITION"
        logger.info(f"\n{'='*15} STEP 5: COMPOSING FINAL VIDEO (PRE-SUBTITLES) {'='*15}")
        composition_successful = create_final_video(
            generated_segment_video_paths, 
            final_voiceover_audio_file, 
            composed_video_no_subs_file, 
            timed_script_segments_with_audio, 
            total_voiceover_duration,
            generated_manim_overlays, 
            script_segments_from_user_plan 
        )
        if not composition_successful: raise RuntimeError("Final video composition (pre-subtitles) failed.")
        if not os.path.exists(composed_video_no_subs_file) or os.path.getsize(composed_video_no_subs_file) < 1024: raise RuntimeError(f"Composed video '{composed_video_no_subs_file}' missing/empty.")
        logger.info(f"Video composition (pre-subtitles) successful: {composed_video_no_subs_file}")
        
        current_pipeline_status = "AUDIO_TRANSCRIPTION"
        logger.info(f"\n{'='*15} STEP 6: TRANSCRIBING AUDIO FOR SUBTITLES {'='*15}")
        transcribed_subtitle_segments = transcribe(final_voiceover_audio_file)
        processed_display_segments_for_ass: List[Dict[str, Any]] = []
        if transcribed_subtitle_segments is None: logger.error("Audio transcription failed critically."); current_pipeline_status = "TRANSCRIPTION_FAILED_CRITICAL"
        elif not transcribed_subtitle_segments: logger.warning("Audio transcription produced no segments."); current_pipeline_status = "TRANSCRIPTION_EMPTY"
        else:
            logger.info("Audio transcription successful. Now splitting into display segments for ASS.")
            processed_display_segments_for_ass = split_transcript_into_display_segments(transcribed_subtitle_segments, timed_script_segments_with_audio) 
            if not processed_display_segments_for_ass: logger.warning("Splitting transcript into display segments resulted in no segments. Subtitles might be empty."); current_pipeline_status = "DISPLAY_SEGMENT_SPLIT_EMPTY"
            else: current_pipeline_status = "SUBTITLE_GENERATION_ASS"
        
        ass_file_generated_successfully = False
        if current_pipeline_status == "SUBTITLE_GENERATION_ASS":
            logger.info(f"\n{'='*15} STEP 7: GENERATING ASS SUBTITLE FILE {'='*15}")
            ass_generation_ok = generate_ass(processed_display_segments_for_ass, script_segments_from_user_plan, subtitles_ass_file, video_title_from_script)
            if not ass_generation_ok: logger.error("Failed to generate ASS file."); current_pipeline_status = "ASS_GENERATION_FAILED"
            else:
                 if os.path.exists(subtitles_ass_file) and os.path.getsize(subtitles_ass_file) > 200: 
                    logger.info(f"ASS file generated: {subtitles_ass_file}"); ass_file_generated_successfully = True; current_pipeline_status = "SUBTITLE_BURN_IN"
                 else: logger.error(f"ASS file missing/too small: {subtitles_ass_file}"); current_pipeline_status = "ASS_FILE_EMPTY_AFTER_GENERATION"
        else: logger.warning(f"Skipping ASS generation due to prior status: {current_pipeline_status}")
        
        if current_pipeline_status == "SUBTITLE_BURN_IN" and ass_file_generated_successfully:
            logger.info(f"\n{'='*15} STEP 8: BURNING SUBTITLES INTO VIDEO {'='*15}")
            burn_successful = burn_subtitles(composed_video_no_subs_file, subtitles_ass_file, final_video_with_subs_file)
            if not burn_successful: logger.error("Failed to burn subtitles."); current_pipeline_status = "SUBTITLE_BURN_FAILED"
            else:
                if not os.path.exists(final_video_with_subs_file) or os.path.getsize(final_video_with_subs_file) < 1024: logger.error("Final video missing/small after burn."); current_pipeline_status = "FINAL_VIDEO_MISSING_AFTER_BURN"
                else: logger.info(f"Subtitles burned. Final video: {final_video_with_subs_file}"); is_pipeline_successful = True; current_pipeline_status = "SUCCESS_WITH_SUBTITLES"
        else:
            logger.warning(f"Skipping subtitle burn-in. Status: {current_pipeline_status}.")
            if os.path.exists(composed_video_no_subs_file):
                 logger.info(f"Renaming '{os.path.basename(composed_video_no_subs_file)}' to '{os.path.basename(final_video_with_subs_file)}' (no subs).")
                 try:
                      if os.path.exists(final_video_with_subs_file): os.remove(final_video_with_subs_file)
                      shutil.move(composed_video_no_subs_file, final_video_with_subs_file) 
                      logger.info(f"Moved composed video (no subs) to final path: '{final_video_with_subs_file}'")
                      composed_abs = os.path.abspath(composed_video_no_subs_file); final_abs = os.path.abspath(final_video_with_subs_file)
                      if composed_abs in files_tracker["composed_video_no_subs"]: files_tracker["composed_video_no_subs"].remove(composed_abs)
                      files_tracker["final_video_with_subs"] = {final_abs}
                      is_pipeline_successful = True; current_pipeline_status = "SUCCESS_NO_SUBTITLES"
                 except OSError as rename_err:
                     logger.error(f"Failed to move composed video (no subs): {rename_err}. Video remains at: '{composed_video_no_subs_file}'")
                     current_pipeline_status = "FINAL_RENAME_FAILED_NO_SUBS"
                     composed_abs = os.path.abspath(composed_video_no_subs_file)
                     if composed_abs not in files_tracker["final_video_with_subs"]: files_tracker["final_video_with_subs"].add(composed_abs) 
            else: logger.critical("Cannot finalize video (no subs): composed video missing."); current_pipeline_status += "_AND_COMPOSED_VIDEO_MISSING"; is_pipeline_successful = False
    except KeyboardInterrupt: logger.warning("\n" + "="*20 + " PIPELINE INTERRUPTED BY USER " + "="*20); current_pipeline_status = "USER_INTERRUPT"; is_pipeline_successful = False
    except RuntimeError as pipeline_error: logger.critical(f"\n{'='*20} PIPELINE HALTED: {pipeline_error} (in step: {current_pipeline_status}) {'='*20}", exc_info=False); current_pipeline_status = f"FAIL_RUNTIME_{current_pipeline_status}"; is_pipeline_successful = False
    except Exception as general_exception: logger.critical(f"\n{'='*20} UNEXPECTED PYTHON ERROR (step: {current_pipeline_status}) {'='*20}\nError: {general_exception}", exc_info=True); current_pipeline_status = f"FAIL_UNEXPECTED_{current_pipeline_status}"; is_pipeline_successful = False
    finally: 
        logger.info(f"\n{'='*15} STEP 9: CLEANUP AND SUMMARY (Status: {current_pipeline_status}) {'='*15}")
        files_to_keep: Set[str] = set()
        final_video_abs_path = os.path.abspath(final_video_with_subs_file)
        composed_no_subs_abs_path = os.path.abspath(composed_video_no_subs_file)
        subtitles_ass_abs_path = os.path.abspath(subtitles_ass_file)
        voiceover_abs_path = os.path.abspath(final_voiceover_audio_file)
        if is_pipeline_successful and os.path.exists(final_video_abs_path):
            files_to_keep.add(final_video_abs_path)
            if current_pipeline_status == "SUCCESS_WITH_SUBTITLES" and os.path.exists(subtitles_ass_abs_path): files_to_keep.add(subtitles_ass_abs_path)
        elif (current_pipeline_status == "SUBTITLE_BURN_FAILED" or current_pipeline_status == "FINAL_RENAME_FAILED_NO_SUBS") and os.path.exists(composed_no_subs_abs_path):
            logger.warning("Keeping composed video without subtitles and related files due to burn/rename failure.")
            files_to_keep.add(composed_no_subs_abs_path)
            if os.path.exists(voiceover_abs_path): files_to_keep.add(voiceover_abs_path)
            if os.path.exists(subtitles_ass_abs_path): files_to_keep.add(subtitles_ass_abs_path)
        else: 
            logger.warning(f"Pipeline incomplete/failed. Keeping major intermediates based on status: {current_pipeline_status}")
            if os.path.exists(voiceover_abs_path): files_to_keep.add(voiceover_abs_path)
            files_to_keep.update(pth for pth in files_tracker["individual_segment_videos"] if pth and os.path.exists(pth))
            if os.path.exists(composed_no_subs_abs_path): files_to_keep.add(composed_no_subs_abs_path)
            if os.path.exists(subtitles_ass_abs_path): files_to_keep.add(subtitles_ass_abs_path)
            if os.path.exists(final_video_abs_path) and final_video_abs_path != composed_no_subs_abs_path : 
                 files_to_keep.add(final_video_abs_path)
        
        if generated_manim_overlays and (is_pipeline_successful or "FAIL" not in current_pipeline_status.upper()):
            for manim_path in generated_manim_overlays.values():
                if os.path.exists(manim_path):
                    files_to_keep.add(manim_path)

        all_tracked_files_abs = set().union(*(files_tracker[key] for key in files_tracker))
        if os.path.exists(composed_no_subs_abs_path) and composed_no_subs_abs_path not in files_to_keep:
            all_tracked_files_abs.add(composed_no_subs_abs_path)

        files_to_remove = all_tracked_files_abs - files_to_keep; cleaned_files_count = 0
        if files_to_remove: logger.info(f"Cleaning up {len(files_to_remove)} intermediate/unused files...")
        for file_path_to_remove in sorted(list(files_to_remove)):
            if file_path_to_remove and os.path.exists(file_path_to_remove):
                try: os.remove(file_path_to_remove); logger.debug(f"  Removed: {os.path.basename(file_path_to_remove)}"); cleaned_files_count += 1
                except OSError as e_remove_final: logger.warning(f"  Failed to remove '{os.path.basename(file_path_to_remove)}': {e_remove_final}")
        logger.info(f"Cleanup: Removed {cleaned_files_count} files.")
        
        temp_dirs_to_clean = {os.path.abspath("temp_audio_segments")} 
        if os.path.isdir(manim_output_dir) and not any(kept_file.startswith(manim_output_dir + os.path.sep) for kept_file in files_to_keep):
            temp_dirs_to_clean.add(manim_output_dir) 

        segment_output_base_abs_dir = os.path.abspath(segment_output_base_dir) 
        try:
            search_dirs_for_raw_temps = [os.path.abspath(output_dir)]
            if os.path.isdir(segment_output_base_abs_dir): 
                search_dirs_for_raw_temps.append(segment_output_base_abs_dir)
            for s_dir in set(search_dirs_for_raw_temps): 
                if not os.path.isdir(s_dir): continue
                for item in os.listdir(s_dir):
                    if item.startswith("temp_") and item.endswith("_raw_clips"):
                        full_item_path = os.path.join(s_dir, item)
                        if os.path.isdir(full_item_path):
                            temp_dirs_to_clean.add(full_item_path)
        except Exception as list_err:
            logger.warning(f"Could not comprehensively list all temp raw clip dirs for cleanup: {list_err}")
        if os.path.isdir(segment_output_base_abs_dir):
             if not any(kept_file.startswith(segment_output_base_abs_dir + os.path.sep) for kept_file in files_to_keep):
                 temp_dirs_to_clean.add(segment_output_base_abs_dir)
             else: logger.info(f"Keeping main segment directory '{segment_output_base_abs_dir}' as it contains kept files.")
        cleaned_dirs_count = 0
        for temp_dir_path in sorted(list(temp_dirs_to_clean), key=len, reverse=True): 
            if os.path.isdir(temp_dir_path):
                is_kept_dir_or_contains_kept = any(kept_file.startswith(temp_dir_path + os.path.sep) for kept_file in files_to_keep) or \
                                             temp_dir_path in files_to_keep or \
                                             temp_dir_path == os.path.abspath(output_dir) 
                if is_kept_dir_or_contains_kept:
                    logger.debug(f"Skipping removal of dir '{temp_dir_path}' (contains kept files, is marked kept, or is main output).")
                    continue
                try:
                    shutil.rmtree(temp_dir_path, ignore_errors=False) 
                    logger.info(f"Removed temp directory: {temp_dir_path}"); cleaned_dirs_count += 1
                except FileNotFoundError: pass 
                except Exception as e_rm_temp_dir: logger.warning(f"Could not clean up temp directory '{temp_dir_path}': {e_rm_temp_dir}")
        logger.info(f"Cleanup: Removed {cleaned_dirs_count} temporary directories.")
        total_runtime_seconds = time.time() - start_tm
        logger.info(f"--- Total Pipeline Runtime: {total_runtime_seconds:.2f}s ({total_runtime_seconds/60:.1f} minutes) ---")
        print("\n" + "="*70 + "\n           AI VIDEO PIPELINE SUMMARY\n" + "="*70)
        print(f" Final Pipeline Status: {current_pipeline_status}"); exit_code = 0 if is_pipeline_successful else 1
        display_output_path_str = "N/A"; display_output_description_str = ""
        if is_pipeline_successful:
             if os.path.exists(final_video_abs_path):
                 display_output_path_str = final_video_abs_path
                 if current_pipeline_status == "SUCCESS_WITH_SUBTITLES": print(f" Outcome:               ✅ SUCCESS (Video with Subtitles)")
                 elif current_pipeline_status == "SUCCESS_NO_SUBTITLES": print(f" Outcome:               ⚠️ SUCCESS (Video WITHOUT Subtitles)")
             else: print(f" Outcome:               ❓ SUCCESS (Status: {current_pipeline_status}, but final file missing!)"); exit_code = 1
        else:
            print(f" Outcome:               ❌ FAILURE or PARTIAL (Process incomplete/interrupted)")
            if os.path.exists(final_video_abs_path): display_output_path_str = final_video_abs_path; display_output_description_str = "(Final video path, check status)"
            elif os.path.exists(composed_no_subs_abs_path): display_output_path_str = composed_no_subs_abs_path; display_output_description_str = "(Composed video, no subs)"
            elif os.path.exists(voiceover_abs_path): display_output_path_str = voiceover_abs_path; display_output_description_str = "(Audio Only)"
        if display_output_path_str != "N/A": print(f" Main Output File:      {display_output_path_str} {display_output_description_str}".strip())
        else: print(f" Main Output File:      N/A (No primary output file generated/located)")
        if files_to_keep:
            print("\n Kept Files:")
            for kept_file_path in sorted(list(files_to_keep)):
                if os.path.exists(kept_file_path):
                    try: print(f"  - {kept_file_path} ({os.path.getsize(kept_file_path) / (1024 * 1024):.2f}MB)")
                    except OSError: print(f"  - {kept_file_path} (File size unavailable)")
                else: print(f"  - {kept_file_path} (File marked to keep, but NOT found!)")
        else: print("\n Kept Files:            None")
        print("="*70 + f"\n Total Runtime:         {total_runtime_seconds:.2f} seconds (~{total_runtime_seconds/60:.1f} minutes)\n" + "="*70 + "\n")
        sys.exit(exit_code)

if __name__ == "__main__":
    if _gemini_available and Config.gemini_key_configured():
        try:
            genai.configure(api_key=Config.GEMINI_API_KEY)
            logger.info("Gemini AI SDK configured globally.")
        except Exception as e:
            logger.error(f"Failed to configure Gemini AI SDK globally: {e}")
    main()
