from fastapi import UploadFile
import os
import shutil
from typing import Optional
import uuid
from datetime import datetime

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff', 'webp'}

def allowed_file(filename: str) -> bool:
    """
    Check if the uploaded file has an allowed extension
    """
    if not filename:
        return False
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

async def save_upload_file(upload_file: UploadFile, destination_dir: str) -> str:
    """
    Save an uploaded file to the specified directory
    Returns the path to the saved file
    """
    # Generate unique filename to avoid conflicts
    file_extension = upload_file.filename.rsplit('.', 1)[1].lower()
    unique_filename = f"{uuid.uuid4()}.{file_extension}"
    file_path = os.path.join(destination_dir, unique_filename)
    
    # Save the file
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(upload_file.file, buffer)
    
    return file_path

def get_file_info(file_path: str) -> dict:
    """
    Get information about a file
    """
    if not os.path.exists(file_path):
        return {}
    
    stat = os.stat(file_path)
    return {
        "size": stat.st_size,
        "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
        "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
        "name": os.path.basename(file_path)
    }

def cleanup_old_files(directory: str, max_age_hours: int = 24) -> int:
    """
    Clean up old files in a directory
    Returns the number of files deleted
    """
    if not os.path.exists(directory):
        return 0
    
    current_time = datetime.now().timestamp()
    deleted_count = 0
    
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path):
            file_age = current_time - os.path.getmtime(file_path)
            if file_age > (max_age_hours * 3600):  # Convert hours to seconds
                try:
                    os.remove(file_path)
                    deleted_count += 1
                except OSError:
                    pass  # File might be in use
    
    return deleted_count

def ensure_directory_exists(directory: str) -> None:
    """
    Ensure that a directory exists, create it if it doesn't
    """
    os.makedirs(directory, exist_ok=True)

def get_supported_formats() -> list:
    """
    Get list of supported image formats
    """
    return list(ALLOWED_EXTENSIONS)
