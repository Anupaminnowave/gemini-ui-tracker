import os
import subprocess
from dotenv import load_dotenv
import vertexai
from vertexai.generative_models import GenerativeModel, Part
from google.cloud import storage

# Load configuration from .env
load_dotenv()
PROJECT_ID = os.getenv("PROJECT_ID")
LOCATION = os.getenv("LOCATION")
BUCKET_NAME = os.getenv("BUCKET_NAME")

# Supported formats
IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.webp'}
VIDEO_EXTENSIONS = {'.mp4', '.mov', '.avi'}

def process_media(input_file):
    """Handles preprocessing: Mutes videos, skips images."""
    _, ext = os.path.splitext(input_file.lower())
    
    if ext in IMAGE_EXTENSIONS:
        print(f"🖼️  Image detected. Skipping mute step for {input_file}.")
        return input_file # Images don't need 'processed' folder muting
    
    if not os.path.exists("processed"):
        os.makedirs("processed")
        
    output_file = os.path.join("processed", f"muted_{os.path.basename(input_file)}")
    
    if os.path.exists(output_file):
        print(f"⏭️  Step 1: Skipping mute, {output_file} already exists.")
        return output_file

    print(f"🔇 Step 1: Muting {input_file}...")
    cmd = f"ffmpeg -i {input_file} -an -vcodec copy {output_file} -y"
    subprocess.run(cmd, shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    return output_file

def upload_to_gcs(local_path):
    """Uploads to GCS, skipping if the file already exists."""
    storage_client = storage.Client(project=PROJECT_ID)
    bucket = storage_client.bucket(BUCKET_NAME)
    
    if not bucket.exists():
        bucket = storage_client.create_bucket(BUCKET_NAME, location=LOCATION)
        
    blob_name = f"uploads/{os.path.basename(local_path)}"
    blob = bucket.blob(blob_name)
    
    if blob.exists():
        print(f"⏭️  Step 2: Skipping upload, {blob_name} already in GCS.")
    else:
        print(f"☁️ Step 2: Uploading to GCS...")
        blob.upload_from_filename(local_path)
        
    return f"gs://{BUCKET_NAME}/{blob_name}"

def analyze_with_gemini(gcs_uri, original_filename):
    """Analyzes with Gemini, caching the report locally."""
    report_path = os.path.join("processed", f"report_{original_filename}.txt")
    
    if os.path.exists(report_path):
        print(f"⏭️  Step 3: Skipping Gemini, existing report found at {report_path}.")
        with open(report_path, "r", encoding="utf-8") as f:
            return f.read()

    vertexai.init(project=PROJECT_ID, location=LOCATION)
    model = GenerativeModel("gemini-2.5-flash")
    
    # Determine MIME type based on extension
    _, ext = os.path.splitext(original_filename.lower())
    mime_type = "video/mp4" if ext in VIDEO_EXTENSIONS else f"image/{ext[1:]}"
    if ext == ".jpg": mime_type = "image/jpeg"

    media_part = Part.from_uri(mime_type=mime_type, uri=gcs_uri)
    
    prompt = """
    Perform a comprehensive UI/UX audit.
    If this is a VIDEO: Document the workflow, look for dead clicks, visual bugs, and lag.
    If this is an IMAGE: Describe everything on the site, identify UI discrepancies, 
    misaligned elements, or overlapping text. Explain the layout clearly.
    """
    
    print(f"🤖 Step 3: Gemini is analyzing the {'video' if ext in VIDEO_EXTENSIONS else 'image'}...")
    response = model.generate_content([media_part, prompt])
    
    if not os.path.exists("processed"): os.makedirs("processed")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(response.text)
    
    return response.text

if __name__ == "__main__":
    # Change this to your filename (e.g., "screenshot.png" or "test.mp4")
    video_filename = "sample_screenshot1.png" 
    input_path = os.path.join("snapshots", video_filename)
    
    if os.path.exists(input_path):
        try:
            processed_path = process_media(input_path)
            gcs_link = upload_to_gcs(processed_path)
            report = analyze_with_gemini(gcs_link, video_filename)
            print(f"\n{'='*20} REPORT {'='*20}\n{report}")
        except Exception as e:
            print(f"❌ Error: {e}")
    else:
        print(f"❌ File not found at {input_path}")