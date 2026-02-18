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

def mute_video(input_file):
    """Removes audio from video using FFmpeg and saves it in the 'processed' folder."""
    if not os.path.exists("processed"):
        os.makedirs("processed")
        
    output_file = os.path.join("processed", f"muted_{os.path.basename(input_file)}")
    print(f"🔇 Step 1: Muting {input_file}...")
    
    # -an removes audio, -vcodec copy keeps quality high
    cmd = f"ffmpeg -i {input_file} -an -vcodec copy {output_file} -y"
    subprocess.run(cmd, shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    return output_file

def upload_to_gcs(local_path):
    """Uploads the muted video to GCS using your storage.admin permissions."""
    storage_client = storage.Client(project=PROJECT_ID)
    bucket = storage_client.bucket(BUCKET_NAME)
    
    # Create bucket if it doesn't exist
    if not bucket.exists():
        print(f"☁️ Creating bucket: {BUCKET_NAME}...")
        bucket = storage_client.create_bucket(BUCKET_NAME, location=LOCATION)
        
    blob = bucket.blob(f"uploads/{os.path.basename(local_path)}")
    print(f"☁️ Step 2: Uploading {local_path} to GCS...")
    blob.upload_from_filename(local_path)
    return f"gs://{BUCKET_NAME}/uploads/{os.path.basename(local_path)}"

def analyze_with_gemini(gcs_uri):
    """Sends the GCS video link to Vertex AI for a full UI defect analysis."""
    vertexai.init(project=PROJECT_ID, location=LOCATION)
    
    # Using Gemini 1.5 Flash for multimodal video analysis
    model = GenerativeModel("gemini-2.5-flash")
    video_part = Part.from_uri(mime_type="video/mp4", uri=gcs_uri)
    
    # Broad UI Testing Prompt
    prompt = """
    Perform a comprehensive UI/UX audit on this screen recording.
    1. Document the end-to-end user workflow, listing every click and interaction.
    2. Identify any 'Dead Clicks' or unresponsive elements where the user interacts but the UI does not change.
    3. Look for visual bugs, such as overlapping text, broken animations, or unexpected error messages.
    4. For every defect found, provide the exact timestamp and a description of what failed.
    5. Evaluate the overall system response time—does the UI feel laggy or snappy?
    """
    
    print("🤖 Step 3: Gemini is analyzing the workflow for defects...")
    response = model.generate_content([video_part, prompt])
    return response.text

if __name__ == "__main__":
    # Ensure this file exists in your recordings/ folder
    video_filename = "test_case1_silent.mp4" 
    input_path = os.path.join("recordings", video_filename)
    
    if os.path.exists(input_path):
        try:
            # Step 1: Pre-process
            muted_path = mute_video(input_path)
            
            # Step 2: Store
            gcs_link = upload_to_gcs(muted_path)
            
            # Step 3: Analyze
            report = analyze_with_gemini(gcs_link)
            
            print("\n" + "="*50)
            print("🔍 QA DEFECT REPORT")
            print("="*50)
            print(report)
            
        except Exception as e:
            print(f"❌ Error during execution: {e}")
    else:
        print(f"❌ Video not found at {input_path}")