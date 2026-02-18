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
    """Removes audio if needed, skipping if the muted file already exists."""
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
    """Uploads to GCS, skipping if the file already exists in the bucket."""
    storage_client = storage.Client(project=PROJECT_ID)
    bucket = storage_client.bucket(BUCKET_NAME)
    
    if not bucket.exists():
        print(f"☁️ Creating bucket: {BUCKET_NAME}...")
        bucket = storage_client.create_bucket(BUCKET_NAME, location=LOCATION)
        
    blob_name = f"uploads/{os.path.basename(local_path)}"
    blob = bucket.blob(blob_name)
    
    # SMART CHECK: If blob exists, don't re-upload
    if blob.exists():
        print(f"⏭️  Step 2: Skipping upload, {blob_name} already in GCS.")
    else:
        print(f"☁️ Step 2: Uploading {local_path} to GCS...")
        blob.upload_from_filename(local_path)
        
    return f"gs://{BUCKET_NAME}/{blob_name}"

def analyze_with_gemini(gcs_uri, original_filename):
    """Analyzes with Gemini, skipping if a local report file already exists."""
    report_path = os.path.join("processed", f"report_{original_filename}.txt")
    
    # SMART CHECK: If report exists locally, just read it
    if os.path.exists(report_path):
        print(f"⏭️  Step 3: Skipping Gemini analysis, existing report found at {report_path}.")
        with open(report_path, "r", encoding="utf-8") as f:
            return f.read()

    vertexai.init(project=PROJECT_ID, location=LOCATION)
    model = GenerativeModel("gemini-2.0-flash-001") # Using stable 2.0 Flash
    video_part = Part.from_uri(mime_type="video/mp4", uri=gcs_uri)
    
    prompt = """
    Perform a comprehensive UI/UX audit on this screen recording.
    1. Document the end-to-end user workflow, listing every click and interaction.
    2. Identify any 'Dead Clicks' or unresponsive elements.
    3. Look for visual bugs, overlapping text, or broken animations.
    4. Provide timestamps for every defect.
    5. Evaluate system response time (laggy vs snappy).
    """
    
    print("🤖 Step 3: Gemini is analyzing the workflow for defects...")
    response = model.generate_content([video_part, prompt])
    
    # Save the report for next time
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(response.text)
    
    return response.text

if __name__ == "__main__":
    video_filename = "test_case2_silent.mp4" 
    input_path = os.path.join("recordings", video_filename)
    
    if os.path.exists(input_path):
        try:
            muted_path = mute_video(input_path)
            gcs_link = upload_to_gcs(muted_path)
            report = analyze_with_gemini(gcs_link, video_filename)
            
            print("\n" + "="*50)
            print("🔍 QA DEFECT REPORT")
            print("="*50)
            print(report)
            
        except Exception as e:
            print(f"❌ Error during execution: {e}")
    else:
        print(f"❌ Video not found at {input_path}")