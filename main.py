import os
import subprocess
import mimetypes
from dotenv import load_dotenv
import vertexai
from vertexai.generative_models import GenerativeModel, Part
from google.cloud import storage

# LangSmith core imports
from langsmith import traceable, Client
from langsmith.run_helpers import get_current_run_tree

# Load configuration from .env
load_dotenv()

PROJECT_ID = os.getenv("PROJECT_ID")
LOCATION = os.getenv("LOCATION")
BUCKET_NAME = os.getenv("BUCKET_NAME")

ls_client = Client()

def get_mime_type(file_path):
    """Detects MIME type for Vertex AI strictness."""
    mime_type, _ = mimetypes.guess_type(file_path)
    if not mime_type:
        ext = os.path.splitext(file_path)[1].lower()
        if ext in ['.jpg', '.jpeg']: return "image/jpeg"
        if ext == '.png': return "image/png"
        if ext == '.mp4': return "video/mp4"
    return mime_type

@traceable
def preprocess_video(input_file):
    """
    Rule 1: Only preprocess (mute) if it's a video and hasn't been done yet.
    Images skip this function entirely in the main logic.
    """
    os.makedirs("processed", exist_ok=True)
    output_file = os.path.join("processed", f"muted_{os.path.basename(input_file)}")
    
    if os.path.exists(output_file):
        print(f"⏭️  Step 1: Video mute skipped (muted file already exists).")
        return output_file

    print(f"🔇 Step 1: Muting video {input_file}...")
    cmd = f"ffmpeg -i {input_file} -an -vcodec copy {output_file} -y"
    subprocess.run(cmd, shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    return output_file

@traceable
def upload_to_gcs(local_path):
    """
    Rule 2: Skip upload if the file already exists in the GCS bucket.
    """
    storage_client = storage.Client(project=PROJECT_ID)
    bucket = storage_client.bucket(BUCKET_NAME)
    blob_name = f"uploads/{os.path.basename(local_path)}"
    blob = bucket.blob(blob_name)
    
    if blob.exists():
        print(f"⏭️  Step 2: GCS skip (File already exists in bucket).")
    else:
        print(f"☁️ Step 2: Uploading to GCS...")
        blob.upload_from_filename(local_path, timeout=600)
        
    return f"gs://{BUCKET_NAME}/{blob_name}"

@traceable(run_type="llm", name="Gemini UI Analysis")
def analyze_with_gemini(gcs_uri, original_filename):
    """
    Rule 3: Skip Gemini API call if a local report already exists.
    """
    report_filename = f"report_{os.path.basename(original_filename)}.txt"
    report_path = os.path.join("processed", report_filename)
    
    if os.path.exists(report_path):
        print(f"⏭️  Step 3: Gemini skip (Local report found: {report_path}).")
        with open(report_path, "r", encoding="utf-8") as f:
            return f.read()

    mime_type = get_mime_type(original_filename)
    vertexai.init(project=PROJECT_ID, location=LOCATION)
    model = GenerativeModel("gemini-2.0-flash") 
    
    print(f"🤖 Step 3: Sending to Gemini API ({mime_type})...")
    media_part = Part.from_uri(mime_type=mime_type, uri=gcs_uri)
    
    prompt = """
    Perform a comprehensive UI/UX audit.
    If this is a VIDEO: Document the workflow, look for dead clicks, visual bugs, and lag.
    If this is an IMAGE: Describe everything on the site, identify UI discrepancies, 
    misaligned elements, or overlapping text. Explain the layout clearly.
    """
    
    response = model.generate_content([media_part, prompt])
    
    # Save the report for future Rule 3 skips
    os.makedirs("processed", exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(response.text)
    
    # LangSmith Token Metadata
    run_tree = get_current_run_tree()
    if run_tree:
        usage = {
            "prompt_tokens": response.usage_metadata.prompt_token_count,
            "completion_tokens": response.usage_metadata.candidates_token_count,
            "total_tokens": response.usage_metadata.total_token_count
        }
        run_tree.end(outputs={"report": response.text})
        run_tree.metadata.update(usage)
    
    return response.text

if __name__ == "__main__":
    if not os.getenv("LANGSMITH_API_KEY"):
        print("❌ Error: LANGSMITH_API_KEY not found.")
    else:
        try:
            target_file = "recordings/test_case1_silent.mp4" # Input file
            
            if not os.path.exists(target_file):
                print(f"❌ File not found: {target_file}")
            else:
                print(f"🚀 Processing: {target_file}")
                
                mime = get_mime_type(target_file)
                
                # BRANCHING LOGIC: Only videos go to Step 1
                if mime and mime.startswith("video"):
                    work_path = preprocess_video(target_file)
                else:
                    print("🖼️  Step 1: Skipping (Images require no preprocessing).")
                    work_path = target_file
                
                # Step 2: Upload (Rule 2 check is inside the function)
                gcs_uri = upload_to_gcs(work_path)
                
                # Step 3: Analyze (Rule 3 check is inside the function)
                report = analyze_with_gemini(gcs_uri, target_file)
                
                print(f"\n{'='*20} AUDIT REPORT {'='*20}\n{report}")
            
        except Exception as e:
            print(f"❌ Error: {e}")
        finally:
            print("\n📤 Finalizing LangSmith connection...")
            ls_client.flush()