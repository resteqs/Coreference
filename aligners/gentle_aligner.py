import time
import requests
import json
import os
def run_gentle_align(audio_file, transcript_file, output_dir, base_name):
    """Run Gentle aligner using Gentle server via Docker."""
    
    gentle_json = os.path.join(output_dir, f"{base_name}_gentle.json")
    print("Waiting 10sec for Docker...")
    time.sleep(10) 
    # Verify files exist
    if not os.path.exists(audio_file) or not os.path.exists(transcript_file):
        print("Error: Input files not found")
        return None

    # Prepare request
    url = "http://localhost:8765/transcriptions?async=false"

    files = {
        'audio': open(audio_file, 'rb'),
        'transcript': open(transcript_file, 'rb')
    }

    try:
        print(f"Sending files to Gentle server for alignment.")
        response = requests.post(url, files=files)
        response.raise_for_status()
        data = response.json()

        # Save output
        with open(gentle_json, 'w', encoding='utf-8') as f:
            json.dump(response.json(), f, indent=2)
            
        return gentle_json
    except requests.exceptions.RequestException as e:
        print(f"Gentle server error: {e}")
        return None
