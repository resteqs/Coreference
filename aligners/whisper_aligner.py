import whisper
from pydub import AudioSegment
import pandas as pd
import time
import os
def verify_mfa_alignment_whisper(audio_file, intervals_from_TxtGrid, model):
    """Verify MFA alignment using Whisper and return results in a DataFrame.

    - Loads a Whisper model for transcription.
    - Segments the audio file based on intervals from the TextGrid.
    - Transcribes each segment and compares it with the expected word. I checked and whisper fills in the short files with silence to match the correct input shape. 
     This should not cause any issues
    - Saves the results, including execution time and timestamps.

    Args:
        audio_file (str): Path to the audio file (.wav).
        intervals_from_TxtGrid (list): Intervals with words and their timings.

    Returns:
        pandas.DataFrame: Contains columns:
            - 'Correct Word': The expected word from the interval.
            - 'Whisper Prediction': The word predicted by Whisper.
            - 'Match': 1 if the words match, 0 otherwise.
            - 'Start Time': Start time of the interval.
            - 'End Time': End time of the interval.
            - 'Execution Time': Cumulative time taken up to this point.
    """
    print("Loading Whisper model...")
    model = whisper.load_model(model) #--------------------------tiny, base, medium, large, turbo---------------------------------
    
    # Load audio file
    print(f"Loading audio file: {audio_file}")
    audio = AudioSegment.from_wav(audio_file)
    
    total_words = len(intervals_from_TxtGrid)
    start_time = time.time()

    results = []
    
    for index, interval in enumerate(intervals_from_TxtGrid, 1):
        word = interval["word"]
        if not word or word.isspace():
            continue
            
        # Extract time segment
        start_ms = int(interval["start_time"] * 1000)
        end_ms = int(interval["end_time"] * 1000)
        segment = audio[start_ms:end_ms]
        
        # Process segment
        temp_file = "temp_segment.wav"
        segment.export(temp_file, format="wav")
        
        # Get Whisper transcription
        result = model.transcribe(temp_file)
        whisper_text = result["text"].strip().lower()
        
        # Compare results
        mfa_word = word.lower().strip()
        is_match = 1 if mfa_word == whisper_text else 0
        
        exec_time = time.time() - start_time
        
        # Append to results list
        results.append({
            'Correct Word': mfa_word,
            'Whisper Prediction': whisper_text,
            'Match': is_match,
            'Start Time': f"{interval['start_time']:.2f}",
            'End Time': f"{interval['end_time']:.2f}",
            'Execution Time': f"{exec_time:.2f}"
        })
        
        # Remove temp file
        os.remove(temp_file)
        
        # Print progress
        progress = (index / total_words) * 100
        print(f"{index}/{total_words} ; {progress:.2f}%")
    
    # Create a pandas DataFrame
    df = pd.DataFrame(results)
    return df