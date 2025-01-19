import torch
import librosa
from transformers import AutoProcessor, AutoModelForCTC
import pandas as pd

def verify_mfa_alignment_mms(audio_file, intervals):


    # Load the processor and model
    processor = AutoProcessor.from_pretrained("mms-meta/mms-zeroshot-300m")
    model = AutoModelForCTC.from_pretrained("mms-meta/mms-zeroshot-300m")

    # Set the model to evaluation mode
    model.eval()

    # Load the audio file
    speech_array, sampling_rate = librosa.load(audio_file, sr=16000)

    results = []
    print("Starting MMS...")
    length = len(intervals)
    for index, interval in enumerate(intervals):
        start_time = interval['start_time']
        end_time = interval['end_time']
        expected_text = interval['word']

        # Extract the audio segment
        start_sample = int(start_time * sampling_rate)
        end_sample = int(end_time * sampling_rate)
        audio_segment = speech_array[start_sample:end_sample]

        # Prepare input values
        inputs = processor(audio_segment, sampling_rate=sampling_rate, return_tensors="pt")

        # Get logits
        with torch.no_grad():
            logits = model(**inputs).logits

        # Decode the predicted ids to text
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.decode(predicted_ids[0])

        # Append the result
        results.append({
            'start_time': start_time,
            'end_time': end_time,
            'expected_text': expected_text,
            'transcribed_text': transcription
        })
        progress = (index / length) * 100
        print(f"{index}/{length} ; {progress:.2f}%")

    # Convert results to a DataFrame
    df_results = pd.DataFrame(results)
    print("MMS Done!")
    return df_results