import os
import pandas as pd
import numpy as np
import re
import librosa
import soundfile as sf

def process_files(input_dir, output_dir=None):
    """Setup directory structure and return list of core file paths for each matching pair."""
    input_dir = os.path.abspath(input_dir)
    
    # Create main results directory if it doesn't exist
    if output_dir is None:
        output_dir = os.path.join(input_dir, "results")
    os.makedirs(output_dir, exist_ok=True)

    # Find all .conll files
    conll_files = [f for f in os.listdir(input_dir) if f.endswith('.conll')]
    if not conll_files:
        raise FileNotFoundError("No .conll files found in input directory")

    # Find matching pairs and create output paths
    file_pairs = []
    for conll_file in conll_files:
        base_name = os.path.splitext(conll_file)[0]
        wav_file = f"{base_name}.wav"
        
        # Check if matching .wav exists
        if os.path.isfile(os.path.join(input_dir, wav_file)):
            # Create book-specific output directory
            book_output_dir = os.path.join(output_dir, base_name)
            os.makedirs(book_output_dir, exist_ok=True)
            
            file_pairs.append({
                'input_dir': input_dir,
                'output_dir': book_output_dir,
                'base_name': base_name,
                'conll': os.path.join(input_dir, conll_file),
                'audio': os.path.join(input_dir, wav_file)
            })

    if not file_pairs:
        raise FileNotFoundError("No matching .conll and .wav pairs found")

    return file_pairs

def convert_conll_to_txt(conll_file, output_dir, base_name):
    """Convert a CoNLL file to a cleaned text file.

    - Extracts words from the CoNLL file.
    - Cleans the text by removing unwanted characters and extra spaces.
    - Writes the cleaned text to an output file.

    Args:
        conll_file (str): Path to the input CoNLL file.
        txt_file 
    """
    txt_file = os.path.join(output_dir, f"{base_name}.txt")
    words = []
    with open(conll_file, 'r', encoding='utf-8') as f_in:
        for line in f_in:
            if line.strip():  # Non-empty line
                tokens = line.strip().split()
                if len(tokens) > 3:  # Ensure enough columns
                    word = clean_text(tokens[3])  # Clean the word
                    if word:  # Only add non-empty words
                        words.append(word)
            else:
                # Add sentence boundary
                if words:
                    words.append('\n')
    
    # Write cleaned text to output file
    with open(txt_file, 'w', encoding='utf-8') as f_out:
        f_out.write(' '.join(words))
    return txt_file
    
def clean_text(text):
    """Remove symbols and special characters from text."""
    return re.sub(r'[^a-zA-Z0-9\s\']', '', text)



def format_audio_for_mfa(input_audio, output_audio):
    """
    Convert audio file to MFA-compatible format
    
    Args:
        input_audio: Path to input audio file
        output_audio: Path to save formatted WAV file
    """
    # Load and resample audio to 16kHz
    y, sr = librosa.load(input_audio, sr=16000)
    # Save as 16-bit WAV
    sf.write(output_audio, y, 16000, subtype='PCM_16')