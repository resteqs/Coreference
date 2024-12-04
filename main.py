import os
import numpy as np
import re
import pandas as pd
import argparse
import subprocess
import tempfile
import shutil
import whisper
from pydub import AudioSegment
from tqdm import tqdm
import csv
import time
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import torch
import librosa
from transformers import AutoProcessor, AutoModelForCTC

def clean_text(text):
    """Remove symbols and special characters from text."""
    return re.sub(r'[^a-zA-Z0-9\s\']', '', text)

def convert_conll_to_txt(conll_file, txt_file):
    """Convert a CoNLL file to a cleaned text file.

    - Extracts words from the CoNLL file.
    - Cleans the text by removing unwanted characters and extra spaces.
    - Writes the cleaned text to an output file.

    Args:
        conll_file (str): Path to the input CoNLL file.
        txt_file 
    """
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


def run_mfa_align(input_dir, output_dir):
    """"Run the Montreal Forced Aligner (MFA) on the input files.

    - Aligns the audio and text files in the input directory.
    - Generates a TextGrid file with alignment data.
    - Saves the TextGrid file in the output directory.

    Args:
        input_dir (str): Path to the directory containing input files.
        output_dir (str): Path to the directory where output files will be saved.
    """
    # Ensure directories exist
    if not os.path.exists(input_dir):
        print(f"Input directory does not exist: {input_dir}")
        return
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Construct the MFA command
    command = [
        'mfa', 'align',
        input_dir,
        'english_mfa',  # Specify the pronunciation dictionary
        'english_mfa',  # Specify the acoustic model
        output_dir
    ]

    try:
        subprocess.run(command, check=True)
        print("Alignment completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error during alignment: {e}")
    except PermissionError as e:
        print(f"Permission error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def extract_textgrid_intervals(textgrid_path):
    """Extract word intervals from a TextGrid file.

    - Parses the TextGrid file to extract words and their timing intervals.
    - Collects intervals where words are present and skips empty intervals.

    Args:
        textgrid_file (str): Path to the TextGrid file.

    Returns:
        list: A list of dictionaries with the following keys:
            - 'word' (str): The word from the interval.
            - 'start_time' (float): Start time of the interval.
            - 'end_time' (float): End time of the interval.
    """
    word_intervals = []
    with open(textgrid_path, 'r' ,encoding='utf-8') as file:
        lines = file.readlines()
    
    is_word_tier = False
    for i, line in enumerate(lines):
        if 'name = "words"' in line:
            is_word_tier = True
            continue
        
        # Extract interval data within the words tier with some regex magic
        if is_word_tier:
            if 'xmin' in line:
                xmin = float(re.search(r"[-+]?\d*\.\d+|\d+", line).group())
            elif 'xmax' in line:
                xmax = float(re.search(r"[-+]?\d*\.\d+|\d+", line).group())
            elif 'text' in line:
                text_match = re.search(r'text = "(.*)"', line)
                text = text_match.group(1).strip() if text_match else ""
                
                # We skip empty text
                if text:
                    word_intervals.append({
                        "word": text,
                        "start_time": xmin,
                        "end_time": xmax
                    })
            
            if 'name' in line and 'name = ' in line and not 'name = "words"' in line:
                break
    
    return word_intervals


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




def align_sequences(conll_words, textgrid_intervals): #Using the Needleman-Wunsch algortihm for alligning sequences, as the files don't fit perfectly
    """Aligns words from the CoNLL file with words from the TextGrid intervals using the *Needleman-Wunsch* algorithm.

    Steps:
    1. **Initialization:**
       - Create a scoring matrix (`score_matrix`) of size (n+1) x (m+1), where n is the length of `conll_words` and m is the length of `textgrid_words`.
       - Initialize the first row and column with cumulative gap penalties to represent alignments with leading gaps.

    2. **Scoring Matrix Construction:**
       - Iterate over each cell in the matrix to compute the optimal score based on:
         - **Match:** If the current words from `conll_words` and `textgrid_words` match, add a positive `match_score`.
         - **Mismatch:** If the words do not match, add a negative `mismatch_penalty`.
         - **Insertion/Deletion (Gaps):** Assign a negative `gap_penalty` for insertions or deletions.
       - Calculate scores for:
         - **Diagonal move (match/mismatch):** `score_matrix[i-1][j-1] + score`
         - **Up move (deletion in TextGrid):** `score_matrix[i-1][j] + gap_penalty`
         - **Left move (insertion in TextGrid):** `score_matrix[i][j-1] + gap_penalty`
       - Select the move with the highest score and store it in `score_matrix[i][j]`.
       - Record the move direction in the `traceback_matrix` for later path reconstruction.

    3. **Traceback:**
       - Start from the bottom-right cell of the matrices (`score_matrix[n][m]`).
       - Reconstruct the optimal alignment by moving in the direction indicated by `traceback_matrix`:
         - **'diag':** Align the current words from both sequences.
         - **'up':** Align the word from `conll_words` with a gap (deletion in `textgrid_words`).
         - **'left':** Align a gap with the word from `textgrid_words` (insertion in `conll_words`).
       - Continue tracing back until reaching the top-left cell.

    4. **Results:**
       - Obtain `aligned_conll` and `aligned_textgrid`, which are the aligned sequences including gaps ('-') where necessary.
       - Collect corresponding time intervals from `textgrid_intervals` for aligned words.
       - Reverse the aligned sequences and times to obtain the correct order.

    Parameters:
        conll_words (list): List of words extracted from the CoNLL file.
        textgrid_intervals (list): List of dictionaries from the TextGrid file, each containing:
                                   - "word": The word string.
                                   - "start_time": Start time of the interval.
                                   - "end_time": End time of the interval.

    Returns:
        aligned_conll (list): Aligned words from the CoNLL file, including gaps.
        aligned_textgrid (list): Aligned words from the TextGrid intervals, including gaps.
        aligned_times (list): List of (start_time, end_time) tuples corresponding to `aligned_textgrid`."""


    # lets get Textgrid words and tinestamps
    textgrid_words = [str(interval["word"]) for interval in textgrid_intervals]
    textgrid_times = [(interval["start_time"], interval["end_time"]) for interval in textgrid_intervals]

    # Initialize scoring parameters
    match_score = 2
    mismatch_penalty = -1
    gap_penalty = -2

    n = len(conll_words)
    m = len(textgrid_words)

    # Initialize the scoring matrix
    score_matrix = np.zeros((n + 1, m + 1))
    traceback_matrix = np.zeros((n + 1, m + 1), dtype='object')

    # Initialize first row and column
    for i in range(1, n + 1):
        score_matrix[i][0] = gap_penalty * i
        traceback_matrix[i][0] = 'up'  # Deletion
    for j in range(1, m + 1):
        score_matrix[0][j] = gap_penalty * j
        traceback_matrix[0][j] = 'left'  # Insertion

    # Fill in the scoring matrix
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            word1 = conll_words[i - 1].lower()
            word2 = textgrid_words[j - 1].lower()

            if word1 == word2:
                score = match_score
            else:
                score = mismatch_penalty

            diag_score = score_matrix[i - 1][j - 1] + score  # Match/Mismatch
            up_score = score_matrix[i - 1][j] + gap_penalty   # Deletion
            left_score = score_matrix[i][j - 1] + gap_penalty # Insertion

            max_score = max(diag_score, up_score, left_score)
            score_matrix[i][j] = max_score

            # Traceback pointers
            if max_score == diag_score:
                traceback_matrix[i][j] = 'diag'
            elif max_score == up_score:
                traceback_matrix[i][j] = 'up'
            else:
                traceback_matrix[i][j] = 'left'

    # Traceback to get the alignment
    aligned_conll = []
    aligned_textgrid = []
    aligned_times = []
    i, j = n, m

    while i > 0 or j > 0:
        direction = traceback_matrix[i][j]

        if direction == 'diag':
            aligned_conll.append(conll_words[i - 1])
            aligned_textgrid.append(textgrid_words[j - 1])
            aligned_times.append(textgrid_times[j - 1])
            i -= 1
            j -= 1
        elif direction == 'up':
            aligned_conll.append(conll_words[i - 1])
            aligned_textgrid.append('-')  # Gap in TextGrid
            aligned_times.append((None, None))
            i -= 1
        else:  # 'left'
            aligned_conll.append('-')  # Gap in CoNLL
            aligned_textgrid.append(textgrid_words[j - 1])
            aligned_times.append(textgrid_times[j - 1])
            j -= 1

    # Reverse the aligned sequences
    aligned_conll = aligned_conll[::-1]
    aligned_textgrid = aligned_textgrid[::-1]
    aligned_times = aligned_times[::-1]

    return aligned_conll, aligned_textgrid, aligned_times



def add_timestamps_to_conll(conll_path, textgrid_intervals, output_path):
    """Add timestamps from TextGrid intervals to a CoNLL file.

    - Aligns words from the CoNLL file with the intervals from the TextGrid.
    - Adds start and end times to each word in the CoNLL file.
    - Writes the updated CoNLL data to a new file.

    Args:
        conll_file (str): Path to the original CoNLL file.
        intervals (list): List of intervals extracted from the TextGrid file.
        output_conll (str): Path to the output CoNLL file with timestamps.
    """
    import pandas as pd

    # Define column names matching your CoNLL file structure
    column_names = ['col0', 'col1', 'col2', 'word', 'col4', 'col5', 'col6', 'col7', 'col8', 'col9', 'col10', 'col11', "col12"]

    # Read the CoNLL file using the adjusted column names
    conll_data = pd.read_csv(
        conll_path,
        sep="\t",
        header=None,
        names=column_names,
        skip_blank_lines=True,
    )

    # Extract the 'word' column
    conll_words = conll_data['word'].astype(str).tolist()

    # Rest of your code remains the same...
    # Align sequences using the conll_words
    aligned_conll, aligned_textgrid, aligned_times = align_sequences(conll_words, textgrid_intervals)

    # Prepare the output data
    start_times = []
    end_times = []
    for conll_word, times in zip(aligned_conll, aligned_times):
        if conll_word != '-':
            start_times.append(times[0])  # May be None
            end_times.append(times[1])    # May be None

    # Add the times to the DataFrame
    conll_data = conll_data.loc[conll_data['word'] != '-'].reset_index(drop=True)
    conll_data['start_time'] = start_times
    conll_data['end_time'] = end_times

    # Save the updated data to the output file
    conll_data.to_csv(output_path, sep="\t", index=False, header=False)

    # Optionally, print or log any mismatches or alignments
    for idx, (c_word, tg_word) in enumerate(zip(aligned_conll, aligned_textgrid)):
        if c_word != tg_word and c_word != '-' and tg_word != '-':
            print(f"Mismatch at position {idx}: CoNLL word '{c_word}' vs. TextGrid word '{tg_word}'")
        elif c_word == '-' or tg_word == '-':
            print(f"Insertion/Deletion at position {idx}: CoNLL word '{c_word}' vs. TextGrid word '{tg_word}'")


def process_files(input_dir, output_dir=None):
    """Process files in the input directory and prepare them for alignment.

    Steps:
    - Check for the required files (.conll and .wav) in the input directory.
    - Use the output directory if provided; otherwise, default to input directory.

    Args:
        input_dir (str): Path to the directory containing input files.
        output_dir (str): Path to the directory where output files will be saved.

    Returns:
        tuple: Contains paths to the following files:
            - conll_file (str): Path to the CoNLL file.
            - audio_file (str): Path to the audio file.
            - txt_file (str): Path to the cleaned text file.
            - textgrid_file (str): Path to the TextGrid file generated by MFA.
            - output_conll (str): Path to the output CoNLL file with timestamps.
    """
    # Normalize and validate paths
    input_dir = os.path.abspath(input_dir)
    output_dir = os.path.abspath(output_dir) if output_dir else input_dir

    # Validate input directory
    if not os.path.isdir(input_dir):
        raise ValueError(f"Input directory does not exist: {input_dir}")

    # Find CoNLL file
    conll_files = [f for f in os.listdir(input_dir) if f.endswith('.conll')]
    if not conll_files:
        raise FileNotFoundError("No .conll file found in input directory")
    
    # Get base name and construct file paths
    base_name = os.path.splitext(conll_files[0])[0]
    
    # Construct file paths using absolute paths
    conll_file = os.path.join(input_dir, conll_files[0])
    audio_file = os.path.join(input_dir, f"{base_name}.wav")
    txt_file = os.path.join(output_dir, f"{base_name}.txt")
    textgrid_file = os.path.join(output_dir, f"{base_name}.TextGrid")
    output_conll = os.path.join(output_dir, f"{base_name}_time.conll")

    # Validate input files exist
    if not os.path.isfile(audio_file):
        raise FileNotFoundError(f"Audio file not found: {audio_file}")
    if not os.path.isfile(conll_file):
        raise FileNotFoundError(f"CoNLL file not found: {conll_file}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    return conll_file, audio_file, txt_file, textgrid_file, output_conll

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



input_dir = "C:/Users/manto/Documents/Coreference-project-NLP/test3"
output_dir = ""

if (output_dir == ""):
    output_dir = input_dir

try:
        conll_file, audio_file, txt_file, textgrid_file, output_conll = process_files(input_dir, output_dir)
        convert_conll_to_txt(conll_file, txt_file) #Uses clean_txt() for removing some chars
        run_mfa_align(input_dir, output_dir) #command = ['mfa', 'align', input_dir, (pronounciation model) 'english_mfa',  (accoustic model) english_mfa', output_dir]
        intervals = extract_textgrid_intervals(textgrid_file) # Process TextGrid and create time-aligned CoNLL
        add_timestamps_to_conll(conll_file, intervals, output_conll) #Adds timestamps from MFA to output_conll file
        mmsFrame = verify_mfa_alignment_mms(audio_file, intervals)
        #whisperFrame = verify_mfa_alignment_whisper(audio_file, intervals, "base") #Verficiation using Whisper Returns a pandaframe
        print(f"Successfully processed files in {input_dir}.Output files saved in: {output_dir} ")
except Exception as e:
        print(f"Error processing files: {e}")

print(mmsFrame)