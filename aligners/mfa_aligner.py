import os
import subprocess
import re

from analysis.alignment_analysis import align_sequences

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
    print(f"Input: {input_dir}, Output: {output_dir}")
    # Construct the MFA command

    command = [
        'mfa', 'align',
        '--clean',  # Clean existing alignments
        '--single_speaker',  # Avoid speaker issues
        input_dir,
        'english_mfa',
        'english_mfa',
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

def add_timestamps_to_conll(conll_path, textgrid_intervals, output_path):
    """Add timestamps from TextGrid intervals to a CoNLL file and create alignment report."""
    import pandas as pd
    import os

    # Define output paths
    output_dir = os.path.dirname(output_path)
    base_name = os.path.splitext(os.path.basename(output_path))[0]
    alignment_report_path = os.path.join(output_dir, f"{base_name}_alignment_report.txt")

    # Read CoNLL file
    column_names = ['col0', 'col1', 'col2', 'word', 'col4', 'col5', 'col6', 
                   'col7', 'col8', 'col9', 'col10', 'col11', "col12"]
    
    conll_data = pd.read_csv(
        conll_path,
        sep="\t",
        header=None,
        names=column_names,
        skip_blank_lines=True,
    )

    # Extract words and align
    conll_words = conll_data['word'].astype(str).tolist()
    aligned_conll, aligned_textgrid, aligned_times = align_sequences(conll_words, textgrid_intervals)

    # Prepare alignment report
    alignment_issues = []
    for idx, (c_word, tg_word) in enumerate(zip(aligned_conll, aligned_textgrid)):
        if c_word != tg_word and c_word != '-' and tg_word != '-':
            alignment_issues.append(
                f"Mismatch at position {idx}: CoNLL word '{c_word}' vs. TextGrid word '{tg_word}'"
            )
        elif c_word == '-' or tg_word == '-':
            alignment_issues.append(
                f"Insertion/Deletion at position {idx}: CoNLL word '{c_word}' vs. TextGrid word '{tg_word}'"
            )

    # Write alignment report
    with open(alignment_report_path, 'w', encoding='utf-8') as f:
        f.write("Alignment Report\n")
        f.write("================\n\n")
        f.write(f"Total words in CoNLL: {len(conll_words)}\n")
        f.write(f"Total words in TextGrid: {len(textgrid_intervals)}\n")
        f.write(f"Total alignment issues: {len(alignment_issues)}\n\n")
        f.write("Detailed Issues:\n")
        f.write("---------------\n")
        for issue in alignment_issues:
            f.write(f"{issue}\n")

    # Prepare timestamps for CoNLL file
    start_times = []
    end_times = []
    for conll_word, times in zip(aligned_conll, aligned_times):
        if conll_word != '-':
            start_times.append(times[0])
            end_times.append(times[1])

    # Add timestamps to DataFrame
    conll_data = conll_data.loc[conll_data['word'] != '-'].reset_index(drop=True)
    conll_data['start_time'] = start_times
    conll_data['end_time'] = end_times

    # Save timestamped CoNLL
    conll_data.to_csv(output_path, sep="\t", index=False, header=False)

    return output_path, alignment_report_path