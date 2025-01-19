import os
import numpy as np
import pandas as pd
import json

def merge_aligners_data(output_conll, gentle_output, output_dir, base_name):
    """
    Merge MFA and Gentle alignment data using MFA as baseline.
    
    Args:
        output_conll (str): Path to MFA output conll file with timestamps
        gentle_output (str): Path to Gentle JSON output
        
    Returns:
        pandas.DataFrame: Contains columns:
            - word: The word
            - mfa_start_time: Start time from MFA
            - mfa_end_time: End time from MFA  
            - gentle_start_time: Start time from Gentle (if matched)
            - gentle_end_time: End time from Gentle (if matched)
    """
    merged_csv = os.path.join(output_dir, f"{base_name}_merged.csv")
    # Parse MFA conll file
    mfa_words = []
    with open(output_conll, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.strip().split('\t')
            if len(parts) < 15:  # Check if line has timestamps
                continue
            word = parts[3]
            start_time = parts[13]
            end_time = parts[14]
            
            if start_time and end_time:  # Only add if timestamps exist
                mfa_words.append({
                    'word': word.lower(),
                    'mfa_start_time': float(start_time),
                    'mfa_end_time': float(end_time)
                })
    
    # Create MFA DataFrame
    mfa_df = pd.DataFrame(mfa_words)
    
    # Parse Gentle JSON
    with open(gentle_output, 'r', encoding='utf-8') as f:
        gentle_data = json.load(f)
    
    gentle_words = []
    for word in gentle_data.get('words', []):
        if word.get('case') == 'success':
            gentle_words.append({
                'word': word.get('alignedWord', '').lower(),
                'gentle_start_time': word.get('start'),
                'gentle_end_time': word.get('end')
            })
    
    gentle_df = pd.DataFrame(gentle_words)
    
    # Add empty columns for gentle timestamps
    mfa_df['gentle_start_time'] = None
    mfa_df['gentle_end_time'] = None
    
    # Match gentle words to MFA words
    for idx, mfa_row in enumerate(mfa_df.itertuples()):
        gentle_match = gentle_df[gentle_df['word'] == mfa_row.word]
        
        if not gentle_match.empty:
            mfa_df.at[idx, 'gentle_start_time'] = gentle_match.iloc[0]['gentle_start_time']
            mfa_df.at[idx, 'gentle_end_time'] = gentle_match.iloc[0]['gentle_end_time']
    
    analysis = analyze_alignment_differences(mfa_df)
    
    mfa_df.to_csv(merged_csv, index=False)
    analysis_file = os.path.join(output_dir, f"{base_name}_alignment_analysis.json")
    with open(analysis_file, 'w', encoding='utf-8') as f:
        # Convert DataFrame to dict for JSON serialization
        analysis['unmatched'] = analysis['unmatched'].to_dict(orient='records')
        json.dump(analysis, f, indent=2)
    
    return merged_csv, analysis_file

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

















def analyze_alignment_differences(df):
    """Analyze differences between MFA and Gentle alignments."""
    # Get matched entries
    matched_df = df.dropna(subset=['gentle_start_time', 'gentle_end_time'])
    
    # Calculate differences for matched entries
    start_diffs = matched_df['gentle_start_time'] - matched_df['mfa_start_time']
    end_diffs = matched_df['gentle_end_time'] - matched_df['mfa_end_time']
    
    # Calculate statistics
    stats = {
        'start_time': {
            'mean_diff': start_diffs.mean(),
            'std_diff': start_diffs.std(),
            'max_diff': start_diffs.max(),
            'min_diff': start_diffs.min()
        },
        'end_time': {
            'mean_diff': end_diffs.mean(),
            'std_diff': end_diffs.std(),
            'max_diff': end_diffs.max(),
            'min_diff': end_diffs.min()
        },
        'total_words': len(df),
        'matched_words': len(matched_df),
        'match_rate': len(matched_df) / len(df) * 100
    }
    
    # Get unmatched entries
    unmatched_df = df[df['gentle_start_time'].isna()].copy()
    unmatched_df['position'] = range(len(unmatched_df))
    
    # Create matched words analysis
    matched_words_analysis = matched_df.apply(
        lambda row: {
            'word': row['word'],
            'position': row.name,
            'mfa_start': row['mfa_start_time'],
            'mfa_end': row['mfa_end_time'],
            'gentle_start': row['gentle_start_time'],
            'gentle_end': row['gentle_end_time'],
            'start_diff': row['gentle_start_time'] - row['mfa_start_time'],
            'end_diff': row['gentle_end_time'] - row['mfa_end_time']
        }, axis=1).tolist()
    
    return {
        'stats': stats,
        'unmatched': unmatched_df[['position', 'word', 'mfa_start_time', 'mfa_end_time']],
        'matched_words': matched_words_analysis
    }
