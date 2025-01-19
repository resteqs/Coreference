from utils.file_utils import process_files, convert_conll_to_txt, format_audio_for_mfa
from utils.docker_utils import start_gentle_container
from aligners.mfa_aligner import run_mfa_align, extract_textgrid_intervals, add_timestamps_to_conll
from aligners.gentle_aligner import run_gentle_align
from analysis.alignment_analysis import merge_aligners_data
import os

def main(input_dir):
    try:
        # Get all file pairs
        file_pairs = process_files(input_dir)
        print(f"Found {len(file_pairs)} matching audio/transcript pairs")

        # Process each pair
        for pair in file_pairs:
            print(f"\nProcessing {pair['base_name']}...")
            
            # Convert and get txt path
            txt_file = convert_conll_to_txt(pair['conll'], pair['input_dir'], pair['base_name'])
            format_audio_for_mfa(pair['audio'], os.path.join(pair['output_dir'], f"{pair['base_name']}_formatted.wav"))
            # Run MFA
            run_mfa_align(pair['input_dir'], pair['output_dir'])
            # Get TextGrid path and extract intervals
            textgrid_file = os.path.join(pair['output_dir'], f"{pair['base_name']}.TextGrid")
            intervals = extract_textgrid_intervals(textgrid_file)
            
            # Add timestamps to new CoNLL
            output_conll = os.path.join(pair['output_dir'], f"{pair['base_name']}_time.conll")
            add_timestamps_to_conll(pair['conll'], intervals, output_conll)
            
            # Run Gentle
            start_gentle_container(pair['input_dir'])
            gentle_json = run_gentle_align(pair['audio'], txt_file, pair['output_dir'], pair['base_name'])
            
            # Merge alignments
            if gentle_json:
                merged_csv, analysis_file = merge_aligners_data(
                    output_conll, 
                    gentle_json,
                    pair['output_dir'],
                    pair['base_name']
                )
                print(f"Successfully processed {pair['base_name']}. Output saved in: {pair['output_dir']}")
            
    except Exception as e:
        print(f"Error processing files: {e}")

if __name__ == "__main__":
    main("C:/Users/manto/Documents/Coreference/.test")