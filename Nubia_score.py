import os
import json
from nubia_score import Nubia

# Initialize NUBIA
nubia = Nubia()

def get_nubia_score(text1, text2):
    try:
        return nubia.score(text1, text2)
    except Exception as e:
        print(f"Error calculating NUBIA score: {e}")
        return None

def parse_entry(entry):
    lines = entry.strip().split('\n')
    data = {}
    for line in lines:
        if ':' in line:
            key, value = line.split(':', 1)
            key = key.strip()
            value = value.strip()
            if key == 'Data':
                try:
                    data[key] = json.loads(value.replace("'", '"'))
                except json.JSONDecodeError:
                    data[key] = value
            else:
                data[key] = value
    return data

# Define the directories
model_directories = {
    #'bitnet': '/work/mahsam/IJCAI/audio_llm_bitnet_results',
    #'gemma': '/work/mahsam/IJCAI/audio_llm_gemma_old_results'
    'llama3_8B': '/work/mahsam/IJCAI/audio_llm_llama3-8B_results',
    'llama3_1B': '/work/mahsam/IJCAI/audio_llm_Llama3.2-1B_results',
    'llama3_3B': '/work/mahsam/IJCAI/audio_llm_Llama3.2-3B_results',
    'phi3': '/work/mahsam/IJCAI/audio_llm_Phi-3_results'
}

output_base_folder = '/work/mahsam/IJCAI/nubia_scores_results'
os.makedirs(output_base_folder, exist_ok=True)

# Iterate over each model directory
for model_name, txt_directory in model_directories.items():
    output_folder = os.path.join(output_base_folder, model_name)
    os.makedirs(output_folder, exist_ok=True)

    txt_files = [f for f in os.listdir(txt_directory) if f.endswith('.txt')]

    for txt_file in txt_files:
        txt_file_path = os.path.join(txt_directory, txt_file)
        output_file = os.path.join(output_folder, f'nubia_scores_{txt_file}')

        print(f"Processing TXT file: {txt_file_path} for model: {model_name}")

        results = []

        with open(txt_file_path, 'r', encoding='utf-8') as txtfile:
            content = txtfile.read()
            entries = content.split('\n\n')  # Assuming entries are separated by blank lines
            
            for entry in entries:
                try:
                    data = parse_entry(entry)
                    question = data.get('Data', {}).get('question', '').strip()
                    response = data.get('Response', '').strip()
                    correct_answer = str(data.get('Correct Answer', '')).strip()

                    if question and response and correct_answer:
                        combined_text = f"Question: {question}\nCorrect Answer: {correct_answer}"
                        score = get_nubia_score(combined_text, response)

                        if score is not None:
                            result = f"Question: {question}\nResponse: {response}\nCorrect Answer: {correct_answer}\nNUBIA Score: {score}\n\n"
                            results.append(result)
                            print(f"Processed question: {question[:30]}... for model: {model_name}")
                    else:
                        print(f"Skipping incomplete entry: {data}")
                except Exception as e:
                    print(f"Error processing entry in {txt_file}: {e}")

        if results:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.writelines(results)
            print(f"NUBIA scores saved to {output_file}")
        else:
            print(f"No results to write for {txt_file}. Check your input TXT file and NUBIA installation.")

        print(f"Total processed entries for {txt_file} in model {model_name}: {len(results)}")

print("Processing complete for all model directories.")
