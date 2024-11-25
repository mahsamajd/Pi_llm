import os
import subprocess
import re
import sys
import json
from fuzzywuzzy import process
from fuzzywuzzy import fuzz

# Path to the whisper.cpp main executable
whisper_main_path = '/work/mahsam/IJCAI/whisper.cpp/main'

# Specify the Whisper models you want to use
model_files = [
    '/work/mahsam/IJCAI/whisper.cpp/models/ggml-small.en.bin',
    '/work/mahsam/IJCAI/whisper.cpp/models/ggml-tiny.en.bin',
    '/work/mahsam/IJCAI/whisper.cpp/models/ggml-medium.en.bin',
    '/work/mahsam/IJCAI/whisper.cpp/models/ggml-base.en.bin'
]

# Check if the specified Whisper models exist
for model_path in model_files:
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Whisper model not found: {model_path}")

print(f"Using Whisper models: {model_files}")

# Directories for audio and JSON files
audio_directory = '/work/mahsam/IJCAI/squad_audio_questions'
audio_files = sorted([os.path.join(audio_directory, file) for file in os.listdir(audio_directory) if file.endswith('.wav')])

output_directory = '/work/mahsam/IJCAI/audio_llm_bitnet_results'
os.makedirs(output_directory, exist_ok=True)

# Paths to the LLM models
bitnet_models = {
    # "bitnet_b1_58-3B": "/work/mahsam/IJCAI/BitNet/models/bitnet_b1_58-3B/ggml-model-i2_s.gguf",
    "bitnet_b1_58_large": "/work/mahsam/IJCAI/BitNet/models/bitnet_b1_58-large/ggml-model-i2_s.gguf",
    "llama3_8B": "/work/mahsam/IJCAI/BitNet/models/Llama3-8B-1.58-100B-tokens/ggml-model-i2_s.gguf"
}

# JSON file for questions and answers
json_file_path = '/work/mahsam/IJCAI/selected_questions.json'

def load_json_data(json_path):
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        print(f"JSON file not found: {json_path}")
        return []

qa_data = load_json_data(json_file_path)

def transcribe_audio(audio_file, whisper_model):
    result = subprocess.run([whisper_main_path, '-m', whisper_model, '-f', audio_file], capture_output=True, text=True)
    transcription = result.stdout.strip()
    transcription = re.sub(r'\[\d{2}:\d{2}:\d{2}\.\d{3} --> \d{2}:\d{2}:\d{2}\.\d{3}\]', '', transcription).strip()
    return transcription

# def run_llm(model_path, context, question, max_tokens=50):
def run_llm(model_path, transcription, max_tokens=50):
    prompt = f"""Based on the following transcription, provide a detailed and specific answer. Make sure your response fully addresses the information given:

Transcription: {transcription}

Answer:"""

    bitnet_script_path = "/work/mahsam/IJCAI/BitNet/run_inference.py"
    if not os.path.exists(bitnet_script_path):
        print(f"Error: run_inference.py not found at {bitnet_script_path}")
        return None

    llm_command = [
        sys.executable,
        bitnet_script_path,
        "-m", model_path,
        "-p", prompt,
        "-n", str(max_tokens),
        "--temp", "0"
    ]
    print(f"Running LLM command: {' '.join(llm_command)}")
    result = subprocess.run(llm_command, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Error during LLM inference: {result.stderr}")
        return None

    full_response = result.stdout.strip()
    response_parts = full_response.split("Answer:", 1)
    response = response_parts[1].strip() if len(response_parts) > 1 else ""
    return response

def clean_response(response):
    response = re.sub(r'^[A-D]\.\s*', '', response)
    response = response.split('.')[0] + '.'
    return response.strip()

# Main processing loop
for whisper_model in model_files:
    whisper_model_name = os.path.splitext(os.path.basename(whisper_model))[0]
    #print(f"\nUsing Whisper model: {os.path.basename(whisper_model)}")
    print(f"\nUsing Whisper model: {whisper_model_name}")    


    def extract_number(filename):
        # Extract the number using a regex pattern
        match = re.search(r'question_(\d+)\.wav', filename)
        return int(match.group(1)) if match else float('inf')

    # Sort the audio files list numerically
    audio_files_sorted = sorted(audio_files, key=extract_number)


    for i, audio in enumerate(audio_files_sorted):
        print(f"audio : {audio}")
        transcription = transcribe_audio(audio, whisper_model)
        print(f"\nTranscribed Text {i + 1} with {os.path.basename(whisper_model)}: {transcription}")
        # print("aq_data",qa_data[i])
        
        
        # # Find the corresponding QA pair in the JSON file
        # qa_pair = qa_data[i] if i < len(qa_data) else {"context": transcription, "question": "No question found"}
        # context = qa_pair.get("context", transcription)
        # question = qa_pair.get("question", "Please summarize the main points.")

        for model_name, model_path in bitnet_models.items():
            # llm_response = run_llm(model_path, context, question)
            llm_response = run_llm(model_path, transcription)
            
            if not llm_response:
                print(f"No response from model {model_name} for audio: {audio}")
                continue

            cleaned_response = llm_response
            # correct_answer = qa_pair.get("answer", "No correct answer provided")
            print(f"Model: {model_name}")
            # print(f"Context: {context[:100]}...")
            # print(f"Question: {question}")
            print(f"Transcription: {transcription}")
            print(f"Response: {cleaned_response}")
            print(f"Correct Answer: {qa_data[i]['answers']}")

            # Create a separate output file for each model
            #model_output_file = os.path.join(output_directory, f"results_{model_name}.txt")
            model_output_file = os.path.join(output_directory, f"results_{whisper_model_name}_{model_name}.txt")

            # Append results to the model-specific output file
            with open(model_output_file, 'a') as file:
                #file.write(f"Whisper Model: {os.path.basename(whisper_model)}\n")
                file.write(f"Whisper Model: {whisper_model_name}\n")

                file.write(f"LLM Model: {model_name}\n")
                # file.write(f"Context: {context}\n")
                # file.write(f"Question: {question}\n")
                file.write(f"Data: {qa_data[i]}\n" )
                file.write(f"Transcription: {transcription}\n")
                file.write(f"Response: {cleaned_response}\n")
                file.write(f"Correct Answer: {qa_data[i]['answers']}\n\n")

            print(f"Results saved to {model_output_file}")
