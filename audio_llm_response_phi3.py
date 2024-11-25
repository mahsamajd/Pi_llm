import os
import subprocess
import re
import json

# Path to the whisper.cpp main executable
whisper_main_path = '/work/mahsam/IJCAI/whisper.cpp/main'

# Specify the Whisper models you want to use
model_files = [
    '/work/mahsam/IJCAI/whisper.cpp/models/ggml-medium.en.bin',
    '/work/mahsam/IJCAI/whisper.cpp/models/ggml-base.en.bin',
    '/work/mahsam/IJCAI/whisper.cpp/models/ggml-small.en.bin',
    '/work/mahsam/IJCAI/whisper.cpp/models/ggml-tiny.en.bin'
]

# Check if the specified Whisper models exist
for model_path in model_files:
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Whisper model not found: {model_path}")

print(f"Using Whisper models: {model_files}")

# Directories for audio and JSON files
audio_directory = '/work/mahsam/IJCAI/squad_audio_questions'
audio_files = sorted([os.path.join(audio_directory, file) for file in os.listdir(audio_directory) if file.endswith('.wav')])

# Output directory for Llama-3.2-3B results
output_directory = '/work/mahsam/IJCAI/audio_llm_Phi-3_results'
os.makedirs(output_directory, exist_ok=True)

# Paths to the Llama-3.2-3B models
llama_models = {
    "phi-3": "/work/mahsam/IJCAI/llama.cpp/models/phi-3.gguf",
    "phi-3-Q8_0": "/work/mahsam/IJCAI/llama.cpp/models/phi-3-Q8_0.gguf",
    "phi-3-Q4_K_M": "/work/mahsam/IJCAI/llama.cpp/models/phi-3-Q4_K_M.gguf",
    "phi-3-Q2_k": "/work/mahsam/IJCAI/llama.cpp/models/phi-3-Q2_k.gguf"
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
    """Transcribes audio using the specified Whisper model."""
    result = subprocess.run([whisper_main_path, '-m', whisper_model, '-f', audio_file], capture_output=True, text=True)
    transcription = result.stdout.strip()
    transcription = re.sub(r'\[\d{2}:\d{2}:\d{2}\.\d{3} --> \d{2}:\d{2}:\d{2}\.\d{3}\]', '', transcription).strip()
    return transcription

def run_llm(model_path, transcription, max_tokens=50):
    """Runs the Llama model using llama-cli with the provided transcription."""
    prompt = f"""Based on the following transcription, provide a detailed and specific answer. Make sure your response fully addresses the information given:

Transcription: {transcription}

Answer:"""

    llama_cli_path = "/work/mahsam/IJCAI/llama.cpp/llama-cli"
    if not os.path.exists(llama_cli_path):
        print(f"Error: llama-cli not found at {llama_cli_path}")
        return None

    llama_command = [
        llama_cli_path,
        "--model", model_path,
        "--prompt", prompt,
        "--n_predict", str(max_tokens)
    ]

    print(f"Running LLM command: {' '.join(llama_command)}")
    result = subprocess.run(llama_command, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Error during LLM inference: {result.stderr}")
        return None

    response = result.stdout.strip()
    response_parts = response.split("Answer:", 1)
    return response_parts[1].strip() if len(response_parts) > 1 else response

# Main processing loop
for whisper_model in model_files:
    whisper_model_name = os.path.splitext(os.path.basename(whisper_model))[0]
    print(f"\nUsing Whisper model: {whisper_model_name}")

    def extract_number(filename):
        """Extracts the question number from the filename."""
        match = re.search(r'question_(\d+)\.wav', filename)
        return int(match.group(1)) if match else float('inf')

    # Sort the audio files numerically
    audio_files_sorted = sorted(audio_files, key=extract_number)

    for i, audio in enumerate(audio_files_sorted):
        print(f"Processing audio file: {audio}")
        transcription = transcribe_audio(audio, whisper_model)
        print(f"\nTranscribed Text {i + 1} with {whisper_model_name}: {transcription}")

        # Get specific fields from the JSON data
        qa_pair = qa_data[i] if i < len(qa_data) else {"context": transcription, "question": "No question found"}
        question = qa_pair.get("question", "No question found")
        correct_answer = qa_pair.get("answers", "No correct answer provided")

        for model_name, model_path in llama_models.items():
            llm_response = run_llm(model_path, transcription)

            if not llm_response:
                print(f"No response from model {model_name} for audio: {audio}")
                continue

            cleaned_response = llm_response
            print(f"Model: {model_name}")
            print(f"Transcription: {transcription}")
            print(f"Response: {cleaned_response}")
            print(f"Correct Answer: {qa_data[i]['answers']}")

            model_output_file = os.path.join(output_directory, f"result_{whisper_model_name}_{model_name}.txt")

            # Save detailed results to a file
            with open(model_output_file, 'a') as file:
                file.write(f"Whisper Model: {whisper_model_name}\n")
                file.write(f"LLM Model: {model_name}\n")
                file.write(f"Data: {qa_data[i]}\n")
                file.write(f"Transcription: {transcription}\n")
                file.write(f"Response: {cleaned_response}\n")
                file.write(f"Correct Answer: {qa_data[i]['answers']}\n\n")

            print(f"Results saved to {model_output_file}")
