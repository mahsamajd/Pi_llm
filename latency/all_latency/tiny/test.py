import os
import csv
import re

def extract_metrics(file_content):
    metrics = {}
    patterns = {
        'Transcription Time': r'Transcription Time: (\d+\.\d+) seconds',
        'Words per Second': r'Words per Second: (\d+\.\d+) WPS',
        'Inference Time': r'Inference Time: (\d+\.\d+) seconds',
        'Tokens per Second': r'Tokens per Second: (\d+\.\d+) TPS'
    }
    
    for metric, pattern in patterns.items():
        match = re.search(pattern, file_content)
        if match:
            metrics[metric] = float(match.group(1))
    
    return metrics

def get_model_name(filename):
    parts = filename.split('_')
    if 'gemma2' in filename:
        return '_'.join(parts[-2:]).split('.')[0]
    elif 'llama3.2' in filename:
        return 'llama3.2' if 'llama3.2.txt' in filename else 'llama3.2_1b'
    elif 'bitnet' in filename:
        return 'bitnet_b1_58-large'
    elif 'Llama3-8B' in filename:
        return 'Llama3-8B-1.58-100B-tokens'
    elif 'phi3_mini' in filename:
        return 'phi3_mini'
    else:
        return parts[-1].split('.')[0]

def process_files(directory):
    models = {}
    
    for filename in os.listdir(directory):
        if filename.startswith('results_') and filename.endswith('.txt'):
            model_name = get_model_name(filename)
            
            if model_name not in models:
                models[model_name] = {
                    'Transcription Time': [],
                    'Words per Second': [],
                    'Inference Time': [],
                    'Tokens per Second': []
                }
            
            with open(os.path.join(directory, filename), 'r') as file:
                content = file.read()
                metrics = extract_metrics(content)
                
                for metric, value in metrics.items():
                    models[model_name][metric].append(value)
    
    return models

def calculate_averages(models):
    averages = {}
    
    for model, metrics in models.items():
        averages[model] = {}
        for metric, values in metrics.items():
            if values:
                averages[model][metric] = sum(values) / len(values)
            else:
                averages[model][metric] = 0
    
    return averages

def save_to_csv(averages, output_file):
    with open(output_file, 'w', newline='') as csvfile:
        fieldnames = ['Model', 'Avg Transcription Time', 'Avg Words per Second', 'Avg Inference Time', 'Avg Tokens per Second']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for model, metrics in averages.items():
            writer.writerow({
                'Model': model,
                'Avg Transcription Time': f"{metrics['Transcription Time']:.2f}",
                'Avg Words per Second': f"{metrics['Words per Second']:.2f}",
                'Avg Inference Time': f"{metrics['Inference Time']:.2f}",
                'Avg Tokens per Second': f"{metrics['Tokens per Second']:.2f}"
            })

# Main execution
directory = '.'  # Current directory, change if needed
output_file = 'model_averages.csv'

models = process_files(directory)
averages = calculate_averages(models)
save_to_csv(averages, output_file)

print(f"Averages have been saved to {output_file}")

