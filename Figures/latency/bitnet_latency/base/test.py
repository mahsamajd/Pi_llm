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

def process_files(directory):
    models = {}
    
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            model_name = filename.split('_')[-1].split('.')[0]
            
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
