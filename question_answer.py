from constants import *
import json 
import os
import argparse
import ecg_plot
import wfdb
import numpy as np 
import torch 
from PIL import Image
import io
import matplotlib.pyplot as plt
from tqdm import tqdm

from model_loader import get_model_loader
from data_utils import get_dataset_loader

class Inferencer():
    def __init__(self, args):
        self.dir = args.dir if args.dir is not None else DIRECTORY
        self.save_dir = args.savedir if args.savedir is not None else OUTPUT_DIRECTORY
        self.dataset_list = args.dataset if args.dataset else DATASET_LIST
        self.model_list = args.model if args.model is not None else MODEL_LIST
        
        self.ecg_base_dir = args.ecg_base_dir 
        self.target_model = self.model_list[0]
    
            
    def ECGVisualizer(self, ecg, sampling_rate):
        ecg = ecg.numpy()
        ecg_plot.plot(ecg, sample_rate=sampling_rate)   

        fig = plt.gcf()
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        image = Image.open(buf).convert('RGB')
        plt.close(fig) 
        return image
        
    def QuestionAnswerer(self, prompt, ecg, ecg_image, model_name, loaded_model_instance):
        if loaded_model_instance:
            return loaded_model_instance.generate(prompt, ecg, ecg_image)
            
        elif model_name == "gpt":
            return "gpt_response" 
        elif model_name == "gemini":
            return "gemini_response" # Add API logic here
        else:
            return f"Error: Model {model_name} not loaded."
    
    def RetrieveAnswer(self, path, model_name, dataset_loader, loaded_model_instance):
        with open(path) as f:
            sample = json.load(f)
            ecg_id = sample["metadata"]["ecg_id"]
            
            try:
                ecg, sampling_rate = dataset_loader.load_signal(ecg_id)
                ecg_image = self.ECGVisualizer(ecg, sampling_rate)
            except Exception as e:
                print(f"Skipping {ecg_id}: {e}")
                return None    
            
            answer_list = []
            for data in sample["data"]:
                text = prompt.format(data["question"], data["options"])
                model_response = self.QuestionAnswerer(text, ecg, ecg_image, model_name, loaded_model_instance)
                if model_response:
                    data["response_raw"] = model_response.strip()
                else:
                    data["response_raw"] = "Error"
                    
                answer_list.append(data)

            sample["metadata"]["model"] = model_name
        
        return sample

    def main(self):
        path_map = {
            "ptbxl": "/nfs_edlab/hschung/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3",
            "mimic_iv_ecg": "path/to/mimic_iv_ecg" 
        }

        model_name = self.target_model
        
        
        # 1. Load Model (Once per process execution)
        current_model_instance = None
        # Add any other local model names to this list
        if model_name in ["gem", "pulse"]:
            try:
                current_model_instance = get_model_loader(model_name)
            except Exception as e:
                print(f"CRITICAL ERROR: Failed to load {model_name}. Exiting process.\nError: {e}")
                return

        # 2. Iterate over Datasets
        for dataset in self.dataset_list:
            print(f"\n--- Processing Dataset: {dataset} ---")

            target_dir = path_map.get(dataset, self.ecg_base_dir)
            print(f"Looking for ECG files in: {target_dir}")

            try:
                current_loader = get_dataset_loader(dataset, target_dir)
            except ValueError as e:
                print(f"Skipping dataset {dataset}: {e}")
                continue

            dataset_path = os.path.join(self.dir, dataset)
            if not os.path.exists(dataset_path):
                print(f"JSON Input directory not found: {dataset_path}")
                continue

            # List all JSON files
            total_path = [os.path.join(dataset_path, p) for p in os.listdir(dataset_path) if p.endswith('.json')]
            
            # 3. Run Inference on Files
            for path in tqdm(total_path, desc=f"Inferencing {dataset} with {model_name}"):
                answer_json = self.RetrieveAnswer(path, model_name, current_loader, current_model_instance)
                
                if answer_json:
                    save_path = os.path.join(self.save_dir, model_name, dataset, os.path.basename(path))
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    with open(save_path, "w") as f:
                        json.dump(answer_json, f, indent=4)

        print(f"\nFinished inference for {model_name}.")

        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-d', '--dir', help = "load directory")
    parser.add_argument('-s', '--savedir', help = "directory for saving result file")
    parser.add_argument('-ds', '--dataset', nargs='*', help = 'mimic_iv_ecg, ptbxl')
    # parser.add_argument('-dx', '--diagnosis', nargs='*', help = '"lateral_myocardial_infarction", "complete_right_bundle_branch_block", "left_posterior_fascicular_block", "premature_atrial_complex", "left_anterior_fascicular_block", "complete_left_bundle_branch_block", "third_degree_av_block", "first_degree_av_block", "inferior_ischemia", "lateral_ischemia", "left_ventricular_hypertrophy", "premature_ventricular_complex", "inferior_myocardial_infarction", "anterior_myocardial_infarction", "second_degree_av_block", "anterior_ischemia", "right_ventricular_hypertrophy"]')
    # parser.add_argument('-l', '--label', nargs='*', help = 'positive, negative')
    parser.add_argument('-m', '--model', nargs='*', help = '"gpt", "gemini", "qwen", "llama", "gem", "pulse", "opentslm", "llava-med", "medgemma","hulu-med"')
    parser.add_argument("--ecg-base-dir", type=str, default="/nfs_edlab/hschung/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3", help="Base directory for ECG signal files")
    args = parser.parse_args()

    inferencer = Inferencer(args)
    inferencer.main()