from constants import *
import json 
import os
import argparse

class Inferencer():
    def __init__(self, dir, save_dir, dataset_list, diagnosis_list, label_list, model_list):
        self.dir = dir if dir is not None else DIRECTORY
        self.save_dir = save_dir if save_dir is not None else dir
        self.dataset_list = dataset_list if dataset_list is not None else DATASET_LIST
        self.diagnosis_list = diagnosis_list if diagnosis_list is not None else DIAGNOSIS_LIST
        self.label_list = label_list if label_list is not None else LABEL_LIST
        self.model_list = model_list if model_list is not None else MODEL_LIST
        

    def PathGatherer(self):
        total_sample_path_list = []
        for dataset in self.dataset_list:
            for diagnosis in self.diagnosis_list:
                for label in self.label_list:
                    current_path = os.path.join(self.dir, dataset, diagnosis, label)
                    path_list = os.listdir(current_path)
                    for path in path_list:
                        sample_list = os.listdir(os.path.join(current_path, path))
                        for sample in sample_list:
                            total_sample_path_list.append(os.path.join(current_path, path, sample))
        
        return total_sample_path_list
    
    def QAplaceHolder(self, prompt, ecg_id, model):
        # ecg id to ecg
        # ecg to ecg image
        if model == "gemini":
            answer = "a"
        if model == "gpt":
            answer = "b"

        return answer
    

    def RetrieveAnswer(self, path, model):
        with open(path) as f:
            sample = json.load(f)
            answer_list = []
            for data in sample["data"]:
                text = prompt.format(data["question"], data["options"])
                model_response = self.QAplaceHolder(text, sample["metadata"]["ecg_id"], model)
                data["response_raw"] = model_response
                answer_list.append(data)

            sample["metadata"]["model"] = model
        
        return sample

    def main(self):
        sample_paths = self.PathGatherer()
        for model in self.model_list:
            for path in sample_paths:
                answer_json = self.RetrieveAnswer(path, model)
                save_path = os.path.join(self.save_dir, model, os.path.relpath(path, self.dir))
                print(save_path)
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                with open(save_path, "w") as f:
                    json.dump(answer_json, f, indent=4)
        pass

        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-d', '--dir', help = "load directory")
    parser.add_argument('-s', '--savedir', help = "directory for saving result file")
    parser.add_argument('-ds', '--dataset', nargs='*', help = 'mimic_iv_ecg, ptbxl')
    parser.add_argument('-dx', '--diagnosis', nargs='*', help = '"lateral_myocardial_infarction", "complete_right_bundle_branch_block", "left_posterior_fascicular_block", "premature_atrial_complex", "left_anterior_fascicular_block", "complete_left_bundle_branch_block", "third_degree_av_block", "first_degree_av_block", "inferior_ischemia", "lateral_ischemia", "left_ventricular_hypertrophy", "premature_ventricular_complex", "inferior_myocardial_infarction", "anterior_myocardial_infarction", "second_degree_av_block", "anterior_ischemia", "right_ventricular_hypertrophy"]')
    parser.add_argument('-l', '--label', nargs='*', help = 'positive, negative')
    parser.add_argument('-m', '--model', nargs='*', help = '"gpt", "gemini", "qwen", "llama", "gem", "pulse", "opentslm", "llava-med", "medgemma","hulu-med"')
    args = parser.parse_args()

    inferencer = Inferencer(args.dir, args.savedir, args.dataset, args.diagnosis, args.label, args.model)
    inferencer.main()