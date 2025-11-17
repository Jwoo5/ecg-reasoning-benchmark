from constants import *
import json 
import os
import argparse

class Inferencer():
    def __init__(self, args):
        self.dir = args.dir if args.dir is not None else DIRECTORY
        self.save_dir = args.savedir if args.savedir is not None else OUTPUT_DIRECTORY
        self.dataset_list = args.dataset if args.dataset else DATASET_LIST
        # self.diagnosis_list = args.diagnosis_list if args.diagnosis_list is not None else DIAGNOSIS_LIST
        # self.label_list = args.label_list if args.label_list is not None else LABEL_LIST
        self.model_list = args.model if args.model is not None else MODEL_LIST
    
    def ECGLoader(self, ecg_id):
        pass

    def ECGVisualizer(self, ecg):
        pass

    def QuestionAnswerer(self, prompt, ecg_id, model):
        ecg = self.ECGLoader(ecg_id)
        ecg_paper = self.ECGVisualizer(ecg)
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
                model_response = self.QuestionAnswerer(text, sample["metadata"]["ecg_id"], model)
                data["response_raw"] = model_response
                answer_list.append(data)

            sample["metadata"]["model"] = model
        
        return sample

    def main(self):

        total_path = []
        for dataset in self.dataset_list:
            temp_path = os.listdir(os.path.join(self.dir, dataset))
            total_path += [os.path.join(self.dir, dataset, path) for path in temp_path]
        
        for model in self.model_list:
            for path in total_path:
                answer_json = self.RetrieveAnswer(path, model)
                save_path = os.path.join(self.save_dir, model, os.path.basename(path))
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                with open(save_path, "w") as f:
                    json.dump(answer_json, f, indent=4)
        pass

        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-d', '--dir', help = "load directory")
    parser.add_argument('-s', '--savedir', help = "directory for saving result file")
    parser.add_argument('-ds', '--dataset', nargs='*', help = 'mimic_iv_ecg, ptbxl')
    # parser.add_argument('-dx', '--diagnosis', nargs='*', help = '"lateral_myocardial_infarction", "complete_right_bundle_branch_block", "left_posterior_fascicular_block", "premature_atrial_complex", "left_anterior_fascicular_block", "complete_left_bundle_branch_block", "third_degree_av_block", "first_degree_av_block", "inferior_ischemia", "lateral_ischemia", "left_ventricular_hypertrophy", "premature_ventricular_complex", "inferior_myocardial_infarction", "anterior_myocardial_infarction", "second_degree_av_block", "anterior_ischemia", "right_ventricular_hypertrophy"]')
    # parser.add_argument('-l', '--label', nargs='*', help = 'positive, negative')
    parser.add_argument('-m', '--model', nargs='*', help = '"gpt", "gemini", "qwen", "llama", "gem", "pulse", "opentslm", "llava-med", "medgemma","hulu-med"')
    args = parser.parse_args()

    inferencer = Inferencer(args)
    inferencer.main()