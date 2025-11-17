import os
import json 
from pathlib import Path
from collections import defaultdict
import pandas as pd 
import argparse   

class Evaluator():
    def __init__(self, args):
        self.dir = args.dir if args.dir is not None else "./"
        self.save_dir = args.save_dir if args.save_dir is not None else dir
        self.keys = args.keys if args.keys is not None else ["model"]
        self.drop_keys = list(set(['id', 'dataset', 'target_dx', 'model', 'label', 'path']) - set(self.keys))

    def AnswerValidator(self, gt, model_answer):
        #mcq
        if gt[1] == model_answer.lower():
            return True
        else:
            return False
        
        #if can choose multiple option
        #binary question
    
    def make_csv(self):
        paths = os.listdir(self.dir)
        rows = []
        for path in paths:
            row = defaultdict(int)
            with open(path) as f:
                continue_flag = True
                result = json.load(f)
                
                rel_parts = Path(path).relative_to(self.dir).parts 
                row["id"] = os.path.basename(path)
                row["data_source"] = result["metadata"]["data_source"]
                row["target_dx"] = result["metadata"]["target_dx"]
                row["model"] = result["metadata"]["model"]
                row["label"] = result["metadata"]["dx_label"]
                row["path_idx"] = result["metadata"]["path_idx"]
                for question in result["data"]:
                    row[f"total_{question['question_type']}s"] += 1
                    if self.AnswerValidatorPlaceHolder(question['answer_str'], question['response_raw']) & continue_flag:
                        row[f"total_correct_{question['question_type']}s"] += 1
                        row[f"consecutive_correct_{question['question_type']}s"] += 1
                    elif self.AnswerValidatorPlaceHolder(question['answer_str'], question['response_raw']) :
                        row[f"total_correct_{question['question_type']}s"] += 1
                    else:
                        continue_flag = False
            
            rows.append(row)
        df = pd.DataFrame(rows).fillna(0)
        return df

    def aggregator(self, df, keys):
        metric_df = df.groupby(keys).sum().drop(self.drop_keys, axis=1)
        for metric in ['finding', 'wave_grounding', 'lead_grounding', 'measurement_grounding']:
            if f'total_{metric}s' in metric_df.columns:
                metric_df[f'average_depth_{metric}']= metric_df[f'consecutive_correct_{metric}s'] / metric_df[f'total_{metric}s']
                metric_df[f'total_accuracy_{metric}']= metric_df[f'total_correct_{metric}s'] / metric_df[f'total_{metric}s']
        return metric_df.fillna(0)
    
    def main(self):
        df = self.make_csv()
        metric = self.aggregator(df, self.keys)
        metric.to_csv(os.path.join(self.save_dir, "metric.csv"))
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-d', '--dir', help = "load directory")
    parser.add_argument('-s', '--save_dir', help = "saving directory")
    parser.add_argument('-k', '--keys', nargs="+", help="keys for aggregation. (dataset, target_dx, model, label, path)")
    args = parser.parse_args()

    evaluator = Evaluator(args)
    evaluator.main()