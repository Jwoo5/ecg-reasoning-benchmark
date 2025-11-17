DIRECTORY = "/data"
DATASET_LIST = ['mimic_iv_ecg', 'ptbxl']
DIAGNOSIS_LIST = ['lateral_myocardial_infarction', 'complete_right_bundle_branch_block', 'left_posterior_fascicular_block', 'premature_atrial_complex', 'left_anterior_fascicular_block', 'complete_left_bundle_branch_block', 'third_degree_av_block', 'first_degree_av_block', 'inferior_ischemia', 'lateral_ischemia', 'left_ventricular_hypertrophy', 'premature_ventricular_complex', 'inferior_myocardial_infarction', 'anterior_myocardial_infarction', 'second_degree_av_block', 'anterior_ischemia', 'right_ventricular_hypertrophy']
LABEL_LIST = ['negative', 'positive']
MODEL_LIST =  ["gpt", "gemini", "qwen", "llama", "gem", "pulse", "opentslm", "llava-med", "medgemma","hulu-med"] 


prompt = """
<Prompt placeholder>

### Question : {}

### Option : {}
"""