# ECG-Reasoning-Benchmark: A Benchmark for Evaluating Clinical Reasoning Capabilities in ECG Interpretation

<p align="center">
    <a href="https://arxiv.org/abs/2603.14326" target="_blank" rel="noopener noreferrer">
        <img src="https://img.shields.io/badge/paper-arXiv_2603.14326-red.svg" alt="arXiv">
    </a>
    <a>
        <img src="https://img.shields.io/badge/-Python_3.12-blue?logo=python&logoColor=white">
    </a>
    <a href="https://huggingface.co/datasets/Jwoo5/ECG-Reasoning-Benchmark" target="_blank" rel="noopener noreferrer">
        <img src="https://img.shields.io/badge/dataset-HuggingFace-yellow.svg" alt="dataset">
    </a>
</p>

This is the official repository for distributing ECG-Reasoning-Benchmark.

> While Multimodal Large Language Models (MLLMs) show promising performance in automated electrocardiogram interpretation, it remains unclear whether they genuinely perform actual step-by-step reasoning or just rely on superficial visual cues. To investigate this, we introduce **ECG-Reasoning-Benchmark**, a novel multi-turn evaluation framework comprising over 6,400 samples to systematically assess step-by-step reasoning across 17 core ECG diagnoses. Our comprehensive evaluation of state-of-the-art models reveals a critical failure in executing multi-step logical deduction. Although models possess the medical knowledge to retrieve clinical criteria for a diagnosis, they exhibit near-zero success rates (< 6\% Completion) in maintaining a complete reasoning chain, primarily failing to ground the corresponding ECG findings to the actual visual evidence in the ECG signal. These results demonstrate that current MLLMs bypass actual visual interpretation, exposing a critical flaw in existing training paradigms and underscoring the necessity for robust, reasoning-centric medical AI.

# Dataset Description

The dataset is organized as follows:
```
data
├── mimic_iv_ecg.jsonl
└── ptbxl.jsonl
```
* Each `.jsonl` file contains the full set of multi-turn QA reasoning samples for its respective data source (i.e., MIMIC-IV-ECG and PTB-XL).
* Each line in the `.jsonl` file represents a single JSON object containing the `metadata` and `data` for one reasoning sample:
> * `metadata`: contains metadata information about the sample:
>   * `id`: a unique integer identifier for **the data sample** (e.g., `0`, `1`, `2`, etc.), **which is used to distinguish different samples in the dataset and can be used as part of the filename for saving the curated model responses for each sample.**
>   * `data_source`: indicates the source of the data (i.e., `"mimic_iv_ecg"` or `"ptbxl"`).
>   * `ecg_id`: a unique identifier for **the ECG sample**, having different formats depending on the data source (e.g., `"41720298"` for MIMIC-IV-ECG and `"21472"` for PTB-XL).
>   * `target_dx`: the target diagnosis for the sample, which is one of the defined 17 core ECG diagnoses (e.g., `"anterior_ischemia"`).
>   * `dx_label`: the GT label for the target diagnosis, where `false` indicates the absence of the diagnosis and `true` indicates its presence.
>   * `path_idx`: the index of the reasoning path, which is used to distinguish different reasoning paths for the same target diagnosis and label. For example, for `third_degree_av_block` with `dx_label` as `false`, there are 3 different reasoning paths, where `path_idx` ranges from `0` to `2`.
>   * `subject_id` **(only for MIMIC-IV-ECG)**: the subject ID of the patient, which can be used to retrieve the corresponding EHR data for the patient from the MIMIC-IV database.
> * `data`: contains the actual multi-turn QA reasoning sample, which is structured as:
>   * `initial_diagnostic_question`: the initial diagnostic question for the sample:
>       * `question`: a question asking for the target diagnosis (e.g., `"Does this ECG suggest the presence of anterior ischemia?"`).
>       * `options`: a list of possible answer options for the question, which is `["Yes", "No"]` for all samples.
>       * `answer`: the correct answer for the question, which is either `"Yes"` or `"No"`.
>       * `question_type`: the type of the question, which is `"initial_diagnostic_question"` for all samples.
>   * `reasoning`: a list of reasoning steps, where each step is a dictionary containing:
>       *  `criterion_selection`
>           * `question`: a question asking for the selection of a specific diagnostic criterion (e.g., `"To accurately diagnose anterior ischemia, which of the following diagnostic criteria should be evaluated?"`).
>           * `options`: a list of possible answer options for the question, where 5 options are provided for this type of question, including 1 correct criterion and 4 distractors.
>           * `answer`: the correct answer for the question, which is one of the options provided in the `options` field.
>           * `answer_idx`: the index of the correct answer in the `options` list, starting from `0`.
>           * `question_type`: the type of the question, which is `"criterion_selection"` for all samples of this reasoning step.
>       * `finding_identification`
>           * `question`: a question asking for the identification of a specific ECG finding related to the selected criterion (e.g., `"Regarding the criterion you selected, does this ECG show ST-segment depression in at least two of the anterior leads, including leads V2, V3, and V4? Note that ST depression is defined as a depression of the J-point greater than 0.1mV (1mm) in lead V2, and greater than 0.07mV (0.7mm) in leads V3 and V4."`).
>           * `options`: a list of possible answer options for the question, which is `["Yes", "No"]` for all samples of this reasoning step.
>           * `answer`: the correct answer for the question, which is either `"Yes"` or `"No"`.
>           * `answer_idx`: the index of the correct answer in the `options` list, where `0` corresponds to `"Yes"` and `1` corresponds to `"No"`.
>           * `question_type`: the type of the question, which is `"finding_identification"` for all samples of this reasoning step.
>       * `ecg_grounding`: a ***list*** of grounding questions where the number of grounding questions depends on the specific criterion being evaluated. Note that this `ecg_grounding` step can be ***empty*** if there is no corresponding grounding question for the criterion being evaluated. Each grounding question is a dictionary containing:
>           * `question`: a question asking for the grounding of a specific ECG finding to the actual visual evidence in the ECG signal (e.g., `"Which of the following leads show ST-segment depression? Select all possible leads from the options below."`).
>           * `options`: a list of possible answer options for the question, depending on the type of the grounding question (i.e., `"lead_grounding"`, `"wave_grounding"`, and `"measurement_grounding"` that will be described below).
>           * `answer`: a ***list*** of the correct answers for the question. **Note: For structural consistency across all grounding types, this field is always formatted as a *list*, even for single-answer questions. However, only `lead_grounding` questions may have multiple correct answers.**
>           * `answer_idx`: a ***list*** of the indices for the correct answers in the `options` list. **Similarly, this is always formatted as a *list* of integers.**
>           * `question_type`: the type of the ECG grounding question, which can be one of the following:
>               * `"lead_grounding"`: a question asking for the grounding of a specific ECG finding to the actual leads in the ECG signal (e.g., `"Which of the following leads show ST-segment depression? Select all possible leads from the options below."`).
>               * `"wave_grounding"`: a question asking for the grounding of a specific ECG finding to the actual waveforms in the ECG signal (e.g., `"Within the selected leads, in which of the following waves can you observe the QRS complex with ST-segment depression? The options below refer to time ranges on the ECG signal, provided in seconds."`).
>               * `"measurement_grounding"`: a question asking for the grounding of a specific ECG finding to the actual measurements in the ECG signal (e.g., `"For the selected segment, which range does the measured QRS duration fall into?"`).
>       * `diagnostic_decision`
>           * `question`: a question asking for the decision of the diagnosis based onthe identified ECG findings (e.g., `"Based onthe finding identified above, does this ECG suggestthe presence of anterior ischemia?"`).
>           * `options`: a list of possible answer options forthe question, which is `["Yes", "No", "Further findings are required to confirmthe diagnosis"]` for all samples of this reasoning step.
>           * `answer`: the correct answer for the question, which is one of the options provided in the `options` field.
>           * `answer_idx`: the index of the correct answer in the `options` list.
>           * `question_type`: the type of the question, which is `"diagnostic_decision"` for all samples of this reasoning step.

## How to Use (Quick Start)

You can easily load the dataset using the Hugging Face Hub or from the local `.jsonl` files provided in this repository.

### Option 1: Loading from Hugging Face Hub
The easiest way to load the dataset is using the Hugging Face `datasets` library.
The dataset is organized into two configurations (`mimic_iv_ecg` and `ptbxl`) and is available under the `test` split.

```python
from datasets import load_dataset

# Load MIMIC-IV-ECG-sourced samples
mimic_dataset = load_dataset("Jwoo5/ECG-Reasoning-Benchmark", "mimic_iv_ecg", split="test")

# Load PTB-XL-sourced samples
ptbxl_dataset = load_dataset("Jwoo5/ECG-Reasoning-Benchmark", "ptbxl", split="test")
```

### Option 2: Loading Local Files Directly

If you cloned this repository and want to load the data locally, you can use the provided `.jsonl` files in the `data/` directory by parsing them line by line as JSON objects.

```python
import json

def load_jsonl(file_path):
    data = []
    with open(file_path, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data

# Load MIMIC-IV-ECG-sourced samples
mimic_dataset = load_jsonl("data/mimic_iv_ecg.jsonl")

# Load PTB-XL-sourced samples
ptbxl_dataset = load_jsonl("data/ptbxl.jsonl")
```

### Example for accessing the first sample in the MIMIC-IV-ECG-sourced dataset

```python
sample = mimic_dataset[0]

print(f"Q: {sample['data']['initial_diagnostic_question']['question']}")
print(f"A: {sample['data']['initial_diagnostic_question']['answer']}")

# Iterate through the reasoning steps
for step in sample["data"]["reasoning"]:
    for q_type in step:
        if q_type == "ecg_grounding":
            for grounding_q in step["ecg_grounding"]:
                print(f"Q: {grounding_q['question']}")
                # Note: 'answer' in ecg_grounding is consistently formatted as a list
                print(f"A: {', '.join(grounding_q['answer'])}")
        else:
            print(f"Q: {step[q_type]['question']}")
            print(f"A: {step[q_type]['answer']}")
```

#### Output:
```
Q: Does this ECG suggest the presence of first degree AV block?
A: Yes
Q: To accurately diagnose first degree AV block, which of the following diagnostic criteria should be evaluated?
A: Evidence of consistent 1:1 atrioventricular conduction
Q: Regarding the criterion you selected, looking at the overall rhythm, is every P wave, excluding those following premature beats, consistently followed by a QRS complex on this ECG?
A: Yes
Q: Based on the finding identified above, does this ECG suggest the presence of first degree AV block?
A: Further findings are required to confirm the diagnosis
Q: In addition to the finding you just identified, which other diagnostic criterion should be evaluated to diagnose first degree AV block?
A: Prolongation of the PR interval
Q: Regarding the criterion you selected, is the PR interval prolonged on this ECG? Note that a PR interval is considered to be prolonged if it is greater than 200 milliseconds.
A: Yes
Q: In which of the following segments can you observe a P wave that demonstrates the prolonged PR interval? The options below refer to time ranges on the ECG signal, provided in seconds.
A: [1.12s - 1.24s]
Q: For the selected segment, which range does the measured PR interval fall into?
A: [230ms - 240ms]
Q: Based on all the findings identified so far, does this ECG suggest the presence of first degree AV block?
A: Yes
```

# Experiments

## Curating Responses From Models for Evaluation

To evaluate the performance of the models on **ECG-Reasoning-Benchmark**, we first need to curate the responses from the models for each sample in the benchmark dataset.
The responses should be curated in the same format as the original samples in the dataset, with only the additional field `model_response` added to each question step (i.e., the same level with the steps including `question` and `answer` fields, such as `initial_diagnostic_question`, `criterion_selection`, `finding_identification`, `ecg_grounding`, and `diagnostic_decision`).
This curation process can be done by running `inference.py` in this repository, which will automatically generate the model responses for each question step and save the curated responses as **individual `.json` files named by the sample's `id` (e.g., `0.json`, `1.json`, ...)**.
These files will be organized within the provided output directory following the structure: `$output_dir/$model_name/$dataset/$target_dx/*.json` (e.g., `$output_dir/$model_name/mimic_iv_ecg/first_degree_av_block/0.json`), where further details can be found in the instructions below.

> [!NOTE]
> When we process a sample in `inference.py`, we record the model response for each question step in the sample, and then proceed with the next question step by appending the current question and the ***GT answer*** to the prompt history regardless of the correctness of the model response for the current question step.
> This makes it possible to evaluate the model performance on the individual stage (e.g., `criterion_selection`, `finding_identification`, `ecg_grounding`, and `diagnostic_decision`), as well as the GT-Reasoning-Based Diagnosis Accuracy reported in the paper.  
> Note that these GT-Prompt-based accuracy for each stage will be reported as `_w_gt` appended to the stage name (e.g., `criterion_selection_accuracy_w_gt`), while other metrics such as `Completion` are still calculated based on the principle that the evaluation terminates upon the first incorrect response in the model's sequential predictions.

### Using Existing Models With the Default Prompt

We have prepared each model implementation with the default prompt for evaluation on **ECG-Reasoning-Benchmark**.
These model implementations include the following models:
* [PULSE](https://arxiv.org/abs/2410.19008)
* [GEM](https://arxiv.org/abs/2503.06073)
* [ECG-R1](https://arxiv.org/abs/2602.04279)
* [OpenTSLM](https://arxiv.org/abs/2510.02410)
* [Hulu-Med](https://arxiv.org/abs/2510.08668v2)
* [MedGemma](https://arxiv.org/abs/2507.05201)
* [Qwen3-VL](https://arxiv.org/abs/2511.21631)
* [Llama-3.2-Vision-Instruct](https://arxiv.org/abs/2407.21783)
* [Gemini](https://arxiv.org/abs/2312.11805)
* [GPT](https://arxiv.org/abs/2601.03267)

We also provide the Python environment configuration files for these models in the [`envs/`](./envs) directory of this repository as these models require different versions of `torch`, `transformers`, or `accelerate` library.
This includes:
* [`env_legacy.yaml`](./envs/env_legacy.yaml): for PULSE and GEM.
* [`env_opentslm.yaml`](./envs/env_opentslm.yaml): for OpenTSLM.
* [`env_hulumed.yaml`](./envs/env_hulumed.yaml): for Hulu-Med.
* [`env_hf.yaml`](./envs/env_hf.yaml): for other models implemented by the huggingface model hub or API endpoints, including ECG-R1, MedGemma, Qwen3-VL, Llama-3.2-Vision-Instruct, Gemini, and GPT.

Of these models, some models are implemented by loading the whole processing pipeline from the huggingface model hub or specific endpoints, while some models are implemented locally in this repository.
Therefore, we provide running scripts for both types of models.

For the locally implemented models (PULSE, GEM, and OpenTSLM), run:
```bash
python inference.py /path/to/data/ \
    --dataset $dataset \
    --model $model_name \
    --ecg-base-dir $ecg_base_dir \
    --output-dir $output_dir \
    --enable-condensed-chat
```
* `/path/to/data` should be consistent with the `data` directory that contains the benchmark dataset (e.g., [`./data`](./data) in this repository).
* `$dataset`: the name of the source dataset, which can be either `mimic_iv_ecg` or `ptbxl`.
* `$model_name`: the name of the model to be evaluated, which can be one of the following: (`pulse`, `gem`, `opentslm`).
* `$ecg_base_dir`: the base directory containing the actual ECG signal files for the samples in the benchmark dataset, which is required for the models to process the ECG signals along with the questions, as the benchmark dataset does not provide the ECG signal files itself.
    * For the `mimic_iv_ecg` source dataset, it should be the directory containing the `files/` directory in the MIMIC-IV-ECG database, which can be downloaded from the [PhysioNet repository for MIMIC-IV-ECG](https://physionet.org/content/mimic-iv-ecg/1.0/).
    * For the `ptbxl` source dataset, it should be the directory containing the `records100/` and `records500/` directories in the PTB-XL database, which can be downloaded from the [PhysioNet repository for PTB-XL](https://physionet.org/content/ptb-xl/1.0.3/).
* `$output_dir`: the directory to save the curated responses from the model for each sample in the benchmark dataset. The results will be saved in `$output_dir/$model_name/$dataset/$target_dx/` directory, where `$target_dx` is the corresponding target diagnosis for each sample (e.g., `first_degree_av_block`).
* `--enable-condensed-chat`: an optional flag to enable the condensed chat format, which makes the prompt history include only the answer for each of the previous questions without `option` fields, which is designed to mitigate the potential issue of exceeding the maximum context length for some models when processing the multi-turn questions.

For other models, run:
```bash
python inference.py /path/to/data/ \
    --dataset $dataset \
    --model $model_name \
    --model-variant $model_variant \
    --ecg-base-dir $ecg_base_dir \
    --output-dir $output_dir \
    --enable-condensed-chat
```
* `$model_name`: can be one of the following: (`hulumed-hf`, `medgemma-hf`, `qwen3-vl-hf`, `llama-3.2-vision-hf`, `gemini`, `gpt`)
* The additional argument `$model_variant` is required for these models, which indicates the specific variant of the model to be evaluated. This identifier will be appended to the predefined `model_id` depending on `$model_name` to load the model from the huggingface model hub or specific endpoints. To check how it works, see `model_id` field in each model implementation class. The example `$model_variant` for each `$model_name` is as follows:
    * `hulumed-hf`: `7B`, `32B`
    * `medgemma-hf`: `4b-it`, `27b-it`, `1.5-4b-it`
    * `qwen3-vl-hf`: `8B-Instruct`, `32B-Instruct`
    * `llama-3.2-vision-hf`: `11B-Vision-Instruct`, `90B-Vision-Instruct`
    * `gemini`: `2.5-flash`, `2.5-pro`, `3-flash-preview`
    * `gpt`: `5-mini`, `5.2`
* `$output_dir`: the results will be saved in `$output_dir/{$model_name}_{$model_variant}/$dataset/$target_dx/` directory.

### Adding New Models

To add new models for evaluation on **ECG-Reasoning-Benchmark**, you can implement a new model class in `models/` directory by following the structure of the existing model classes, and then run `inference.py` with the corresponding `$model_name` and `$model_variant`.
Follow the instructions below to implement a new model class:
1. Create a new directory and a new Python file for the model implementation under the `models/` directory. Implement the model class in the new Python file by following the structure of the existing model classes, which should extend the `BaseModel` class defined in `models/model.py`. You also need to create `__init__.py` in that directory to import the new model class for registration.
2. This new model class should be decorated with `@register_model(model_name)` to register the model with a unique name for loading the model.
3. If the base modality of the model is not the `"image"` (i.e., Vision-Language model), you should clarify the base modality of the model by setting the `self.ecg_modality_base` field in the `__init__` method of the model class. Note that we only support the base modality of `"signal"` and `"image"` for now, where the former will input the ECG signal as a 500Hz 12-lead 1D signal array, while the latter will input the ECG signal as a 12-lead ECG chart image by converting the 500Hz signal using [`ecg-plot` Python library](https://github.com/dy1901/ecg_plot).
4. For the image-based models (i.e., Vision-Language models), you should also clarify if the model requires base64 encoding for the input ECG image setting the `require_base64_image` method to return `True` in the model class.
5. Implement classmethod `build_model`, which builds the model instance. This can call the `__init__` method of the model class to initialize the model instance, and also include any additional processing steps for building the model before calling the `__init__` method.
6. Implement `get_response` method, which generates a response based on the conversation history. This method should take the `utils.Conversation` instance as input, and return the generated response as a string. The conversation history (`Conversations.conversation`) is a list of dictionaries, where each dictionary contains the following fields:
    * `role`: the role of the speaker, which can be either one of `"system"`, `"user"`, and `"model"`.
    * For the `system` or `model` role,
        * `text`: the text content of the conversation turn. In other words, the system prompt for the `system` role, and the model response for the `model` role.
    * For the `user` role,
        * `question`: the question asking for the model response.
        * `options`: the list of options for the question.
        * `signal` (optional): the ECG signal input, which is a 500Hz 12-lead 1D signal array. Only provided for the signal-based models, and for the very first question turn (i.e., the `initial_diagnostic_question` step) in the conversation history.
        * `image` (optional): the ECG image input, which is a 12-lead ECG chart image as a PIL image object or base64-encoded string depending on the model requirement. Only provided for the image-based models, and for the very first question turn (i.e., the `initial_diagnostic_question` step) in the conversation history.
> [!NOTE]
> Note that the first turn of the conversation history (`Conversation.conversation`) is always the system prompt, and the final turn is always the current user question turn asking for the model response. As aforementioned, the very first user question turn (i.e., `Conversation.conversation[1]`) contains `image` or `signal` field for the ECG input.  
> We strongly recommend you to refer to other pre-existing model implementations for this method to see how to process the conversation history to make the full prompt for the model input.
7. You can also implement any other methods for the model class as needed, such as additional helper methods for processing the ECG input or generating the model response.

### Modifying the Prompt Design

The system prompt is defined in the `inference.py` file as a global variable `system_prompt`, which is used as the initial system prompt for all models by default.
In addition, we also append another default prompt for image-based models in the `initial_diagnostic_question` step to give the information about the ECG paper rate (also known as ECG paper speed), which is defined in the `inference` method of the `Inferencer` class in `inference.py` file.
You can modify these prompts as you need to potentially improve the model performance on the benchmark dataset.

For other types of prompts such as the question prompts for each question step to build the prompt history, they are defined in each model class (mainly in `get_response` method for the pre-defined models).
Therefore, You can design your own question prompts and implement them in the `get_response` method of your model class.

## Evaluating the Model Performance

After curating the model responses, you can evaluate the model performance by running `evaluate.py` in this repository, which will automatically calculate the evaluation metrics and save the evaluation results in a CSV file for each model and dataset.
The judgment for the correctness of the model response with respect to the GT answer is either done by heuristic string matching or by Gemini from Google, depending on the evaluation settings specified by the user.  

The heuristic string matching is based on the exact string matching between the model response and the GT answer, with handling for some known cases to avoid the issue of minor variations in the model response (e.g., "Yes." vs "Yes", or "Yes" vs "\*\*Yes\*\*").
Note that it only includes some known cases based on the manual analysis of the model responses by the authors, and it may not cover all the possible variations in the model responses, which can potentially lead to some incorrect judgments.
However, this can be a useful method for a quick evaluation without the need for additional API calls to Gemini, which can be costly and time-consuming when evaluating a large number of samples.
For using this heuristic string matching method, run:
```bash
python evaluation.py /path/to/results/ \
    --dataset $dataset_list \
    --model $model_name_list \
    --evaluator heuristic \
    --save-dir $save_dir
```
* `/path/to/results`: should be consistent with `$output_dir` provided in the previous step for curating the model responses, which contains the curated responses for each model and dataset.
* `$dataset_list`: a list of dataset names to be evaluated, separated by whitespace (e.g., `--dataset ptbxl mimic_iv_ecg`).
* `$model_name_list`: a list of **full** model names separated by whitespace, each of which should be consistent with the directory name in the `/path/to/results` directory where the curated responses from the model are saved (e.g., `--model pulse gemini_2.5-flash`).
* `$save_dir`: the directory to save the evaluation results, which will be saved in `$save_dir/$evaluator_name/$dataset/` directory. Note that the evaluation results from all the models specified in `$model_name_list` will be pooled together in the same CSV files. These CSV files are composed of results for each target diagnosis, as well as the overall results in `total.csv`.


On the other hand, the Gemini evaluator is based on prompting Gemini to judge the correctness of the model response with respect to the GT answer, which can potentially provide a more accurate judgment by understanding the semantic meaning of the model response and the GT answer, handling the variations in the model responses.
For using this Gemini evaluator, run:
```bash
python evaluation.py /path/to/results/ \
    --dataset $dataset_list \
    --model $model_name_list \
    --evaluator gemini \
    --gemini-model $gemini_model \
    --use-cache \
    --save-cache \
    --load-cache \
    --save-cache-interval 1 \
    --save-dir $save_dir
```
* `$gemini_model`: the specific variant of Gemini to be used as the evaluator (e.g., `gemini-2.5-flash`, `gemini-3-flash-preview`).
* `--use-cache`: an optional flag to indicate whether to use the cache for storing the evaluation results from Gemini, which can avoid the redundant API calls to Gemini for the same evaluation samples to save the cost and time for evaluation. Specifically, when this flag is enabled, the evaluation result for each (model response, GT answer) pair will be stored in the internal cache directory in the `GeminiEvaluator` instance. Then, when the same (model response, GT answer) pair appears in the evaluation samples, the evaluation result will be retrieved from the cache instead of making API calls to Gemini. Enabling this functionality is highly recommended when evaluating a large number of samples, which can significantly reduce the cost and time for evaluation.
* `--save-cache`: an optional flag to indicate whether to save the cache to the disk during the evaluation process, which can save the cache for future evaluation runs. The cache will be saved in `~/.cache/ecg-reasoning-benchmark/` directory by default, with the filename encoded with the hash of the evaluator name (e.g., the hash of `gemini-2.5-flash` or `gemini-3-flash-preview`) to distinguish the cache for different Gemini variants.
* `--load-cache`: an optional flag to indicate whether to load the cache from the disk before the evaluation process.
* `--save-cache-interval`: an optional argument to specify the interval for saving the cache to the disk during the evaluation process, which can avoid the potential issue of losing the cache due to unexpected interruption during the evaluation process. For example, when `--save-cache-interval 1` is specified, the cache will be saved to the disk for every single (model response, GT answer) pair.

# Citation
We will update the citation information soon.