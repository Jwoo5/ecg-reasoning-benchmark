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

# Release Notes

> [!IMPORTANT]
> **Re-evaluation guidance for the latest release (0.0.2)**
>
> If you have already curated model responses against the previous release, please note that re-evaluation is needed to reflect the changes introduced in 0.0.2:
> * For the diagnoses covered by the **diagnosis-wide sample updates** below (LAFB, LPFB, LVH, 3AVB, PAC, PVC), re-evaluation is **required** -- prior results on these diagnoses are no longer comparable under the updated samples.
> * For the **individually re-sampled samples from the manual inspection pass**, the original samples were not erroneous, so prior results remain technically valid; however, we still **recommend** re-running on the updated samples for consistency with the latest release.
> * For **any model evaluated with the 0.0.1 inference code**, re-running inference with the 0.0.2 code is **required**. The 0.0.1 release contained a bug -- accidentally introduced during a code cleanup -- where the GT answer of the `initial_diagnostic_question` step was injected into the dialogue history, contrary to the documented GT-substitution exception for that step. This bug has been fixed in 0.0.2. Note that **the experimental results reported in the paper are unaffected**, since the bug was introduced only after those experiments had already been completed.

<details open>
  <summary><b><i>0.0.2 (Pre-release)</i></b></summary>

* **Diagnosis-wide sample updates.** For the following diagnoses, every sample was regenerated to reflect improvements to the underlying question templates or to correct systematic errors present in the previous release:
  * `left_anterior_fascicular_block` (LAFB) and `left_posterior_fascicular_block` (LPFB): The Criterion Selection question asking about the LBBB-related rule-out condition was reworded for clarity (e.g., _"To accurately diagnose left anterior fascicular block, which of the following diagnostic criteria should be evaluated?"_ → _"To accurately diagnose left anterior fascicular block, which of the following conditions must be ruled out before diagnosis?"_).
  * `left_ventricular_hypertrophy` (LVH): Fixed an error where certain grounding segments were being sampled incorrectly, and re-sampled all samples from scratch. In addition, the Finding Identification question for amplitude-based criteria (e.g., _"Max R amplitude in aVL > 1.1mV"_) now explicitly instructs that premature beats should be excluded from consideration.
  * `third_degree_av_block` (3AVB): The distractors in the Criterion Selection question concerning _"Presence of atrial fibrillation"_ were refined to be less ambiguous (e.g., distractors such as _"1:1 AV Conduction"_ were replaced with clearer alternatives).
  * `premature_atrial_complex` (PAC) and `premature_ventricular_complex` (PVC): Fixed an error where the measurement grounding for some samples was being sampled incorrectly, and re-sampled all the samples from scratch.
* **Dataset-wide manual inspection pass.** To guarantee the best data quality, every sample in the dataset was manually inspected. Individual samples that did not contain outright errors but had any minor concerns were re-sampled while preserving the overall dataset distribution. The affected sample IDs are listed below.

  <blockquote>
    <details>
      <summary><b>Re-sampled individual samples from the manual inspection</b></summary>
      <ul>
        <li>
          <code>complete_left_bundle_branch_block</code> (CLBBB)
        <ul>
          <li>
          <b><i>(source: PTB-XL)</i></b> 414, 421, 424, 427, 429, 430, 435, 436, 438, 439, 449, 450, 455, 461, 462, 463, 464, 465, 467, 474, 489, 494, 506, 509, 513, 590, 601, 602, 606, 610
          </li>
          <li>
          <b><i>(source: MIMIC-IV-ECG)</i></b> 553, 554, 555, 559, 565, 566, 567, 571, 575, 581, 585, 589, 593, 602, 604, 614, 620, 628, 636, 638, 641, 644, 728, 730, 738, 746
          </li>
        </ul>
        <li>
          <code>complete_right_bundle_branch_block</code> (CRBBB)
        </li>
        <ul>
          <li>
          <b><i>(source: PTB-XL)</i></b> 775, 776, 780, 783, 784, 788, 800, 804, 807, 811
          </li>
          <li>
          <b><i>(source: MIMIC-IV-ECG)</i></b> 928, 929, 930, 934, 938, 942, 946, 950
          </li>
        </ul>
        <li>
          <code>left_anterior_fascicular_block</code> (LAFB)
        </li>
        <ul>
          <li>
            <b><i>(source: PTB-XL)</i></b> 858, 892, 896, 969, 980
          </li>
          <li>
            <b><i>(source: MIMIC-IV-ECG)</i></b> 956, 1010, 1114
          </li>
        </ul>
        <li>
          <code>`left_posterior_fascicular_block`</code> (LPFB)
        </li>
        <ul>
          <li>
            <b><i>(source: PTB-XL)</i></b> 1086
          </li>
        </ul>
        <li>
          <code>`anterior_ischemia`</code> (ISCAN)
        </li>
        <ul>
          <li>
            <b><i>(source: PTB-XL)</i></b> 2548
          </li>
        </ul>
        <li>
            <code>`inferior_ischemia`</code> (ISCIN)
        </li>
        <ul>
          <li>
            <b><i>(source: PTB-XL)</i></b> 2703, 2705, 2706, 2707, 2714, 2715, 2717, 2732, 2733, 2734, 2739, 2741, 2742, 2743, 2746, 2747, 2755, 2759, 2764, 2766, 2776, 2793, 2796, 2823, 2838, 2839, 2841, 2844, 2874, 2882
          </li>
          <li>
            <b><i>(source: MIMIC-IV-ECG)</i></b> 3013, 3022, 3024, 3025, 3030, 3032, 3036, 3039, 3040, 3042, 3045, 3046, 3057, 3065, 3073, 3084, 3103, 3137
          </li>
        </ul>
        <li>
          <code>`lateral_ischemia`</code> (ISCLA)
        </li>
        <ul>
          <li>
            <b><i>(source: PTB-XL)</i></b> 2952, 2981, 3032
          </li>
          <li>
            <b><i>(source: MIMIC-IV-ECG)</i></b> 3224
          </li>
        </ul>
      </ul>
    </details>
  </blockquote>

</details>

<details><summary><b><i>0.0.1 (Pre-release)</i></b></summary>

* Initial pre-release of the dataset

</details>

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

## Installation

The benchmark can be installed as a Python package for programmatic use:

```bash
pip install -e .
```

This enables importing the benchmark modules from external projects:

```python
from ecg_reasoning_benchmark import Inferencer, load_data, evaluate_results
from ecg_reasoning_benchmark.models import BaseModel, register_model
from ecg_reasoning_benchmark.evaluators import get_evaluator_cls
```

After installation, the following CLI tools are available:
* `ecg-reasoning-benchmark-inference`: run model inference on benchmark samples to curate model responses (supports both default and `forced-commit` inference modes).
* `ecg-reasoning-benchmark-evaluate`: compute evaluation metrics (e.g., per-stage accuracy, Completion (Hard / Soft), Depth, GT-RDA) from the curated responses.
* `ecg-reasoning-benchmark-forced-commit-plot`: render the finding-coverage vs. diagnosis-accuracy curves from `gemini-forced-commit` evaluator outputs.

Run any of them with `--help` to see the full list of arguments:
```bash
ecg-reasoning-benchmark-inference --help
ecg-reasoning-benchmark-evaluate --help
ecg-reasoning-benchmark-forced-commit-plot --help
```

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
This curation process can be done by running the inference CLI, which will automatically generate the model responses for each question step and save the curated responses as **individual `.json` files named by the sample's `id` (e.g., `0.json`, `1.json`, ...)**.
These files will be organized within the provided output directory following the structure: `$output_dir/$model_name/$dataset/$target_dx/*.json` (e.g., `$output_dir/$model_name/mimic_iv_ecg/first_degree_av_block/0.json`), where further details can be found in the instructions below.

> [!NOTE]
> When we process a sample in `inference.py`, we record the model response for each question step in the sample, and then proceed with the next question step by appending the current question and the ***GT answer*** to the prompt history regardless of the correctness of the model response for the current question step.
> This makes it possible to evaluate the model performance on the individual stage (e.g., `criterion_selection`, `finding_identification`, `ecg_grounding`, and `diagnostic_decision`), as well as the GT-Reasoning-Based Diagnosis Accuracy reported in the paper.  
> Note that these GT-Prompt-based accuracy for each stage will be reported as `_w_gt` appended to the stage name (e.g., `criterion_selection_accuracy_w_gt`), while other metrics such as `Completion` are still calculated based on the principle that the evaluation terminates upon the first incorrect response in the model's sequential predictions.
>
> **Exception for `initial_diagnostic_question`:** The GT-substitution rule above does ***not*** apply to the `initial_diagnostic_question` step. For this very first step, the model's actual response is kept verbatim in the prompt history (the GT answer is ***not*** injected), and the subsequent reasoning steps proceed from there regardless of whether the model answered the initial diagnostic question correctly. This is because the initial diagnostic question is treated as the model's own standalone diagnostic judgment rather than as a reasoning step to be teacher-forced, so the reasoning chain should unfold on top of the model's real answer.

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

> [!IMPORTANT]
> The benchmark is structured as an installable Python package. All internal imports use relative paths (e.g., `from .models import ...`), so modules **cannot** be run directly as standalone scripts (e.g., `python inference.py` will fail with `ImportError`). Instead, use one of the following execution methods:
> 1. **CLI entry points** (after `pip install -e .`): `ecg-reasoning-benchmark-inference`, `ecg-reasoning-benchmark-evaluate`
> 2. **Module execution**: `python -m ecg_reasoning_benchmark.inference`, `python -m ecg_reasoning_benchmark.evaluation`

For the locally implemented models (PULSE, GEM, and OpenTSLM), run:
```bash
ecg-reasoning-benchmark-inference /path/to/data/ \
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
ecg-reasoning-benchmark-inference /path/to/data/ \
    --dataset $dataset \
    --model $model_name \
    --model-variant $model_variant \
    --ecg-base-dir $ecg_base_dir \
    --output-dir $output_dir \
    --enable-condensed-chat
```
* `$model_name`: can be one of the following: (`hulumed-hf`, `medgemma-hf`, `qwen3-vl-hf`, `llama-3.2-vision-hf`, `gemini`, `gpt`)
* The additional argument `$model_variant` is required for these models, which indicates the specific variant of the model to be evaluated. This identifier will be appended to the predefined `model_id` depending on `$model_name` to load the model from the huggingface model hub or specific endpoints. To check how it works, see `model_id` field in each model implementation class. The example `$model_variant` for each `$model_name` is as follows:
    * `ecg-r1-hf`: `SFT`, `RL`
    * `hulumed-hf`: `7B`, `32B`
    * `medgemma-hf`: `4b-it`, `27b-it`, `1.5-4b-it`
    * `qwen3-vl-hf`: `8B-Instruct`, `32B-Instruct`
    * `llama-3.2-vision-hf`: `11B-Vision-Instruct`, `90B-Vision-Instruct`
    * `gemini`: `2.5-flash`, `2.5-pro`, `3-flash-preview`
    * `gpt`: `5-mini`, `5.2`
* `$output_dir`: the results will be saved in `$output_dir/{$model_name}_{$model_variant}/$dataset/$target_dx/` directory.

### Uniform Random-Guess Baseline

We additionally provide `random-guess`, a content-blind baseline that ignores the ECG and the question and picks **uniformly at random** among the offered options at every step (for the multi-answer lead-grounding step, each option is included independently with probability 1/2). It establishes the chance-level floor for every metric, so that any model's score can be read relative to no-information guessing.

Because it never reads the ECG, it declares `ecg_modality_base = "text"`, which makes the inference pipeline skip ECG signal loading and image rendering entirely -- so it needs neither a GPU/API nor the PhysioNet ECG files (any placeholder `--ecg-base-dir` works). Runs are made reproducible with `--seed`:
```bash
ecg-reasoning-benchmark-inference /path/to/data/ \
    --dataset $dataset \
    --model random-guess \
    --model-variant run0 \
    --seed 0 \
    --ecg-base-dir /tmp \
    --output-dir $output_dir
```
* `--seed`: random seed for stochastic baselines (deterministic models ignore it). Vary it across runs (e.g., `run0`/`--seed 0`, `run1`/`--seed 1`, ...) to estimate the baseline's mean and confidence interval.
* Score it with the `heuristic` evaluator: its responses are verbatim option text, so string matching is exact, deterministic, and cost-free.

### Adding New Models

To add new models for evaluation on **ECG-Reasoning-Benchmark**, you can implement a new model class in `ecg_reasoning_benchmark/models/` directory by following the structure of the existing model classes, and then run the inference CLI with the corresponding `$model_name` and `$model_variant`.
Follow the instructions below to implement a new model class:
1. Create a new directory and a new Python file for the model implementation under the `ecg_reasoning_benchmark/models/` directory. Implement the model class in the new Python file by following the structure of the existing model classes, which should extend the `BaseModel` class defined in `ecg_reasoning_benchmark/models/model.py`. You also need to create `__init__.py` in that directory to import the new model class for registration.
2. This new model class should be decorated with `@register_model(model_name)` to register the model with a unique name for loading the model.
3. If the base modality of the model is not the `"image"` (i.e., Vision-Language model), you should clarify the base modality of the model by setting the `self.ecg_modality_base` field in the `__init__` method of the model class. The supported modalities are:
    * `"image"` (default): the ECG signal is converted to a 12-lead ECG chart image using [`ecg-plot`](https://github.com/dy1901/ecg_plot) and passed to the model as a PIL Image or base64-encoded string.
    * `"signal"`: the ECG signal is passed as a 500Hz 12-lead 1D signal array (torch.Tensor).
    * `"text"`: the ECG signal loading and visualization are skipped entirely. Instead, only the wfdb record path is passed to the model via `ecg_path` kwarg in `get_response()`. The model is responsible for loading and processing the ECG signal on its own. This is useful for models that have their own ECG analysis pipeline (e.g., signal-processing-based agents).
4. For the image-based models (i.e., Vision-Language models), you should also clarify if the model requires base64 encoding for the input ECG image by setting the `require_base64_image` method to return `True` in the model class.
5. Implement classmethod `build_model`, which builds the model instance. This can call the `__init__` method of the model class to initialize the model instance, and also include any additional processing steps for building the model before calling the `__init__` method. All CLI arguments are forwarded as keyword arguments via `build_model(model_name, **vars(args))`, so models can accept additional configuration (e.g., API keys, model-specific parameters) through the CLI without modifying the core inference code.
6. Implement `get_response` method, which generates a response based on the conversation history. This method should take the `utils.Conversation` instance as input, and return the generated response as a string. Additional keyword arguments are passed through, including `ecg_path` (the wfdb record path without extension, available for all modalities).

    **Accessing conversation turns for prompt construction:** Use `conversation.get_turns_for_prompt()` instead of directly accessing `conversation.conversation[1:]`. This helper method automatically handles the exclusion of the initial diagnostic Q&A from the conversation history in subsequent reasoning turns, while preserving the ECG data (image/signal) from the first user turn. When the conversation has progressed beyond the initial diagnostic question (more than 3 turns including the system prompt), the method returns an ECG-only stub (containing only `image`/`signal` keys, without a `role` key) followed by the remaining turns. Model implementations should use `turn.get("role")` instead of `turn["role"]` to safely handle this stub.

    The conversation turns contain the following fields:
    * `role`: the role of the speaker, which can be either one of `"system"`, `"user"`, and `"model"`. Note that the ECG-only stub returned by `get_turns_for_prompt()` does not have a `role` key.
    * For the `system` or `model` role,
        * `text`: the text content of the conversation turn. In other words, the system prompt for the `system` role, and the model response for the `model` role.
    * For the `user` role,
        * `question`: the question asking for the model response.
        * `options`: the list of options for the question.
        * `signal` (optional): the ECG signal input, which is a 500Hz 12-lead 1D signal array. Only provided for the signal-based models, and for the very first question turn (i.e., the `initial_diagnostic_question` step) in the conversation history.
        * `image` (optional): the ECG image input, which is a 12-lead ECG chart image as a PIL image object or base64-encoded string depending on the model requirement. Only provided for the image-based models, and for the very first question turn (i.e., the `initial_diagnostic_question` step) in the conversation history.
> [!NOTE]
> Note that the first turn of the raw conversation history (`Conversation.conversation`) is always the system prompt, and the final turn is always the current user question turn asking for the model response. The very first user question turn (i.e., `Conversation.conversation[1]`) contains `image` or `signal` field for the ECG input.
> However, **always use `conversation.get_turns_for_prompt()` for building prompts** rather than accessing `conversation.conversation` directly -- this ensures the initial diagnostic Q&A is properly excluded from subsequent reasoning turns.
> We strongly recommend you to refer to other pre-existing model implementations for this method to see how to process the conversation history to make the full prompt for the model input.
7. You can also implement any other methods for the model class as needed, such as additional helper methods for processing the ECG input or generating the model response.

### Forced-Commit Inference Mode

The default inference protocol described above is intended for computing all stage-level metrics (Completion, per-stage `_w_gt` accuracies, GT-Reasoning-Based Diagnosis Accuracy, etc.) from a single curated chain per sample.

For a complementary experiment that characterizes **how diagnosis accuracy responds to progressively revealed findings**, we provide an alternative protocol via `--inference-mode forced-commit`:

```bash
ecg-reasoning-benchmark-inference /path/to/data/ \
    --dataset $dataset \
    --model $model_name \
    [--model-variant $model_variant] \
    --ecg-base-dir $ecg_base_dir \
    --output-dir $output_dir_forced_commit \
    --inference-mode forced-commit
```

Under this protocol:
* The `initial_diagnostic_question` step is run identically to the default protocol (the model's actual response is kept in history, ECG modality is attached to that first user turn).
* For **each reasoning loop**, a fresh inference is issued where the prompt contains GT-teacher-forced history through that loop's `criterion_selection` / `finding_identification` / `ecg_grounding`, and the loop's `diagnostic_decision` is asked with the options **restricted to `["Yes", "No"]`** (the `"Further findings are required to confirm the diagnosis"` option is omitted, forcing the model to commit to a binary diagnosis given the evidence it has seen so far).
* A sample with *N* reasoning loops therefore triggers *N + 1* inference calls (IDQ + one per loop).

> [!NOTE]
> The output JSONs are saved in the same directory layout as the default mode, but with three marker fields:
> * `metadata.inference_mode` is set to `"forced-commit"`.
> * Only `initial_diagnostic_question` and each loop's `diagnostic_decision` carry a `model_response` field (the intermediate `criterion_selection` / `finding_identification` / `ecg_grounding` steps are GT-teacher-forced and not inferred by the model).
> * Each loop's `diagnostic_decision` additionally records `options_restricted: ["Yes", "No"]`.
>
> ***Use a different `--output-dir` from your default-mode runs.*** The two schemas are not interchangeable and the forced-commit evaluator described below will refuse to process default-mode outputs (and vice versa).

### Modifying the Prompt Design

The system prompt is defined in `ecg_reasoning_benchmark/inference.py` as a global variable `system_prompt`, which is used as the initial system prompt for all models by default.
In addition, we also append another default prompt for image-based models in the `initial_diagnostic_question` step to give the information about the ECG paper rate (also known as ECG paper speed), which is defined in the `inference` method of the `Inferencer` class in `ecg_reasoning_benchmark/inference.py`.
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
ecg-reasoning-benchmark-evaluate /path/to/results/ \
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
ecg-reasoning-benchmark-evaluate /path/to/results/ \
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

## Evaluating the Forced-Commit Experiment

For model responses curated with `--inference-mode forced-commit` (see *Forced-Commit Inference Mode* above), we provide a dedicated evaluator `gemini-forced-commit` that uses Gemini to judge whether each forced-commit Yes/No response matches the sample-level ground-truth diagnosis (`metadata.dx_label`).
Unlike the default evaluators, it buckets each data point by **finding-coverage ratio** `(loop_idx + 1) / total_loops` into one of four quartile bins `(0, 25%] / (25%, 50%] / (50%, 75%] / (75%, 100%]` and reports per-bin accuracy + counts, along with the IDQ accuracy as a separate "no-reasoning baseline".

```bash
ecg-reasoning-benchmark-evaluate /path/to/forced_commit_results/ \
    --dataset $dataset_list \
    --model $model_name_list \
    --evaluator gemini-forced-commit \
    --gemini-model $gemini_model \
    --use-cache --save-cache --load-cache --save-cache-interval 1 \
    --save-dir $save_dir
```
* The Gemini-specific flags (`--use-cache`, etc.) behave identically to the default Gemini evaluator. The Gemini API cache is shared across both evaluators when the same `--gemini-model` is used, so forced-commit runs reuse any matching cache hits from default-mode runs.
* The evaluator asserts `metadata.inference_mode == "forced-commit"` on every sample; pointing it at default-mode outputs will raise an `AssertionError` immediately.
* CSV output is written to `$save_dir/gemini-forced-commit_$gemini_model/$dataset/{total,$dx}.csv`. Each row corresponds to one model and contains: `idq_{total, correct, accuracy}` and, for each quartile bin, `bin{1..4}_{lo, hi, total, correct, accuracy}`.

### Plotting the Diagnosis-Accuracy vs. Finding-Coverage Curve

After evaluating, use `ecg-reasoning-benchmark-forced-commit-plot` to render the "finding-coverage → diagnosis-accuracy" curve from the evaluator's CSVs:

```bash
ecg-reasoning-benchmark-forced-commit-plot \
    --eval-dir $save_dir/gemini-forced-commit_$gemini_model \
    --dataset $dataset \
    [--dx total | --all-dx] \
    [--models $m1 $m2 ...] \
    [--layout {combined|separate}] \
    [--output figure.pdf]
```
* `--dx` selects which CSV under `<eval-dir>/<dataset>/` to plot (`total` by default; pass a specific diagnosis name to plot per-dx).
* `--all-dx` renders a separate PDF for **every** CSV under `<eval-dir>/<dataset>/` in a single invocation (ignores `--dx`). When combined with `--output`, that path is interpreted as an **output directory** and each PDF is written to `<output>/<dx>{,_separate}.pdf`; otherwise each PDF is written next to its source CSV.
* `--models` filters which rows of the CSV are drawn (default: all rows).
* `--layout combined` (default) plots every requested model on a single axes; `--layout separate` produces a subplot grid with one subplot per model.
* For each model, the per-bin accuracy is drawn as a curve, and the model's IDQ accuracy is drawn as a transparent horizontal dashed line in the same color (a visual anchor for the "no-reasoning baseline"). A neutral-gray dashed entry labeled `IDQ baseline` is added to the legend so the dashed-line convention is self-explanatory in the figure.
* The figure is saved to `--output` (default: `<eval-dir>/<dataset>/<dx>.pdf` for combined, `<dx>_separate.pdf` for separate).
* Figures are styled for direct paper inclusion: colorblind-safe palette (Wong / seaborn `colorblind`), distinct per-model markers for B/W-print legibility, top/right spines removed with a y-only dashed grid, and TrueType font embedding (`pdf.fonttype=42`) so PDFs pass paper-submission font checks.

# Citation
Please cite this work as:
```
@article{oh2026ecg,
  title={ECG-Reasoning-Benchmark: A Benchmark for Evaluating Clinical Reasoning Capabilities in ECG Interpretation},
  author={Oh, Jungwoo and Chung, Hyunseung and Lee, Junhee and Kim, Min-Gyu and Yoon, Hangyul and Lee, Ki Seong and Lee, Youngchae and Yeo, Muhan and Choi, Edward},
  journal={arXiv preprint arXiv:2603.14326},
  year={2026}
}
```
