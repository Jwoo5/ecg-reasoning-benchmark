"""Works in src directory of OpenTSLM repo. Based on ECGQACoTQADataset of OpenTSLM repo."""

import sys
import os
import json
import torch
import numpy as np
import wfdb
import pdb 

from huggingface_hub import hf_hub_download

from prompt.full_prompt import FullPrompt
from prompt.text_prompt import TextPrompt
from prompt.text_time_series_prompt import TextTimeSeriesPrompt
from model.llm.OpenTSLMFlamingo import OpenTSLMFlamingo

def load_and_process_ecg_ptbxl(path, target_len=1000):
    """
    Load a PTB-XL ECG file via wfdb and return a (12, target_len) numpy array,
    normalized per lead (mean/std) similar to ECGQACoTQADataset._process_ecg_lead.
    """
    record = wfdb.rdrecord(path)
    ecg_signal = record.p_signal  # shape: (samples, leads)

    if ecg_signal.ndim != 2 or ecg_signal.shape[1] < 12:
        raise ValueError(f"Expected at least 12 leads, got shape {ecg_signal.shape}")

    # Transpose to (leads, samples)
    ecg = ecg_signal.T.astype(np.float32)  # (C, L)

    # Downsample if > 1000 samples (assume 500 Hz -> ~100 Hz) How they did it in ECG-QA-CoT
    L = ecg.shape[1]
    if L > target_len:
        stride = L // target_len
        if stride <= 0:
            stride = 1
        ecg = ecg[:, ::stride]
        L = ecg.shape[1]

    # Pad or cut to exactly target_len
    if L < target_len:
        new_ecg = np.zeros((ecg.shape[0], target_len), dtype=np.float32)
        new_ecg[:, :L] = ecg
        ecg = new_ecg
    elif L > target_len:
        ecg = ecg[:, :target_len]

    # Per-lead normalization
    norm_ecg = np.zeros_like(ecg)
    means = []
    stds = []
    for i in range(ecg.shape[0]):
        lead = ecg[i]
        m = float(lead.mean())
        s = float(lead.std())
        if s < 1e-6:
            norm = lead - m
        else:
            norm = (lead - m) / s
        norm_ecg[i] = norm
        means.append(m)
        stds.append(s)

    return norm_ecg, means, stds

def extract_questions_from_json(json_obj):
    """
    Extract all questions from the JSON structure into a flat list.
    Each entry is a dict containing:
      - id: a unique identifier within the file
      - question_type
      - question
      - options
    """
    questions = []
    data = json_obj.get("data", {})

    # 1) initial_diagnostic_question
    if "initial_diagnostic_question" in data:
        q = data["initial_diagnostic_question"]
        questions.append({
            "id": "initial_diagnostic_question",
            "question_type": q.get("question_type", "initial_diagnostic_question"),
            "question": q["question"],
            "options": q.get("options", []),
            "answer": q.get("answer")
        })

    # 2) path_* blocks (e.g., path_1)
    for path_key, path_val in data.items():
        if not path_key.startswith("path_"):
            continue

        for step_idx, step in enumerate(path_val):
            for key, node in step.items():
                if key == "grounding":

                    for g_idx, gq in enumerate(node):
                        qid = f"{path_key}_step{step_idx}_{key}{g_idx}"
                        questions.append({
                            "id": qid,
                            "question_type": gq.get("question_type", "grounding"),
                            "question": gq["question"],
                            "options": gq.get("options", []),
                            "answer": gq.get("answer")
                        })
                else:
                    # criterion_selection / finding / decision
                    qid = f"{path_key}_step{step_idx}_{key}"
                    questions.append({
                        "id": qid,
                        "question_type": node.get("question_type", key),
                        "question": node["question"],
                        "options": node.get("options", []),
                        "answer": node.get("answer")
                    })

    return questions

def build_prompt_for_question(metadata, q_spec, norm_ecg, means, stds):
    """
    Build (FullPrompt) for a single question spec using ECG-QA-CoT-style formatting.
    """
    # Clinical context 
    ecg_id = metadata.get("ecg_id", "Unknown")
    target_dx = metadata.get("target_dx", "Unknown diagnosis")

    clinical_context = (
        f"12-lead clinical ECG recording."
    )

    question_text = q_spec["question"]
    options = q_spec.get("options", [])

    pre_prompt_str = f"""You are an expert cardiologist analyzing an ECG (electrocardiogram). 

Clinical Context: {clinical_context}

Your task is to examine the ECG signal and answer the following medical question:

Question: {question_text}

Instructions:
- Begin by analyzing the time series without assuming a specific answer.
- Think step-by-step about what the observed patterns suggest regarding the cardiac condition.
- Write your rationale as a single, natural paragraph — do not use bullet points, numbered steps, or section headings.
- Do **not** mention any final answer until the very end.
- Consider the ECG morphology, intervals, and any abnormalities that relate to the question.
"""

    # Time-series prompts: one per lead
    lead_names = ["I","II","III","aVR","aVL","aVF","V1","V2","V3","V4","V5","V6"]
    ts_prompts = []
    for idx, lead_signal in enumerate(norm_ecg):
        lead_name = lead_names[idx] if idx < len(lead_names) else f"Lead_{idx}"
        mean_val = means[idx]
        std_val = stds[idx]
        label = (
            f"This is ECG Lead {lead_name}, it has mean {mean_val:.4f} "
            f"and std {std_val:.4f}:"
        )
        ts_prompts.append(
            TextTimeSeriesPrompt(label, lead_signal.tolist())
        )

    # Post-prompt: options + "Answer: "
    if options:
        options_text = ", ".join(options)
        post_prompt_str = f"""
Based on your analysis of the ECG data, select your answer from the following options. You must select from the options:
{options_text}

Make sure that your last word is the answer. You MUST end your response with "Answer: "
""".strip()
    else:
        post_prompt_str = """
Based on your analysis of the ECG data, provide your answer.
Make sure that your last word is the answer. You MUST end your response with "Answer: "
""".strip()
    # pdb.set_trace()
    full_prompt = FullPrompt(
        TextPrompt(pre_prompt_str),
        ts_prompts,
        TextPrompt(post_prompt_str)
    )
    return full_prompt

def split_reasoning_and_answer(output: str):
    """
    Split the model output into:
      - reasoning: everything before the last 'Answer:'
      - answer: everything after the last 'Answer:' on that line

    If 'Answer:' is not found, reasoning = output, answer = "".
    """
    if output is None:
        return "", ""

    marker = "Answer:"
    idx = output.rfind(marker)
    if idx == -1:
        # No "Answer:" found
        return output.strip(), ""

    reasoning = output[:idx].strip()
    answer_part = output[idx + len(marker):].strip()

    answer_part = answer_part.split("\n", 1)[0].strip()

    if answer_part.endswith((".", "!", "?")):
        answer_part = answer_part[:-1].strip()

    return reasoning, answer_part

def main():
    device = "cuda"
    BASE_LLM_ID = "meta-llama/Llama-3.2-3B"
    CHECKPOINT_REPO_ID = "OpenTSLM/llama3b-ecg-flamingo"
    CHECKPOINT_FILENAME = "best_model.pt"
    ecg_base_path = "/nfs_edlab/hschung/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"
    data_dir = "/nfs_edlab/hschung/ecg-reasoning-benchmark/data/ptbxl/anterior_ischemia" 
    results_dir = "./result"  
    os.makedirs(results_dir, exist_ok=True)

    # 1) Init model once
    print(f"Using device: {device}")
    print(f"Initializing model architecture using base: {BASE_LLM_ID}...")
    model = OpenTSLMFlamingo(
        device=device,
        llm_id=BASE_LLM_ID
    )
    print("Model architecture built.")

    print(f"Downloading checkpoint from {CHECKPOINT_REPO_ID}...")
    checkpoint_path = hf_hub_download(
        repo_id=CHECKPOINT_REPO_ID,
        filename=CHECKPOINT_FILENAME
    )
    # uses map_location="cpu" internally to avoid OOM
    model.load_from_file(checkpoint_path)
    model.to(device)
    model.eval()

    # 2) Iterate over all JSON files in data_dir
    for fname in sorted(os.listdir(data_dir)):
        if not fname.endswith(".json"):
            continue

        json_path = os.path.join(data_dir, fname)
        print(f"\nProcessing file: {json_path}")

        with open(json_path, "r") as f:
            qa_json = json.load(f)

        metadata = qa_json.get("metadata", {})
        ecg_id = metadata.get("ecg_id")
        ecg_num_int = int(ecg_id)
        dir_num = (ecg_num_int // 1000) * 1000
        path = os.path.join(ecg_base_path, "records100", f"{dir_num:05d}", f"{ecg_num_int:05d}_lr")
        
        norm_ecg, means, stds = load_and_process_ecg_ptbxl(path)

        questions = extract_questions_from_json(qa_json)
        print(f"  Found {len(questions)} questions in this JSON.")

        # Run model for each question and collect outputs
        results = {
            "metadata": metadata,
            "model_name": CHECKPOINT_REPO_ID,
            "base_llm": BASE_LLM_ID,
            "responses": []
        }

        for q_spec in questions:
            qid = q_spec["id"]
            print(f"   - Running question: {qid} ({q_spec['question_type']})")

            full_prompt = build_prompt_for_question(
                metadata, q_spec, norm_ecg, means, stds
            )

            with torch.inference_mode(), torch.cuda.amp.autocast(dtype=torch.float16):
                answer_text = model.eval_prompt(
                    full_prompt,
                    max_new_tokens=500
                )

            model_reasoning, model_answer = split_reasoning_and_answer(answer_text)

            results["responses"].append({
                "id": qid,
                "question_type": q_spec["question_type"],
                "question": q_spec["question"],
                "options": q_spec.get("options", []),
                "answer": q_spec.get("answer"),
                "model_output": answer_text,       
                "model_reasoning": model_reasoning, 
                "model_answer": model_answer      
            })

        # Save results JSON next to input name
        out_path = os.path.join(results_dir, fname)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  Saved results to: {out_path}")

if __name__ == "__main__":
    main()
