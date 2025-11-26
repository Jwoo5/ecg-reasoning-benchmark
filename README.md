# ecg-reasoning-benchmark
1. Open-TSLM sota model for ecg (OpenTSLM/llama3b-ecg-flamingo) OpenTSLM_Inference.py currently working code in OpenTSLM repo src directory, based on ECGQACoTQADataset of OpenTSLM repo.

2. Note: 100hz sampled ecgs used as input for OpenTSLM model (12, 1000), can use 100hz data for PTB-XL but must downsample for mimic-iv-ecg. Many of the outputs produce none despite providing options and requiring the model to choose from options. 

3. Todo: Integrate into current framework