# ecg-reasoning-benchmark
1. Open-TSLM sota model for ecg (OpenTSLM/llama3b-ecg-flamingo) integrated into branch 

2. Note: 100hz sampled ecgs used as input for OpenTSLM model (12, 1000), using 100hz data for PTB-XL but must downsample for mimic-iv-ecg (based on the ECG-QA-Cot dataset in OpenTSLM, using every 5th timestep of the 500hz sampled datset). Many of the outputs produce none despite providing options and requiring the model to choose from options. 

3. OpenTSLM may be overfitting to ECG-QA-CoT, which was based on PTB-XL and not Mimic-iv-ECG (the authors used the original ECG-QA dataset based on PTB-XL to make ECG-QA-CoT)

4. https://github.com/StanfordBDHG/OpenTSLM/issues/27 what someone mentioned about the output of OpenTSLM, which I agree: Hello, I've experimented with several of the pre-trained models and found them to be very over-fit to the 'autoregressive language modeling' training objective. ECG-trained models prompted with other medical data streams or only slightly different prompts than the ones used during training will consistently output gibberish even when in-context examples are provided.
