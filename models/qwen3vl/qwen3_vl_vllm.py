import os
import torch
from transformers import AutoProcessor

from .. import BaseModel, register_model

@register_model("qwen3-vl-vllm")
class Qwen3VLVLLMModel(BaseModel):
    def __init__(
        self,
        model_variant: str = "235B-A22B-Instruct-FP8",
    ):
        # Import vllm here to avoid issues if not using this model
        from vllm import LLM, SamplingParams

        os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

        self.model_variant = model_variant

        model_id = f"Qwen/Qwen3-VL-{model_variant}"

        self.model = LLM(
            model=model_id,
            mm_encoder_tp_mode="data",
            enable_expert_parallel=True,
            tensor_parallel_size=torch.cuda.device_count(),
            seed=0,
        )
        self.processor = AutoProcessor.from_pretrained(model_id)

        self.sampling_params = SamplingParams(
            temperature=0,
            max_tokens=300,
            top_k=-1,
            stop_token_ids=[],
        )

    def prepare_inputs_for_vllm(self, messages, processor):
        # Import process_vision_info here to avoid issues if not using this model
        from qwen_vl_utils import process_vision_info

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs, video_kwargs = process_vision_info(
            messages,
            image_patch_size=processor.image_processor.patch_size,
            return_video_kwargs=True,
            return_video_metadata=True,
        )

        mm_data = {}
        if image_inputs is not None:
            mm_data["image"] = image_inputs
            
        return {
            "prompt": text,
            "multi_modal_data": mm_data,
            "mm_processor_kwargs": video_kwargs
        }

    def get_response(self, conversation, enable_condensed_chat: bool = False, verbose: bool = False) -> str:
        assert (
            conversation.conversation[0]["role"] == "system"
        ), "The first turn in the conversation must be from the system."
        assert (
            conversation.conversation[-1]["role"] == "user"
        ), "The last turn in the conversation must be from the user."
        assert (
            "image" in conversation.conversation[1]
        ), "The conversation must contain an ECG image in the first user turn."

        system = conversation.conversation[0]["text"]
        messages = [{"role": "system", "content": [{"type": "text", "text": system}]}]
        for i, turn in enumerate(conversation.conversation[1:]):
            if turn["role"] == "user":
                user_text = f"Question: {turn['question']}\n\n"

                do_add_options = False
                # do not add options in previous turns to reserve context length
                if enable_condensed_chat:
                    if i == len(conversation.conversation[1:]) - 1:
                        do_add_options = True
                else:
                    do_add_options = True

                if do_add_options:
                    if i == 0:
                        user_text += "Options:\n"
                    elif "select all possible leads" in turn["question"].lower():
                        user_text += (
                            "This question may have multiple correct answers from the following options:\n"
                        )
                    else:
                        user_text += "This question has one of the following options as the correct answer:\n"
                    for option in turn["options"]:
                        user_text += f"- {option}\n"
                    user_text += "Your response must be **ONLY** the full text of the selected option. Do not "
                    user_text += "include any uncertainty, explanation, reasoning, or extra words."

                if i == 0:
                    user = {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": f"data:image/png;base64,{turn['image']}"},
                            {"type": "text", "text": user_text},
                        ],
                    }
                else:
                    user = {"role": "user", "content": [{"type": "text", "text": user_text}]}
                messages.append(user)
            elif turn["role"] == "model":
                messages.append({"role": "assistant", "content": [{"type": "text", "text": turn["text"]}]})

        if verbose:
            print(f"\nQuestion: {conversation.conversation[-1]['question']}")

        response = self.generate(messages)

        if verbose:
            print(f"Response: {response}")

        return response

    def generate(self, messages, **kwargs):
        inputs = [self.prepare_inputs_for_vllm(message, self.processor) for message in [messages]]

        outputs = self.model.generate(inputs, sampling_params=self.sampling_params)

        response = outputs[0].outputs[0].text

        return response

    @classmethod
    def build_model(cls, model_variant="32B-Instruct", **kwargs):
        return cls(model_variant=model_variant)