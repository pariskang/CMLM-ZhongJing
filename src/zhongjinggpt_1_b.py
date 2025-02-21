# -*- coding: utf-8 -*-
"""ZhongJingGPT-1.B.ipynb

Automatically generated by Colaboratory.


Original file is located at
    https://colab.research.google.com/drive/1DCPomUsfTxqkqxKpK-AIGvBSPbkOm7R3

# ZhongJingGPT-2-1.8b

A Traditional Chinese Medicine large language model, inspired by the wisdom of the eminent representative of ancient Chinese medical scholars, Zhang Zhongjing. This model aims to illuminate the profound knowledge of Traditional Chinese Medicine, bridging the gap between ancient wisdom and modern technology, and providing a reliable and professional tool for the Traditional Chinese Medical fields. However, all generated results are for reference only and should be provided by experienced professionals for diagnosis and treatment results and suggestions.
"""

import torch
print(torch.cuda.is_available())

!pip install transformers huggingface_hub accelerate peft

"""# You should restart colab and the run the following code."""

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Set the device
device = "cuda"  # replace with your device: "cpu", "cuda", "mps"

# Initialize model and tokenizer
peft_model_id = "CMLL/ZhongJing-2-1_8b"
base_model_id = "Qwen/Qwen1.5-1.8B-Chat"
model = AutoModelForCausalLM.from_pretrained(base_model_id, device_map="auto")
model.load_adapter(peft_model_id)
tokenizer = AutoTokenizer.from_pretrained(
    "CMLL/ZhongJing-2-1_8b",
    padding_side="right",
    trust_remote_code=True,
    pad_token=''
)

def get_model_response(question, context):
    # Create the prompt
    prompt = f"Question: {question}\nContext: {context}"
    messages = [
        {"role": "system", "content": "You are a helpful TCM assistant named 仲景中医大语言模型."},
        {"role": "user", "content": prompt}
    ]

    # Prepare the input
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    # Generate the response
    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=512
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    # Decode the response
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response

# Loop to get user input and provide model response
while True:
    user_question = input("Enter your question (or type 'exit' to stop): ")
    if user_question.lower() == 'exit':
        break
    user_context = input("Enter context (or type 'none' if no context): ")
    if user_context.lower() == 'none':
        user_context = ""

    print("Model is generating a response, please wait...")
    model_response = get_model_response(user_question, user_context)
    print("Model's response:", model_response)

