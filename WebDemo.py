from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import gradio as gr
import os


os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"
USE_CUDA = torch.cuda.is_available()
device_ids_parallel = [0]
device = torch.device("cuda:{}".format(device_ids_parallel[0]) if USE_CUDA else "cpu")

# 初始化
peft_model_id = "CMLM/ZhongJing-2-1_8b"
base_model_id = "Qwen/Qwen1.5-1.8B-Chat"
model = AutoModelForCausalLM.from_pretrained(base_model_id, device_map="auto")
model.load_adapter(peft_model_id)
tokenizer = AutoTokenizer.from_pretrained(
    "CMLM/ZhongJing-2-1_8b",
    padding_side="right",
    trust_remote_code=True,
    pad_token=''
)

#单轮
def single_turn_chat(question):
    prompt = f"Question: {question}"
    messages = [
        {"role": "system", "content": "You are a helpful TCM medical assistant named 仲景中医大语言模型, created by 医哲未来 of Fudan University."},
        {"role": "user", "content": prompt}
    ]
    input = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([input], return_tensors="pt").to(device)
    generated_ids = model.generate( model_inputs.input_ids,max_new_tokens=512)
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response

#多轮
def multi_turn_chat(question, chat_history=None):
    if not isinstance(question, str):
        raise ValueError("The question must be a string.")
    
    if chat_history is None or chat_history == []:
        chat_history = [{"role": "system", "content": "You are a helpful TCM medical assistant named 仲景中医大语言模型, created by 医哲未来 of Fudan University."}]
    
    chat_history.append({"role": "user", "content": question})
    
    # Apply the chat template and prepare the input
    inputs = tokenizer.apply_chat_template(chat_history, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([inputs], return_tensors="pt").to(device)
    
    try:
        # Generate the response from the model
        outputs = model.generate(model_inputs.input_ids, max_new_tokens=512)
        generated_ids = outputs[:, model_inputs.input_ids.shape[-1]:]
        response = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    except Exception as e:
        raise RuntimeError("Error in model generation: " + str(e))
    
    # Append the assistant's response to the chat history
    chat_history.append({"role": "assistant", "content": response})
    
    # Format the chat history for output
    # formatted_history = [(entry['role'], entry['content']) for entry in chat_history]
    tempass = ""
    tempuser = ""
    formatted_history = []
    for entry in chat_history:
        if entry['role'] == 'user':
            tempuser = entry['content']
        elif entry['role'] == 'assistant':
            tempass = entry['content']
            temp = tempuser,tempass
            formatted_history.append(temp)
    
    return formatted_history, chat_history


def clear_history():
    return [], []

# 单轮界面
single_turn_interface = gr.Interface(
    fn=single_turn_chat,
    inputs=["text"],
    outputs="text",
    title="仲景GPT-V2-1.8B 单轮对话",
    description="博极医源，精勤不倦。Unlocking the Wisdom of Traditional Chinese Medicine with AI."
)

# 多轮界面
with gr.Blocks() as multi_turn_interface:
    chatbot = gr.Chatbot(label="仲景GPT-V2-1.8B 多轮对话")
    state = gr.State([])
    with gr.Row():
        with gr.Column(scale=6):
            user_input = gr.Textbox(label="输入", placeholder="输入你的问题")
        with gr.Column(scale=1):
            submit_button = gr.Button("提交")
    submit_button.click(multi_turn_chat, inputs=[user_input, state], outputs=[chatbot, state])
    clear_button = gr.Button("清除聊天记录")
    clear_button.click(clear_history, outputs=[chatbot, state])

# 主界面
def switch_mode(mode):
    if mode == "单轮对话":
        return gr.update(visible=True), gr.update(visible=False)
    else:
        return gr.update(visible=False), gr.update(visible=True)
    
switch_button = gr.Radio(["单轮对话", "多轮对话"], label="选择对话模式", value="单轮对话")

demo = gr.Blocks()

with demo:
    gr.Markdown("# 仲景GPT 网页版 Demo")
    with gr.Row():
        switch_button.render()
    with gr.Row():
        with gr.Column(visible=True) as single_turn_col:
            single_turn_interface.render()
        with gr.Column(visible=False) as multi_turn_col:
            multi_turn_interface.render()
    switch_button.change(switch_mode, inputs=switch_button, outputs=[single_turn_col, multi_turn_col])

demo.launch(share=True)
