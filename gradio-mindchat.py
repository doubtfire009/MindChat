import gradio as gr
from vllm import LLM, SamplingParams
from modelscope import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import os


os.environ["no_proxy"] = "localhost,127.0.0.1,::1"
os.environ['VLLM_USE_MODELSCOPE']='True'

class mindchat_llm():
    def __init__(self):
        # embedding_model_path = '/www/llm_model/X-D-Lab/MindChat-Qwen-7B-v2'
        llm_model_path = '/www/llm_model/X-D-Lab/MindChat-Qwen-7B-v2'
        device = 'GPU'
        print("=== load llm ===")
        # sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
        # llm = LLM(model=llm_model_path, trust_remote_code=True, gpu_memory_utilization=0.9)
        self.tokenizer = AutoTokenizer.from_pretrained(llm_model_path, revision='v1.0.1', trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(llm_model_path, revision='v1.0.1', device_map="auto", trust_remote_code=True, fp16=True).eval()
        model.generation_config = GenerationConfig.from_pretrained(llm_model_path, revision='v1.0.1', trust_remote_code=True) # 可指定不同的生成长度、top_p等相关超参
        self.mindchat_model = model
    def create_history_chat(self, question, bot):
        chat_history = []
        for item in bot:
            chat_history.append((item[0], item[1]))

        response, history = self.model.chat(self.tokenizer, question, history=None)
        print("------test------")
        print(response)
        print("-----history-------")
        print(history)
        return response

mt_llm = mindchat_llm()

def doChatbot(message, bot):

    answer = mt_llm.create_history_chat(message, bot)
    # if "我: " in answer:
    #     res = answer.split("我: ")[-1]
    # elif "AI:" in answer:
    #     res = answer.split("AI:")[-1]
    # elif "？" in answer:
    #     res = answer.split("？")[-1]
    # elif "?" in answer:
    #     res = answer.split("?")[-1]
    # else:
    #     res = answer
    return res

def start_chatbot():
    gr.ChatInterface(
        fn=doChatbot,
        chatbot=gr.Chatbot(height=500, value=[]),
        textbox=gr.Textbox(placeholder="请输入您的问题", container=False, scale=7),
        title="老白心灵抚慰助手（MindChat）",
        theme="soft",
        submit_btn="发送",
        clear_btn="清空"
    ).queue().launch(server_port=7000, server_name='0.0.0.0')

# if __name__ == "__main__":


start_chatbot()