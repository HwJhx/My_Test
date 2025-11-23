import os
import asyncio
from pathlib import Path
# Get keys for your project from the project settings page: https://cloud.langfuse.com
from dotenv import load_dotenv

# 获取当前文件所在目录，确保能找到 .env 文件
env_path = Path(__file__).parent / '.env'
load_dotenv(dotenv_path=env_path)

# Keys are now loaded from .env file




from langfuse import get_client

langfuse = get_client()
 
# Verify connection
if langfuse.auth_check():
    print("Langfuse client is authenticated and ready!")
else:
    print("Authentication failed. Please check your credentials and host.")



from openinference.instrumentation.dspy import DSPyInstrumentor
 
DSPyInstrumentor().instrument()


import dspy
lm = dspy.LM(
        model="openai/moonshotai/Kimi-K2-Thinking",
        # api_key=openai_api_key,
        api_base="https://api.siliconflow.cn",
        model_type="chat",
        max_tokens=4096,
        cache=False,
        temperature=0.1,  # 进一步增加随机性
        )
dspy.configure(lm=lm, usage_tracker=DSPyInstrumentor().instrument())




from langfuse import observe, propagate_attributes
def llm_infer(input):
    math = dspy.ChainOfThought("question -> answer: str")
    return math(question=input)
    

def llm_analysis(input):
    math = dspy.ChainOfThought("question -> answer: str")
    # 在问题中添加中文回答的要求
    question_with_instruction = f"{input}\n请用中文详细回答。"
    return math(question=question_with_instruction)

def update_langfuse_usage(info:str):
    """
    从 DSPy 的历史记录中提取 token 使用情况，并更新到 Langfuse
    """
    # Debug: Check if DSPy captured usage in history
    print("\n{}".format(info))
    if lm.history:
        last_interaction = lm.history[-1]
        usage = last_interaction.get('usage')
        completion_tokens = usage.get('completion_tokens')
        prompt_tokens = usage.get('prompt_tokens')
        total_tokens = usage.get('total_tokens')
        print(f"Last Usage in History: {last_interaction.get('usage')}")
        print(f"Last Response Keys: {last_interaction.keys()}")

        # 构建 langfuse 所需的 usage 格式
        usage = {
            "input": prompt_tokens,  # 输入token数(包含prompt和原始输入)
            "output": completion_tokens,  # 输出token数
            "total": total_tokens  # 总token数
        }
        
        generation = langfuse.update_current_generation(
            usage_details=usage
        )
        
        # langfuse.flush()
    else:
        print("No history found in lm.")



@observe(as_type="generation", name="llm-infer-call")
async def my_llm_pipeline():
    # Add additional attributes (user_id, session_id, metadata, version, tags) to all spans created within this execution scope
 
    input = "请写一个20字左右的散文"
    # YOUR APPLICATION CODE HERE
    result = llm_infer(input)

    # Update the trace input and output
    langfuse.update_current_trace(
        input=input,
        output=result,
    )
    update_langfuse_usage(input)

@observe(as_type="generation", name="llm-analysis-call")
async def my_llm_pipeline222():

    input = "Two dice are tossed. What is the probability that the sum equals two?"
    result = llm_analysis(input)
    # Update the trace input and output
    langfuse.update_current_trace(
        input=input,
        output=result,
    )
    update_langfuse_usage(input)

@observe()
async def main():
    # Add additional attributes (user_id, session_id, metadata, version, tags) to all spans created within this execution scope
    with propagate_attributes(
        user_id="user_123",
        session_id="session_abc",
        tags=["agent", "my-trace"],
        metadata={"email": "user@langfuse.com"},
        version="1.0.0"
    ):
        await my_llm_pipeline()
        await my_llm_pipeline222()

asyncio.run(main())

