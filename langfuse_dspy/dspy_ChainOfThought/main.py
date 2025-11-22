import os
import asyncio
# Get keys for your project from the project settings page: https://cloud.langfuse.com
os.environ["LANGFUSE_PUBLIC_KEY"] = "pk-lf-96a76ed5-18dc-45f9-984e-1954963b0e6d" 
os.environ["LANGFUSE_SECRET_KEY"] = "sk-lf-200016fe-78f8-4ee8-b652-95323fa2256a" 
os.environ["LANGFUSE_BASE_URL"] = "https://us.cloud.langfuse.com" # 🇺🇸 US region
 
# Your OpenAI key
os.environ["OPENAI_API_KEY"] = "sk-eqzeuqzxixvegmqxvzotaeulteeciduvmrttalyizwylotfv" 




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



# math = dspy.ChainOfThought("question -> answer: float")
# math(question="Two dice are tossed. What is the probability that the sum equals two?")


# math = dspy.ChainOfThought("question -> answer: str")
# math(question="请写一个300字左右的中文散文")

from langfuse import observe, propagate_attributes
def llm_infer(input):
    math = dspy.ChainOfThought("question -> answer: str")
    return math(question=input)
    



@observe(as_type="generation")
async def my_llm_pipeline():
    # Add additional attributes (user_id, session_id, metadata, version, tags) to all spans created within this execution scope
    with propagate_attributes(
        user_id="user_123",
        session_id="session_abc",
        tags=["agent", "my-trace"],
        metadata={"email": "user@langfuse.com"},
        version="1.0.0"
    ):
 
        input = "请写一个20字左右的散文"
        # YOUR APPLICATION CODE HERE
        result = llm_infer(input)
 
        # Update the trace input and output
        langfuse.update_current_trace(
            input=input,
            output=result,
        )
        # Debug: Check if DSPy captured usage in history
        print("\n--- Debug Info ---")
        if lm.history:
            last_interaction = lm.history[-1]
            usage = last_interaction.get('usage')
            completion_tokens = usage.get('completion_tokens')
            prompt_tokens = usage.get('prompt_tokens')
            total_tokens = usage.get('total_tokens')
            print(f"Last Usage in History: {last_interaction.get('usage')}")  # <--- 这里打印出来的
            print(f"Last Response Keys: {last_interaction.keys()}")
            #     # Mock token usage data
            # usage = {
            #     "prompt_tokens": 50,  # Replace with actual token count
            #     "completion_tokens": 49,  # Replace with actual token count
            #     "total_tokens": 99  # Replace with actual token count
            # }

            # Mock token usage data
            usage = {
                "input": 50,  # Replace with actual token count
                "output": 49,  # Replace with actual token count
                "total": 99  # Replace with actual token count
            }
            
            generation = langfuse.update_current_generation(
                usage_details=usage
            )
            
            # langfuse.flush()
        else:
            print("No history found in lm.")

asyncio.run(my_llm_pipeline())

