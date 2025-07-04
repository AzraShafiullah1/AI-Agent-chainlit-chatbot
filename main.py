import os
from dotenv import load_dotenv
import chainlit as cl
from agents import Agent, AsyncOpenAI, OpenAIChatCompletionsModel, RunConfig, Runner
from openai.types.responses import ResponseTextDeltaEvent

# Load environment variables
load_dotenv()

# Get Gemini API key
gemini_api_key = os.getenv("GEMINI_API_KEY")

# Setup Gemini-compatible client
external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

# Model setup
model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client
)

config = RunConfig(
    model=model,
    model_provider=external_client,
    tracing_disabled=True
)

frontend_agent = Agent(
    name='Frontend Expert Agent',
    instructions="You are a frontend expert. You help with UI/UX using HTML, CSS, JavaScript, TypeScript, React Native, React, Next.js, React.js and Tailwind CSS, Python."
)

@cl.on_chat_start
async def handle_start():
    # Initialize user session history
    cl.user_session.set("history", [])
    await cl.Message(content="Welcome to the Frontend Expert Agent! How can I assist you today?").send()

@cl.on_message
async def handle_message(message: cl.Message):

    history = cl.user_session.get("history")

    history.append({"role": "user", "content": message.content})

    message = cl.Message(content= "")
    await message.send()

    result=  Runner.run_streamed(
        frontend_agent,
        input=message.content,
        run_config=config
    )
     
    async for event in result.stream_events():
        if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
            await message.stream_token(event.data.delta)
            
    history.append({"role": "assistant", "content": result.final_output})
    cl.user_session.set("history", history)
    await cl.Message(content=result.final_output).send()
    print("Assistant:", result.final_output)

  