import os
import chainlit as cl
from dotenv import load_dotenv
from openai import OpenAI
from agents import Agent, Runner, function_tool


load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


@function_tool
def answer_question(query: str) -> str:
    """Answer academic questions as a helpful tutor."""
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful academic tutor for students."},
            {"role": "user", "content": query}
        ]
    )
    return response.choices[0].message.content.strip()


@function_tool
def study_tips(topic: str) -> str:
    """Provide effective study tips for a given topic."""
    prompt = f"Give 5 effective study tips for {topic}."
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are an educational coach for students."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content.strip()


@function_tool
def summarize_text(passage: str) -> str:
    """Summarize a given text in 3-4 lines."""
    prompt = f"Summarize the following text in 3-4 lines:\n\n{passage}"
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a summarization assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content.strip()


student_agent = Agent(
    name="Smart Student Assistant",
    instructions="You are a helpful AI assistant for students. You can answer academic questions, provide study tips, and summarize content.",
    tools=[answer_question, study_tips, summarize_text],
)

@cl.on_chat_start
async def start_chat():
    await cl.Message(
        content=(
            "# Smart Student Assistant\n"
            "Welcome! This assistant can:\n"
            "- Answer academic questions\n"
            "- Provide effective study tips\n"
            "- Summarize short text passages\n\n"
            "Please enter your query to begin."
        )
    ).send()


@cl.on_message
async def handle_message(message: cl.Message):
    query = message.content
    result = Runner.run_sync(student_agent, query)
    await cl.Message(content=result.final_output).send()
