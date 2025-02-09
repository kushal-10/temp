from datetime import timedelta
from typing import List
from pydantic import BaseModel
from restack_ai.agent import agent, import_functions, log

def create_query(message):
    try:
        if not message.chapters:
            return f"This is my goal - {message.goal}, this this the expected timeline - {message.timeline}. Based on this information, find the relevant chapters from the book."
        else:
            return f"This is my goal - {message.goal}, this this the expected timeline - {message.timeline}. Based on this information, find the relevant passages from the book. Here are the chapters: {message.chapters}"
    except Exception as e:
        log.error(f"Error in create_query: {e}")
        return ""
    
# def create_query(message):

#     return f"This is my goal - {message.goal}, this this the expected timeline - {message.timeline}. Based on this information, find the relevant passages from the book."


with import_functions():
    from src.functions.llm_chat import llm_chat, LlmChatInput, Message
    from src.functions.book1 import lookup_book


class MessageEvent(BaseModel):
    content: str


class EndEvent(BaseModel):
    end: bool


@agent.defn()
class AgentRag:
    def __init__(self) -> None:
        self.end = False
        self.messages = []

    @agent.event
    async def message(self, message: MessageEvent) -> List[Message]:
        log.info(f"Received message: {message.content}")

        query = create_query(message)
        print(f"query: {query}")
        book_info = await agent.step(
            lookup_book,
            start_to_close_timeout=timedelta(seconds=120)
        )


        system_content = f"You are a helpful assistant given a goal {message.goal}, a timeline {message.timeline} and relevant chapters from a book {book_info}, Given these three things, generate a JSON file with an action plan. The output should be in JSON format. The JSON should have following fields - day, goal, and action. The day is number of day from the timeline - Day 1, Day 2 etc.. The goal should contain a specific subgoal based on the original goal. The action should contain an achievable task that can be done towards goal of the day."

        self.messages.append(Message(role="user", content=message.content or ""))

        completion = await agent.step(
            llm_chat,
            LlmChatInput(messages=self.messages, system_content=system_content),
            start_to_close_timeout=timedelta(seconds=120),
        )
        # temp
        log.info(f"completion: {completion}")

        self.messages.append(
            Message(
                role="assistant", content=completion.choices[0].message.content or ""
            )
        )

        return self.messages

    @agent.event
    async def end(self, end: EndEvent) -> EndEvent:
        log.info("Received end")
        self.end = True
        return {"end": True}

    @agent.run
    async def run(self, input: dict):
        await agent.condition(lambda: self.end)
        return
