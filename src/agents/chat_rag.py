from datetime import timedelta
from typing import List
from pydantic import BaseModel
from restack_ai.agent import agent, import_functions, log
from src.books.atomic_habits import atomic_habits
from src.books.deep_work import deep_work
from src.books.think_and_grow_rich import think_and_grow_rich

def create_query(message):
    if not message.chapters:
        return f"This is my goal - {message.goal}, this this the expected timeline - {message.timeline}. Based on this information, find the relevant chapters from the book."
    else:
        return f"This is my goal - {message.goal}, this this the expected timeline - {message.timeline}. Based on this information, find the relevant passages from the book. Here are the chapters: {message.chapters}"

with import_functions():
    from src.functions.llm_chat import llm_chat, LlmChatInput, Message
    from src.functions.lookup_sales import lookupSales
    from src.functions.book1 import lookup_book

BOOKS = {
    "think-and-grow-rich": think_and_grow_rich,
    "atomic-habits": atomic_habits, 
    "deep-work": deep_work
}


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

        book_info = await agent.step(
            lookup_book(create_query(message)), start_to_close_timeout=timedelta(seconds=120)
        )

        system_content = f"You are a helpful assistant given a goal {message.goal}, a timeline {message.timeline} and relevant chapters from a book {book_info}, Given these three things, generate a JSON file with an action plan."

        self.messages.append(Message(role="user", content=message.content or ""))

        completion = await agent.step(
            llm_chat,
            LlmChatInput(messages=self.messages, system_content=system_content),
            start_to_close_timeout=timedelta(seconds=120),
        )

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
