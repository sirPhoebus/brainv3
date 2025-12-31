import asyncio
from typing import Dict, List, Callable, Any
import structlog

logger = structlog.get_logger()

class MessageBus:
    """
    A simple in-memory pub/sub system for agent communication.
    """
    
    def __init__(self):
        self.subscribers: Dict[str, List[Callable]] = {}
        
    def subscribe(self, topic: str, callback: Callable):
        if topic not in self.subscribers:
            self.subscribers[topic] = []
        self.subscribers[topic].append(callback)
        logger.debug("subscribed_to_topic", topic=topic)
        
    async def publish(self, topic: str, message: Any):
        if topic not in self.subscribers:
            return
            
        logger.debug("publishing_message", topic=topic)
        tasks = []
        for callback in self.subscribers[topic]:
            if asyncio.iscoroutinefunction(callback):
                tasks.append(callback(message))
            else:
                callback(message)
        
        if tasks:
            await asyncio.gather(*tasks)
