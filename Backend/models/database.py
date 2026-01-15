from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import MongoClient
from config import MONGODB_URI, MONGODB_NAME
from typing import Optional

class Database:
    client: Optional[AsyncIOMotorClient] = None
    
    @classmethod
    async def connect_db(cls):
        """Connect to MongoDB"""
        cls.client = AsyncIOMotorClient(MONGODB_URI)
        print("Connected to MongoDB")
    
    @classmethod
    async def close_db(cls):
        """Close MongoDB connection"""
        if cls.client:
            cls.client.close()
            print("Closed MongoDB connection")
    
    @classmethod
    def get_db(cls):
        """Get database instance"""
        if cls.client is None:
            raise Exception("Database not connected")
        return cls.client[MONGODB_NAME]

db = Database()
