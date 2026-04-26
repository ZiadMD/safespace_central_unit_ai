from app.config import config
from app.utils.logger import logger
from app.schemas.analysis_result import AnalysisResult

# Placeholder tables ready to be filled once schema is decided.
# Example:
# from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
# from sqlalchemy.orm import sessionmaker, declarative_base
# Base = declarative_base()
# class DBAnalysisResult(Base): ...

class DBService:
    def __init__(self):
        self.enabled = config.ENABLE_DB_STORAGE
        if self.enabled:
            pass
            # self.engine = create_async_engine(config.DATABASE_URL, echo=False)
            # self.SessionLocal = sessionmaker(self.engine, class_=AsyncSession, expire_on_commit=False)

    async def store_result(self, result: AnalysisResult) -> None:
        if not self.enabled:
            return
            
        logger.info(f"Storing AnalysisResult for Incident {result.incidentId} to DB")
        # async with self.SessionLocal() as session:
        #     # Insert logic here
        #     await session.commit()
            
    async def store_incident(self, *args, **kwargs) -> None:
        if not self.enabled:
            return
            
        # Placeholder for storing raw incident references to Database
        pass

db_service = DBService()
