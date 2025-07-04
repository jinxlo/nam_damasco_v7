# namwoo_app/utils/db_utils.py

import logging
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, scoped_session, Session as SQLAlchemySession
from sqlalchemy.exc import SQLAlchemyError, OperationalError
from contextlib import contextmanager
from typing import Optional, Generator, List, Dict

import datetime
from datetime import timezone

from ..models import Base
from ..models.conversation_pause import ConversationPause
from ..config import Config

logger = logging.getLogger(__name__)

db_session: scoped_session = None
_engine = None
_SessionFactory = None


def init_db(app) -> bool:
    """
    Initialize the SQLAlchemy engine and session factories using app config.
    This "lazy" version creates the engine but does NOT test the connection,
    allowing the test environment to load its configuration first.
    """
    global _engine, _SessionFactory, db_session

    # Prevent re-initialization
    if _engine is not None:
        return True

    db_uri = app.config.get('SQLALCHEMY_DATABASE_URI')
    if not db_uri:
        logger.error("SQLALCHEMY_DATABASE_URI not configured. Database features will fail.")
        return False

    try:
        # We still log the URI for debugging purposes
        db_uri_parts = db_uri.split('@')
        loggable_db_uri = db_uri_parts[-1] if len(db_uri_parts) > 1 else db_uri
        logger.info(f"Initializing database engine for: {loggable_db_uri}")

        _engine = create_engine(
            db_uri,
            pool_pre_ping=True,
            pool_recycle=3600,
            echo=app.config.get('SQLALCHEMY_ECHO', False)
        )
        
        # ==============================================================================
        # === THIS IS THE CHANGE =======================================================
        # The `with _engine.connect() as connection:` block has been removed.
        # We no longer try to connect to the database during initialization.
        # The connection will be established by the pool on the first query.
        # ==============================================================================

        _SessionFactory = sessionmaker(autocommit=False, autoflush=False, bind=_engine)
        db_session = scoped_session(_SessionFactory)
        logger.info("SQLAlchemy engine and session factory have been configured successfully.")
        return True

    except Exception as e:
        logger.error(f"Database initialization failed: {e}", exc_info=True)
        _engine = None; _SessionFactory = None; db_session = None
        return False

@contextmanager
def get_db_session() -> Generator[Optional[SQLAlchemySession], None, None]:
    """
    Yields a SQLAlchemy Session, handles rollback on error, and always removes the session.
    """
    if not db_session:
        logger.error("db_session (ScopedSessionFactory) not initialized. Cannot create DB session.")
        yield None 
        return

    session: SQLAlchemySession = db_session()
    logger.debug(f"DB Session {id(session)} acquired from ScopedSessionFactory.")
    try:
        yield session
    except SQLAlchemyError as e:
        logger.error(f"DB Session {id(session)} SQLAlchemy error: {e}", exc_info=True)
        session.rollback()
        logger.debug(f"DB Session {id(session)} rolled back due to SQLAlchemyError.")
        raise
    except Exception as e:
        logger.error(f"DB Session {id(session)} unexpected error: {e}", exc_info=True)
        session.rollback()
        logger.debug(f"DB Session {id(session)} rolled back due to unexpected error.")
        raise
    finally:
        logger.debug(f"DB Session {id(session)} scope ending. Calling db_session.remove().")
        db_session.remove()
        logger.debug(f"DB Session {id(session)} removed from current scope by ScopedSessionFactory.")

def create_all_tables(app) -> bool:
    """
    Create all tables from SQLAlchemy models and ensure pgvector is present.
    """
    if not _engine:
        logger.error("Database engine not initialized. Cannot create tables.")
        return False
    try:
        logger.info("Attempting to create tables from SQLAlchemy models (if they don't already exist)...")
        Base.metadata.create_all(bind=_engine)
        logger.info("SQLAlchemy Base.metadata.create_all() executed.")

        with _engine.connect() as connection:
            with connection.begin():
                logger.info("Ensuring pgvector extension exists in the database...")
                connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
                logger.info("pgvector extension check complete.")
        return True
    except Exception as e:
        logger.error(f"Error during create_all_tables: {e}", exc_info=True)
        return False

def fetch_history(session_id: str) -> Optional[List[Dict]]:
    """
    Fetches conversation history for a given session_id.
    """
    with get_db_session() as session:
        if not session: return None
        from ..models.history import ConversationHistory
        try:
            record = session.query(ConversationHistory).filter_by(session_id=session_id).first()
            if record and record.history_data:
                return record.history_data
            return []
        except Exception as e:
            logger.exception(f"Error fetching history for session {session_id}: {e}")
            return None

def save_history(session_id: str, history_list: List[Dict]) -> bool:
    """
    Saves or updates conversation history for a given session_id.
    """
    with get_db_session() as session:
        if not session: return False
        from ..models.history import ConversationHistory
        try:
            record = session.query(ConversationHistory).filter_by(session_id=session_id).first()
            if record:
                record.history_data = history_list
            else:
                record = ConversationHistory(session_id=session_id, history_data=history_list)
                session.add(record)
            session.commit()
            return True
        except Exception as e:
            logger.exception(f"Error saving history for session {session_id}: {e}")
            return False

# --- CONVERSATION PAUSE MANAGEMENT ---

def is_conversation_paused(conversation_id: str) -> bool:
    """Checks if a conversation is currently paused."""
    with get_db_session() as session:
        if not session:
            logger.error(f"Cannot check pause status for conv {conversation_id}: DB session not available.")
            return False
        try:
            now_utc = datetime.datetime.now(timezone.utc)
            pause_record = session.query(ConversationPause)\
                .filter(ConversationPause.conversation_id == conversation_id)\
                .filter(ConversationPause.paused_until > now_utc)\
                .first()
            if pause_record:
                logger.debug(f"Conv {conversation_id} IS paused until {pause_record.paused_until.isoformat()}")
                return True
            else:
                logger.debug(f"Conv {conversation_id} is NOT actively paused.")
                return False
        except Exception as e:
            logger.exception(f"Error checking pause status for conversation {conversation_id}: {e}")
            return False

def get_pause_record(conversation_id: str) -> Optional[ConversationPause]:
    """Retrieves the current active pause record for a conversation, if any."""
    with get_db_session() as session:
        if not session:
            logger.error(f"Cannot get pause record for conv {conversation_id}: DB session not available.")
            return None
        try:
            now_utc = datetime.datetime.now(timezone.utc)
            pause_record = session.query(ConversationPause)\
                .filter(ConversationPause.conversation_id == conversation_id)\
                .filter(ConversationPause.paused_until > now_utc)\
                .first()
            return pause_record
        except Exception as e:
            logger.exception(f"Error getting pause record for conversation {conversation_id}: {e}")
            return None

def pause_conversation_for_duration(conversation_id: str, duration_seconds: int):
    """Sets or updates a pause for a conversation."""
    with get_db_session() as session:
        if not session:
            logger.error(f"Cannot pause conv {conversation_id}: DB session not available.")
            return
        try:
            pause_until_time = datetime.datetime.now(timezone.utc) + datetime.timedelta(seconds=duration_seconds)
            pause_record = session.query(ConversationPause).filter_by(conversation_id=conversation_id).first()
            if pause_record:
                pause_record.paused_until = pause_until_time
                logger.info(f"Updating pause for conversation {conversation_id} until {pause_until_time.isoformat()}.")
            else:
                pause_record = ConversationPause(conversation_id=conversation_id, paused_until=pause_until_time)
                session.add(pause_record)
                logger.info(f"Setting new pause for conversation {conversation_id} until {pause_until_time.isoformat()}.")
            session.commit()
            logger.debug(f"Pause set/updated in DB for conversation {conversation_id}.")
        except Exception as e:
            logger.exception(f"Error pausing conversation {conversation_id}: {e}")

def unpause_conversation(conversation_id: str):
    """Removes any active pause for a conversation."""
    with get_db_session() as session:
        if not session:
            logger.error(f"Cannot unpause conv {conversation_id}: DB session not available.")
            return
        try:
            pause_record = session.query(ConversationPause).filter_by(conversation_id=conversation_id).first()
            if pause_record:
                session.delete(pause_record)
                logger.info(f"Deleted pause record for conversation {conversation_id}, effectively unpausing it.")
                session.commit()
            else:
                logger.info(f"No pause record found to delete for conversation {conversation_id}.")
        except Exception as e:
            logger.exception(f"Error unpausing conversation {conversation_id}: {e}")