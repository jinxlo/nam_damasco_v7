# /tests/conftest.py
import sys
import os
import pytest
from flask import Flask
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

# --- Path Fix (Still necessary for the import below) ---
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the app factory
from namwoo_app import create_app

# ==============================================================================
# === THE CORE FIX: A DEDICATED TEST CONFIGURATION =============================
# ==============================================================================
# We define a simple dictionary with the specific settings needed for our tests.
# This completely avoids any issues with .env file loading order.
TEST_CONFIG = {
    "TESTING": True,
    "DEBUG": False,
    "SQLALCHEMY_DATABASE_URI": "postgresql://namwoo:damasco2025!@localhost:5433/namwoo_test",
    # We also define DATABASE_URL to satisfy our own safety check later.
    "DATABASE_URL": "postgresql://namwoo:damasco2025!@localhost:5433/namwoo_test",
}
# ==============================================================================

@pytest.fixture(scope='session')
def app() -> Flask:
    """ Creates the test application instance using your factory. """
    # Create the app instance using your existing factory
    test_app = create_app()
    # Force the app to use our explicit test configuration
    test_app.config.from_mapping(TEST_CONFIG)
    yield test_app

# The rest of the file remains the same, as it now receives a
# correctly configured app instance.

@pytest.fixture(scope='session')
def _db_engine(app: Flask):
    """ Creates a database engine for the entire test session. """
    # This will now get the correct URI from the app config we just forced
    db_uri = app.config.get('DATABASE_URL')
    
    # The safety check will now pass
    if not db_uri or 'test' not in db_uri:
        pytest.fail("DANGER: Test DATABASE_URL is not set or does not contain 'test'.")
        
    engine = create_engine(db_uri)
    yield engine
    engine.dispose()

@pytest.fixture(scope='session', autouse=True)
def _setup_database(_db_engine):
    """
    Assumes schema is pre-loaded from an SQL file and just verifies extensions.
    """
    print("\n--- SKIPPING automatic table creation. Assuming schema is pre-loaded. ---")
    with _db_engine.connect() as connection:
        with connection.begin():
            connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
            connection.execute(text("CREATE EXTENSION IF NOT EXISTS unaccent;"))
    print("--- TEST DATABASE READY ---\n")
    yield
    print("\n--- SKIPPING automatic table teardown. ---")

@pytest.fixture(scope='function')
def db_session(_db_engine):
    """ Provides a clean, transactional database session for each test. """
    connection = _db_engine.connect()
    transaction = connection.begin()
    Session = sessionmaker(bind=connection)
    session = Session()
    yield session
    session.close()
    transaction.rollback()
    connection.close()

@pytest.fixture(scope='function')
def client(app: Flask):
    """ Provides a Flask test client. """
    return app.test_client()