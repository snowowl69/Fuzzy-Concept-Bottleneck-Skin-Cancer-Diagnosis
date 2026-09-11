from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, JSON
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime

Base = declarative_base()

class DiagnosisRecord(Base):
    """
    Amazon RDS table mapping for storing diagnosis history, prediction results,
    fuzzy ABCDE scores, and image references.
    """
    __tablename__ = 'diagnosis_history'

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    # User authentication tracking (Amazon Cognito Integration)
    user_id = Column(String(255), nullable=True)
    
    # S3 references
    original_image_s3_url = Column(String(1024), nullable=True)
    mask_image_s3_url = Column(String(1024), nullable=True)
    
    # Extracted ABCDE scores
    asymmetry_score = Column(Float, nullable=True)
    border_irregularity = Column(Float, nullable=True)
    color_variation = Column(Float, nullable=True)
    diameter_score = Column(Float, nullable=True)
    
    # Diagnosis Prediction
    predicted_class = Column(String(100), nullable=False)
    confidence = Column(Float, nullable=False)
    class_probabilities = Column(JSON, nullable=True)
    
    # Explanations / Counterfactual tracking
    explanation_text = Column(String, nullable=True)

def get_db_engine(db_credentials):
    """Create SQLAlchemy engine using credentials from AWS Secrets Manager."""
    if not db_credentials:
        # Fallback to local SQLite for local testing
        return create_engine('sqlite:///local_diagnosis.db')
        
    db_user = db_credentials.get('username')
    db_pass = db_credentials.get('password')
    db_host = db_credentials.get('host')
    db_port = db_credentials.get('port', 5432)
    db_name = db_credentials.get('dbname', 'skincancerdb')
    
    db_url = f"postgresql://{db_user}:{db_pass}@{db_host}:{db_port}/{db_name}"
    return create_engine(db_url)

def init_db(engine):
    """Initialize the database schema."""
    Base.metadata.create_all(engine)

def get_session(engine):
    Session = sessionmaker(bind=engine)
    return Session()
