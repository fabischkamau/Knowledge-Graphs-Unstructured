import os
import yaml
import json
import pickle
import logging
import hashlib
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import networkx as nx
from dataclasses import dataclass, asdict
from enum import Enum
from dotenv import load_dotenv
from pydantic import BaseModel, Field, model_validator
from typing import List, Dict, Any, Optional, Tuple
from tqdm import tqdm
import re
# PDF processing - try different import approaches
try:
    import pymupdf as fitz  # Newer PyMuPDF versions
except ImportError:
    try:
        import fitz  # Older PyMuPDF versions
    except ImportError:
        print("Warning: PyMuPDF not available. PDF processing will be disabled.")
        fitz = None

# DOCX processing - handle potential import issues
try:
    from docx import Document as DocxDocument  # python-docx for DOCX processing
    docx_available = True
except ImportError:
    print("Warning: python-docx not available. DOCX processing will be disabled.")
    DocxDocument = None
    docx_available = False

# LangChain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_core.prompts import ChatPromptTemplate

# LLM imports
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Embedding imports
from sentence_transformers import SentenceTransformer
import numpy as np

# Neo4j imports
from neo4j import GraphDatabase
import neo4j.exceptions

# Set up colored logging
import colorlog

def load_env_file(env_file_path: str = ".env"):
    """Load environment variables from .env file."""
    env_path = Path(env_file_path)
    if env_path.exists():
        load_dotenv(env_path)
        return True
    return False


class LogLevel(Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


def setup_colored_logging(log_level: str = "INFO", log_to_file: bool = False, log_directory: str = "./logs") -> logging.Logger:
    """Setup colored logging for the pipeline."""
    logger = logging.getLogger("kg_pipeline")
    logger.setLevel(getattr(logging, log_level.upper()))
    
    if not logger.handlers:
        # Console handler with colors
        console_handler = colorlog.StreamHandler()
        console_handler.setFormatter(colorlog.ColoredFormatter(
            '%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            log_colors={
                'DEBUG': 'cyan',
                'INFO': 'green',
                'WARNING': 'yellow',
                'ERROR': 'red',
                'CRITICAL': 'red,bg_white',
            }
        ))
        logger.addHandler(console_handler)
        
        # File handler if enabled
        if log_to_file:
            log_dir = Path(log_directory)
            log_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = log_dir / f"kg_pipeline_{timestamp}.log"
            
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            ))
            logger.addHandler(file_handler)
            
            logger.info(f"Logging to file: {log_file}")
    
    return logger


@dataclass
class PipelineState:
    """Track the state of pipeline execution."""
    processed_files: List[str]
    failed_files: List[str]
    extracted_entities: Dict[str, Any]
    last_execution_time: datetime
    total_chunks_processed: int
    total_files_to_process: int
    
    def save_state(self, state_file: Path):
        """Save the current state to a pickle file."""
        with open(state_file, 'wb') as f:
            pickle.dump(asdict(self), f)
    
    @classmethod
    def load_state(cls, state_file: Path) -> 'PipelineState':
        """Load state from a pickle file."""
        if state_file.exists():
            with open(state_file, 'rb') as f:
                data = pickle.load(f)
                data['last_execution_time'] = datetime.fromisoformat(data['last_execution_time']) if isinstance(data['last_execution_time'], str) else data['last_execution_time']
                return cls(**data)
        return cls([], [], {}, datetime.now(), 0, 0)


class Entity(BaseModel):
    """Schema for extracted entities."""
    type: str = Field(description="Type of entity")
    name: str = Field(description="Name of the entity")
    properties: Dict[str, Any] = Field(
        default_factory=dict, 
        description="Additional properties"
    )

    @model_validator(mode='after')
    def ensure_name_in_properties(self):
        if 'name' not in self.properties:
            self.properties['name'] = self.name
        return self


class Relationship(BaseModel):
    """Schema for extracted relationships."""
    source: str = Field(description="Name of the source entity")
    target: str = Field(description="Name of the target entity")
    type: str = Field(description="Type of relationship")
    properties: Dict[str, Any] = Field(
        default_factory=dict, 
        description="Additional properties"
    )


class EntityRelationshipModel(BaseModel):
    """Pydantic model for structured entity-relationship extraction."""
    entities: List[Entity] = Field(description="List of extracted entities")
    relationships: List[Relationship] = Field(description="List of relationships between entities")


class DocumentProcessor:
    """Handle different document formats."""
    
    @staticmethod
    def extract_text_from_pdf(file_path: Path) -> str:
        """Extract text from PDF file."""
        try:
            doc = fitz.open(file_path)
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()
            return text
        except Exception as e:
            raise Exception(f"Failed to extract text from PDF {file_path}: {e}")
    
    @staticmethod
    def extract_text_from_docx(file_path: Path) -> str:
        """Extract text from DOCX file."""
        try:
            doc = docx.Document(file_path)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        except Exception as e:
            raise Exception(f"Failed to extract text from DOCX {file_path}: {e}")
    
    @staticmethod
    def extract_text_from_txt(file_path: Path) -> str:
        """Extract text from TXT file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            raise Exception(f"Failed to extract text from TXT {file_path}: {e}")
    
    @staticmethod
    def extract_text_from_md(file_path: Path) -> str:
        """Extract text from Markdown file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            raise Exception(f"Failed to extract text from MD {file_path}: {e}")
    
    @classmethod
    def extract_text(cls, file_path: Path) -> str:
        """Extract text based on file extension."""
        suffix = file_path.suffix.lower()
        
        if suffix == '.pdf':
            return cls.extract_text_from_pdf(file_path)
        elif suffix == '.docx':
            return cls.extract_text_from_docx(file_path)
        elif suffix == '.txt':
            return cls.extract_text_from_txt(file_path)
        elif suffix == '.md':
            return cls.extract_text_from_md(file_path)
        else:
            raise ValueError(f"Unsupported file format: {suffix}")


class EmbeddingProcessor:
    """Handle text embeddings."""
    
    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.embeddings_model = None
        
        if config.get('enable_embeddings', False):
            self._initialize_embeddings()
    
    def _initialize_embeddings(self):
        """Initialize embedding model based on configuration."""
        provider = self.config.get('provider', 'openai').lower()
        model_name = self.config.get('model', 'text-embedding-ada-002')
        
        if provider == 'openai':
            api_key = self.config.get('api_key')
            self.embeddings_model = OpenAIEmbeddings(
                model=model_name,
                openai_api_key=api_key
            )
            self.logger.info(f"Initialized OpenAI embeddings with model: {model_name}")
        
        elif provider == 'sentence-transformers':
            self.embeddings_model = SentenceTransformer(model_name)
            self.logger.info(f"Initialized SentenceTransformer with model: {model_name}")
        
        else:
            raise ValueError(f"Unsupported embedding provider: {provider}")
    
    def embed_text(self, text: str) -> Optional[List[float]]:
        """Generate embedding for text."""
        if not self.embeddings_model:
            return None
        
        try:
            if isinstance(self.embeddings_model, OpenAIEmbeddings):
                return self.embeddings_model.embed_query(text)
            else:  # SentenceTransformer
                embedding = self.embeddings_model.encode(text)
                return embedding.tolist()
        except Exception as e:
            self.logger.error(f"Failed to generate embedding: {e}")
            return None
    
    def embed_texts(self, texts: List[str]) -> List[Optional[List[float]]]:
        """Generate embeddings for multiple texts."""
        if not self.embeddings_model:
            return [None] * len(texts)
        
        try:
            if isinstance(self.embeddings_model, OpenAIEmbeddings):
                return self.embeddings_model.embed_documents(texts)
            else:  # SentenceTransformer
                embeddings = self.embeddings_model.encode(texts)
                return [emb.tolist() for emb in embeddings]
        except Exception as e:
            self.logger.error(f"Failed to generate embeddings: {e}")
            return [None] * len(texts)


class KnowledgeGraphPipeline:
    """Main pipeline class for building knowledge graphs from documents."""
    
    # Default allowed entity and relationship types
    DEFAULT_ENTITY_TYPES = [
        "Person", "Organization", "Location", "Event", "Concept",
        "Document", "Date", "Technology", "Product", "Topic"
    ]
    
    DEFAULT_RELATIONSHIP_TYPES = [
        # General relationships
        "MENTIONS", "RELATED_TO", "PART_OF", "CONTAINS", "BELONGS_TO",
        
        # Person relationships
        "WORKS_FOR", "KNOWS", "COLLABORATES_WITH", "LEADS", "REPORTS_TO",
        
        # Event relationships
        "PARTICIPATES_IN", "OCCURS_IN", "CAUSES", "FOLLOWS", "PRECEDES",
        
        # Location relationships
        "LOCATED_IN", "NEAR", "TRAVELS_TO",
        
        # Document relationships
        "AUTHORED_BY", "REFERENCES", "CITES", "DISCUSSES"
    ]
    
    def __init__(self, config_path: str, env_file_path: str = ".env"):
        """Initialize the pipeline with configuration and environment file."""
        # Load environment variables from .env file
        env_loaded = load_env_file(env_file_path)
        
        self.config = self._load_config(config_path)
        self.logger = setup_colored_logging(
            self.config.get('log_level', 'INFO'),
            self.config.get('log_to_file', False),
            self.config.get('log_directory', './logs')
        )
        
        if env_loaded:
            self.logger.info(f"Loaded environment variables from {env_file_path}")
        else:
            self.logger.warning(f"Environment file {env_file_path} not found. Using system environment variables.")
        
        # Initialize LLM
        self.llm = self._initialize_llm()
        
        # Initialize embeddings processor
        self.embedding_processor = EmbeddingProcessor(
            self.config.get('embeddings', {}), 
            self.logger
        )
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.get('chunk_size', 2000),
            chunk_overlap=self.config.get('chunk_overlap', 200),
            length_function=len,
        )
        
        # Setup paths
        self.input_dir = Path(self.config['input_directory'])
        self.output_dir = Path(self.config['output_directory'])
        self.state_file = Path(self.config.get('state_file', 'pipeline_state.pkl'))
        self.cache_dir = Path(self.config.get('cache_directory', 'cache'))
        
        # Create directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Load or initialize state
        self.state = PipelineState.load_state(self.state_file)
        
        # Entity and relationship types
        self.allowed_entity_types = self.config.get('allowed_entity_types', self.DEFAULT_ENTITY_TYPES)
        self.allowed_relationship_types = self.config.get('allowed_relationship_types', self.DEFAULT_RELATIONSHIP_TYPES)
        
        # Supported formats
        self.supported_formats = self.config.get('supported_formats', ['txt', 'pdf', 'docx', 'md'])
        
        # NetworkX graph for analysis
        self.graph = nx.Graph()
        
        # Progress tracking
        self.progress_logging = self.config.get('progress_logging', True)
        
        self.logger.info("Knowledge Graph Pipeline initialized successfully")
        
    def _resolve_env_variables(self, value: str) -> str:
        """Resolve environment variables in configuration values."""
        if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
            env_var = value[2:-1]
            env_value = os.environ.get(env_var)
            if env_value is None:
                self.logger.warning(f"Environment variable {env_var} not found")
                return value
            return env_value
        return value
    
    def _process_config_env_vars(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Process all environment variables in the configuration."""
        processed_config = {}
        
        for key, value in config.items():
            if isinstance(value, dict):
                processed_config[key] = self._process_config_env_vars(value)
            elif isinstance(value, list):
                processed_config[key] = [
                    self._process_config_env_vars(item) if isinstance(item, dict) else 
                    self._resolve_env_variables(item) if isinstance(item, str) else item
                    for item in value
                ]
            elif isinstance(value, str):
                processed_config[key] = self._resolve_env_variables(value)
            else:
                processed_config[key] = value
        
        return processed_config
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file with environment variable support."""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Process environment variables in config
            processed_config = self._process_config_env_vars(config)
            return processed_config
            
        except Exception as e:
            raise ValueError(f"Error loading configuration: {e}")
    
    def _initialize_llm(self):
        """Initialize the LLM based on configuration."""
        provider = self.config['llm']['provider'].lower()
        model = self.config['llm']['model']
        api_key = self.config['llm']['api_key']
        
        try:
            if provider == 'anthropic':
                self.logger.info(f"Initializing LLM with provider: Anthropic, model: {model}")
                return ChatAnthropic(
                    model=model,
                    anthropic_api_key=api_key,
                    temperature=self.config['llm'].get('temperature', 0.1)
                )
            elif provider == 'openai':
                self.logger.info(f"Initializing LLM with provider: OpenAI, model: {model}")
                return ChatOpenAI(
                    model=model,
                    openai_api_key=api_key,
                    temperature=self.config['llm'].get('temperature', 0.1)
                )
            else:
                raise ValueError(f"Unsupported LLM provider: {provider}")
        except Exception as e:
            self.logger.error(f"Failed to initialize LLM with provider: {provider}, model: {model}. Error: {e}")
            raise
    
    def _get_file_hash(self, file_path: Path) -> str:
        """Generate hash for file to check if it has changed."""
        with open(file_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    
    def _get_cache_path(self, file_path: Path) -> Path:
        """Get cache file path for a given input file."""
        file_hash = self._get_file_hash(file_path)
        return self.cache_dir / f"{file_path.stem}_{file_hash}.json"
    
    def _load_from_cache(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """Load extracted entities from cache if available."""
        cache_path = self._get_cache_path(file_path)
        if cache_path.exists():
            try:
                with open(cache_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.warning(f"Failed to load cache for {file_path}: {e}")
        return None
    
    def _save_to_cache(self, file_path: Path, extracted_data: Dict[str, Any]):
        """Save extracted entities to cache."""
        cache_path = self._get_cache_path(file_path)
        try:
            with open(cache_path, 'w') as f:
                json.dump(extracted_data, f, indent=2)
            self.logger.debug(f"Saved cache for {file_path}")
        except Exception as e:
            self.logger.warning(f"Failed to save cache for {file_path}: {e}")
    
    def _create_extraction_prompt(self) -> ChatPromptTemplate:
        """Create the prompt template for entity-relationship extraction."""
        domain_context = ""
        extraction_context = self.config.get('extraction_context', {})
        
        if extraction_context:
            domain = extraction_context.get('domain', 'general')
            focus_areas = extraction_context.get('focus_areas', [])
            
            domain_context = f"\nDomain Context: {domain}"
            if focus_areas:
                domain_context += f"\nFocus on these areas: {', '.join(focus_areas)}"
        
        prompt_text = f"""
You are an expert at extracting structured information from documents. Extract entities and relationships from the following text chunk.

IMPORTANT CONSISTENCY RULES:
1. ALL entities must have a 'name' property
2. Use consistent property names across all entities of the same type
3. Use canonical names for entities (e.g., full names for people, official names for organizations)
4. Ensure entity types and relationships match the allowed lists exactly
5. DO NOT create Chunk or Document entities - they are handled separately
6. Be precise and factual - only extract information that is explicitly stated or clearly implied

Allowed Entity Types: {{allowed_entities}}
Allowed Relationship Types: {{allowed_relationships}}
{domain_context}

Text to analyze:
{{text}}

Context Information:
- Document: {{document_name}}
- Chunk Number: {{chunk_number}}

Focus on extracting:
1. People, organizations, and their relationships
2. Important events and their participants
3. Locations and their significance
4. Key concepts, topics, and themes
5. Dates, technologies, products mentioned
6. Clear relationships between entities

Be authoritative and consistent with naming conventions. Extract only information that adds value to understanding the document's content.
"""
        
        return ChatPromptTemplate.from_template(prompt_text)
    
    def _extract_from_chunk(self, chunk: Document, document_name: str, chunk_number: int) -> Dict[str, Any]:
        """Extract entities and relationships from a single chunk."""
        prompt = self._create_extraction_prompt()
        
        try:
            structured_llm = self.llm.with_structured_output(schema=EntityRelationshipModel, method="function_calling")
            chain = prompt | structured_llm

            # Execute LLM extraction
            result = chain.invoke({
                "text": chunk.page_content,
                "document_name": document_name,
                "chunk_number": chunk_number,
                "allowed_entities": ", ".join(self.allowed_entity_types),
                "allowed_relationships": ", ".join(self.allowed_relationship_types)
            })

            entities_dict = []
            relationships_dict = []

            chunk_entity_name = f"{document_name}_chunk_{chunk_number}"
            document_entity_exists = False
            chunk_entity_exists = False

            # Add all extracted entities
            for entity in result.entities:
                entity_dict = {
                    "type": entity.type,
                    "properties": {"name": entity.name, **entity.properties}
                }

                if entity.type == "Document":
                    document_entity_exists = True
                    if not entity_dict["properties"].get("name") or entity_dict["properties"]["name"] == "unknown":
                        entity_dict["properties"]["name"] = document_name

                if entity.type == "Chunk" and entity.name == chunk_entity_name:
                    chunk_entity_exists = True

                entities_dict.append(entity_dict)

            # Add Document entity if missing
            if not document_entity_exists:
                entities_dict.append({
                    "type": "Document",
                    "properties": {
                        "name": document_name
                    }
                })

            # Add Chunk entity if missing
            chunk_text = chunk.page_content
            chunk_embedding = None
            
            # Generate embedding for chunk if enabled
            if self.config.get('embeddings', {}).get('chunk_embeddings', False):
                chunk_embedding = self.embedding_processor.embed_text(chunk_text)
            
            chunk_properties = {
                "name": chunk_entity_name,
                "text": chunk_text,
                "chunk_number": chunk_number
            }
            
            if chunk_embedding:
                chunk_properties["embedding"] = chunk_embedding

            if not chunk_entity_exists:
                entities_dict.append({
                    "type": "Chunk",
                    "properties": chunk_properties
                })

            # Add relationships from extraction
            chunk_document_link_exists = False

            for rel in result.relationships:
                rel_dict = {
                    "source": rel.source,
                    "target": rel.target,
                    "type": rel.type,
                    "properties": rel.properties
                }

                # Check for chunk-document link
                if (rel.source == chunk_entity_name and rel.target == document_name and rel.type == "BELONGS_TO") or \
                   (rel.source == document_name and rel.target == chunk_entity_name and rel.type == "CONTAINS"):
                    chunk_document_link_exists = True

                relationships_dict.append(rel_dict)

            # Add BELONGS_TO relationship if missing
            if not chunk_document_link_exists:
                relationships_dict.append({
                    "source": chunk_entity_name,
                    "target": document_name,
                    "type": "BELONGS_TO",
                    "properties": {}
                })

            # Link chunk to each extracted entity with MENTIONS relationship
            for entity in entities_dict:
                entity_name = entity["properties"]["name"]
                entity_type = entity["type"]

                if entity_type not in {"Chunk", "Document"}:
                    relationships_dict.append({
                        "source": chunk_entity_name,
                        "target": entity_name,
                        "type": "MENTIONS",
                        "properties": {
                            "chunk_number": chunk_number
                        }
                    })

            return {
                "entities": entities_dict,
                "relationships": relationships_dict
            }

        except Exception as e:
            self.logger.error(f"Failed to extract from chunk {chunk_number}: {e}")
            # On failure, fallback to just chunk and document
            chunk_text = chunk.page_content
            chunk_embedding = None
            
            if self.config.get('embeddings', {}).get('chunk_embeddings', False):
                chunk_embedding = self.embedding_processor.embed_text(chunk_text)
            
            chunk_properties = {
                "name": f"{document_name}_chunk_{chunk_number}",
                "text": chunk_text,
                "chunk_number": chunk_number
            }
            
            if chunk_embedding:
                chunk_properties["embedding"] = chunk_embedding
            
            return {
                "entities": [
                    {
                        "type": "Chunk",
                        "properties": chunk_properties
                    },
                    {
                        "type": "Document",
                        "properties": {
                            "name": document_name
                        }
                    }
                ],
                "relationships": [
                    {
                        "source": f"{document_name}_chunk_{chunk_number}",
                        "target": document_name,
                        "type": "BELONGS_TO",
                        "properties": {}
                    }
                ]
            }
    
    def _process_file(self, file_path: Path) -> Dict[str, Any]:
        """Process a single file and extract entities."""
        document_name = file_path.stem
        
        # Check cache first
        cached_result = self._load_from_cache(file_path)
        if cached_result:
            self.logger.info(f"Loaded {document_name} from cache")
            return cached_result
        
        self.logger.info(f"Processing {document_name} ({file_path.suffix})")
        
        try:
            # Extract text based on file type
            text = DocumentProcessor.extract_text(file_path)
            
            # Split into chunks
            chunks = self.text_splitter.create_documents([text])
            self.logger.info(f"Split {document_name} into {len(chunks)} chunks")
            
            # Extract from chunks
            all_entities = []
            all_relationships = []
            
            # Use progress bar if enabled
            chunk_iterator = enumerate(chunks)
            if self.progress_logging:
                chunk_iterator = tqdm(
                    chunk_iterator, 
                    total=len(chunks),
                    desc=f"Processing {document_name}",
                    unit="chunk"
                )
            
            # Process chunks with threading
            with ThreadPoolExecutor(max_workers=self.config.get('max_workers', 4)) as executor:
                # Create futures for all chunks
                future_to_chunk = {
                    executor.submit(self._extract_from_chunk, chunk, document_name, i): i
                    for i, chunk in enumerate(chunks)
                }
                
                for future in as_completed(future_to_chunk):
                    chunk_num = future_to_chunk[future]
                    try:
                        result = future.result()
                        all_entities.extend(result.get('entities', []))
                        all_relationships.extend(result.get('relationships', []))
                        self.state.total_chunks_processed += 1
                        self.logger.debug(f"Processed chunk {chunk_num} for {document_name}")
                    except Exception as e:
                        self.logger.error(f"Chunk {chunk_num} failed for {document_name}: {e}")
            
            # Combine results
            extracted_data = {
                "document_name": document_name,
                "entities": all_entities,
                "relationships": all_relationships,
                "extraction_time": datetime.now().isoformat(),
                "total_chunks": len(chunks)
            }
            
            # Save to cache
            self._save_to_cache(file_path, extracted_data)
            
            self.logger.info(f"Extracted {len(all_entities)} entities and {len(all_relationships)} relationships from {document_name}")
            return extracted_data
            
        except Exception as e:
            self.logger.error(f"Failed to process {file_path}: {e}")
            self.state.failed_files.append(str(file_path))
            raise
    
    def _discover_files(self) -> List[Path]:
        """Discover all supported files in the input directory."""
        files_to_process = []
        
        # Get all supported file extensions with dot prefix
        supported_extensions = [f".{ext}" for ext in self.supported_formats]
        
        # Recursively find all supported files
        for file_path in self.input_dir.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
                if str(file_path) not in self.state.processed_files:
                    files_to_process.append(file_path)
                    self.logger.debug(f"Found new file: {file_path}")
        
        self.logger.info(f"Discovered {len(files_to_process)} files to process")
        return files_to_process
    
    def _load_all_cached_data(self) -> List[Dict[str, Any]]:
        """Load all cached extraction data for Cypher generation."""
        self.logger.info("Loading cached extraction data from cache directory")
        cached_data = []
        
        try:
            cache_files = list(self.cache_dir.glob("*.json"))
            if not cache_files:
                self.logger.warning("No cache files found")
                return []
            
            # Use progress bar for loading cache files
            if self.progress_logging:
                cache_files = tqdm(cache_files, desc="Loading cache files", unit="file")
            
            for cache_file in cache_files:
                try:
                    with open(cache_file, 'r') as f:
                        data = json.load(f)
                        cached_data.append(data)
                        self.logger.debug(f"Loaded cached data from {cache_file}")
                except Exception as e:
                    self.logger.warning(f"Failed to load cache file {cache_file}: {e}")
        
            self.logger.info(f"Loaded {len(cached_data)} cached extraction files")
            return cached_data
        except Exception as e:
            self.logger.error(f"Error loading cached data: {e}")
            return []
    
    def generate_cypher_from_cache(self):
        """Generate Cypher from all cached extraction data and optionally import to Neo4j."""
        try:
            # Load all cached data
            all_cached_data = self._load_all_cached_data()
            
            if not all_cached_data:
                self.logger.warning("No cached data found for Cypher generation")
                return None
            
            # Verify and repair chunk text in cached data
            all_cached_data = self._verify_and_repair_chunk_text(all_cached_data)
            
            # Merge data from all cache files
            self.logger.info("Merging cached data")
            merged_data = self._merge_entities(all_cached_data)
            
            # Update NetworkX graph
            self._update_networkx_graph(merged_data)
            
            # Generate Cypher queries
            self.logger.info("Generating Cypher queries from cached data")
            cypher_queries = self._generate_cypher(merged_data)
            
            # Save Cypher queries
            cypher_file = self._save_cypher(cypher_queries)
            
            # Auto-import to Neo4j if configured
            if self.config.get('auto_import_neo4j', False):
                self.logger.info("Auto-importing to Neo4j")
                self._import_to_neo4j(cypher_file)
            
            self.logger.info(f"Generated Cypher from {len(all_cached_data)} cached files")
            self.logger.info(f"Total entities: {len(merged_data['entities'])}")
            self.logger.info(f"Total relationships: {len(merged_data['relationships'])}")
            
            return cypher_file
        except Exception as e:
            self.logger.error(f"Failed to generate Cypher from cache: {e}")
            return None
        
    def _merge_entities(self, all_extracted_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Merge entities from all files and resolve duplicates, preserving chunk text."""
        merged_entities = {}
        merged_relationships = []
        chunks_by_document = {}

        # Progress tracking for merging
        total_files = len(all_extracted_data)
        
        if self.progress_logging:
            all_extracted_data = tqdm(all_extracted_data, desc="Merging entities", unit="file")

        for data in all_extracted_data:
            document_name = data.get('document_name', 'unknown')
            
            for entity in data.get('entities', []):
                entity_type = entity['type']
                entity_props = entity.get('properties', {})
                entity_name = entity_props.get('name', 'unknown')

                # Normalize Document names
                if entity_type == "Document":
                    entity_name = entity_name.strip()
                    entity_props["name"] = entity_name

                # Normalize key
                entity_key = f"{entity_type}_{entity_name.strip().lower().replace(' ', '_')}"

                # Track Chunk text for repairs
                if entity_type == "Chunk":
                    chunk_number = entity_props.get('chunk_number')
                    
                    if document_name not in chunks_by_document:
                        chunks_by_document[document_name] = {}

                    if chunk_number is not None and "text" in entity_props:
                        chunks_by_document[document_name][chunk_number] = entity_props["text"]

                    # Avoid overwriting text if already stored
                    if entity_key in merged_entities and "text" not in merged_entities[entity_key].get("properties", {}):
                        if "text" in entity_props:
                            merged_entities[entity_key]["properties"]["text"] = entity_props["text"]

                # Merge or add new entity
                if entity_key in merged_entities:
                    existing_props = merged_entities[entity_key]["properties"]
                    for k, v in entity_props.items():
                        if k not in existing_props or not existing_props[k]:
                            existing_props[k] = v
                else:
                    merged_entities[entity_key] = {
                        "type": entity_type,
                        "properties": entity_props
                    }

            # Add relationships
            merged_relationships.extend(data.get('relationships', []))

        # Repair missing chunk text
        for entity_key, entity in merged_entities.items():
            if entity["type"] == "Chunk":
                props = entity.get("properties", {})
                chunk_name = props.get("name", "")
                if "text" not in props or not props["text"]:
                    if "_chunk_" in chunk_name:
                        parts = chunk_name.split("_chunk_")
                        if len(parts) == 2:
                            document_name = parts[0]
                            try:
                                chunk_number = int(parts[1])
                                if document_name in chunks_by_document and chunk_number in chunks_by_document[document_name]:
                                    props["text"] = chunks_by_document[document_name][chunk_number]
                            except (ValueError, IndexError):
                                continue

        return {
            "entities": list(merged_entities.values()),
            "relationships": merged_relationships
        }
    
    def _generate_cypher(self, merged_data: Dict[str, Any]) -> str:
        """Generate Cypher queries for Neo4j import with proper text sanitization and vector indexes."""
        import json
        import re
        cypher_queries = []

        def escape_text(value):
            if value is None:
                return "null"
            elif isinstance(value, bool):
                return str(value).lower()
            elif isinstance(value, (int, float)):
                return str(value)
            elif isinstance(value, list):
                if all(isinstance(x, (int, float)) for x in value):
                    return json.dumps(value)
                else:
                    return "[{}]".format(", ".join([escape_text(x) for x in value]))
            elif isinstance(value, str):
                escaped = value.replace("\\", "\\\\").replace("'", "\\'").replace('"', '\\"')
                if len(escaped) > 32000:
                    escaped = escaped[:32000] + "..."
                return f"'{escaped}'"
            else:
                return escape_text(str(value))

        entity_types_with_embeddings = set()

        entities = merged_data['entities']
        if self.progress_logging:
            entities = tqdm(entities, desc="Generating entity queries", unit="entity")

        for entity in entities:
            entity_type = entity['type']
            properties = entity.get('properties', {})

            if 'embedding' in properties and properties['embedding']:
                entity_types_with_embeddings.add(entity_type)

            if entity_type == "Chunk":
                chunk_name = properties.get('name', '')
                chunk_number = properties.get('chunk_number')
                if chunk_number is None and "_chunk_" in chunk_name:
                    try:
                        chunk_number = int(chunk_name.split("_chunk_")[1])
                        properties["chunk_number"] = chunk_number
                    except (ValueError, IndexError):
                        pass
                if "text" not in properties or not properties["text"]:
                    properties["text"] = f"Text for chunk {chunk_number} (needs repair)"

            prop_strings = []
            for key, value in properties.items():
                if value is None or (isinstance(value, str) and not value.strip()):
                    continue
                sanitized_value = escape_text(value)
                prop_strings.append(f"{key}: {sanitized_value}")

            if not prop_strings:
                continue

            prop_string = ", ".join(prop_strings)
            cypher = f"MERGE (n:{entity_type} {{{prop_string}}})"
            cypher_queries.append(cypher)

        relationships = merged_data['relationships']
        if self.progress_logging:
            relationships = tqdm(relationships, desc="Generating relationship queries", unit="relationship")

        for rel in relationships:
            source_name = rel['source']
            target_name = rel['target']
            rel_type = rel['type']
            properties = rel.get('properties', {})

            if not source_name or not target_name:
                continue

            source = escape_text(source_name)[1:-1]  # Remove outer quotes
            target = escape_text(target_name)[1:-1]

            prop_strings = []
            for key, value in properties.items():
                if value is None or (isinstance(value, str) and not value.strip()):
                    continue
                sanitized_value = escape_text(value)
                prop_strings.append(f"{key}: {sanitized_value}")

            prop_string = f" {{{', '.join(prop_strings)}}}" if prop_strings else ""

            cypher = f"""
            MATCH (a {{name: '{source}'}})
            MATCH (b {{name: '{target}'}})
            MERGE (a)-[r:{rel_type}{prop_string}]->(b)
            """.strip()
            cypher_queries.append(cypher)

        if entity_types_with_embeddings and self.config.get('embeddings', {}).get('enable_embeddings', False):
            embedding_config = self.config.get('embeddings', {})
            dimensions = embedding_config.get('dimensions', 1536)
            similarity_function = embedding_config.get('similarity_function', 'cosine')

            for entity_type in entity_types_with_embeddings:
                vector_index_query = f"""
                CREATE VECTOR INDEX {entity_type.lower()}_embeddings IF NOT EXISTS
                FOR (n:{entity_type}) ON n.embedding
                OPTIONS {{
                    indexConfig: {{
                        `vector.dimensions`: {dimensions},
                        `vector.similarity_function`: '{similarity_function}'
                    }}
                }}
                """.strip()
                cypher_queries.append(vector_index_query)

        return ";\n\n".join(cypher_queries) + ";"


    def _verify_and_repair_chunk_text(self, cached_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Verify and repair missing chunk text in cached data."""
        self.logger.info("Verifying chunk text in cached data")
        
        # Track all chunks by document and number
        all_chunks = {}
        
        # First pass: collect all available chunk texts
        for data in cached_data:
            document_name = data.get('document_name', 'unknown')
            
            for entity in data.get('entities', []):
                if entity['type'] == 'Chunk':
                    props = entity.get('properties', {})
                    chunk_number = props.get('chunk_number')
                    chunk_text = props.get('text')
                    
                    if chunk_number is not None and chunk_text:
                        if document_name not in all_chunks:
                            all_chunks[document_name] = {}
                        
                        all_chunks[document_name][chunk_number] = chunk_text
        
        # Second pass: fix missing texts
        chunks_repaired = 0
        for data in cached_data:
            document_name = data.get('document_name', 'unknown')
            
            for entity in data.get('entities', []):
                if entity['type'] == 'Chunk':
                    props = entity.get('properties', {})
                    chunk_number = props.get('chunk_number')
                    
                    if chunk_number is not None and (not props.get('text') or props.get('text') == ''):
                        if document_name in all_chunks and chunk_number in all_chunks[document_name]:
                            props['text'] = all_chunks[document_name][chunk_number]
                            entity['properties'] = props
                            chunks_repaired += 1
        
        if chunks_repaired > 0:
            self.logger.info(f"Repaired text for {chunks_repaired} chunks")
        
        return cached_data
        
    def _save_cypher(self, cypher_queries: str):
        """Save Cypher queries to file with timestamp."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.output_dir / f"knowledge_graph_{timestamp}.cypher"
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(cypher_queries)
            self.logger.info(f"Saved Cypher queries to {output_file}")
            return output_file
        except Exception as e:
            self.logger.error(f"Failed to save Cypher queries: {e}")
            raise
    
    def _import_to_neo4j(self, cypher_file: Path):
        """Import Cypher to Neo4j safely with query validation and escaping."""
        if not self.config.get('neo4j'):
            self.logger.warning("Neo4j config missing; skipping import.")
            return

        neo4j_config = self.config['neo4j']

        try:
            driver = GraphDatabase.driver(
                neo4j_config['uri'],
                auth=(neo4j_config['username'], neo4j_config['password'])
            )

            with driver.session() as session:
                with open(cypher_file, 'r', encoding='utf-8') as f:
                    content = f.read()

                # Split and filter only valid-looking Cypher queries
                queries = [
                    q.strip() for q in content.split(';')
                    if q.strip() and re.match(r'^(MERGE|MATCH|CREATE|CALL|WITH)', q.strip(), re.IGNORECASE)
                ]

                if self.progress_logging:
                    queries = tqdm(queries, desc="Importing to Neo4j", unit="query")

                success, failure = 0, 0
                for q in queries:
                    try:
                        session.run(q)
                        success += 1
                    except Exception as e:
                        failure += 1
                        self.logger.error(f"Cypher error in query: {q[:100]}...: {e}")

            self.logger.info(f"Neo4j import done — ✅ {success} success | ❌ {failure} failed")

        except Exception as e:
            self.logger.critical(f"Neo4j connection/import failed: {e}")

    
    def _update_networkx_graph(self, merged_data: Dict[str, Any]):
        """Update NetworkX graph for analysis."""
        # Clear existing graph
        self.graph.clear()
        
        # Add nodes
        for entity in merged_data['entities']:
            node_id = entity.get('properties', {}).get('name', f"unknown_{entity['type']}")
            self.graph.add_node(node_id, **entity)
        
        # Add edges
        for rel in merged_data['relationships']:
            source = rel['source']
            target = rel['target']
            if self.graph.has_node(source) and self.graph.has_node(target):
                self.graph.add_edge(source, target, **rel)
    
    def run_pipeline(self):
        """Run the complete pipeline with progress tracking."""
        start_time = time.time()
        self.logger.info("Starting Knowledge Graph Pipeline")
        
        try:
            # Discover files to process
            files_to_process = self._discover_files()
            
            if not files_to_process:
                self.logger.info("No new files to process")
                return
            
            self.logger.info(f"Found {len(files_to_process)} files to process")
            self.state.total_files_to_process = len(files_to_process)
            
            # Process files with progress tracking
            all_extracted_data = []
            processed_count = 0
            
            if self.progress_logging:
                files_to_process = tqdm(files_to_process, desc="Processing files", unit="file")
            
            for file_path in files_to_process:
                try:
                    extracted_data = self._process_file(file_path)
                    all_extracted_data.append(extracted_data)
                    self.state.processed_files.append(str(file_path))
                    processed_count += 1
                    
                    # Log progress percentage
                    progress_pct = (processed_count / self.state.total_files_to_process) * 100
                    self.logger.info(f"Progress: {progress_pct:.1f}% ({processed_count}/{self.state.total_files_to_process} files)")
                    
                    # Save state after each file
                    self.state.save_state(self.state_file)
                    
                except Exception as e:
                    self.logger.error(f"Failed to process {file_path}: {e}")
                    continue
            
            if not all_extracted_data:
                self.logger.warning("No data was successfully extracted")
                return
            
            # Merge all extracted data
            self.logger.info("Merging extracted data")
            merged_data = self._merge_entities(all_extracted_data)
            
            # Update NetworkX graph
            self._update_networkx_graph(merged_data)
            
            # Generate Cypher queries
            self.logger.info("Generating Cypher queries")
            cypher_queries = self._generate_cypher(merged_data)
            
            # Save Cypher queries
            cypher_file = self._save_cypher(cypher_queries)
            
            # Import to Neo4j if configured
            if self.config.get('auto_import_neo4j', False):
                self.logger.info("Auto-importing to Neo4j")
                self._import_to_neo4j(cypher_file)
            
            # Update final state
            self.state.last_execution_time = datetime.now()
            self.state.save_state(self.state_file)
            
            # Log summary
            execution_time = time.time() - start_time
            self.logger.info("="*60)
            self.logger.info("PIPELINE COMPLETION SUMMARY")
            self.logger.info("="*60)
            self.logger.info(f"✅ Pipeline completed successfully in {execution_time:.2f} seconds")
            self.logger.info(f"📁 Processed {len(all_extracted_data)} files")
            self.logger.info(f"🔍 Total chunks processed: {self.state.total_chunks_processed}")
            self.logger.info(f"🏷️  Total entities: {len(merged_data['entities'])}")
            self.logger.info(f"🔗 Total relationships: {len(merged_data['relationships'])}")
            self.logger.info(f"💾 Cypher file saved: {cypher_file}")
            if self.config.get('auto_import_neo4j', False):
                self.logger.info("🗄️  Data imported to Neo4j")
            self.logger.info("="*60)
            
        except Exception as e:
            self.logger.critical(f"Pipeline failed: {e}")
            raise
    
    def get_pipeline_stats(self) -> Dict[str, Any]:
        """Get pipeline execution statistics."""
        return {
            "processed_files": len(self.state.processed_files),
            "failed_files": len(self.state.failed_files),
            "total_chunks_processed": self.state.total_chunks_processed,
            "total_files_to_process": self.state.total_files_to_process,
            "last_execution": self.state.last_execution_time.isoformat(),
            "graph_nodes": self.graph.number_of_nodes(),
            "graph_edges": self.graph.number_of_edges(),
            "supported_formats": self.supported_formats,
            "entity_types": self.allowed_entity_types,
            "relationship_types": self.allowed_relationship_types
        }


# Rename the class to be more general
class KnowledgeGraphPipeline(KnowledgeGraphPipeline):
    """General-purpose Knowledge Graph Pipeline for processing documents."""
    pass


def main():
    """Main function to run the pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description="General Knowledge Graph Pipeline")
    parser.add_argument("--config", default="config.yaml", help="Path to configuration YAML file")
    parser.add_argument("--env", default=".env", help="Path to environment file")
    parser.add_argument("--stats", action="store_true", help="Show pipeline statistics")
    parser.add_argument("--generate-cypher", action="store_true", help="Generate Cypher from cached data without processing new files")
    parser.add_argument("--import-neo4j", action="store_true", help="Import generated Cypher to Neo4j")
    
    args = parser.parse_args()
    
    # Initialize pipeline with env file
    pipeline = KnowledgeGraphPipeline(args.config, args.env)
    
    if args.stats:
        # Show statistics
        stats = pipeline.get_pipeline_stats()
        print(json.dumps(stats, indent=2))
    elif args.generate_cypher:
        # Generate Cypher from cached data
        cypher_file = pipeline.generate_cypher_from_cache()
        
        # Import to Neo4j if requested
        if cypher_file and args.import_neo4j:
            pipeline.logger.info("Importing generated Cypher to Neo4j")
            pipeline._import_to_neo4j(cypher_file)
    else:
        # Run regular pipeline
        pipeline.run_pipeline()
        
        # Import to Neo4j if requested
        if args.import_neo4j and not pipeline.config.get('auto_import_neo4j', False):
            latest_cypher = max(pipeline.output_dir.glob("knowledge_graph_*.cypher"), key=os.path.getctime, default=None)
            if latest_cypher:
                pipeline.logger.info(f"Importing latest Cypher file {latest_cypher} to Neo4j")
                pipeline._import_to_neo4j(latest_cypher)
            else:
                pipeline.logger.warning("No Cypher file found for Neo4j import")


if __name__ == "__main__":
    main()