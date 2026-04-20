# Test Suite

Test suite for odoo-ai-assistant with 80% coverage requirement.

## Test Files

### Core Modules (One test file per module)

- **`test_retriever.py`** - Tests for FAISS retriever (10 tests)
  - Module import verification
  - Initialization with index and corpus
  - Query functionality with top-k results
  - Environment variable configuration
  - Prompt building
  - Error handling

- **`test_llm.py`** - Tests for LLM module (8 tests)
  - Module import verification
  - ModelResponse dataclass
  - Bedrock API integration (mocked)
  - Error handling
  - Multi-model support
  - Region configuration

- **`test_indexer.py`** - Tests for index building (8 tests)
  - Module import verification
  - Text chunking with overlap
  - FAISS index creation
  - Corpus generation
  - Output directory handling

- **`test_ingest.py`** - Tests for data ingestion (8 tests)
  - Module import verification
  - URL fetching (success/failure/timeout)
  - JSONL file creation
  - Source processing
  - Error handling

- **`test_app.py`** - Tests for Gradio app (8 tests)
  - Module import verification
  - Context recall calculation
  - Chat functionality
  - Conversation history
  - App creation

## Coverage Configuration

### Excluded from Coverage
- `scripts/*` - Utility scripts (e.g., `list_models.py`)
- `generate_qa.py` - Unused module (replaced by manual data curation)
- `app.py` - Gradio UI code (core business logic tested separately, UI verified manually)
- `tests/*` - Test files themselves
- Build artifacts and virtual environments

### Coverage Target
- **Minimum**: 80%
- **Current**: ~88% (excluding Gradio UI code)

### Coverage by Module
- `retriever.py`: 100%
- `__init__.py`: 100%
- `indexer.py`: 90%
- `ingest.py`: 86%
- `llm.py`: 82%

## Running Tests

```bash
# Run all tests with coverage
uv run pytest --cov=src --cov-fail-under=80

# Run specific test file
uv run pytest tests/test_retriever.py -v

# Run with coverage report
uv run pytest --cov=src --cov-report=term-missing

# Run with verbose output
uv run pytest -v
```

## Test Strategy

- **Unit tests**: Mock external dependencies (Bedrock API, file I/O)
- **Integration tests**: Use temporary directories for file operations
- **Fixtures**: Reusable test data (FAISS indices, mock embedders)
- **Mocking**: Boto3 clients, HTTP requests, sentence transformers

## Adding New Tests

When adding new functionality:
1. Create corresponding test file (`test_<module>.py`)
2. Mock external dependencies
3. Test both success and error paths
4. Ensure coverage stays above 80%
5. Run `uv run pytest` before committing
