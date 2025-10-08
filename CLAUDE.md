# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a FastAPI-based OpenAI Math Chat application that provides a web interface for mathematical conversations with AI. The application features LaTeX mathematical expression rendering using KaTeX, streaming responses from OpenAI's API, and a modern, production-ready architecture with proper resource management and security configurations.

## Architecture

### Backend (`app.py`)
- **Modern FastAPI application** with async lifespan management and dependency injection patterns
- **Streaming chat endpoint** (`/chat`) that proxies requests to OpenAI API using Server-Sent Events
- **Health check endpoint** (`/health`) for monitoring and deployment
- **Structured configuration management** with environment-based settings
- **Production-ready error handling** with proper HTTP status codes and logging
- **Security-hardened CORS** configuration restricting origins and methods
- **Resource management** with proper OpenAI client initialization and cleanup

### Frontend (`web/index.html`)
- **Single-page application** with vanilla JavaScript and modern UI design
- **Real-time streaming** chat interface with Server-Sent Events
- **Enhanced LaTeX rendering** supporting multiple delimiter types:
  - Standard: `$...$` (inline) and `$$...$$` (display)
  - LaTeX-style: `\(...\)` (inline) and `\[...\]` (display)
- **Smart markdown processing** with LaTeX expression protection during parsing
- **Responsive design** with gradient backgrounds and modern styling

### Key Dependencies
- `fastapi` - Modern web framework with async support
- `uvicorn` - High-performance ASGI server
- `openai` - Official OpenAI API client with streaming support
- `python-dotenv` - Environment variable management
- `pydantic` - Data validation and serialization

## Development Commands

### Running the Application
```bash
# Start the development server (recommended)
python app.py

# Alternative using uvicorn directly
uvicorn app:app --host 0.0.0.0 --port 8000 --reload

# For debugging in VS Code
# Use the "Debug FastAPI" configuration in .vscode/launch.json
```

### Environment Setup
```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment (Windows)
.venv\Scripts\activate

# Install dependencies
pip install fastapi uvicorn openai python-dotenv
```

### Configuration
- Create `.env` file with `OPENAI_API_KEY=your_api_key_here`
- Optional: Set `HOST` and `PORT` environment variables
- The application will run without API key but with limited functionality

## File Structure
```
app.py              # Main FastAPI application (refactored)
web/                # Frontend assets
  index.html        # Single-page chat interface (enhanced)
.vscode/            # VS Code development configuration
  launch.json       # Debug configurations for FastAPI
.env                # Environment variables (API keys)
.gitignore          # Git ignore patterns
CLAUDE.md           # This documentation file
```

## API Endpoints

- `GET /` - Serves the main chat interface
- `POST /chat` - Streaming chat endpoint accepting `{"message": "...", "model": "gpt-4o-mini"}`
- `GET /health` - Health check endpoint returning service status and configuration

## Frontend Integration Notes

The JavaScript frontend features:
- **Enhanced LaTeX processing** supporting four delimiter types before markdown parsing
- **Placeholder-based protection** preventing markdown conflicts with math expressions
- **Real-time streaming** with accumulative content display and smooth rendering
- **Automatic KaTeX rendering** after each content update with error tolerance
- **Modern UI/UX** with animations, gradients, and responsive design
- **Robust error handling** for network issues and malformed responses

## Recent Updates (Updated: 2025-09-22)

### Major Refactoring (Latest)
- **Architecture overhaul**: Implemented modern FastAPI patterns with async lifespan management
- **Security hardening**: Restricted CORS to localhost origins and specific HTTP methods
- **Resource management**: Added proper OpenAI client initialization and cleanup
- **Enhanced logging**: Unified logging format with uvicorn-style colored output
- **Production readiness**: Added health checks, structured error responses, and configuration management

### LaTeX Rendering Improvements
- **Multi-delimiter support**: Added support for `\(...\)` and `\[...\]` LaTeX delimiters
- **Enhanced processing**: Fixed math expression parsing conflicts with markdown
- **Pattern matching**: Improved regex patterns for reliable LaTeX expression detection
- **Display fixes**: Resolved rendering issues with complex mathematical notation

### Development Experience
- **VS Code integration**: Added debug configurations for FastAPI development
- **Better error messages**: Enhanced startup feedback and error reporting
- **Documentation**: Comprehensive project documentation and setup instructions

## Important Development Notes

### Code Patterns
- Use dependency injection patterns instead of global state when possible
- Follow async/await patterns throughout the application
- Implement proper error handling with structured responses
- Use Pydantic models for all API request/response validation

### Security Considerations
- CORS is restricted to localhost - update for production deployment
- API keys are loaded from environment variables only
- All user input is validated through Pydantic models
- Error responses don't expose internal implementation details

### Testing Strategy
- Unit tests should mock the OpenAI client using dependency injection
- Integration tests can use the health check endpoint
- Frontend tests should verify LaTeX rendering with various delimiter types
- Load testing should focus on the streaming chat endpoint