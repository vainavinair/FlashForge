# FlashForge: Smart Flashcards with Spaced Repetition

The aim of this project is to design and develop an intelligent web application that automates the creation and scheduling of flashcards. Unlike existing tools that primarily focus on static flashcard generation, this app integrates an LLM (Large Language Model) for dynamic flashcard creation from user-uploaded material (text or handwritten notes), and implements spaced repetition techniques using a probabilistic algorithm (Bayesian-inspired) for optimal memory retention.

## Features

- 📚 **AI-Powered Flashcard Generation** - Uses Google Gemini to create flashcards from uploaded documents
- 🔍 **OCR Support** - Extract text from scanned PDFs and handwritten notes using Tesseract OCR
- 🧠 **Spaced Repetition** - Bayesian-inspired algorithm for optimal learning schedules
- 📊 **Analytics Dashboard** - Track your learning progress and performance metrics
- 📄 **Multiple File Formats** - Support for PDF, TXT, and DOCX files

## Quick Start

### Local Development

1. Clone the repository
2. Install dependencies: `pip install -r requirements-deploy.txt`
3. Set up environment variables (see `.env.example`)
4. Run the app: `python app.py`

### Deployment

See [DEPLOYMENT.md](DEPLOYMENT.md) for complete deployment instructions on Render.com.

**Quick Deploy Steps:**
1. Push code to GitHub
2. Connect repository to Render.com
3. Configure environment variables
4. Add persistent disk (1 GB)
5. Deploy!

## Technology Stack

- **Backend**: Flask, Python 3.12
- **Database**: SQLite with persistent storage
- **AI/ML**: Google Gemini API, Hybrid Scheduler (Thompson Sampling + Knowledge Tracing)
- **OCR**: Tesseract OCR
- **Production**: Gunicorn WSGI server

## Documentation

- [Deployment Guide](DEPLOYMENT.md) - Complete Render.com deployment instructions
- [Evaluation Guide](EVALUATION_GUIDE.md) - Research evaluation metrics and methodology

## License

This project is for educational purposes.