# AI Ethics Research Platform for DSMBs, HECs, and IRBs

This platform processes and evaluates AI model responses to ethical case scenarios in research oversight, specifically focusing on Data Safety Monitoring Boards (DSMBs), Healthcare Ethics Committees (HECs), and Institutional Review Boards (IRBs).

## Project Structure

```
.
├── code/                      # Core processing scripts
│   ├── ai_ethics_processor.py # Main script for processing cases through AI models
│   ├── analyze_results.py     # Analysis script for AI responses
│   ├── requirements.txt       # Python dependencies for processing
│   └── README.md             # Processing module documentation
├── shared/                    # Shared modules
│   └── db_adapter.py         # Database adapter for SQLite/Supabase
├── data/                      # Data storage
│   ├── results.db            # SQLite database (if using SQLite)
│   ├── analysis_output/      # Analysis visualizations and reports
│   └── evaluation_results/   # Human evaluation results and analysis
├── research/                  # Research materials
│   ├── scenarios/            # Case scenarios
│   └── prompt/               # Prompt templates
├── webapp/                    # Web interface for human evaluation
│   ├── app.py               # Flask application
│   ├── analyze_evaluations.py# Evaluation analysis script
│   ├── requirements.txt     # Python dependencies for webapp
│   ├── static/             # Static files (CSS, JS)
│   └── templates/          # HTML templates
├── Dockerfile               # Container configuration for webapp
├── .env.template           # Template for environment variables
└── README.md               # This file
```

## Features

- **Multi-Model Support**: Processes cases through multiple AI models:
  - OpenAI GPT-4
  - Google Gemini
  - Anthropic Claude
  - Grok
- **Database Flexibility**: Supports both SQLite and Supabase PostgreSQL
- **Web Interface**: Flask-based evaluation platform
- **Analysis Tools**: Comprehensive analysis scripts for both AI responses and human evaluations

## Setup

1. Clone the repository:
\`\`\`bash
git clone [repository-url]
cd sbu-ai-ethics-research-dsmb-hec-irb
\`\`\`

2. Create and activate a virtual environment:
\`\`\`bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\\Scripts\\activate
\`\`\`

3. Install dependencies:
\`\`\`bash
pip install -r code/requirements.txt
pip install -r webapp/requirements.txt
\`\`\`

4. Configure environment variables:
\`\`\`bash
cp .env.template .env
\`\`\`
Edit .env with your:
- API keys (OpenAI, Google, Anthropic, Grok)
- Database configuration
- Flask settings

## Database Configuration

The platform supports two database types:

### SQLite (Default)
- Set in .env: \`DB_TYPE=sqlite\`
- Default location: \`data/results.db\`
- No additional configuration needed

### Supabase PostgreSQL
- Set in .env: \`DB_TYPE=supabase\`
- Required variables:
  - SUPABASE_DB_NAME
  - SUPABASE_USER
  - SUPABASE_PASSWORD
  - SUPABASE_HOST
  - SUPABASE_PORT

## Usage

### Processing AI Responses
\`\`\`bash
cd code
python ai_ethics_processor.py [--model MODEL] [--iterations ITERATIONS]
\`\`\`

### Running Analysis
\`\`\`bash
python code/analyze_results.py
\`\`\`

### Running the Web Application

#### Local Development
\`\`\`bash
python webapp/app.py
\`\`\`

#### Docker Deployment
\`\`\`bash
# Build the image
docker build -t ai-ethics-webapp .

# Run the container
docker run -p 5005:5005 --env-file .env ai-ethics-webapp
\`\`\`

The web interface will be available at http://localhost:5005

## Analysis Outputs

Analysis scripts generate various visualizations and reports in:
- \`data/analysis_output/\`: AI response analysis
- \`data/evaluation_results/\`: Human evaluation analysis

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

TBD

## Contact

Hants Williams