# AI Ethics Research Processor

This tool processes ethical case scenarios through multiple Large Language Models (LLMs) and stores the results in a database for analysis. Supports both SQLite and Supabase PostgreSQL databases.

## Setup

1. Install the required Python packages:

```bash
pip install -r requirements.txt
```

2. Set up your API keys by creating a `.env` file:

```bash
# Copy the template file
cp .env.template .env

# Edit the .env file with your actual API keys
nano .env  # or use any text editor
```

Your `.env` file should contain:

```
# API Keys
OPENAI_API_KEY=your_openai_api_key_here
GEMINI_API_KEY=your_gemini_api_key_here
CLAUDE_API_KEY=your_claude_api_key_here
GROK_API_KEY=your_grok_api_key_here

# Model configurations (optional)
OPENAI_MODEL=gpt-4.1              # Default OpenAI model to use
GEMINI_MODEL=gemini-1.5-pro       # Default Gemini model to use
CLAUDE_MODEL=claude-3-opus-20240229 # Default Claude model to use
GROK_MODEL=grok-1                 # Default GROK model to use

# Database configuration
# Choose database type: "sqlite" or "supabase"
DB_TYPE=sqlite

# SQLite configuration (used when DB_TYPE=sqlite)
# DB_PATH=custom/path/to/results.db

# Supabase PostgreSQL configuration (used when DB_TYPE=supabase)
# Get these values from your Supabase project settings > Database > Connection Info
SUPABASE_URL=db.eigtivpuaudjkdadajfb.supabase.co
SUPABASE_KEY=your-database-password
```

## Database Configuration

The system supports two database types:

1. **SQLite** (default) - A file-based database that doesn't require additional setup. 
   - Set `DB_TYPE=sqlite` in your `.env` file.
   - Optionally specify a custom database location with `DB_PATH`.

2. **Supabase PostgreSQL** - A cloud-based PostgreSQL database.
   - Set `DB_TYPE=supabase` in your `.env` file.
   - Set `SUPABASE_URL` to your Supabase database host value from the Connection Info panel.
     - It should look like: `db.eigtivpuaudjkdadajfb.supabase.co`
     - If you only provide the project ID, the adapter will automatically format it correctly.
   - Set `SUPABASE_KEY` to your database password from the Connection Info panel.
   - Make sure your Supabase project allows database connections from your IP address.

## Model Configuration

The script is configured to use:
- OpenAI: "gpt-4.1" by default
- Gemini: "gemini-1.5-pro" by default
- Claude: "claude-3-opus-20240229" by default
- GROK: "grok-1" by default

You can change these by setting the environment variables in your `.env` file.

## Usage

Run the script with default settings (all models, 1 iteration per case):

```bash
python ai_ethics_processor.py
```

Specify a single model:

```bash
python ai_ethics_processor.py --model openai
```

or

```bash
python ai_ethics_processor.py --model gemini
```

or

```bash
python ai_ethics_processor.py --model claude
```

or

```bash
python ai_ethics_processor.py --model grok
```

Change the number of iterations:

```bash
python ai_ethics_processor.py --iterations 5
```

Use a custom database location (only applies to SQLite):

```bash
python ai_ethics_processor.py --db-path /path/to/custom/database.db
```

Clean up any incorrect vendor names in the database:

```bash
python ai_ethics_processor.py --cleanup
```

## Analysis

After collecting responses, run the analysis script:

```bash
python analyze_results.py
```

Or with a custom database path (for SQLite):

```bash
python analyze_results.py --db-path /path/to/custom/database.db
```

## Database Schema

Results are stored in a database with the following schema:

Table: `responses`
- `id`: Auto-incremented primary key
- `case_id`: Identifier of the case
- `scenario_filename`: Filename of the scenario (e.g., "1_case.md")
- `vendor`: The AI provider (e.g., "OpenAI", "Google", "Anthropic", "GROK")
- `model`: The specific model name
- `model_version`: The model version as returned by the API
- `iteration`: Iteration number
- `timestamp`: When the response was generated
- `prompt`: The full prompt sent to the model
- `full_response`: Complete response from the model
- `recommended_decision`: Extracted primary recommendation
- `alternative_decision`: Extracted secondary recommendation
- `least_recommended_decision`: Extracted least recommended option
- `processing_time`: Time taken to generate the response in seconds

## Directory Structure

The script expects the following directory structure:

```
sbu-ai-ethics-research-dsmb-hec-irb/
├── code/
│   ├── ai_ethics_processor.py
│   ├── db_adapter.py
│   └── requirements.txt
├── data/
│   └── results.db (created automatically when using SQLite)
├── research/
│   ├── prompt/
│   │   └── prompt_v1.md
│   └── scenarios/
│       ├── 1_case.md
│       ├── 1_discussion.md
│       ├── 2_case.md
│       └── ...
└── manuscript/
    ├── intro.md
    └── methods.md
```

## Analyzing Results

After running the script, you can analyze the results using SQL queries or by connecting to the database using Python and pandas:

### For SQLite:

```python
import sqlite3
import pandas as pd

# Connect to the database
conn = sqlite3.connect('../data/results.db')

# Load all responses into a pandas DataFrame
df = pd.read_sql_query("SELECT * FROM responses", conn)

# Perform analysis
model_comparison = df.groupby(['vendor', 'model']).agg({
    'processing_time': ['mean', 'std'],
    'id': 'count'
}).reset_index()

print(model_comparison)

# Close the connection
conn.close()
```

### For Supabase PostgreSQL:

```python
import psycopg2
from psycopg2.extras import RealDictCursor
import pandas as pd
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get connection details from environment variables
supabase_host = os.environ["SUPABASE_URL"]
supabase_password = os.environ["SUPABASE_KEY"]

# Connect to the Supabase PostgreSQL database
conn = psycopg2.connect(
    user="postgres",
    password=supabase_password,
    host=supabase_host,
    port="5432",
    dbname="postgres",
    cursor_factory=RealDictCursor
)

# Load all responses into a pandas DataFrame
df = pd.read_sql_query("SELECT * FROM responses", conn)

# Perform analysis
model_comparison = df.groupby(['vendor', 'model']).agg({
    'processing_time': ['mean', 'std'],
    'id': 'count'
}).reset_index()

print(model_comparison)

# Close the connection
conn.close()
```

## Required Packages

The main script requires the following Python packages:

```
openai
google-generativeai
anthropic
python-dotenv
psycopg2-binary (for Supabase)
pandas (for analysis)
```

## Logging

The script logs its activity to both the console and a file named `ai_ethics_processor.log` in the current directory.
