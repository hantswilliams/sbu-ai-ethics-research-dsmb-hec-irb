# AI Ethics Research Processor

This tool processes ethical case scenarios through multiple Large Language Models (LLMs) and stores the results in a SQLite database for analysis.

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
OPENAI_API_KEY=your_openai_api_key_here
GEMINI_API_KEY=your_gemini_api_key_here

# Optional configurations
OPENAI_MODEL=gpt-4                # Default OpenAI model to use
GEMINI_MODEL=gemini-1.5-pro       # Default Gemini model to use
# DB_PATH=custom/path/to/results.db
```

## Model Configuration

The script is configured to use:
- OpenAI: "gpt-4" by default
- Gemini: "gemini-1.5-pro" by default

You can change these by setting the `OPENAI_MODEL` and `GEMINI_MODEL` environment variables in your `.env` file.

## Usage

Run the script with default settings (both models, 3 iterations per case):

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

Change the number of iterations:

```bash
python ai_ethics_processor.py --iterations 5
```

Use a custom database location:

```bash
python ai_ethics_processor.py --db-path /path/to/custom/database.db
```

## Analysis

After collecting responses, run the analysis script:

```bash
python analyze_results.py
```

Or with a custom database path:

```bash
python analyze_results.py --db-path /path/to/custom/database.db
```

## Database

Results are stored in a SQLite database at `../data/results.db` relative to the script location. The database contains a single table `responses` with the following columns:

- `id`: Auto-incremented primary key
- `case_id`: Identifier of the case
- `scenario_filename`: Filename of the scenario (e.g., "1_case.md")
- `vendor`: The AI provider (e.g., "OpenAI", "Google")
- `model`: The specific model name (e.g., "gpt-4", "gemini-1.5-pro")
- `model_version`: The model version as returned by the API
- `iteration`: Iteration number
- `timestamp`: When the response was generated
- `prompt`: The full prompt sent to the model
- `full_response`: Complete response from the model
- `recommended_decision`: Extracted primary recommendation
- `alternative_decision`: Extracted secondary recommendation
- `least_recommended_decision`: Extracted least recommended option
- `processing_time`: Time taken to generate the response in seconds
- `alternative_decision`: Extracted secondary recommendation
- `least_recommended_decision`: Extracted least recommended option
- `processing_time`: Time taken to generate the response in seconds

## Directory Structure

The script expects the following directory structure:

```
sbu-ai-ethics-research-dsmb-hec-irb/
├── code/
│   ├── ai_ethics_processor.py
│   └── requirements.txt
├── data/
│   └── results.db (created automatically)
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

```python
import sqlite3
import pandas as pd

# Connect to the database
conn = sqlite3.connect('../data/results.db')

# Load all responses into a pandas DataFrame
df = pd.read_sql_query("SELECT * FROM responses", conn)

# Perform analysis
model_comparison = df.groupby('model').agg({
    'processing_time': ['mean', 'std'],
    'id': 'count'
}).reset_index()

print(model_comparison)

# Close the connection
conn.close()
```

## Logging

The script logs its activity to both the console and a file named `ai_ethics_processor.log` in the current directory.
