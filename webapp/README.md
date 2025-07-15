# AI Ethics Research Evaluation Web Application

This Flask web application provides a platform for human evaluators to assess AI-generated responses to healthcare ethics scenarios. It's part of a research project studying the effectiveness of different AI models in handling complex ethical decisions in clinical settings.

## UI Implementation 

The UI has been modernized using [Tailwind CSS](https://tailwindcss.com/), a utility-first CSS framework. Tailwind is loaded via CDN in the `base.html` template.

Key features of our Tailwind implementation:

- **Custom Theme**: We've defined a custom theme with SBU colors in `base.html`
- **Component Classes**: Common components like buttons and flash messages have custom Tailwind component classes
- **Responsive Design**: All pages are responsive and work well on mobile and desktop devices
- **Dynamic Model Handling**: UI dynamically adapts to any number of models/iterations
- **Blinded Evaluation**: Interface masks model identities to prevent bias

### Template Structure

- `base.html` - Base template with Tailwind configuration and layout
- `index.html` - Sign-in page
- `dashboard.html` - Main dashboard showing available scenarios
- `evaluate.html` - Evaluation form for AI responses

## Features

- Evaluator sign-in system
- Blind evaluation of AI responses (evaluators don't know which AI model generated the response)
- Assessment based on the SummEval four-way rubric:
  - Relevance
  - Correctness/Consistency
  - Fluency
  - Coherence
- Scoring on a 1-5 scale for each criterion
- Optional comments for additional feedback
- Dashboard showing completed and pending evaluations

## Setup

1. Install the required Python packages:

```bash
pip install -r requirements.txt
```

2. Set up your configuration by creating a `.env` file:

```bash
# Copy the template file
cp .env.template .env

# Edit the .env file with your settings
nano .env  # or use any text editor
```

3. Make sure the SQLite database exists and has the necessary tables:

```bash
# The app will automatically create the evaluation tables
# but it needs the responses table to already exist
```

## Running the Application

Start the Flask application:

```bash
python app.py
```

By default, the application will run on http://localhost:5000.

## Database Schema

The application uses the following tables:

### Existing Tables (from the main research project)
- `responses`: Contains AI-generated responses to ethical scenarios

### New Tables (created by this web app)
- `evaluators`: Information about human evaluators
  - `id`: Auto-incremented primary key
  - `name`: Evaluator's name
  - `email`: Optional email address
  - `created_at`: Timestamp of account creation

- `evaluations`: Individual response evaluations
  - `id`: Auto-incremented primary key
  - `evaluator_id`: Foreign key to evaluators table
  - `response_id`: Foreign key to responses table
  - `case_id`: Case identifier
  - `scenario_filename`: Scenario filename
  - `relevance_score`: Score from 1-5
  - `correctness_score`: Score from 1-5
  - `fluency_score`: Score from 1-5
  - `coherence_score`: Score from 1-5
  - `comments`: Optional feedback
  - `created_at`: Timestamp of evaluation

## Analyzing Results

After collecting evaluations, you can analyze the results using SQL queries or by connecting to the database with Python and pandas.

Example analysis query:

```sql
-- Average scores by vendor
SELECT r.vendor, 
       AVG(e.relevance_score) as avg_relevance,
       AVG(e.correctness_score) as avg_correctness,
       AVG(e.fluency_score) as avg_fluency,
       AVG(e.coherence_score) as avg_coherence,
       AVG((e.relevance_score + e.correctness_score + e.fluency_score + e.coherence_score) / 4.0) as avg_overall
FROM evaluations e
JOIN responses r ON e.response_id = r.id
GROUP BY r.vendor;
```
