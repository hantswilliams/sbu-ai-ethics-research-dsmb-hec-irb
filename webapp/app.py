#!/usr/bin/env python3
"""
AI Ethics Research - Evaluation Web Application

This Flask application provides a web interface for human evaluators to assess AI-generated
responses to ethical case scenarios.

Usage:
    python app.py
"""

import os
import uuid
import random
import logging
from datetime import datetime
from pathlib import Path
from flask import Flask, render_template, request, redirect, url_for, session, flash
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("webapp.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("ai_ethics_webapp")

# Import the database adapter
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shared.db_adapter import get_db_adapter

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", str(uuid.uuid4()))

# Set base path
BASE_PATH = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Initialize database adapter
db_adapter = get_db_adapter()
db_adapter.init_db()

# Add context processor for all templates
@app.context_processor
def inject_globals():
    return {
        'current_year': datetime.now().year
    }

def get_db_connection():
    """Create a connection to the database using the configured adapter"""
    return db_adapter.get_connection()

def init_db():
    """Initialize the database with evaluation tables if they don't exist"""
    # The database is now initialized when the adapter is created
    logger.info("Database initialization handled by adapter")
    
    # We can add any additional initialization logic here if needed
    pass

@app.route('/')
def index():
    """Home page - Sign in form"""
    return render_template('index.html')

@app.route('/signin', methods=['POST'])
def signin():
    """Process sign in form and create/retrieve evaluator record"""
    email = request.form.get('email', '').strip().lower()
    
    # Validate email
    if not email:
        flash('Email is required')
        return redirect(url_for('index'))
    
    # Check if email is from stonybrook.edu domain
    if not email.endswith('@stonybrook.edu'):
        flash('Please use your Stony Brook University email (@stonybrook.edu)')
        return redirect(url_for('index'))
    
    # Extract username from email to use as name
    name = email.split('@')[0]
    
    # Store in database
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Check if evaluator already exists with this email
    if db_adapter.type == "sqlite":
        cursor.execute('SELECT id FROM evaluators WHERE email = ?', (email,))
    else:
        cursor.execute('SELECT id FROM evaluators WHERE email = %s', (email,))
    evaluator = cursor.fetchone()
    
    if evaluator:
        evaluator_id = evaluator['id']
        logger.info(f"Existing evaluator signed in: {email} (ID: {evaluator_id})")
    else:
        if db_adapter.type == "sqlite":
            cursor.execute('INSERT INTO evaluators (name, email, created_at) VALUES (?, ?, ?)',
                          (name, email, datetime.now().isoformat()))
            conn.commit()
            evaluator_id = cursor.lastrowid
        else:
            cursor.execute('INSERT INTO evaluators (name, email, created_at) VALUES (%s, %s, %s) RETURNING id',
                          (name, email, datetime.now().isoformat()))
            evaluator_id = cursor.fetchone()['id']
            conn.commit()
        logger.info(f"New evaluator created: {email} (ID: {evaluator_id})")
    
    conn.close()
    
    # Store in session
    session['evaluator_id'] = evaluator_id
    session['name'] = name
    session['email'] = email
    
    return redirect(url_for('dashboard'))

@app.route('/dashboard')
def dashboard():
    """Show dashboard with scenarios to evaluate"""
    if 'evaluator_id' not in session:
        return redirect(url_for('index'))
    
    conn = get_db_connection()
    
    # Get all unique case scenarios
    cursor = conn.cursor()
    cursor.execute('''
    SELECT DISTINCT case_id, scenario_filename 
    FROM responses 
    ORDER BY case_id, scenario_filename
    ''')
    scenarios = cursor.fetchall()
    
    # Get completed evaluations for this evaluator
    if db_adapter.type == "sqlite":
        cursor.execute('''
        SELECT DISTINCT r.case_id, r.scenario_filename, r.vendor, r.model  
        FROM evaluations e
        JOIN responses r ON e.response_id = r.id
        WHERE e.evaluator_id = ?
        ''', (session['evaluator_id'],))
    else:
        cursor.execute('''
        SELECT DISTINCT r.case_id, r.scenario_filename, r.vendor, r.model  
        FROM evaluations e
        JOIN responses r ON e.response_id = r.id
        WHERE e.evaluator_id = %s
        ''', (session['evaluator_id'],))
    completed = {(row['case_id'], row['scenario_filename'], row['vendor'], row['model']) for row in cursor.fetchall()}
    
    # Mark scenarios as completed or not
    scenario_list = []
    for scenario in scenarios:
        # Get all distinct vendor/model combinations for this case
        if db_adapter.type == "sqlite":
            cursor.execute('''
            SELECT DISTINCT vendor, model
            FROM responses
            WHERE case_id = ? AND scenario_filename = ?
            ORDER BY vendor, model
            ''', (scenario['case_id'], scenario['scenario_filename']))
        else:
            cursor.execute('''
            SELECT DISTINCT vendor, model
            FROM responses
            WHERE case_id = %s AND scenario_filename = %s
            ORDER BY vendor, model
            ''', (scenario['case_id'], scenario['scenario_filename']))
        models = cursor.fetchall()
        
        # Count how many models this evaluator has already evaluated
        evaluated_models = [
            (scenario['case_id'], scenario['scenario_filename'], model['vendor'], model['model'])
            in completed for model in models
        ]
        evaluated_count = sum(evaluated_models)
        
        # Mark as complete only if all models have been evaluated
        is_complete = (evaluated_count >= len(models))
        
        # Get model breakdown
        model_breakdown = []
        for idx, model in enumerate(models, 1):
            is_evaluated = (scenario['case_id'], scenario['scenario_filename'], model['vendor'], model['model']) in completed
            model_breakdown.append({
                'vendor': f"Masked Model {idx}",  # Mask the actual vendor name
                'model': "",  # Hide the model name entirely
                'evaluated': is_evaluated,
                'model_number': idx  # Add a model number for reference
            })
        
        scenario_list.append({
            'case_id': scenario['case_id'],
            'scenario_filename': scenario['scenario_filename'],
            'completed': is_complete,
            'progress': f"{evaluated_count}/{len(models)}",
            'model_breakdown': model_breakdown
        })
    
    conn.close()
    logger.info(f"Dashboard loaded for evaluator {session.get('email')} (ID: {session['evaluator_id']})")
    
    return render_template('dashboard.html', 
                          email=session.get('email'),
                          scenarios=scenario_list)

@app.route('/evaluate/<case_id>/<scenario_filename>')
def evaluate(case_id, scenario_filename):
    """Show evaluation form for a specific scenario"""
    if 'evaluator_id' not in session:
        return redirect(url_for('index'))
    
    # First try to read scenario content directly from file
    scenario_path = BASE_PATH / "research" / "scenarios" / scenario_filename
    case_content = None
    
    if scenario_path.exists():
        try:
            with open(scenario_path, 'r') as f:
                case_content = f.read()
            logger = logging.getLogger(__name__)
            logger.info(f"Successfully read scenario content from file: {scenario_path}")
        except Exception as e:
            print(f"Error reading scenario file: {e}")
            case_content = f"Error: Could not read scenario file. {str(e)}"
    else:
        # If file doesn't exist, check if it's in the cases table as fallback
        conn = get_db_connection()
        cursor = conn.cursor()
        
        try:
            if db_adapter.type == "sqlite":
                cursor.execute('''
                SELECT content 
                FROM cases 
                WHERE case_id = ? AND filename = ?
                ''', (case_id, scenario_filename))
            else:
                cursor.execute('''
                SELECT content 
                FROM cases 
                WHERE case_id = %s AND filename = %s
                ''', (case_id, scenario_filename))
            case = cursor.fetchone()
            
            if case:
                case_content = case['content']
                print(f"Retrieved scenario content from database")
            else:
                case_content = f"Error: Scenario file not found at {scenario_path} and no record in database"
        except Exception as e:
            print(f"Error checking database for scenario: {e}")
            case_content = f"Error: Could not retrieve scenario content. {str(e)}"
    
    # Try to read the ethics committee discussion file
    discussion_content = None
    discussion_filename = scenario_filename.replace('_case.md', '_discussion.md')
    discussion_path = BASE_PATH / "research" / "scenarios" / discussion_filename
    
    if discussion_path.exists():
        try:
            with open(discussion_path, 'r') as f:
                discussion_content = f.read()
            logger = logging.getLogger(__name__)
            logger.info(f"Successfully read discussion content from file: {discussion_path}")
        except Exception as e:
            print(f"Error reading discussion file: {e}")
            discussion_content = f"Error: Could not read discussion file. {str(e)}"
    else:
        discussion_content = "Ethics committee discussion not available for this scenario."
        logger.warning(f"Discussion file not found at {discussion_path}")
    
    # Get a random response for this case that hasn't been evaluated by this evaluator
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        # First, get all available models and vendors for this case
        if db_adapter.type == "sqlite":
            cursor.execute('''
            SELECT DISTINCT vendor, model
            FROM responses
            WHERE case_id = ? AND scenario_filename = ?
            ''', (case_id, scenario_filename))
        else:
            cursor.execute('''
            SELECT DISTINCT vendor, model
            FROM responses
            WHERE case_id = %s AND scenario_filename = %s
            ''', (case_id, scenario_filename))
        
        available_models = cursor.fetchall()
        
        # Get all responses already evaluated by this evaluator for this case
        if db_adapter.type == "sqlite":
            cursor.execute('''
            SELECT r.vendor, r.model
            FROM evaluations e
            JOIN responses r ON e.response_id = r.id
            WHERE e.evaluator_id = ? AND r.case_id = ? AND r.scenario_filename = ?
            ''', (session['evaluator_id'], case_id, scenario_filename))
        else:
            cursor.execute('''
            SELECT r.vendor, r.model
            FROM evaluations e
            JOIN responses r ON e.response_id = r.id
            WHERE e.evaluator_id = %s AND r.case_id = %s AND r.scenario_filename = %s
            ''', (session['evaluator_id'], case_id, scenario_filename))
        
        evaluated_models = {(row['vendor'], row['model']) for row in cursor.fetchall()}
        
        # Find models that haven't been evaluated yet
        unevaluated_models = []
        for model in available_models:
            if (model['vendor'], model['model']) not in evaluated_models:
                unevaluated_models.append((model['vendor'], model['model']))
        
        # If there are unevaluated models, select one randomly
        if unevaluated_models:
            # Randomly select a vendor/model combination that hasn't been evaluated
            vendor, model_name = random.choice(unevaluated_models)
            
            # Get a response for this vendor/model combination
            if db_adapter.type == "sqlite":
                cursor.execute('''
                SELECT r.id, r.full_response, r.vendor, r.model, r.iteration 
                FROM responses r
                LEFT JOIN evaluations e ON r.id = e.response_id AND e.evaluator_id = ?
                WHERE r.case_id = ? AND r.scenario_filename = ? 
                AND r.vendor = ? AND r.model = ?
                AND e.id IS NULL
                ORDER BY RANDOM()
                LIMIT 1
                ''', (session['evaluator_id'], case_id, scenario_filename, vendor, model_name))
            else:
                cursor.execute('''
                SELECT r.id, r.full_response, r.vendor, r.model, r.iteration 
                FROM responses r
                LEFT JOIN evaluations e ON r.id = e.response_id AND e.evaluator_id = %s
                WHERE r.case_id = %s AND r.scenario_filename = %s 
                AND r.vendor = %s AND r.model = %s
                AND e.id IS NULL
                ORDER BY RANDOM()
                LIMIT 1
                ''', (session['evaluator_id'], case_id, scenario_filename, vendor, model_name))
        else:
            # If all models have been evaluated, check if there are any responses left
            if db_adapter.type == "sqlite":
                cursor.execute('''
                SELECT r.id, r.full_response, r.vendor, r.model, r.iteration 
                FROM responses r
                LEFT JOIN evaluations e ON r.id = e.response_id AND e.evaluator_id = ?
                WHERE r.case_id = ? AND r.scenario_filename = ? AND e.id IS NULL
                ORDER BY RANDOM()
                LIMIT 1
                ''', (session['evaluator_id'], case_id, scenario_filename))
            else:
                cursor.execute('''
                SELECT r.id, r.full_response, r.vendor, r.model, r.iteration 
                FROM responses r
                LEFT JOIN evaluations e ON r.id = e.response_id AND e.evaluator_id = %s
                WHERE r.case_id = %s AND r.scenario_filename = %s AND e.id IS NULL
                ORDER BY RANDOM()
                LIMIT 1
                ''', (session['evaluator_id'], case_id, scenario_filename))
        
        response = cursor.fetchone()
    except Exception as e:
        print(f"Error querying responses: {e}")
        logger.error(f"Error querying responses: {e}")
        response = None
    
    conn.close()
    
    if not response:
        flash('No responses available for this scenario or you have already evaluated all of them.')
        return redirect(url_for('dashboard'))
    
    # Mask vendor/model information to ensure blind evaluation
    return render_template('evaluate.html', 
                          case_id=case_id,
                          scenario_filename=scenario_filename,
                          case_content=case_content,
                          discussion_content=discussion_content,
                          response=response)

@app.route('/submit_evaluation', methods=['POST'])
def submit_evaluation():
    """Process evaluation form submission"""
    if 'evaluator_id' not in session:
        return redirect(url_for('index'))
    
    # Get form data
    response_id = request.form.get('response_id')
    case_id = request.form.get('case_id')
    scenario_filename = request.form.get('scenario_filename')
    iteration = request.form.get('iteration')
    relevance_score = request.form.get('relevance_score')
    correctness_score = request.form.get('correctness_score')
    fluency_score = request.form.get('fluency_score')
    coherence_score = request.form.get('coherence_score')
    comments = request.form.get('comments', '')
    
    # Validate data
    if not all([response_id, case_id, scenario_filename, iteration,
               relevance_score, correctness_score, fluency_score, coherence_score]):
        flash('All scores are required')
        return redirect(url_for('evaluate', case_id=case_id, scenario_filename=scenario_filename))
    
    # Store in database
    conn = get_db_connection()
    cursor = conn.cursor()
    
    if db_adapter.type == "sqlite":
        cursor.execute('''
        INSERT INTO evaluations (
            evaluator_id, response_id, case_id, scenario_filename, iteration,
            relevance_score, correctness_score, fluency_score, coherence_score,
            comments, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            session['evaluator_id'], response_id, case_id, scenario_filename, iteration,
            relevance_score, correctness_score, fluency_score, coherence_score,
            comments, datetime.now().isoformat()
        ))
    else:
        cursor.execute('''
        INSERT INTO evaluations (
            evaluator_id, response_id, case_id, scenario_filename, iteration,
            relevance_score, correctness_score, fluency_score, coherence_score,
            comments, created_at
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ''', (
            session['evaluator_id'], response_id, case_id, scenario_filename, iteration,
            relevance_score, correctness_score, fluency_score, coherence_score,
            comments, datetime.now()
        ))
    
    conn.commit()
    conn.close()
    
    flash('Evaluation submitted successfully!')
    
    # Check if there are more responses to evaluate for this scenario
    return redirect(url_for('dashboard'))

@app.route('/logout')
def logout():
    """Clear session and redirect to home"""
    session.clear()
    return redirect(url_for('index'))

if __name__ == '__main__':
    # Initialize database
    init_db()
    
    # Run the app
    debug_mode = os.environ.get("FLASK_DEBUG", "True").lower() in ("true", "1", "t")
    app.run(debug=debug_mode, host='0.0.0.0', port=int(os.environ.get("PORT", 5005)))
