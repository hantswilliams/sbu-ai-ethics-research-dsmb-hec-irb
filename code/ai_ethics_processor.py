#!/usr/bin/env python3
"""
AI Ethics Research - LLM API Handler

This script processes ethical case scenarios through multiple LLMs (ChatGPT, Google Gemini, Anthropic Claude, and GROK)
and stores the results in a SQLite database.

Usage:
    python ai_ethics_processor.py [--model MODEL] [--iterations ITERATIONS] [--cleanup]

Options:
    --model MODEL          Specify which model to use: 'openai', 'gemini', 'claude', 'grok', or 'all' (default)
    --iterations ITERATIONS Number of iterations per scenario (default: 1)
    --cleanup              Clean up the database by fixing any incorrect vendor names
"""

import os
import json
import time
import argparse
from datetime import datetime
from pathlib import Path
import logging
import re
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Import database adapter
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shared.db_adapter import get_db_adapter

# Third-party packages for API access
import openai
from google.generativeai import GenerativeModel, configure
import google.generativeai as genai

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("ai_ethics_processor.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("ai_ethics")

class AIEthicsProcessor:
    def __init__(self, base_path=None, openai_api_key=None, gemini_api_key=None, claude_api_key=None, grok_api_key=None, db_path=None):
        """
        Initialize the AI Ethics Processor
        
        Args:
            base_path: Path to the project root
            openai_api_key: OpenAI API key (if None, will look for OPENAI_API_KEY env var)
            gemini_api_key: Google Gemini API key (if None, will look for GEMINI_API_KEY env var)
            claude_api_key: Anthropic Claude API key (if None, will look for CLAUDE_API_KEY env var)
            grok_api_key: GROK API key (if None, will look for GROK_API_KEY env var)
            db_path: Custom path to database (if None, will use the configured adapter)
        """
        # Set base path
        if base_path is None:
            self.base_path = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        else:
            self.base_path = Path(base_path)
        
        # Initialize API clients
        self._init_openai(openai_api_key)
        self._init_gemini(gemini_api_key)
        self._init_claude(claude_api_key)
        self._init_grok(grok_api_key)
        
        # Setup database
        self.db_adapter = get_db_adapter()
        self.db_adapter.init_db()
        logger.info(f"Using {self.db_adapter.type} database")
        
        # Load prompt template
        self.prompt_template = self._load_prompt_template()
        
        # Scan for case files
        self.case_files = self._scan_case_files()
        logger.info(f"Found {len(self.case_files)} case files")

    def _init_openai(self, api_key=None):
        """Initialize OpenAI client"""
        if api_key is None:
            api_key = os.environ.get("OPENAI_API_KEY")
        
        if api_key:
            self.openai_client = openai.OpenAI(api_key=api_key)
            model_name = os.environ.get("OPENAI_MODEL", "gpt-4.1")
            logger.info(f"OpenAI client initialized with model {model_name}")
        else:
            self.openai_client = None
            logger.warning("OpenAI API key not found. OpenAI functionality will be disabled.")

    def _init_gemini(self, api_key=None):
        """Initialize Google Gemini client"""
        if api_key is None:
            api_key = os.environ.get("GEMINI_API_KEY")
        
        if api_key:
            configure(api_key=api_key)
            # Get model name from environment variable or use default
            model_name = os.environ.get("GEMINI_MODEL", "gemini-1.5-pro")
            self.gemini_model = GenerativeModel(model_name)
            logger.info(f"Google Gemini client initialized with model {model_name}")
        else:
            self.gemini_model = None
            logger.warning("Google Gemini API key not found. Gemini functionality will be disabled.")

    def _init_claude(self, api_key=None):
        """Initialize Anthropic Claude client"""
        try:
            import anthropic
            
            if api_key is None:
                api_key = os.environ.get("CLAUDE_API_KEY")
            
            if api_key:
                self.claude_client = anthropic.Anthropic(api_key=api_key)
                # Get model name from environment variable or use default
                model_name = os.environ.get("CLAUDE_MODEL", "claude-3-opus-20240229")
                self.claude_model_name = model_name
                logger.info(f"Anthropic Claude client initialized with model {model_name}")
            else:
                self.claude_client = None
                logger.warning("Claude API key not found. Claude functionality will be disabled.")
        except ImportError:
            logger.error("Anthropic package not installed. Please install with 'pip install anthropic'")
            self.claude_client = None

    def _init_grok(self, api_key=None):
        """Initialize GROK client"""
        try:
            if api_key is None:
                api_key = os.environ.get("GROK_API_KEY")
            
            if api_key:
                self.grok_api_key = api_key
                model_name = os.environ.get("GROK_MODEL", "grok-1")
                self.grok_model_name = model_name
                logger.info(f"GROK client initialized with model {model_name}")
            else:
                self.grok_api_key = None
                logger.warning("GROK API key not found. GROK functionality will be disabled.")
        except Exception as e:
            logger.error(f"Error initializing GROK client: {e}")
            self.grok_api_key = None

    def _init_database(self):
        """Initialize the database"""
        # Database is now handled by the adapter
        pass

    def _load_prompt_template(self):
        """Load the prompt template from the research/prompt directory"""
        prompt_path = self.base_path / "research" / "prompt" / "prompt_v1.md"
        try:
            with open(prompt_path, 'r') as f:
                prompt_template = f.read()
                logger.info(f"Loaded prompt template from {prompt_path}")
                return prompt_template
        except Exception as e:
            logger.error(f"Error loading prompt template: {e}")
            raise

    def _scan_case_files(self):
        """Scan for case files in the scenarios directory"""
        scenarios_dir = self.base_path / "research" / "scenarios"
        case_files = []
        
        for file in scenarios_dir.glob("*_case.md"):
            case_id = file.stem.split('_')[0]  # Extract case ID from filename
            case_files.append({
                "id": case_id,
                "path": file
            })
        
        return sorted(case_files, key=lambda x: x["id"])

    def _load_case_content(self, case_path):
        """Load the content of a case file"""
        try:
            with open(case_path, 'r') as f:
                return f.read()
        except Exception as e:
            logger.error(f"Error loading case file {case_path}: {e}")
            raise

    def _build_full_prompt(self, case_content):
        """Build the full prompt by combining the template and case content"""
        # Template already has the context, role, etc. - just append the case
        full_prompt = f"{self.prompt_template}\n\nClinical Scenario for Analysis:\n{case_content}"
        return full_prompt

    def query_openai(self, prompt):
        """Query the OpenAI API with the given prompt"""
        if not self.openai_client:
            raise ValueError("OpenAI client not initialized")
        
        # Get model name from environment variable or use default
        model_name = os.environ.get("OPENAI_MODEL", "gpt-4")
        
        start_time = time.time()
        try:
            response = self.openai_client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "You are an AI Ethics Advisor embedded within a Hospital Ethics Committee."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=4000
            )
            processing_time = time.time() - start_time
            
            # Extract model version from response if available
            model_version = response.model
            
            return response.choices[0].message.content, processing_time, model_name, model_version
        except Exception as e:
            logger.error(f"Error querying OpenAI: {e}")
            raise

    def query_gemini(self, prompt):
        """Query the Google Gemini API with the given prompt"""
        if not self.gemini_model:
            raise ValueError("Gemini model not initialized")
        
        # Get the model name from environment or use default
        model_name = os.environ.get("GEMINI_MODEL", "gemini-1.5-pro")
        
        start_time = time.time()
        try:
            response = self.gemini_model.generate_content(prompt)
            processing_time = time.time() - start_time
            
            # Get model version if available, otherwise use the model name
            try:
                model_version = response.candidates[0].safety_ratings[0].model_version if hasattr(response, 'candidates') else model_name
            except (AttributeError, IndexError):
                model_version = model_name
                
            return response.text, processing_time, model_name, model_version
        except Exception as e:
            logger.error(f"Error querying Gemini: {e}")
            raise

    def query_claude(self, prompt):
        """Query the Anthropic Claude API with the given prompt"""
        if not self.claude_client:
            raise ValueError("Claude client not initialized")
        
        model_name = self.claude_model_name
        
        start_time = time.time()
        try:
            response = self.claude_client.messages.create(
                model=model_name,
                max_tokens=4000,
                temperature=0.7,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                system="You are an AI Ethics Advisor embedded within a Hospital Ethics Committee."
            )
            processing_time = time.time() - start_time
            
            # Extract model version
            model_version = response.model
            
            return response.content[0].text, processing_time, model_name, model_version
        except Exception as e:
            logger.error(f"Error querying Claude: {e}")
            raise

    def query_grok(self, prompt):
        """Query the GROK API with the given prompt"""
        if not self.grok_api_key:
            raise ValueError("GROK API key not initialized")
        
        import requests
        
        model_name = self.grok_model_name
        
        # GROK API endpoint - updated to correct API endpoint
        url = "https://api.x.ai/v1/chat/completions"
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.grok_api_key}"
        }
        
        data = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": "You are an AI Ethics Advisor embedded within a Hospital Ethics Committee."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.7,
            "max_tokens": 4000
        }
        
        start_time = time.time()
        try:
            # Add verify=True to enforce SSL verification and timeout to prevent hanging
            response = requests.post(url, headers=headers, json=data, verify=True, timeout=30)
            response.raise_for_status()  # Raise exception for HTTP errors
            
            result = response.json()
            processing_time = time.time() - start_time
            
            # Extract content and model version
            content = result["choices"][0]["message"]["content"]
            model_version = result.get("model", model_name)
            
            return content, processing_time, model_name, model_version
        except Exception as e:
            logger.error(f"Error querying GROK: {e}")
            raise

    def _extract_decisions(self, response_text):
        """Extract the three decisions from the response text"""
        # Initialize default values
        recommended = alternative = least_recommended = "Not found"
        
        # Look for sections with the three recommendations
        recommended_pattern = r"Recommended Decision \(Best Medical Option\)[:\s]*(.*?)(?=Alternative Decision|\Z)"
        alternative_pattern = r"Alternative Decision \(Second-Best Medical Option\)[:\s]*(.*?)(?=Least-Recommended Decision|\Z)"
        least_pattern = r"Least-Recommended Decision \(Third Medical Option\)[:\s]*(.*?)(?=\n\n|\Z)"
        
        # Try to extract each decision
        recommended_match = re.search(recommended_pattern, response_text, re.DOTALL)
        if recommended_match:
            recommended = recommended_match.group(1).strip()
            
        alternative_match = re.search(alternative_pattern, response_text, re.DOTALL)
        if alternative_match:
            alternative = alternative_match.group(1).strip()
            
        least_match = re.search(least_pattern, response_text, re.DOTALL)
        if least_match:
            least_recommended = least_match.group(1).strip()
            
        return {
            "recommended": recommended,
            "alternative": alternative,
            "least_recommended": least_recommended
        }

    def save_response(self, case_id, scenario_filename, vendor, model, model_version, iteration, prompt, response, processing_time):
        """Save a response to the database"""
        decisions = self._extract_decisions(response)
        
        conn = self.db_adapter.get_connection()
        
        response_id = self.db_adapter.insert_response(
            conn,
            case_id,
            scenario_filename, 
            vendor,
            model,
            model_version,
            iteration, 
            datetime.now().isoformat(), 
            prompt, 
            response,
            decisions["recommended"],
            decisions["alternative"],
            decisions["least_recommended"],
            processing_time
        )
        
        self.db_adapter.close_connection(conn)
        logger.info(f"Saved response for case {case_id} ({scenario_filename}), {vendor} {model} ({model_version}), iteration {iteration}")

    def process_case(self, case, model="all", iterations=1):
        """Process a single case through the specified model(s)"""
        case_id = case["id"]
        case_path = case["path"]
        scenario_filename = case_path.name
        case_content = self._load_case_content(case_path)
        full_prompt = self._build_full_prompt(case_content)
        
        # Convert legacy "both" parameter to "all" for backward compatibility
        if model == "both":
            model = "all"
            
        logger.info(f"Processing case {case_id} ({scenario_filename}) with model(s): {model}")
        
        if model in ["openai", "all"] and self.openai_client:
            for i in range(1, iterations + 1):
                logger.info(f"Running OpenAI iteration {i}/{iterations} for case {case_id}")
                try:
                    response, processing_time, model_name, model_version = self.query_openai(full_prompt)
                    self.save_response(
                        case_id=case_id,
                        scenario_filename=scenario_filename,
                        vendor="OpenAI",
                        model=model_name, 
                        model_version=model_version,
                        iteration=i, 
                        prompt=full_prompt, 
                        response=response, 
                        processing_time=processing_time
                    )
                    # Add delay to avoid rate limits
                    time.sleep(2)
                except Exception as e:
                    logger.error(f"Error in OpenAI processing: {e}")
        
        if model in ["gemini", "all"] and self.gemini_model:
            for i in range(1, iterations + 1):
                logger.info(f"Running Gemini iteration {i}/{iterations} for case {case_id}")
                try:
                    response, processing_time, model_name, model_version = self.query_gemini(full_prompt)
                    self.save_response(
                        case_id=case_id,
                        scenario_filename=scenario_filename,
                        vendor="Google",
                        model=model_name, 
                        model_version=model_version,
                        iteration=i, 
                        prompt=full_prompt, 
                        response=response, 
                        processing_time=processing_time
                    )
                    # Add delay to avoid rate limits
                    time.sleep(2)
                except Exception as e:
                    logger.error(f"Error in Gemini processing: {e}")
                    
        if model in ["claude", "all"] and self.claude_client:
            for i in range(1, iterations + 1):
                logger.info(f"Running Claude iteration {i}/{iterations} for case {case_id}")
                try:
                    response, processing_time, model_name, model_version = self.query_claude(full_prompt)
                    self.save_response(
                        case_id=case_id,
                        scenario_filename=scenario_filename,
                        vendor="Anthropic",
                        model=model_name, 
                        model_version=model_version,
                        iteration=i, 
                        prompt=full_prompt, 
                        response=response, 
                        processing_time=processing_time
                    )
                    # Add delay to avoid rate limits
                    time.sleep(2)
                except Exception as e:
                    logger.error(f"Error in Claude processing: {e}")
                    
        if model in ["grok", "all"] and self.grok_api_key:
            for i in range(1, iterations + 1):
                logger.info(f"Running GROK iteration {i}/{iterations} for case {case_id}")
                try:
                    response, processing_time, model_name, model_version = self.query_grok(full_prompt)
                    self.save_response(
                        case_id=case_id,
                        scenario_filename=scenario_filename,
                        vendor="GROK",
                        model=model_name, 
                        model_version=model_version,
                        iteration=i, 
                        prompt=full_prompt, 
                        response=response, 
                        processing_time=processing_time
                    )
                    # Add delay to avoid rate limits
                    time.sleep(2)
                except Exception as e:
                    logger.error(f"Error in GROK processing: {e}")
                    logger.warning("Attempting to use fallback for GROK...")
                    # Use the fallback method if the GROK API fails
                    self.fallback_for_grok(case_id, scenario_filename, i, full_prompt)

    def process_all_cases(self, model="all", iterations=1):
        """Process all available cases"""
        logger.info(f"Starting processing of {len(self.case_files)} cases with {model} model(s), {iterations} iteration(s) each")
        for case in self.case_files:
            self.process_case(case, model, iterations)
        logger.info("Completed processing all cases")

    def generate_summary_stats(self):
        """Generate summary statistics from the database"""
        conn = self.db_adapter.get_connection()
        cursor = conn.cursor()
        
        # Count by vendor
        cursor.execute("SELECT vendor, COUNT(*) FROM responses GROUP BY vendor")
        vendor_counts = cursor.fetchall()
        
        # Count by specific model
        cursor.execute("SELECT vendor, model, COUNT(*) FROM responses GROUP BY vendor, model")
        model_counts = cursor.fetchall()
        
        # Count by case
        cursor.execute("SELECT case_id, COUNT(*) FROM responses GROUP BY case_id")
        case_counts = cursor.fetchall()
        
        # Count by scenario filename
        cursor.execute("SELECT scenario_filename, COUNT(*) FROM responses GROUP BY scenario_filename")
        scenario_counts = cursor.fetchall()
        
        # Average processing time by vendor
        cursor.execute("SELECT vendor, AVG(processing_time) FROM responses GROUP BY vendor")
        vendor_avg_times = cursor.fetchall()
        
        # Average processing time by specific model
        cursor.execute("SELECT vendor, model, AVG(processing_time) FROM responses GROUP BY vendor, model")
        model_avg_times = cursor.fetchall()
        
        self.db_adapter.close_connection(conn)
        
        # Convert to dictionaries based on adapter type
        if self.db_adapter.type == "sqlite":
            vendor_counts_dict = dict(vendor_counts)
            case_counts_dict = dict(case_counts)
            scenario_counts_dict = dict(scenario_counts) if scenario_counts else {}
            vendor_avg_times_dict = dict(vendor_avg_times)
            model_counts_list = model_counts
            model_avg_times_list = model_avg_times
        else:
            # PostgreSQL returns dictionaries, we need to extract the values
            vendor_counts_dict = {row['vendor']: row['count'] for row in vendor_counts}
            case_counts_dict = {row['case_id']: row['count'] for row in case_counts}
            scenario_counts_dict = {row['scenario_filename']: row['count'] for row in scenario_counts} if scenario_counts else {}
            vendor_avg_times_dict = {row['vendor']: row['avg'] for row in vendor_avg_times}
            model_counts_list = [(row['vendor'], row['model'], row['count']) for row in model_counts]
            model_avg_times_list = [(row['vendor'], row['model'], row['avg']) for row in model_avg_times]
        
        # Calculate total count
        total_count = sum(vendor_counts_dict.values())
        
        logger.info("Summary Statistics:")
        logger.info(f"Total responses: {total_count}")
        logger.info(f"By vendor: {vendor_counts_dict}")
        logger.info(f"By model: {model_counts_list}")
        logger.info(f"By case: {case_counts_dict}")
        logger.info(f"By scenario: {scenario_counts_dict}")
        logger.info(f"Average processing times by vendor: {vendor_avg_times_dict}")
        logger.info(f"Average processing times by model: {model_avg_times_list}")
        
        return {
            "vendor_counts": vendor_counts_dict,
            "model_counts": {f"{vendor}_{model}": count for vendor, model, count in model_counts_list},
            "case_counts": case_counts_dict,
            "scenario_counts": scenario_counts_dict,
            "vendor_avg_times": vendor_avg_times_dict,
            "model_avg_times": {f"{vendor}_{model}": time for vendor, model, time in model_avg_times_list}
        }

    def fallback_for_grok(self, case_id, scenario_filename, iteration, prompt):
        """Fallback method if GROK API has issues - use OpenAI to simulate GROK response"""
        if not self.openai_client:
            logger.error("Cannot use fallback for GROK: OpenAI client not initialized")
            return
            
        logger.warning(f"Using OpenAI as fallback for GROK on case {case_id}")
        
        try:
            # Create a special prompt that asks OpenAI to simulate GROK
            fallback_prompt = f"""I need you to simulate how GROK AI would respond to this ethics scenario.
            Respond in GROK's style and format. The original prompt is:
            
            {prompt}
            
            Remember to format your response as GROK would, while providing an ethics committee recommendation.
            """
            
            response, processing_time, model_name, model_version = self.query_openai(fallback_prompt)
            
            # Save the response with vendor marked as GROK (simulated)
            self.save_response(
                case_id=case_id,
                scenario_filename=scenario_filename,
                vendor="GROK (simulated)",
                model="grok-1-simulated", 
                model_version="simulated-by-openai",
                iteration=iteration, 
                prompt=prompt, 
                response=response, 
                processing_time=processing_time
            )
            logger.info(f"Successfully saved simulated GROK response for case {case_id}")
        except Exception as e:
            logger.error(f"Error in GROK fallback: {e}")
            return

    def cleanup_database(self):
        """Clean up the database by fixing incorrect vendor names"""
        conn = self.db_adapter.get_connection()
        cursor = conn.cursor()
        
        # Update any "Claude" vendor entries to "Anthropic"
        if self.db_adapter.type == "sqlite":
            cursor.execute('''
            UPDATE responses 
            SET vendor = "Anthropic" 
            WHERE vendor = "Claude"
            ''')
        else:
            cursor.execute('''
            UPDATE responses 
            SET vendor = 'Anthropic' 
            WHERE vendor = 'Claude'
            ''')
        
        rows_affected = cursor.rowcount
        conn.commit()
        self.db_adapter.close_connection(conn)
        
        logger.info(f"Database cleanup complete: {rows_affected} rows updated (Claude -> Anthropic)")
        return rows_affected


def main():
    parser = argparse.ArgumentParser(description='Process ethical cases through AI models')
    parser.add_argument('--model', choices=['openai', 'gemini', 'claude', 'grok', 'all'], default='all',
                        help='Which model provider to use: openai, gemini, claude, grok, or all')
    parser.add_argument('--iterations', type=int, default=1,
                        help='Number of iterations per scenario')
    parser.add_argument('--db-path', type=str, default=None,
                        help='Custom path to SQLite database')
    parser.add_argument('--cleanup', action='store_true',
                        help='Clean up the database by fixing vendor names')
    args = parser.parse_args()
    
    try:
        processor = AIEthicsProcessor(db_path=args.db_path)
        
        # Run cleanup if requested
        if args.cleanup:
            rows_affected = processor.cleanup_database()
            logger.info(f"Database cleanup completed. {rows_affected} rows affected.")
            return
            
        processor.process_all_cases(model=args.model, iterations=args.iterations)
        processor.generate_summary_stats()
        logger.info("Processing completed successfully")
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise


if __name__ == "__main__":
    main()
