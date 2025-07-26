#!/usr/bin/env python3
"""
AI Ethics Research - Evaluation Analysis Script

This script analyzes the human evaluations of AI-generated responses
and generates insights and visualizations.

Usage:
    python analyze_evaluations.py
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
import argparse
from dotenv import load_dotenv

# Add shared module to path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shared.db_adapter import get_db_adapter

# Load environment variables from .env file
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("evaluation_analysis.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("evaluation_analysis")

class EvaluationAnalyzer:
    def __init__(self, base_path=None):
        """
        Initialize the Evaluation Analyzer
        
        Args:
            base_path: Path to the project root
        """
        # Set base path
        if base_path is None:
            self.base_path = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        else:
            self.base_path = Path(base_path)
        
        # Initialize database adapter
        self.db_adapter = get_db_adapter()
        logger.info(f"Using {self.db_adapter.type} database")
            
        # Output directory for visualizations
        self.output_dir = self.base_path / "data" / "evaluation_results"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Load data
        self.responses_df = self._load_responses()
        self.evaluations_df = self._load_evaluations()
        self.scenario_details = self._load_scenario_details()
        
        # Join responses and evaluations
        if not self.evaluations_df.empty:
            self.df = pd.merge(
                self.evaluations_df, 
                self.responses_df, 
                left_on='response_id', 
                right_on='id', 
                suffixes=('', '_response')
            )
            logger.info(f"Loaded {len(self.df)} evaluations from database")
        else:
            self.df = pd.DataFrame()
            logger.warning("No evaluations found in database")
        
        # Load scenario details
        self.scenario_details = self._load_scenario_details()

    def _fetch_query(self, query):
        """Execute a query and return results as a pandas DataFrame"""
        conn = self.db_adapter.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(query)
            columns = [desc[0] for desc in cursor.description]
            data = cursor.fetchall()
            
            # Convert dict rows to list of values in same order as columns
            if self.db_adapter.type == "supabase":
                data = [[row[col] for col in columns] for row in data]
                
            return pd.DataFrame(data, columns=columns)
        except Exception as e:
            logger.error(f"Error executing query: {e}")
            return pd.DataFrame()
        finally:
            conn.close()

    def _load_responses(self):
        """Load responses data from database"""
        query = "SELECT * FROM responses"
        df = self._fetch_query(query)
        logger.info(f"Loaded {len(df)} responses from database")
        return df

    def _load_evaluations(self):
        """Load evaluations data from database"""
        query = "SELECT * FROM evaluations"
        df = self._fetch_query(query)
        logger.info(f"Loaded {len(df)} evaluations from database")
        return df

    def _load_scenario_details(self):
        """Load scenario details from the CSV file"""
        try:
            scenario_details_path = self.base_path / "research" / "scenarios" / "scenario_details.csv"
            logger.info(f"Loading scenario details from: {scenario_details_path}")
            
            if not scenario_details_path.exists():
                logger.error(f"Scenario details file not found at: {scenario_details_path}")
                return {}
                
            # Read directly with pandas, explicitly setting encoding
            df = pd.read_csv(scenario_details_path, encoding='utf-8')
            logger.info(f"Successfully loaded CSV with {len(df)} rows")
            logger.info("CSV contents:")
            logger.info(df.to_string())
            
            # Create mapping from case numbers to descriptions
            scenario_map = {}
            for _, row in df.iterrows():
                # Extract just the number from case_file (e.g., "1_case" -> "1")
                case_num = str(row['case_file']).split('_')[0].strip()
                description = str(row['simple_groups']).strip()
                
                # Store both string and numeric versions of the key
                scenario_map[case_num] = description
                scenario_map[int(case_num)] = description
                
                logger.info(f"Created mapping: {case_num} (and {int(case_num)}) -> {description}")
            
            logger.info("Final scenario mappings created:")
            for k, v in scenario_map.items():
                logger.info(f"  {k}: {v}")
            
            return scenario_map
            
        except Exception as e:
            logger.error(f"Error loading scenario details: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return {}
            
        except Exception as e:
            logger.error(f"Error loading scenario details: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return {}
                
            scenario_df = pd.read_csv(scenario_details_path)
            logger.info(f"Loaded CSV with columns: {scenario_df.columns.tolist()}")
            logger.info("First few rows of scenario_details.csv:")
            logger.info(scenario_df.head().to_string())
            
            # Create a mapping from case number to simple group description
            scenario_map = {}
            for _, row in scenario_df.iterrows():
                # Map both the full case name (e.g., "1_case") and just the number
                case_full = row['case_file']
                case_num = case_full.split('_')[0]
                description = row['simple_groups'].strip()
                
                # Log each mapping being created
                logger.info(f"Creating mapping for case {case_full}: {description}")
                
                scenario_map[case_full] = description  # Map full case name
                scenario_map[case_num + '_case'] = description  # Map with _case
                scenario_map[case_num] = description  # Map just the number
            
            logger.info("Final scenario mappings:")
            for key, value in scenario_map.items():
                logger.info(f"  {key}: {value}")
                
            return scenario_map
            
        except Exception as e:
            logger.error(f"Error loading scenario details: {str(e)}")
            return {}
        except Exception as e:
            logger.warning(f"Could not load scenario details: {e}")
            return {}

    def generate_basic_stats(self):
        """Generate basic statistics about the evaluations"""
        if self.df.empty:
            logger.warning("No data available for analysis")
            return None
            
        # Count evaluations by evaluator
        evaluator_counts = self.df.groupby('evaluator_id').size()
        
        # Count evaluations by scenario
        scenario_counts = self.df.groupby(['case_id', 'scenario_filename', 'iteration']).size()
        
        # Average scores by vendor
        vendor_scores = self.df.groupby('vendor').agg({
            'relevance_score': ['mean', 'std', 'count'],
            'correctness_score': ['mean', 'std', 'count'],
            'fluency_score': ['mean', 'std', 'count'],
            'coherence_score': ['mean', 'std', 'count']
        })
        
        # Average scores by model
        model_scores = self.df.groupby(['vendor', 'model']).agg({
            'relevance_score': ['mean', 'std', 'count'],
            'correctness_score': ['mean', 'std', 'count'],
            'fluency_score': ['mean', 'std', 'count'],
            'coherence_score': ['mean', 'std', 'count']
        })
        
        # Calculate overall scores
        self.df['overall_score'] = (
            self.df['relevance_score'] +
            self.df['correctness_score'] +
            self.df['fluency_score'] +
            self.df['coherence_score']
        ) / 4.0
        
        # Overall average by vendor
        vendor_overall = self.df.groupby('vendor')['overall_score'].agg(['mean', 'std', 'count'])
        
        # Overall average by model
        model_overall = self.df.groupby(['vendor', 'model'])['overall_score'].agg(['mean', 'std', 'count'])
        
        # Log the results
        logger.info(f"Evaluations by evaluator:\n{evaluator_counts}")
        logger.info(f"Evaluations by scenario:\n{scenario_counts}")
        logger.info(f"Average scores by vendor:\n{vendor_scores}")
        logger.info(f"Overall average by vendor:\n{vendor_overall}")
        
        # Save to CSV
        evaluator_counts.to_csv(self.output_dir / "evaluator_counts.csv")
        scenario_counts.to_csv(self.output_dir / "scenario_evaluation_counts.csv")
        vendor_scores.to_csv(self.output_dir / "vendor_scores.csv")
        vendor_overall.to_csv(self.output_dir / "vendor_overall_scores.csv")
        model_scores.to_csv(self.output_dir / "model_scores.csv")
        model_overall.to_csv(self.output_dir / "model_overall_scores.csv")
        
        return {
            'evaluator_counts': evaluator_counts,
            'scenario_counts': scenario_counts,
            'vendor_scores': vendor_scores,
            'vendor_overall': vendor_overall,
            'model_scores': model_scores,
            'model_overall': model_overall
        }

    def plot_score_distributions(self):
        """Plot the distributions of evaluation scores"""
        if self.df.empty:
            logger.warning("No data available for plotting")
            return
            
        # Set up the figure
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        # Plot each metric
        metrics = ['relevance_score', 'correctness_score', 'fluency_score', 'coherence_score']
        titles = ['Relevance', 'Correctness', 'Fluency', 'Coherence']
        
        for i, (metric, title) in enumerate(zip(metrics, titles)):
            sns.histplot(self.df[metric], kde=True, ax=axes[i], bins=5)
            axes[i].set_title(f'{title} Score Distribution')
            axes[i].set_xlabel(f'{title} Score (1-5)')
            axes[i].set_ylabel('Count')
            
        plt.tight_layout()
        plt.savefig(self.output_dir / "score_distributions.png")
        plt.close()
        
        # Plot overall score distribution
        plt.figure(figsize=(10, 6))
        sns.histplot(self.df['overall_score'], kde=True, bins=10)
        plt.title('Overall Score Distribution')
        plt.xlabel('Overall Score (1-5)')
        plt.ylabel('Count')
        plt.tight_layout()
        plt.savefig(self.output_dir / "overall_score_distribution.png")
        plt.close()
        
        logger.info("Score distribution plots saved")

    def plot_vendor_comparisons(self):
        """Plot comparisons between vendors"""
        if self.df.empty or len(self.df['vendor'].unique()) < 2:
            logger.warning("Not enough data for vendor comparison")
            return
            
        # Set up the figure
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        # Plot each metric
        metrics = ['relevance_score', 'correctness_score', 'fluency_score', 'coherence_score']
        titles = ['Relevance', 'Correctness', 'Fluency', 'Coherence']
        
        for i, (metric, title) in enumerate(zip(metrics, titles)):
            sns.boxplot(x='vendor', y=metric, data=self.df, ax=axes[i])
            axes[i].set_title(f'{title} by Vendor')
            axes[i].set_xlabel('Vendor')
            axes[i].set_ylabel(f'{title} Score (1-5)')
            
        plt.tight_layout()
        plt.savefig(self.output_dir / "vendor_score_comparisons.png")
        plt.close()
        
        # Plot overall comparison
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='vendor', y='overall_score', data=self.df)
        plt.title('Overall Score by Vendor')
        plt.xlabel('Vendor')
        plt.ylabel('Overall Score (1-5)')
        plt.tight_layout()
        plt.savefig(self.output_dir / "vendor_overall_comparison.png")
        plt.close()
        
        # Create bar chart of averages
        plt.figure(figsize=(12, 8))
        metrics_df = self.df.groupby('vendor')[metrics].mean().reset_index()
        metrics_df = pd.melt(metrics_df, id_vars=['vendor'], value_vars=metrics, 
                          var_name='Metric', value_name='Average Score')
        
        sns.barplot(x='Metric', y='Average Score', hue='vendor', data=metrics_df)
        plt.title('Average Scores by Metric and Vendor')
        plt.xlabel('Evaluation Metric')
        plt.ylabel('Average Score (1-5)')
        plt.ylim(1, 5)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(self.output_dir / "vendor_metric_averages.png")
        plt.close()
        
        logger.info("Vendor comparison plots saved")

    def plot_model_comparisons(self):
        """Plot comparisons between models"""
        if self.df.empty or len(self.df.groupby(['vendor', 'model'])) < 2:
            logger.warning("Not enough data for model comparison")
            return
            
        # Create a model identifier column
        self.df['model_id'] = self.df['vendor'] + ' ' + self.df['model']
        
        # Set up the figure
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        # Plot each metric
        metrics = ['relevance_score', 'correctness_score', 'fluency_score', 'coherence_score']
        titles = ['Relevance', 'Correctness', 'Fluency', 'Coherence']
        
        for i, (metric, title) in enumerate(zip(metrics, titles)):
            sns.boxplot(x='model_id', y=metric, data=self.df, ax=axes[i])
            axes[i].set_title(f'{title} by Model')
            axes[i].set_xlabel('Model')
            axes[i].set_ylabel(f'{title} Score (1-5)')
            axes[i].set_xticklabels(axes[i].get_xticklabels(), rotation=45, ha='right')
            
        plt.tight_layout()
        plt.savefig(self.output_dir / "model_score_comparisons.png")
        plt.close()
        
        # Plot overall comparison
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='model_id', y='overall_score', data=self.df)
        plt.title('Overall Score by Model')
        plt.xlabel('Model')
        plt.ylabel('Overall Score (1-5)')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(self.output_dir / "model_overall_comparison.png")
        plt.close()
        
        logger.info("Model comparison plots saved")

    def analyze_by_iteration(self):
        """Analyze scores by iteration to see if responses improve over time"""
        if self.df.empty:
            logger.warning("No data available for iteration analysis")
            return None
        
        # Group by iteration number and calculate mean scores
        iteration_scores = self.df.groupby('iteration').agg({
            'relevance_score': ['mean', 'std', 'count'],
            'correctness_score': ['mean', 'std', 'count'],
            'fluency_score': ['mean', 'std', 'count'],
            'coherence_score': ['mean', 'std', 'count'],
            'overall_score': ['mean', 'std', 'count']
        })
        
        # More detailed analysis by vendor and iteration
        vendor_iteration_scores = self.df.groupby(['vendor', 'iteration']).agg({
            'overall_score': ['mean', 'std', 'count']
        })
        
        # Plot iteration comparison
        plt.figure(figsize=(10, 6))
        ax = sns.barplot(x='iteration', y='overall_score', data=self.df)
        ax.set_title('Overall Score by Iteration')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Average Score (1-5)')
        plt.tight_layout()
        
        output_path = self.output_dir / 'iteration_comparison.png'
        plt.savefig(output_path)
        plt.close()
        
        return {
            'iteration_scores': iteration_scores,
            'vendor_iteration_scores': vendor_iteration_scores,
            'plot_path': output_path
        }

    def analyze_by_scenario(self):
        """Analyze scores by individual case scenarios"""
        if self.df.empty:
            logger.warning("No data available for scenario analysis")
            return None
            
        logger.info("Starting scenario analysis...")
        logger.info(f"Available scenario mappings: {self.scenario_details}")
            
        # Create scenario identifier
        self.df['scenario_id'] = self.df['case_id'].astype(str) + '-' + self.df['scenario_filename']
        
        # Map case IDs to descriptions
        def get_scenario_description(case_id):
            try:
                # Try both integer and string versions for lookup
                str_id = str(case_id)
                int_id = int(case_id)
                
                # Log what we're trying to match
                logger.info(f"Looking up case {case_id} (str: {str_id}, int: {int_id})")
                
                # Try both versions
                if int_id in self.scenario_details:
                    desc = self.scenario_details[int_id]
                    logger.info(f"Found match using int {int_id}: {desc}")
                    return desc
                if str_id in self.scenario_details:
                    desc = self.scenario_details[str_id]
                    logger.info(f"Found match using string {str_id}: {desc}")
                    return desc
                    
                logger.warning(f"No mapping found for case {case_id}")
                return 'Unknown'
            except Exception as e:
                logger.error(f"Error getting scenario description for {case_id}: {str(e)}")
                return 'Unknown'
        
        # Map case IDs to descriptions
        self.df['scenario_group'] = self.df['case_id'].apply(get_scenario_description)
        self.df['scenario_label'] = 'Case ' + self.df['case_id'].astype(str) + ' - ' + self.df['scenario_group']
        
        # Log the mappings we created
        logger.info("Final scenario mappings:")
        unique_mappings = self.df[['case_id', 'scenario_group']].drop_duplicates().sort_values('case_id')
        logger.info(unique_mappings.to_string())
        
        self.df['scenario_group'] = self.df['case_id'].apply(get_scenario_description)
        
        # Create final label
        self.df['scenario_label'] = 'Case ' + self.df['case_id'] + ' - ' + self.df['scenario_group']
        
        # Log the unique mappings to help with debugging
        unique_mappings = self.df[['case_id', 'scenario_group', 'scenario_label']].drop_duplicates().sort_values('case_id')
        logger.info("Final scenario mappings created:")
        logger.info(unique_mappings.to_string())
        
        # Calculate metrics by scenario
        scenario_metrics = self.df.groupby('scenario_id').agg({
            'relevance_score': ['mean', 'std', 'count'],
            'correctness_score': ['mean', 'std', 'count'],
            'fluency_score': ['mean', 'std', 'count'],
            'coherence_score': ['mean', 'std', 'count'],
            'overall_score': ['mean', 'std', 'count']
        })
        
        # Calculate metrics by scenario and model
        scenario_model_metrics = self.df.groupby(['scenario_id', 'vendor', 'model']).agg({
            'relevance_score': ['mean', 'std'],
            'correctness_score': ['mean', 'std'],
            'fluency_score': ['mean', 'std'],
            'coherence_score': ['mean', 'std'],
            'overall_score': ['mean', 'std']
        })
        
        # Plot scenario comparisons
        plt.figure(figsize=(15, 8))
        sns.boxplot(x='scenario_id', y='overall_score', data=self.df)
        plt.title('Overall Scores by Scenario')
        plt.xlabel('Scenario')
        plt.ylabel('Overall Score (1-5)')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(self.output_dir / "scenario_overall_scores.png")
        plt.close()
        
        # Plot heatmap of average scores by scenario and metric
        metrics = ['relevance_score', 'correctness_score', 'fluency_score', 'coherence_score']
        scenario_avg = self.df.groupby('scenario_label')[metrics].mean()
        
        plt.figure(figsize=(14, 10))  # Increased figure size for better readability
        sns.heatmap(scenario_avg, annot=True, cmap='YlOrRd', fmt='.2f')
        plt.title('Average Scores by Scenario and Metric')
        plt.ylabel('Scenario (Type)')
        plt.xlabel('Metric')
        plt.tight_layout()
        plt.savefig(self.output_dir / "scenario_metric_heatmap.png", bbox_inches='tight')
        plt.close()
        
        # Plot model performance by scenario
        plt.figure(figsize=(15, 8))
        scenario_model_overall = self.df.groupby(['scenario_id', 'model_id'])['overall_score'].mean().unstack()
        sns.heatmap(scenario_model_overall, annot=True, cmap='YlOrRd', fmt='.2f')
        plt.title('Model Performance by Scenario')
        plt.ylabel('Scenario')
        plt.xlabel('Model')
        plt.tight_layout()
        plt.savefig(self.output_dir / "scenario_model_heatmap.png")
        plt.close()
        
        # Create line charts for each metric
        metrics = {
            'relevance_score': 'Relevance',
            'correctness_score': 'Correctness',
            'fluency_score': 'Fluency',
            'coherence_score': 'Coherence',
            'overall_score': 'Overall'
        }
        
        for metric, metric_name in metrics.items():
            # Calculate mean scores by scenario and vendor
            vendor_scenario_scores = self.df.pivot_table(
                index='vendor',
                columns='scenario_id',
                values=metric,
                aggfunc='mean'
            )
            
            # Create the line plot
            plt.figure(figsize=(15, 8))
            for vendor in vendor_scenario_scores.index:
                plt.plot(
                    range(len(vendor_scenario_scores.columns)),
                    vendor_scenario_scores.loc[vendor],
                    marker='o',
                    label=vendor,
                    linewidth=2
                )
            
            plt.title(f'{metric_name} Scores by Scenario and Vendor')
            plt.xlabel('Scenario ID')
            plt.ylabel(f'{metric_name} Score (1-5)')
            # Create descriptive labels for scenarios
            scenario_labels = []
            for col in vendor_scenario_scores.columns:
                case_num = col.split('-')[0]  # Get the case number from scenario_id
                desc = self.scenario_details.get(case_num, 'Unknown')
                scenario_labels.append(f"Case {case_num}\n({desc})")
                
            plt.xticks(
                range(len(vendor_scenario_scores.columns)),
                scenario_labels,
                rotation=45,
                ha='right'
            )
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend(title='Vendor', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.ylim(1, 5)
            
            # Adjust layout to prevent label cutoff
            plt.tight_layout()
            plt.savefig(self.output_dir / f"scenario_{metric.lower()}_line_chart.png", 
                       bbox_inches='tight',
                       dpi=300)
            plt.close()
            
            # Also save the data used for the plot
            vendor_scenario_scores.to_csv(
                self.output_dir / f"scenario_{metric.lower()}_by_vendor.csv"
            )
        
        logger.info("Scenario analysis and line charts completed and saved")
        
        return {
            'scenario_metrics': scenario_metrics,
            'scenario_model_metrics': scenario_model_metrics
        }

    def generate_comprehensive_report(self):
        """Generate a comprehensive analysis report"""
        if self.df.empty:
            logger.warning("No data available for report generation")
            return
            
        # Run all analyses
        basic_stats = self.generate_basic_stats()
        self.plot_score_distributions()
        self.plot_vendor_comparisons()
        self.plot_model_comparisons()
        iteration_analysis = self.analyze_by_iteration()
        scenario_analysis = self.analyze_by_scenario()
        scenario_analysis = self.analyze_by_scenario()
        
        # Get unique vendors and models
        vendors = sorted(self.df['vendor'].unique())
        models = [f"{row['vendor']} {row['model']}" for _, row in 
                 self.df[['vendor', 'model']].drop_duplicates().iterrows()]
        
        # Create summary data
        summary = pd.DataFrame({
            'Total Evaluations': len(self.df),
            'Unique Evaluators': self.df['evaluator_id'].nunique(),
            'Unique Cases': self.df['case_id'].nunique(),
            'Unique Scenarios': self.df.groupby(['case_id', 'scenario_filename', 'iteration']).ngroups,
            'Vendors': len(vendors),
            'Models': len(models),
        }, index=['Summary'])
        
        summary.to_csv(self.output_dir / "evaluation_summary.csv")
        
        # Create the HTML report
        html_report = f"""<html>
        <head>
            <title>AI Ethics Research Evaluation Analysis</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2, h3 {{ color: #333366; }}
                table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                img {{ max-width: 100%; height: auto; margin: 10px 0; }}
                .section {{ margin-bottom: 30px; }}
            </style>
        </head>
        <body>
            <h1>AI Ethics Research Evaluation Analysis</h1>
            <p>Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <div class="section">
                <h2>Overview</h2>
                <p>Total evaluations analyzed: {len(self.df)}</p>
                <p>Vendors included: {', '.join(vendors)}</p>
                <p>Models included: {', '.join(models)}</p>
                {summary.to_html()}
            </div>
            
            <div class="section">
                <h2>Score Distributions</h2>
                <p>Distribution of scores across all evaluations:</p>
                <img src="score_distributions.png" alt="Score Distributions">
                <img src="overall_score_distribution.png" alt="Overall Score Distribution">
            </div>
            
            <div class="section">
                <h2>Vendor Comparisons</h2>
                <h3>Average Scores by Vendor</h3>
                {basic_stats['vendor_overall'].to_html()}
                
                <h3>Detailed Scores by Vendor</h3>
                {basic_stats['vendor_scores'].to_html()}
                
                <h3>Visual Comparisons</h3>
                <img src="vendor_score_comparisons.png" alt="Vendor Score Comparisons">
                <img src="vendor_overall_comparison.png" alt="Vendor Overall Comparison">
                <img src="vendor_metric_averages.png" alt="Vendor Metric Averages">
            </div>
            
            <div class="section">
                <h2>Model Comparisons</h2>
                <h3>Average Scores by Model</h3>
                {basic_stats['model_overall'].to_html()}
                
                <h3>Detailed Scores by Model</h3>
                {basic_stats['model_scores'].to_html()}
                
                <h3>Visual Comparisons</h3>
                <img src="model_score_comparisons.png" alt="Model Score Comparisons">
                <img src="model_overall_comparison.png" alt="Model Overall Comparison">
            </div>
            
            <div class="section">
                <h2>Scenario Coverage</h2>
                <p>Number of evaluations by scenario:</p>
                {basic_stats['scenario_counts'].to_frame().to_html()}
            </div>
            
            <div class="section">
                <h2>Iteration Analysis</h2>
                <h3>Scores by Iteration Number</h3>
                {iteration_analysis['iteration_scores'].to_html()}
                
                <h3>Vendor Performance by Iteration</h3>
                {iteration_analysis['vendor_iteration_scores'].to_html()}
                
                <p>Comparison of overall scores across iterations:</p>
                <img src="iteration_comparison.png" alt="Iteration Comparison">
            </div>
            
            <div class="section">
                <h2>Scenario Analysis</h2>
                <h3>Scores by Scenario</h3>
                {scenario_analysis['scenario_metrics'].to_html()}
                
                <h3>Model Performance by Scenario</h3>
                {scenario_analysis['scenario_model_metrics'].to_html()}
                
                <p>Visual comparisons of scores by scenario:</p>
                <img src="scenario_overall_scores.png" alt="Scenario Overall Scores">
                <img src="scenario_metric_heatmap.png" alt="Scenario Metric Heatmap">
                <img src="scenario_model_heatmap.png" alt="Model Performance by Scenario">
            </div>
            
            <div class="section">
                <h2>Evaluator Participation</h2>
                <p>Number of evaluations by evaluator ID:</p>
                {basic_stats['evaluator_counts'].to_frame().to_html()}
            </div>

            <div class="section">
                <h2>Detailed Scenario Analysis</h2>
                
                <h3>Scenario Types</h3>
                <table class="scenario-types">
                    <tr>
                        <th>Case Number</th>
                        <th>Type</th>
                    </tr>
                    {"".join([f"<tr><td>Case {k}</td><td>{v}</td></tr>" for k, v in self.scenario_details.items()])}
                </table>
                
                <h3>Overall Metrics by Scenario</h3>
                {scenario_analysis['scenario_metrics'].to_html()}
                
                <h3>Model Performance by Scenario</h3>
                {scenario_analysis['scenario_model_metrics'].to_html()}
                
                <h3>Visual Analysis</h3>
                <h4>Overall Performance</h4>
                <img src="scenario_overall_scores.png" alt="Scenario Overall Scores">
                <img src="scenario_metric_heatmap.png" alt="Scenario Metric Heatmap">
                <img src="scenario_model_heatmap.png" alt="Scenario Model Performance Heatmap">
                
                <h4>Detailed Metric Analysis by Scenario</h4>
                <p>The following charts show how each vendor performed across different scenarios for each evaluation metric:</p>
                
                <div class="metric-charts">
                    <h5>Relevance Scores</h5>
                    <img src="scenario_relevance_score_line_chart.png" alt="Relevance Scores by Scenario and Vendor">
                    
                    <h5>Correctness Scores</h5>
                    <img src="scenario_correctness_score_line_chart.png" alt="Correctness Scores by Scenario and Vendor">
                    
                    <h5>Fluency Scores</h5>
                    <img src="scenario_fluency_score_line_chart.png" alt="Fluency Scores by Scenario and Vendor">
                    
                    <h5>Coherence Scores</h5>
                    <img src="scenario_coherence_score_line_chart.png" alt="Coherence Scores by Scenario and Vendor">
                    
                    <h5>Overall Scores</h5>
                    <img src="scenario_overall_score_line_chart.png" alt="Overall Scores by Scenario and Vendor">
                </div>
                
                <p>This section provides a detailed breakdown of how different models performed across various scenarios,
                allowing us to identify patterns in model performance across different types of cases. The line charts
                above show the progression of scores across scenarios for each vendor, making it easy to identify:
                <ul>
                    <li>Which vendors consistently perform better for specific metrics</li>
                    <li>How performance varies across different scenarios</li>
                    <li>Where there are significant gaps in performance between vendors</li>
                    <li>Which scenarios are particularly challenging or easy for specific vendors</li>
                </ul>
                </p>
            </div>
        </body>
        </html>"""
        
        # Write the HTML report to a file
        with open(self.output_dir / "evaluation_report.html", "w") as f:
            f.write(html_report)
            
        logger.info(f"Comprehensive evaluation report saved to {self.output_dir / 'evaluation_report.html'}")


def main():
    parser = argparse.ArgumentParser(description='Analyze AI ethics evaluations')
    parser.add_argument('--base-path', type=str, default=None,
                        help='Custom path to project root directory')
    args = parser.parse_args()
    
    try:
        analyzer = EvaluationAnalyzer(base_path=args.base_path)
        analyzer.generate_comprehensive_report()
        logger.info("Evaluation analysis completed successfully")
    except Exception as e:
        logger.error(f"Error in analysis: {e}")
        raise


if __name__ == "__main__":
    main()
