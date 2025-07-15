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
                <h2>Evaluator Participation</h2>
                <p>Number of evaluations by evaluator ID:</p>
                {basic_stats['evaluator_counts'].to_frame().to_html()}
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
