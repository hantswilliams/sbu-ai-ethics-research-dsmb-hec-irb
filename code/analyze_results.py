#!/usr/bin/env python3
"""
AI Ethics Research - Analysis Script

This script analyzes the responses stored in the database and generates
insights and visualizations.

Usage:
    python analyze_results.py
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
import re
from collections import Counter
import argparse
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Import database adapter
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shared.db_adapter import get_db_adapter

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("analysis.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("analysis")

class ResultsAnalyzer:
    def __init__(self, base_path=None):
        """
        Initialize the Results Analyzer
        
        Args:
            base_path: Path to the project root
        """
        # Set base path
        if base_path is None:
            self.base_path = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        else:
            self.base_path = Path(base_path)
            
        # Get the database adapter
        self.db_adapter = get_db_adapter()
        logger.info(f"Using {self.db_adapter.type} database")
            
        # Output directory for visualizations
        self.output_dir = self.base_path / "data" / "analysis_output"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Load data
        self.df = self._load_data()
        logger.info(f"Loaded {len(self.df)} responses from database")

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
            self.db_adapter.close_connection(conn)

    def _load_data(self):
        """Load data from database into a pandas DataFrame"""
        query = "SELECT * FROM responses"
        return self._fetch_query(query)

    def generate_basic_stats(self):
        """Generate basic statistics about the responses"""
        # Count by vendor
        vendor_counts = self.df['vendor'].value_counts()
        
        # Count by model
        model_counts = self.df.groupby(['vendor', 'model']).size().reset_index(name='count')
        
        # Count by case
        case_counts = self.df['case_id'].value_counts()
        
        # Count by scenario filename if available
        scenario_counts = self.df['scenario_filename'].value_counts() if 'scenario_filename' in self.df.columns else None
        
        # Processing time stats by vendor
        vendor_time_stats = self.df.groupby('vendor')['processing_time'].agg(['mean', 'std', 'min', 'max'])
        
        # Processing time stats by model
        model_time_stats = self.df.groupby(['vendor', 'model'])['processing_time'].agg(['mean', 'std', 'min', 'max'])
        
        # Log the results
        logger.info(f"Response counts by vendor:\n{vendor_counts}")
        logger.info(f"Response counts by model:\n{model_counts}")
        logger.info(f"Response counts by case:\n{case_counts}")
        if scenario_counts is not None:
            logger.info(f"Response counts by scenario filename:\n{scenario_counts}")
        logger.info(f"Processing time statistics by vendor (seconds):\n{vendor_time_stats}")
        logger.info(f"Processing time statistics by model (seconds):\n{model_time_stats}")
        
        # Create a summary dataframe
        summary = pd.DataFrame({
            'Total Responses': len(self.df),
            'Unique Cases': self.df['case_id'].nunique(),
            'Unique Scenarios': self.df['scenario_filename'].nunique() if 'scenario_filename' in self.df.columns else self.df['case_id'].nunique(),
            'Unique Vendors': self.df['vendor'].nunique(),
            'Unique Models': self.df[['vendor', 'model']].drop_duplicates().shape[0],
        }, index=['Summary'])
        
        # Save to CSV
        summary.to_csv(self.output_dir / "summary_stats.csv")
        vendor_time_stats.to_csv(self.output_dir / "vendor_time_stats.csv")
        model_time_stats.to_csv(self.output_dir / "model_time_stats.csv")
        
        return {
            'vendor_counts': vendor_counts,
            'model_counts': model_counts,
            'case_counts': case_counts,
            'scenario_counts': scenario_counts,
            'vendor_time_stats': vendor_time_stats,
            'model_time_stats': model_time_stats,
            'summary': summary
        }

    def plot_processing_times(self):
        """Plot processing times by vendor and model"""
        # By vendor
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='vendor', y='processing_time', data=self.df)
        plt.title('Processing Time by Vendor')
        plt.ylabel('Time (seconds)')
        plt.xlabel('Vendor')
        plt.tight_layout()
        plt.savefig(self.output_dir / "processing_times_by_vendor.png")
        plt.close()
        
        # By model within vendor
        plt.figure(figsize=(12, 6))
        sns.boxplot(x='model', y='processing_time', hue='vendor', data=self.df)
        plt.title('Processing Time by Model and Vendor')
        plt.ylabel('Time (seconds)')
        plt.xlabel('Model')
        plt.xticks(rotation=45)
        plt.legend(title='Vendor')
        plt.tight_layout()
        plt.savefig(self.output_dir / "processing_times_by_model.png")
        plt.close()
        
        # Plot processing time by case and vendor
        plt.figure(figsize=(14, 8))
        sns.boxplot(x='case_id', y='processing_time', hue='vendor', data=self.df)
        plt.title('Processing Time by Case and Vendor')
        plt.ylabel('Time (seconds)')
        plt.xlabel('Case ID')
        plt.tight_layout()
        plt.savefig(self.output_dir / "processing_times_by_case_vendor.png")
        plt.close()
        
        # Create a grouped bar chart showing average processing time by model
        avg_times = self.df.groupby(['vendor', 'model'])['processing_time'].mean().reset_index()
        plt.figure(figsize=(12, 6))
        sns.barplot(x='model', y='processing_time', hue='vendor', data=avg_times)
        plt.title('Average Processing Time by Model and Vendor')
        plt.ylabel('Avg Time (seconds)')
        plt.xlabel('Model')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(self.output_dir / "avg_processing_times.png")
        plt.close()
        
        logger.info(f"Processing time plots saved to {self.output_dir}")

    def analyze_response_length(self):
        """Analyze the length of responses"""
        # Add a column for response length
        self.df['response_length'] = self.df['full_response'].apply(len)
        
        # Basic stats by vendor
        vendor_length_stats = self.df.groupby('vendor')['response_length'].agg(['mean', 'std', 'min', 'max'])
        
        # Basic stats by model
        model_length_stats = self.df.groupby(['vendor', 'model'])['response_length'].agg(['mean', 'std', 'min', 'max'])
        
        logger.info(f"Response length statistics by vendor:\n{vendor_length_stats}")
        logger.info(f"Response length statistics by model:\n{model_length_stats}")
        
        # Plot by vendor
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='vendor', y='response_length', data=self.df)
        plt.title('Response Length by Vendor')
        plt.ylabel('Length (characters)')
        plt.xlabel('Vendor')
        plt.tight_layout()
        plt.savefig(self.output_dir / "response_lengths_by_vendor.png")
        plt.close()
        
        # Plot by model within vendor
        plt.figure(figsize=(12, 6))
        sns.boxplot(x='model', y='response_length', hue='vendor', data=self.df)
        plt.title('Response Length by Model and Vendor')
        plt.ylabel('Length (characters)')
        plt.xlabel('Model')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(self.output_dir / "response_lengths_by_model.png")
        plt.close()
        
        # Create a grouped bar chart showing average response length by model
        avg_lengths = self.df.groupby(['vendor', 'model'])['response_length'].mean().reset_index()
        plt.figure(figsize=(12, 6))
        sns.barplot(x='model', y='response_length', hue='vendor', data=avg_lengths)
        plt.title('Average Response Length by Model and Vendor')
        plt.ylabel('Avg Length (characters)')
        plt.xlabel('Model')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(self.output_dir / "avg_response_lengths.png")
        plt.close()
        
        # Save stats
        vendor_length_stats.to_csv(self.output_dir / "vendor_response_length_stats.csv")
        model_length_stats.to_csv(self.output_dir / "model_response_length_stats.csv")
        
        return vendor_length_stats, model_length_stats

    def analyze_ethical_principles(self):
        """Analyze mentions of ethical principles in responses"""
        # Define ethical principles to look for
        principles = {
            'autonomy': r'\b(?:autonomy|autonomous|self-determination)\b',
            'beneficence': r'\b(?:beneficence|benefit|welfare)\b',
            'non-maleficence': r'\b(?:non-maleficence|nonmaleficence|do no harm|harm avoidance)\b',
            'justice': r'\b(?:justice|fairness|equity|equality)\b'
        }
        
        # Count mentions of each principle
        for principle, pattern in principles.items():
            self.df[f'{principle}_mentions'] = self.df['full_response'].apply(
                lambda x: len(re.findall(pattern, x, re.IGNORECASE))
            )
        
        # Calculate statistics by vendor
        vendor_principle_stats = self.df.groupby('vendor')[
            [f'{p}_mentions' for p in principles.keys()]
        ].agg(['mean', 'std', 'sum'])
        
        # Calculate statistics by model
        model_principle_stats = self.df.groupby(['vendor', 'model'])[
            [f'{p}_mentions' for p in principles.keys()]
        ].agg(['mean', 'std', 'sum'])
        
        logger.info(f"Ethical principle mention statistics by vendor:\n{vendor_principle_stats}")
        logger.info(f"Ethical principle mention statistics by model:\n{model_principle_stats}")
        
        # Create plots
        # 1. Vendor comparison
        principle_mentions_by_vendor = pd.melt(
            self.df, 
            id_vars=['vendor', 'case_id', 'iteration'],
            value_vars=[f'{p}_mentions' for p in principles.keys()],
            var_name='principle',
            value_name='mentions'
        )
        principle_mentions_by_vendor['principle'] = principle_mentions_by_vendor['principle'].str.replace('_mentions', '')
        
        plt.figure(figsize=(12, 8))
        sns.barplot(x='principle', y='mentions', hue='vendor', data=principle_mentions_by_vendor)
        plt.title('Average Mentions of Ethical Principles by Vendor')
        plt.ylabel('Average Mentions per Response')
        plt.xlabel('Ethical Principle')
        plt.tight_layout()
        plt.savefig(self.output_dir / "ethical_principle_mentions_by_vendor.png")
        plt.close()
        
        # 2. Model comparison
        principle_mentions_by_model = pd.melt(
            self.df, 
            id_vars=['vendor', 'model', 'case_id', 'iteration'],
            value_vars=[f'{p}_mentions' for p in principles.keys()],
            var_name='principle',
            value_name='mentions'
        )
        principle_mentions_by_model['principle'] = principle_mentions_by_model['principle'].str.replace('_mentions', '')
        
        # Aggregate by model and principle
        model_principle_avg = principle_mentions_by_model.groupby(['vendor', 'model', 'principle'])['mentions'].mean().reset_index()
        
        plt.figure(figsize=(16, 10))
        g = sns.catplot(
            data=model_principle_avg,
            kind="bar",
            x="principle", y="mentions", hue="model", col="vendor",
            height=6, aspect=1.5
        )
        g.set_axis_labels("Ethical Principle", "Average Mentions")
        g.set_titles("Vendor: {col_name}")
        plt.tight_layout()
        plt.savefig(self.output_dir / "ethical_principle_mentions_by_model.png")
        plt.close()
        
        # Save stats
        vendor_principle_stats.to_csv(self.output_dir / "vendor_ethical_principle_stats.csv")
        model_principle_stats.to_csv(self.output_dir / "model_ethical_principle_stats.csv")
        
        return vendor_principle_stats, model_principle_stats

    def analyze_recommendation_consistency(self):
        """Analyze consistency of recommendations across iterations"""
        # Group by case_id, vendor, and model
        groups = self.df.groupby(['case_id', 'vendor', 'model'])
        
        consistency_results = []
        
        for (case_id, vendor, model), group in groups:
            if len(group) <= 1:
                continue  # Skip if only one iteration
                
            # Check for consistency in recommendations
            rec_consistent = len(group['recommended_decision'].unique()) == 1
            alt_consistent = len(group['alternative_decision'].unique()) == 1
            least_consistent = len(group['least_recommended_decision'].unique()) == 1
            
            consistency_results.append({
                'case_id': case_id,
                'vendor': vendor,
                'model': model,
                'iterations': len(group),
                'rec_consistent': rec_consistent,
                'alt_consistent': alt_consistent,
                'least_consistent': least_consistent,
                'all_consistent': rec_consistent and alt_consistent and least_consistent
            })
        
        # Convert to DataFrame
        consistency_df = pd.DataFrame(consistency_results)
        
        if consistency_df.empty:
            logger.warning("No consistency data available - need multiple iterations per model")
            # Create a dummy plot with a message
            plt.figure(figsize=(10, 6))
            plt.text(0.5, 0.5, "Insufficient data for consistency analysis.\nRequires multiple iterations of the same case-vendor-model.", 
                     horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
            plt.axis('off')
            plt.savefig(self.output_dir / "recommendation_consistency.png")
            plt.close()
            
            # Create empty dummy plots for the other visualizations too
            plt.figure(figsize=(12, 6))
            plt.text(0.5, 0.5, "Insufficient data for vendor consistency analysis", 
                     horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
            plt.axis('off')
            plt.savefig(self.output_dir / "recommendation_consistency_by_vendor.png")
            plt.close()
            
            plt.figure(figsize=(16, 10))
            plt.text(0.5, 0.5, "Insufficient data for model consistency analysis", 
                     horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
            plt.axis('off')
            plt.savefig(self.output_dir / "recommendation_consistency_by_model.png")
            plt.close()
            
            return None
            
        # Calculate overall consistency by vendor
        vendor_consistency = consistency_df.groupby('vendor').agg({
            'rec_consistent': 'mean',
            'alt_consistent': 'mean',
            'least_consistent': 'mean',
            'all_consistent': 'mean'
        })
        
        # Calculate overall consistency by model
        model_consistency = consistency_df.groupby(['vendor', 'model']).agg({
            'rec_consistent': 'mean',
            'alt_consistent': 'mean',
            'least_consistent': 'mean',
            'all_consistent': 'mean'
        })
        
        # Calculate overall consistency by case
        case_consistency = consistency_df.groupby('case_id').agg({
            'rec_consistent': 'mean',
            'alt_consistent': 'mean',
            'least_consistent': 'mean',
            'all_consistent': 'mean'
        })
        
        logger.info(f"Recommendation consistency by vendor:\n{vendor_consistency}")
        logger.info(f"Recommendation consistency by model:\n{model_consistency}")
        logger.info(f"Recommendation consistency by case:\n{case_consistency}")
        
        # Create a scenario-specific plot (this was missing)
        plt.figure(figsize=(12, 6))
        sns.barplot(x=case_consistency.index, y='all_consistent', data=case_consistency.reset_index())
        plt.title('Overall Recommendation Consistency by Scenario')
        plt.ylabel('Consistency Rate')
        plt.xlabel('Scenario ID')
        plt.ylim(0, 1)
        plt.tight_layout()
        plt.savefig(self.output_dir / "recommendation_consistency.png")
        plt.close()
        
        # Plot by vendor
        consistency_melted_vendor = pd.melt(
            consistency_df,
            id_vars=['vendor', 'case_id'],
            value_vars=['rec_consistent', 'alt_consistent', 'least_consistent', 'all_consistent'],
            var_name='recommendation_type',
            value_name='is_consistent'
        )
        
        plt.figure(figsize=(12, 6))
        sns.barplot(x='recommendation_type', y='is_consistent', hue='vendor', data=consistency_melted_vendor)
        plt.title('Recommendation Consistency by Vendor')
        plt.ylabel('Consistency Rate')
        plt.xlabel('Recommendation Type')
        plt.ylim(0, 1)
        plt.tight_layout()
        plt.savefig(self.output_dir / "recommendation_consistency_by_vendor.png")
        plt.close()
        
        # Plot by model
        consistency_melted_model = pd.melt(
            consistency_df,
            id_vars=['vendor', 'model', 'case_id'],
            value_vars=['rec_consistent', 'alt_consistent', 'least_consistent', 'all_consistent'],
            var_name='recommendation_type',
            value_name='is_consistent'
        )
        
        # Aggregate by model and recommendation type
        model_consistency_avg = consistency_melted_model.groupby(['vendor', 'model', 'recommendation_type'])['is_consistent'].mean().reset_index()
        
        plt.figure(figsize=(16, 10))
        g = sns.catplot(
            data=model_consistency_avg,
            kind="bar",
            x="recommendation_type", y="is_consistent", hue="model", col="vendor",
            height=6, aspect=1.5
        )
        g.set_axis_labels("Recommendation Type", "Consistency Rate")
        g.set_titles("Vendor: {col_name}")
        plt.tight_layout()
        plt.savefig(self.output_dir / "recommendation_consistency_by_model.png")
        plt.close()
        
        # Save stats
        consistency_df.to_csv(self.output_dir / "recommendation_consistency.csv")
        vendor_consistency.to_csv(self.output_dir / "vendor_consistency_summary.csv")
        model_consistency.to_csv(self.output_dir / "model_consistency_summary.csv")
        case_consistency.to_csv(self.output_dir / "case_consistency_summary.csv")  # New
        
        return {
            'vendor_consistency': vendor_consistency,
            'model_consistency': model_consistency,
            'case_consistency': case_consistency,  # New
            'consistency_df': consistency_df
        }

    def analyze_by_scenario(self):
        """Analyze responses by scenario filename"""
        if 'scenario_filename' not in self.df.columns:
            logger.warning("scenario_filename column not found in database. Skipping scenario analysis.")
            return None
            
        # Basic stats by scenario
        scenario_counts = self.df['scenario_filename'].value_counts()
        logger.info(f"Response counts by scenario filename:\n{scenario_counts}")
        
        # Add this analysis to visualizations
        plt.figure(figsize=(12, 6))
        sns.countplot(y='scenario_filename', data=self.df, order=scenario_counts.index)
        plt.title('Response Counts by Scenario Filename')
        plt.ylabel('Scenario Filename')
        plt.xlabel('Count')
        plt.tight_layout()
        plt.savefig(self.output_dir / "responses_by_scenario.png")
        plt.close()
        
        # Response length by scenario
        scenario_length_stats = self.df.groupby('scenario_filename')['response_length'].agg(['mean', 'std', 'min', 'max'])
        logger.info(f"Response length statistics by scenario:\n{scenario_length_stats}")
        
        # Save stats
        scenario_length_stats.to_csv(self.output_dir / "scenario_response_length_stats.csv")
        scenario_counts.to_csv(self.output_dir / "scenario_counts.csv")
        
        # Generate HTML snippet for the report
        html_snippet = f"""
            <div class="section">
                <h2>Scenario Analysis</h2>
                
                <h3>Response Counts by Scenario</h3>
                {scenario_counts.to_frame().to_html()}
                
                <h3>Response Length by Scenario</h3>
                {scenario_length_stats.to_html()}
                
                <h3>Scenario Visualizations</h3>
                <img src="responses_by_scenario.png" alt="Responses by Scenario">
            </div>
        """
        
        return {
            'scenario_counts': scenario_counts,
            'scenario_length_stats': scenario_length_stats,
            'html_snippet': html_snippet
        }

    def generate_comprehensive_report(self):
        """Generate a comprehensive analysis report"""
        # Run all analyses
        basic_stats = self.generate_basic_stats()
        self.plot_processing_times()
        vendor_length_stats, model_length_stats = self.analyze_response_length()
        ethical_stats = self.analyze_ethical_principles()
        recommendation_stats = self.analyze_recommendation_consistency()
        
        # Run scenario analysis if the column exists
        scenario_analysis = self.analyze_by_scenario()
        
        # Prepare recommendation stats HTML
        if recommendation_stats is not None:
            recommendation_stats_html = f"""
                <h3>By Vendor</h3>
                {recommendation_stats['vendor_consistency'].to_html()}
                <img src="recommendation_consistency_by_vendor.png" alt="Recommendation Consistency by Vendor">
                
                <h3>By Model</h3>
                {recommendation_stats['model_consistency'].to_html()}
                <img src="recommendation_consistency_by_model.png" alt="Recommendation Consistency by Model">
            """
        else:
            recommendation_stats_html = """
                <p>No consistency analysis available. This analysis requires multiple iterations of the same model for each case.</p>
            """
        
        # List of unique cases
        cases = sorted(self.df['case_id'].unique())
        
        # List of unique vendors and models
        vendors = sorted(self.df['vendor'].unique())
        models = [f"{vendor} {model}" for vendor, model in 
                  self.df[['vendor', 'model']].drop_duplicates().values]
        
        # Create the HTML report
        html_report = f"""<html>
        <head>
            <title>AI Ethics Research Analysis Report</title>
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
            <h1>AI Ethics Research Analysis Report</h1>
            <p>Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <div class="section">
                <h2>Overview</h2>
                <p>Total responses analyzed: {len(self.df)}</p>
                <p>Vendors included: {', '.join(vendors)}</p>
                <p>Models included: {', '.join(models)}</p>
                <p>Cases analyzed: {', '.join(cases)}</p>
            </div>
            
            <div class="section">
                <h2>Basic Statistics</h2>
                
                <h3>Response Counts by Vendor</h3>
                {basic_stats['vendor_counts'].to_frame().to_html()}
                
                <h3>Response Counts by Model</h3>
                {basic_stats['model_counts'].to_html()}
                
                <h3>Processing Time Statistics by Vendor (seconds)</h3>
                {basic_stats['vendor_time_stats'].to_html()}
                
                <h3>Processing Time Statistics by Model (seconds)</h3>
                {basic_stats['model_time_stats'].to_html()}
                
                <h3>Processing Time Visualizations</h3>
                <img src="processing_times_by_vendor.png" alt="Processing Times by Vendor">
                <img src="processing_times_by_model.png" alt="Processing Times by Model">
                <img src="avg_processing_times.png" alt="Average Processing Times">
            </div>
            
            <div class="section">
                <h2>Response Length Analysis</h2>
                
                <h3>Response Length Statistics by Vendor</h3>
                {vendor_length_stats.to_html()}
                
                <h3>Response Length Statistics by Model</h3>
                {model_length_stats.to_html()}
                
                <h3>Response Length Visualizations</h3>
                <img src="response_lengths_by_vendor.png" alt="Response Lengths by Vendor">
                <img src="response_lengths_by_model.png" alt="Response Lengths by Model">
                <img src="avg_response_lengths.png" alt="Average Response Lengths">
            </div>
            
            {scenario_analysis['html_snippet'] if scenario_analysis else ""}
            
            <div class="section">
                <h2>Ethical Principles Analysis</h2>
                <p>Average mentions of ethical principles per response:</p>
                
                <h3>By Vendor</h3>
                <img src="ethical_principle_mentions_by_vendor.png" alt="Ethical Principle Mentions by Vendor">
                
                <h3>By Model</h3>
                <img src="ethical_principle_mentions_by_model.png" alt="Ethical Principle Mentions by Model">
            </div>
            
            <div class="section">
                <h2>Recommendation Consistency Analysis</h2>
                <p>Consistency of recommendations across iterations for the same case:</p>
                
                {recommendation_stats_html}
            </div>
            
        </body>
        </html>"""
        
        # Write the HTML report to a file
        with open(self.output_dir / "analysis_report.html", "w") as f:
            f.write(html_report)
            
        logger.info(f"Comprehensive analysis report saved to {self.output_dir / 'analysis_report.html'}")
        
        return {
            'basic_stats': basic_stats,
            'vendor_length_stats': vendor_length_stats,
            'model_length_stats': model_length_stats,
            'ethical_stats': ethical_stats,
            'recommendation_stats': recommendation_stats,
            'scenario_analysis': scenario_analysis
        }
def main():
    parser = argparse.ArgumentParser(description='Analyze AI ethics responses')
    parser.add_argument('--base-path', type=str, default=None,
                        help='Base path for the project')
    args = parser.parse_args()
    
    try:
        analyzer = ResultsAnalyzer(base_path=args.base_path)
        analyzer.generate_comprehensive_report()
        logger.info("Analysis completed successfully")
    except Exception as e:
        logger.error(f"Error in analysis: {e}")
        raise


if __name__ == "__main__":
    main()
