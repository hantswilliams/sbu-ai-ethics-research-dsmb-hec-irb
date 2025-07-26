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
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
import argparse
from dotenv import load_dotenv
from scipy import stats
from scipy.stats import f_oneway, pearsonr
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from sklearn.decomposition import PCA

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

    def run_anova_and_tukey_tests(self):
        """Perform ANOVA and Tukey HSD tests for between-model comparisons"""
        if self.df.empty:
            logger.warning("No data available for ANOVA and Tukey tests")
            return None
            
        metrics = ['relevance_score', 'correctness_score', 'fluency_score', 'coherence_score', 'overall_score']
        titles = ['Relevance', 'Correctness', 'Fluency', 'Coherence', 'Overall']
        
        anova_results = {}
        tukey_results = {}
        
        for metric, title in zip(metrics, titles):
            # Group data by vendor
            groups = [self.df[self.df['vendor'] == vendor][metric].values for vendor in self.df['vendor'].unique()]
            
            # Perform one-way ANOVA
            f_stat, p_value = f_oneway(*groups)
            anova_results[metric] = {
                'F-statistic': f_stat,
                'p-value': p_value,
                'significant': p_value < 0.05
            }
            
            # Perform Tukey HSD post-hoc test if ANOVA is significant
            if p_value < 0.05:
                # Create arrays for Tukey test
                values = self.df[metric].values
                labels = self.df['vendor'].values
                
                # Run Tukey HSD
                tukey = pairwise_tukeyhsd(values, labels, alpha=0.05)
                tukey_results[metric] = tukey
                
                # Log the results
                logger.info(f"ANOVA for {title}: F={f_stat:.4f}, p={p_value:.4f}")
                logger.info(f"Tukey HSD for {title}:\n{tukey}")
            else:
                logger.info(f"ANOVA for {title}: F={f_stat:.4f}, p={p_value:.4f} (not significant)")
        
        # Group data by model (more granular than vendor)
        self.df['model_identifier'] = self.df['vendor'] + ' ' + self.df['model']
        model_anova_results = {}
        model_tukey_results = {}
        
        for metric, title in zip(metrics, titles):
            # Group data by model
            groups = [self.df[self.df['model_identifier'] == model][metric].values 
                     for model in self.df['model_identifier'].unique()]
            
            # Perform one-way ANOVA
            f_stat, p_value = f_oneway(*groups)
            model_anova_results[metric] = {
                'F-statistic': f_stat,
                'p-value': p_value,
                'significant': p_value < 0.05
            }
            
            # Perform Tukey HSD post-hoc test if ANOVA is significant
            if p_value < 0.05:
                # Create arrays for Tukey test
                values = self.df[metric].values
                labels = self.df['model_identifier'].values
                
                # Run Tukey HSD
                tukey = pairwise_tukeyhsd(values, labels, alpha=0.05)
                model_tukey_results[metric] = tukey
                
                # Log the results
                logger.info(f"Model-level ANOVA for {title}: F={f_stat:.4f}, p={p_value:.4f}")
                logger.info(f"Model-level Tukey HSD for {title}:\n{tukey}")
            else:
                logger.info(f"Model-level ANOVA for {title}: F={f_stat:.4f}, p={p_value:.4f} (not significant)")
        
        # Save results to CSV
        anova_df = pd.DataFrame(anova_results).T
        anova_df.to_csv(self.output_dir / "anova_results.csv")
        
        model_anova_df = pd.DataFrame(model_anova_results).T
        model_anova_df.to_csv(self.output_dir / "model_anova_results.csv")
        
        # Create a summary of significant differences
        significant_pairs = []
        for metric, tukey in model_tukey_results.items():
            reject = tukey.reject
            
            # Handle the data structure safely
            if hasattr(tukey, 'data'):
                for i in range(len(reject)):
                    if reject[i]:
                        # Check if this is a proper data structure
                        if isinstance(tukey.data, list) and len(tukey.data) >= 2:
                            group1 = tukey.data[0]
                            group2 = tukey.data[1]
                            
                            # Make sure we can access these as iterables
                            if hasattr(group1, '__iter__') and hasattr(group2, '__iter__'):
                                g1 = group1[i] if i < len(group1) else "Unknown"
                                g2 = group2[i] if i < len(group2) else "Unknown"
                                
                                significant_pairs.append({
                                    'Metric': metric,
                                    'Group1': g1,
                                    'Group2': g2,
                                    'Mean Difference': tukey.meandiffs[i],
                                    'p-value': tukey.pvalues[i],
                                    'Lower CI': tukey.confint[i][0],
                                    'Upper CI': tukey.confint[i][1]
                                })
                        else:
                            # If data structure is unexpected, add a basic entry
                            significant_pairs.append({
                                'Metric': metric,
                                'Group1': "Group pair " + str(i),
                                'Group2': "Unknown",
                                'Mean Difference': tukey.meandiffs[i] if hasattr(tukey, 'meandiffs') else 0,
                                'p-value': tukey.pvalues[i] if hasattr(tukey, 'pvalues') else 0,
                                'Lower CI': tukey.confint[i][0] if hasattr(tukey, 'confint') else 0,
                                'Upper CI': tukey.confint[i][1] if hasattr(tukey, 'confint') else 0
                            })
        
        if significant_pairs:
            sig_df = pd.DataFrame(significant_pairs)
            sig_df.to_csv(self.output_dir / "significant_model_differences.csv")
        
        return {
            'vendor_anova': anova_results,
            'vendor_tukey': tukey_results,
            'model_anova': model_anova_results,
            'model_tukey': model_tukey_results,
            'significant_pairs': significant_pairs if significant_pairs else None
        }
    
    def calculate_dimension_correlations(self):
        """Calculate Pearson correlation coefficients between evaluation dimensions"""
        if self.df.empty:
            logger.warning("No data available for correlation analysis")
            return None
            
        # Define the metrics
        metrics = ['relevance_score', 'correctness_score', 'fluency_score', 'coherence_score', 'overall_score']
        
        # Calculate correlation matrix
        corr_matrix = self.df[metrics].corr(method='pearson')
        logger.info(f"Pearson correlation matrix:\n{corr_matrix}")
        
        # Calculate p-values for correlations
        p_values = pd.DataFrame(np.zeros_like(corr_matrix), 
                               index=corr_matrix.index, 
                               columns=corr_matrix.columns)
        
        for i, col_i in enumerate(metrics):
            for j, col_j in enumerate(metrics):
                if i != j:  # Skip diagonal (self-correlations)
                    corr, p = pearsonr(self.df[col_i], self.df[col_j])
                    p_values.loc[col_i, col_j] = p
        
        logger.info(f"Correlation p-values:\n{p_values}")
        
        # Create a heatmap of correlations
        plt.figure(figsize=(10, 8))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, 
                   mask=mask, square=True, fmt='.2f')
        plt.title('Pearson Correlation Between Evaluation Dimensions')
        plt.tight_layout()
        plt.savefig(self.output_dir / "dimension_correlations.png")
        plt.close()
        
        # Save results to CSV
        corr_matrix.to_csv(self.output_dir / "dimension_correlations.csv")
        p_values.to_csv(self.output_dir / "correlation_p_values.csv")
        
        return {
            'correlation_matrix': corr_matrix,
            'p_values': p_values
        }
    
    def run_principal_component_analysis(self):
        """Perform principal component analysis on evaluation dimensions"""
        if self.df.empty:
            logger.warning("No data available for PCA")
            return None
            
        # Define the metrics
        metrics = ['relevance_score', 'correctness_score', 'fluency_score', 'coherence_score']
        
        # Extract and standardize the data
        X = self.df[metrics].values
        X_std = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
        
        # Run PCA
        pca = PCA()
        principal_components = pca.fit_transform(X_std)
        
        # Get explained variance ratio
        explained_variance = pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance)
        
        logger.info(f"PCA explained variance ratio: {explained_variance}")
        logger.info(f"PCA cumulative variance: {cumulative_variance}")
        
        # Plot explained variance
        plt.figure(figsize=(10, 6))
        plt.bar(range(1, len(explained_variance) + 1), explained_variance, alpha=0.8, 
               label='Individual explained variance')
        plt.step(range(1, len(cumulative_variance) + 1), cumulative_variance, where='mid',
                label='Cumulative explained variance')
        plt.axhline(y=0.95, color='r', linestyle='--', label='95% explained variance threshold')
        plt.xlabel('Principal Components')
        plt.ylabel('Explained Variance Ratio')
        plt.title('PCA: Explained Variance by Components')
        plt.legend(loc='best')
        plt.tight_layout()
        plt.savefig(self.output_dir / "pca_explained_variance.png")
        plt.close()
        
        # Plot first two principal components
        plt.figure(figsize=(12, 10))
        plt.scatter(principal_components[:, 0], principal_components[:, 1],
                   c=self.df['overall_score'], cmap='viridis', alpha=0.8, s=50)
        plt.colorbar(label='Overall Score')
        
        # Add vendor information as labels
        for i, model in enumerate(self.df['vendor'].unique()):
            model_mask = self.df['vendor'] == model
            centroid_x = np.mean(principal_components[model_mask, 0])
            centroid_y = np.mean(principal_components[model_mask, 1])
            plt.annotate(model, (centroid_x, centroid_y), 
                        fontsize=12, fontweight='bold',
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
        
        # Plot component loadings (feature vectors)
        feature_vectors = pca.components_.T
        for i, feature in enumerate(metrics):
            plt.arrow(0, 0, feature_vectors[i, 0] * 5, feature_vectors[i, 1] * 5, 
                     head_width=0.2, head_length=0.2, fc='red', ec='red')
            plt.text(feature_vectors[i, 0] * 5.2, feature_vectors[i, 1] * 5.2, feature, 
                    color='red', fontsize=12)
        
        plt.xlabel(f'Principal Component 1 ({explained_variance[0]:.2%} variance)')
        plt.ylabel(f'Principal Component 2 ({explained_variance[1]:.2%} variance)')
        plt.title('PCA: Evaluation Dimensions')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(self.output_dir / "pca_components.png")
        plt.close()
        
        # Save PCA results
        pca_results = pd.DataFrame(data=principal_components[:, :2],
                                  columns=['PC1', 'PC2'])
        pca_results['vendor'] = self.df['vendor'].values
        pca_results['model'] = self.df['model'].values
        pca_results['overall_score'] = self.df['overall_score'].values
        
        pca_results.to_csv(self.output_dir / "pca_results.csv")
        
        # Save component loadings
        loadings = pd.DataFrame(
            pca.components_.T,
            columns=[f'PC{i+1}' for i in range(pca.components_.shape[0])],
            index=metrics
        )
        loadings.to_csv(self.output_dir / "pca_loadings.csv")
        
        return {
            'explained_variance': explained_variance,
            'cumulative_variance': cumulative_variance,
            'loadings': loadings,
            'pca_results': pca_results
        }
    
    def calculate_confidence_intervals(self):
        """Calculate confidence intervals for mean scores"""
        if self.df.empty:
            logger.warning("No data available for confidence interval calculation")
            return None
            
        metrics = ['relevance_score', 'correctness_score', 'fluency_score', 'coherence_score', 'overall_score']
        titles = ['Relevance', 'Correctness', 'Fluency', 'Coherence', 'Overall']
        
        # Calculate vendor-level confidence intervals
        vendor_ci = {}
        for vendor in self.df['vendor'].unique():
            vendor_data = self.df[self.df['vendor'] == vendor]
            vendor_ci[vendor] = {}
            
            for metric, title in zip(metrics, titles):
                # Get the mean
                mean = vendor_data[metric].mean()
                # Get the standard error
                std_err = vendor_data[metric].sem()
                # Calculate 95% confidence interval
                ci_95 = stats.t.interval(0.95, vendor_data[metric].count() - 1, 
                                        loc=mean, scale=std_err)
                
                vendor_ci[vendor][metric] = {
                    'mean': mean,
                    'std_err': std_err,
                    'ci_lower': ci_95[0],
                    'ci_upper': ci_95[1],
                    'count': vendor_data[metric].count()
                }
        
        # Calculate model-level confidence intervals
        self.df['model_identifier'] = self.df['vendor'] + ' ' + self.df['model']
        model_ci = {}
        
        for model in self.df['model_identifier'].unique():
            model_data = self.df[self.df['model_identifier'] == model]
            model_ci[model] = {}
            
            for metric, title in zip(metrics, titles):
                # Get the mean
                mean = model_data[metric].mean()
                # Get the standard error
                std_err = model_data[metric].sem()
                # Calculate 95% confidence interval
                ci_95 = stats.t.interval(0.95, model_data[metric].count() - 1, 
                                       loc=mean, scale=std_err)
                
                model_ci[model][metric] = {
                    'mean': mean,
                    'std_err': std_err,
                    'ci_lower': ci_95[0],
                    'ci_upper': ci_95[1],
                    'count': model_data[metric].count()
                }
        
        # Convert to DataFrames for easier handling
        vendor_ci_df = pd.DataFrame.from_dict({(vendor, metric): values 
                                             for vendor, metrics in vendor_ci.items() 
                                             for metric, values in metrics.items()}, 
                                             orient='index')
        
        model_ci_df = pd.DataFrame.from_dict({(model, metric): values 
                                            for model, metrics in model_ci.items() 
                                            for metric, values in metrics.items()}, 
                                            orient='index')
        
        # Save to CSV
        vendor_ci_df.to_csv(self.output_dir / "vendor_confidence_intervals.csv")
        model_ci_df.to_csv(self.output_dir / "model_confidence_intervals.csv")
        
        # Create plots with confidence intervals
        # Plot for vendors
        for metric, title in zip(metrics, titles):
            plt.figure(figsize=(12, 6))
            
            # Extract data for this metric
            plot_data = []
            for vendor in self.df['vendor'].unique():
                plot_data.append({
                    'vendor': vendor,
                    'mean': vendor_ci[vendor][metric]['mean'],
                    'ci_lower': vendor_ci[vendor][metric]['ci_lower'],
                    'ci_upper': vendor_ci[vendor][metric]['ci_upper']
                })
            
            plot_df = pd.DataFrame(plot_data)
            
            # Plot with error bars
            plt.errorbar(plot_df['vendor'], plot_df['mean'], 
                        yerr=[plot_df['mean'] - plot_df['ci_lower'], 
                              plot_df['ci_upper'] - plot_df['mean']],
                        fmt='o', capsize=5, capthick=2, elinewidth=2, markersize=8)
            
            plt.title(f'{title} Scores by Vendor with 95% CI')
            plt.xlabel('Vendor')
            plt.ylabel(f'{title} Score (1-5)')
            plt.ylim(1, 5)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig(self.output_dir / f"vendor_{metric}_with_ci.png")
            plt.close()
        
        # Plot for overall score with all vendors
        plt.figure(figsize=(12, 6))
        plot_data = []
        for vendor in self.df['vendor'].unique():
            plot_data.append({
                'vendor': vendor,
                'mean': vendor_ci[vendor]['overall_score']['mean'],
                'ci_lower': vendor_ci[vendor]['overall_score']['ci_lower'],
                'ci_upper': vendor_ci[vendor]['overall_score']['ci_upper']
            })
        
        plot_df = pd.DataFrame(plot_data)
        
        # Sort by mean score for better visualization
        plot_df = plot_df.sort_values('mean', ascending=False)
        
        # Plot with error bars
        plt.errorbar(plot_df['vendor'], plot_df['mean'], 
                   yerr=[plot_df['mean'] - plot_df['ci_lower'], 
                         plot_df['ci_upper'] - plot_df['mean']],
                   fmt='o', capsize=5, capthick=2, elinewidth=2, markersize=10)
        
        plt.title('Overall Scores by Vendor with 95% Confidence Intervals')
        plt.xlabel('Vendor')
        plt.ylabel('Overall Score (1-5)')
        plt.ylim(1, 5)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(self.output_dir / "vendor_overall_with_ci.png")
        plt.close()
        
        return {
            'vendor_ci': vendor_ci,
            'model_ci': model_ci
        }
        
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
        
        # Run the new statistical analyses
        anova_results = self.run_anova_and_tukey_tests()
        correlation_results = self.calculate_dimension_correlations()
        pca_results = self.run_principal_component_analysis()
        ci_results = self.calculate_confidence_intervals()
        
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
                
                <h3>Statistical Significance (ANOVA and Tukey HSD)</h3>
                <p>One-way ANOVA tests were conducted to determine if there are statistically significant 
                differences between vendors across evaluation dimensions:</p>
                <table>
                    <tr>
                        <th>Dimension</th>
                        <th>F-statistic</th>
                        <th>p-value</th>
                        <th>Significant</th>
                    </tr>
                    {"".join([f"<tr><td>{metric.replace('_score', '').capitalize()}</td><td>{results['F-statistic']:.4f}</td><td>{results['p-value']:.4f}</td><td>{'Yes' if results['significant'] else 'No'}</td></tr>" for metric, results in anova_results['vendor_anova'].items()])}
                </table>
                
                <p>Where significant differences were found, Tukey HSD post-hoc tests were conducted to 
                identify which specific vendor pairs differed significantly.</p>
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
                
                <h3>Statistical Significance (ANOVA and Tukey HSD)</h3>
                <p>One-way ANOVA tests were conducted to determine if there are statistically significant 
                differences between models across evaluation dimensions:</p>
                <table>
                    <tr>
                        <th>Dimension</th>
                        <th>F-statistic</th>
                        <th>p-value</th>
                        <th>Significant</th>
                    </tr>
                    {"".join([f"<tr><td>{metric.replace('_score', '').capitalize()}</td><td>{results['F-statistic']:.4f}</td><td>{results['p-value']:.4f}</td><td>{'Yes' if results['significant'] else 'No'}</td></tr>" for metric, results in anova_results['model_anova'].items()])}
                </table>
                
                <p>Where significant differences were found, Tukey HSD post-hoc tests were conducted to 
                identify which specific model pairs differed significantly.</p>
            </div>
            
            <div class="section">
                <h2>Confidence Intervals</h2>
                <p>95% confidence intervals for mean scores by vendor:</p>
                <img src="vendor_overall_with_ci.png" alt="Vendor Overall Scores with Confidence Intervals">
                
                <p>These confidence intervals provide a range within which we can be 95% confident that the true mean score lies.</p>
            </div>
            
            <div class="section">
                <h2>Dimension Correlations</h2>
                <p>Pearson correlation analysis between evaluation dimensions:</p>
                <img src="dimension_correlations.png" alt="Dimension Correlations Heatmap">
                
                <p>This correlation matrix shows how strongly the different evaluation dimensions 
                are related to each other. Higher values (closer to 1) indicate stronger positive correlations.</p>
            </div>
            
            <div class="section">
                <h2>Principal Component Analysis</h2>
                <p>PCA was performed to identify underlying patterns in evaluator ratings:</p>
                <img src="pca_explained_variance.png" alt="PCA Explained Variance">
                <img src="pca_components.png" alt="PCA Components">
                
                <p>The PCA plot shows how the evaluation dimensions relate to each other in a reduced 
                dimensional space. Points represent individual evaluations, and their colors indicate 
                the overall score. The arrows show how the original dimensions contribute to the principal components.</p>
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
        
        # Run the basic analyses first
        analyzer.generate_basic_stats()
        analyzer.plot_score_distributions()
        analyzer.plot_vendor_comparisons()
        analyzer.plot_model_comparisons()
        analyzer.analyze_by_iteration()
        analyzer.analyze_by_scenario()
        
        # Run the new statistical analyses
        analyzer.run_anova_and_tukey_tests()
        analyzer.calculate_dimension_correlations()
        analyzer.run_principal_component_analysis()
        analyzer.calculate_confidence_intervals()
        
        # Generate the comprehensive report
        analyzer.generate_comprehensive_report()
        
        logger.info("Evaluation analysis completed successfully")
    except Exception as e:
        logger.error(f"Error in analysis: {e}")
        raise


if __name__ == "__main__":
    main()
