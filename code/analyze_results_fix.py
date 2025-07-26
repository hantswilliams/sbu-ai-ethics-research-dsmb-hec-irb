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
