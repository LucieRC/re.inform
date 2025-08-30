#!/usr/bin/env python3
"""
EVS (European Values Study) Data Analysis Tool
This script provides functionality to analyze EVS survey data with:
- Data loading from various formats
- Data cleaning and preprocessing
- Filtering for French data only (cntry=FR)
- Exploratory data analysis
- Statistical analysis
- Data visualization
- Variable documentation integration
- Thematic analysis based on variable categories
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
import pyreadstat  # For reading SPSS files which are common for survey data
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import statsmodels.api as sm
import argparse
import json
import re
from matplotlib.colors import LinearSegmentedColormap
# from wordcloud import WordCloud  # You may need to install this: pip install wordcloud


def load_data(file_path):
    """Load EVS data from various formats (CSV, SPSS, Stata)"""
    if file_path.endswith('.csv'):
        data = pd.read_csv(file_path)
    elif file_path.endswith('.sav'):  # SPSS format
        data, meta = pyreadstat.read_sav(file_path)
    elif file_path.endswith('.dta'):  # Stata format
        data, meta = pyreadstat.read_dta(file_path)  # Changed from read_stata to read_dta
    else:
        raise ValueError("Unsupported file format. Please provide CSV, SPSS, or Stata file.")
    
    print(f"Data loaded with {data.shape[0]} rows and {data.shape[1]} columns.")
    
    # Filter to keep only French data
    if 'cntry' in data.columns:
        # Store original count for comparison
        original_count = len(data)
        
        # Filter for French data only (France ISO code is 250)
        data = data[data['cntry'] == 250]
        
        # Report the filtering
        print(f"Filtered data to include only French responses (cntry=250).")
        print(f"Retained {len(data)} rows out of {original_count} ({len(data)/original_count*100:.1f}%).")
    else:
        print("Warning: 'cntry' column not found. Could not filter for French data.")
    
    return data


def clean_data(data):
    """Basic data cleaning for EVS dataset"""
    # Check for missing values
    missing_values = data.isnull().sum()
    print(f"Missing values per column:\n{missing_values[missing_values > 0]}")
    
    # Remove rows with all missing values
    data = data.dropna(how='all')
    
    # For demonstration, replace missing values in numeric columns with mean
    numeric_columns = data.select_dtypes(include=[np.number]).columns
    for col in numeric_columns:
        data[col] = data[col].fillna(data[col].mean())
    
    # Convert categorical variables to category type
    categorical_columns = data.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        data[col] = data[col].astype('category')
    
    print(f"Data cleaned. New shape: {data.shape}")
    return data


def identify_metadata_columns(data):
    """Identify metadata columns that appear before 'cntry'"""
    if 'cntry' in data.columns:
        cntry_index = data.columns.get_loc('cntry')
        metadata_cols = list(data.columns[:cntry_index])
        print(f"Identified {len(metadata_cols)} metadata columns before 'cntry':")
        print(", ".join(metadata_cols))
        return metadata_cols
    else:
        print("Warning: 'cntry' column not found. Cannot identify metadata columns.")
        return []


def load_variable_documentation(doc_path=None):
    """Load variable documentation from a CSV or JSON file"""
    var_docs = {}
    
    if doc_path is None:
        print("No variable documentation provided. Proceeding without variable descriptions.")
        return var_docs
        
    try:
        if doc_path.endswith('.csv'):
            docs_df = pd.read_csv(doc_path)
            # Assuming the CSV has 'variable' and 'description' columns
            var_docs = dict(zip(docs_df['variable'], docs_df['description']))
        elif doc_path.endswith('.json'):
            with open(doc_path, 'r', encoding='utf-8') as f:
                var_docs = json.load(f)
        else:
            print(f"Unsupported documentation format: {doc_path}. Proceeding without variable descriptions.")
            
        print(f"Loaded documentation for {len(var_docs)} variables.")
    except Exception as e:
        print(f"Failed to load variable documentation: {str(e)}")
    
    return var_docs


def categorize_variables(data, var_docs=None, metadata_cols=None):
    """Categorize variables based on their names or documentation"""
    categories = {
        'metadata': metadata_cols if metadata_cols else [],
        'demographics': [],
        'values': [],
        'beliefs': [],
        'social_attitudes': [],
        'political': [],
        'economic': [],
        'family': [],
        'environment': [],
        'religion': [],
        'other': []
    }
    
    # Keywords for categorization
    keywords = {
        'demographics': ['age', 'sex', 'gender', 'education', 'income', 'marital', 'employ', 'occupation', 'region', 'ethnic'],
        'values': ['value', 'important', 'priority', 'moral', 'ethic', 'principle'],
        'beliefs': ['believe', 'belief', 'think', 'opinion', 'view', 'perceive'],
        'social_attitudes': ['social', 'society', 'community', 'neighbor', 'tolerance', 'trust', 'diversity'],
        'political': ['politic', 'democra', 'govern', 'party', 'vote', 'election', 'policy'],
        'economic': ['econom', 'money', 'financ', 'job', 'work', 'income', 'business'],
        'family': ['family', 'child', 'parent', 'marriage', 'relationship', 'partner'],
        'environment': ['environment', 'climate', 'nature', 'ecolog', 'sustain', 'green'],
        'religion': ['relig', 'church', 'god', 'faith', 'spirit', 'pray', 'worship']
    }
    
    # Skip metadata columns if they're already categorized
    columns_to_categorize = [col for col in data.columns if col not in categories['metadata']]
    
    # Loop through remaining columns
    for col in columns_to_categorize:
        assigned = False
        col_lower = col.lower()
        
        # First check documentation if available
        if var_docs and col in var_docs:
            desc = var_docs[col].lower()
            for category, terms in keywords.items():
                if any(term in desc for term in terms):
                    categories[category].append(col)
                    assigned = True
                    break
        
        # If not assigned by documentation, try using the variable name
        if not assigned:
            for category, terms in keywords.items():
                if any(term in col_lower for term in terms):
                    categories[category].append(col)
                    assigned = True
                    break
        
        # Default category
        if not assigned:
            categories['other'].append(col)
    
    # Print categorization summary
    print("\nVariable categorization summary:")
    for category, vars_list in categories.items():
        if vars_list:
            print(f"  {category}: {len(vars_list)} variables")
    
    return categories


def exploratory_analysis(data):
    """Perform exploratory data analysis"""
    # Basic statistics for numeric columns
    numeric_summary = data.describe()
    print("\nNumeric Variables Summary:")
    print(numeric_summary)
    
    # Frequency tables for categorical columns
    categorical_columns = data.select_dtypes(include=['category']).columns
    for col in categorical_columns[:5]:  # Show only first 5 for brevity
        print(f"\nFrequency table for {col}:")
        print(data[col].value_counts(normalize=True) * 100)  # Show as percentages
    
    return numeric_summary


def enhanced_exploratory_analysis(data, var_docs=None, categories=None):
    """Perform enhanced exploratory analysis with variable documentation"""
    results = {}
    
    # Run basic exploratory analysis
    numeric_summary = exploratory_analysis(data)
    results['numeric_summary'] = numeric_summary
    
    # Since we're focusing on French data, add a note about that
    print("\nAnalysis of French EVS Data (cntry=FR)")
    print("-" * 40)
    
    # Analyze key demographics if available
    demographic_vars = []
    if categories and 'demographics' in categories:
        demographic_vars = categories['demographics']
    
    if demographic_vars:
        print("\nKey Demographic Variables Summary (French respondents):")
        for var in demographic_vars[:5]:  # First 5 demographics
            if var in data.columns:
                if data[var].dtype.name in ['category', 'object']:
                    print(f"\n{var}:")
                    if var_docs and var in var_docs:
                        print(f"  Description: {var_docs[var]}")
                    freq = data[var].value_counts()
                    print(freq)
                else:
                    print(f"\n{var}:")
                    if var_docs and var in var_docs:
                        print(f"  Description: {var_docs[var]}")
                    print(data[var].describe())
    
    # Add France-specific insights if available
    region_vars = [col for col in data.columns if 'region' in col.lower()]
    if region_vars:
        print("\nFrench Regional Distribution:")
        for var in region_vars:
            counts = data[var].value_counts()
            print(f"\n{var}:")
            print(counts)
    
    # Look for wave/year variables to identify the time period
    year_vars = [col for col in data.columns if 'year' in col.lower() or 'wave' in col.lower()]
    
    if year_vars:
        print("\nTime Period Distribution:")
        for var in year_vars:
            counts = data[var].value_counts()
            print(f"\n{var}:")
            print(counts)
            results['years'] = counts.to_dict()
    
    return results


def visualize_data(data, output_dir='evs_plots'):
    """Create visualizations for EVS data"""
    # Set up the visualizations
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Create output directory for plots
    os.makedirs(output_dir, exist_ok=True)
    
    # Distribution of numeric variables
    numeric_columns = data.select_dtypes(include=[np.number]).columns[:5]  # First 5 columns
    
    if len(numeric_columns) > 0:
        fig, axes = plt.subplots(len(numeric_columns), 1, figsize=(10, 3*len(numeric_columns)))
        if len(numeric_columns) == 1:
            axes = [axes]  # Make it iterable when only one column
            
        for i, col in enumerate(numeric_columns):
            sns.histplot(data[col].dropna(), ax=axes[i])
            axes[i].set_title(f'Distribution of {col}')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/numeric_distributions.png')
        plt.close()
    
    # Bar plots for categorical variables
    categorical_columns = data.select_dtypes(include=['category']).columns[:5]  # First 5 columns
    
    for col in categorical_columns:
        plt.figure(figsize=(10, 6))
        counts = data[col].value_counts().sort_values(ascending=False)
        sns.barplot(x=counts.index.astype(str), y=counts.values)
        plt.title(f'Frequency of {col}')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/{col}_frequencies.png')
        plt.close()
    
    # Correlation heatmap
    numeric_data = data.select_dtypes(include=[np.number])
    if numeric_data.shape[1] > 1:  # Only if we have multiple numeric columns
        plt.figure(figsize=(12, 10))
        corr = numeric_data.corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
        plt.title('Correlation Matrix of Numeric Variables')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/correlation_heatmap.png')
        plt.close()
    
    print(f"Visualizations saved in '{output_dir}' directory.")


def statistical_analysis(data):
    """Perform basic statistical tests on the data"""
    results = {}
    
    # Select numeric and categorical columns
    numeric_columns = data.select_dtypes(include=[np.number]).columns
    categorical_columns = data.select_dtypes(include=['category']).columns
    
    # One-way ANOVA: Compare means of numeric variable across categories
    if len(numeric_columns) > 0 and len(categorical_columns) > 0:
        num_col = numeric_columns[0]
        cat_col = categorical_columns[0]
        
        categories = data[cat_col].dropna().unique()
        if len(categories) > 1:
            try:
                anova_data = [data[data[cat_col] == cat][num_col].dropna() for cat in categories]
                anova_data = [data_group for data_group in anova_data if len(data_group) > 0]
                
                if len(anova_data) > 1:  # Need at least 2 groups
                    f_stat, p_value = stats.f_oneway(*anova_data)
                    
                    results['anova'] = {
                        'variables': f"{num_col} by {cat_col}",
                        'f_statistic': f_stat,
                        'p_value': p_value,
                        'significant': p_value < 0.05
                    }
            except Exception as e:
                print(f"ANOVA failed: {str(e)}")
    
    # Chi-square test for two categorical variables
    if len(categorical_columns) >= 2:
        cat1 = categorical_columns[0]
        cat2 = categorical_columns[1]
        
        try:
            contingency_table = pd.crosstab(data[cat1], data[cat2])
            chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
            
            results['chi_square'] = {
                'variables': f"{cat1} vs {cat2}",
                'chi2_statistic': chi2,
                'p_value': p,
                'dof': dof,
                'significant': p < 0.05
            }
        except Exception as e:
            print(f"Chi-square test failed: {str(e)}")
    
    return results


def thematic_analysis(data, categories, var_docs=None, output_dir='evs_plots'):
    """Perform thematic analysis based on variable categories"""
    os.makedirs(output_dir, exist_ok=True)
    results = {}
    
    # Define key categories to analyze
    key_categories = ['values', 'beliefs', 'social_attitudes', 'political', 'religion']
    
    for category in key_categories:
        if category not in categories or not categories[category]:
            continue
        
        cat_vars = categories[category]
        print(f"\n--- {category.upper()} ANALYSIS (FRENCH DATA) ---")
        
        # Pick representative variables (numeric and categorical separately)
        num_vars = [var for var in cat_vars if var in data.columns and pd.api.types.is_numeric_dtype(data[var])]
        cat_vars = [var for var in cat_vars if var in data.columns and (data[var].dtype.name == 'category' or pd.api.types.is_object_dtype(data[var]))]
        
        # Analyze numeric variables in this category
        if num_vars:
            # Basic statistics
            cat_num_stats = data[num_vars].describe().transpose()
            print(f"\nNumeric {category} variables statistics:")
            print(cat_num_stats)
            results[f'{category}_numeric_stats'] = cat_num_stats.to_dict()
            
            # Correlation matrix for these variables
            if len(num_vars) > 1:
                corr = data[num_vars].corr()
                plt.figure(figsize=(10, 8))
                mask = np.triu(np.ones_like(corr, dtype=bool))
                cmap = sns.diverging_palette(230, 20, as_cmap=True)
                sns.heatmap(corr, mask=mask, cmap=cmap, annot=True, fmt=".2f", linewidths=0.5)
                plt.title(f'Correlation Matrix of {category.title()} Variables')
                plt.tight_layout()
                plt.savefig(f'{output_dir}/{category}_correlation.png')
                plt.close()
                print(f"Correlation matrix saved to {output_dir}/{category}_correlation.png")
        
        # Analyze categorical variables in this category
        if cat_vars and len(cat_vars) > 0:
            print(f"\nCategorical {category} variables:")
            # Take up to 3 categorical variables for visualization
            for var in cat_vars[:3]:
                if var_docs and var in var_docs:
                    print(f"\n{var}: {var_docs[var]}")
                else:
                    print(f"\n{var}")
                
                # Frequency table
                freq = data[var].value_counts().reset_index()
                freq.columns = [var, 'count']
                print(freq)
                
                # Bar plot
                plt.figure(figsize=(10, 6))
                ax = sns.barplot(x=var, y='count', data=freq)
                plt.title(f'Distribution of {var}')
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                plt.savefig(f'{output_dir}/{var}_distribution.png')
                plt.close()
                
                results[f'{var}_distribution'] = freq.set_index(var)['count'].to_dict()
    
    # If we have country and values variables, perform a cross-country comparison of values
    country_vars = [col for col in data.columns if 'country' in col.lower() or 'nation' in col.lower()]
    values_vars = categories.get('values', [])
    
    if country_vars and values_vars and len(values_vars) > 0:
        country_var = country_vars[0]  # Take the first country variable
        print("\n--- CROSS-COUNTRY VALUES COMPARISON ---")
        
        # Select a numeric value variable for cross-country comparison
        value_var = None
        for var in values_vars:
            if var in data.columns and pd.api.types.is_numeric_dtype(data[var]):
                value_var = var
                break
        
        if value_var:
            # Create a boxplot comparing the value across countries
            plt.figure(figsize=(12, 8))
            ax = sns.boxplot(x=country_var, y=value_var, data=data)
            plt.title(f'Cross-Country Comparison of {value_var}')
            plt.xticks(rotation=90)
            plt.tight_layout()
            plt.savefig(f'{output_dir}/cross_country_{value_var}.png')
            plt.close()
            print(f"Cross-country comparison saved to {output_dir}/cross_country_{value_var}.png")
            
            # ANOVA test to see if there are significant differences
            countries = data[country_var].unique()
            if len(countries) > 1:
                try:
                    groups = [data[data[country_var] == country][value_var].dropna() for country in countries]
                    groups = [g for g in groups if len(g) > 0]
                    
                    if len(groups) > 1:
                        f_stat, p_value = stats.f_oneway(*groups)
                        print(f"\nANOVA test for {value_var} across countries:")
                        print(f"F-statistic: {f_stat:.4f}")
                        print(f"p-value: {p_value:.4f}")
                        print(f"Significant differences: {p_value < 0.05}")
                        
                        results['cross_country_anova'] = {
                            'variable': value_var,
                            'f_statistic': float(f_stat),
                            'p_value': float(p_value),
                            'significant': p_value < 0.05
                        }
                except Exception as e:
                    print(f"Cross-country ANOVA failed: {str(e)}")
    
    return results


def create_wordcloud_from_text_responses(data, text_columns, output_dir='evs_plots'):
    """Create wordclouds from text responses if available"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if we have any potential text columns (more than 20 unique values)
    potential_text_cols = []
    
    if not text_columns:
        for col in data.columns:
            if data[col].dtype == 'object':
                unique_vals = data[col].nunique()
                if unique_vals > 20 and unique_vals < 1000:  # Arbitrary thresholds to identify text
                    potential_text_cols.append(col)
        
        text_columns = potential_text_cols
    
    if not text_columns:
        print("No suitable text columns found for wordcloud analysis")
        return
    
    print("\n--- TEXT ANALYSIS ---")
    for col in text_columns:
        try:
            # Combine all text into one string
            text = ' '.join(data[col].dropna().astype(str).tolist())
            
            if len(text) > 100:  # Only proceed if we have enough text
                # Create and save wordcloud
                wordcloud = WordCloud(width=800, height=400, background_color='white', 
                                     max_words=100, contour_width=3).generate(text)
                
                plt.figure(figsize=(10, 6))
                plt.imshow(wordcloud, interpolation='bilinear')
                plt.axis('off')
                plt.title(f'Word Cloud - {col}')
                plt.tight_layout()
                plt.savefig(f'{output_dir}/{col}_wordcloud.png')
                plt.close()
                
                print(f"Word cloud for {col} saved to {output_dir}/{col}_wordcloud.png")
        except Exception as e:
            print(f"Wordcloud generation for {col} failed: {str(e)}")


def demographic_insights(data, categories, output_dir='evs_plots'):
    """Generate insights based on demographic variables"""
    if 'demographics' not in categories or not categories['demographics']:
        print("No demographic variables identified for analysis")
        return {}
    
    demo_vars = categories['demographics']
    os.makedirs(output_dir, exist_ok=True)
    results = {}
    
    # Find key demographic variables
    age_var = next((var for var in demo_vars if 'age' in var.lower()), None)
    gender_var = next((var for var in demo_vars if any(term in var.lower() for term in ['sex', 'gender'])), None)
    edu_var = next((var for var in demo_vars if any(term in var.lower() for term in ['edu', 'school'])), None)
    
    # Analyze distribution of demographic variables
    if age_var and age_var in data.columns:
        try:
            plt.figure(figsize=(10, 6))
            sns.histplot(data[age_var].dropna(), kde=True)
            plt.title('Age Distribution')
            plt.xlabel('Age')
            plt.savefig(f'{output_dir}/age_distribution.png')
            plt.close()
            
            age_stats = data[age_var].describe()
            print("\nAge Statistics:")
            print(age_stats)
            results['age_stats'] = age_stats.to_dict()
        except Exception as e:
            print(f"Age analysis failed: {str(e)}")
    
    if gender_var and gender_var in data.columns:
        try:
            gender_counts = data[gender_var].value_counts()
            plt.figure(figsize=(8, 6))
            gender_counts.plot(kind='pie', autopct='%1.1f%%')
            plt.title('Gender Distribution')
            plt.savefig(f'{output_dir}/gender_distribution.png')
            plt.close()
            
            print("\nGender Distribution:")
            print(gender_counts)
            results['gender_distribution'] = gender_counts.to_dict()
        except Exception as e:
            print(f"Gender analysis failed: {str(e)}")
    
    # Cross-tabulation of demographics with key variables from other categories
    if (gender_var or age_var) and categories.get('values'):
        # Take first value variable for analysis
        value_vars = [var for var in categories['values'] if var in data.columns]
        if value_vars:
            value_var = value_vars[0]
            
            print(f"\nAnalyzing {value_var} by demographics:")
            
            if gender_var:
                try:
                    # Analysis by gender
                    plt.figure(figsize=(10, 6))
                    sns.boxplot(x=gender_var, y=value_var, data=data)
                    plt.title(f'{value_var} by Gender')
                    plt.savefig(f'{output_dir}/{value_var}_by_gender.png')
                    plt.close()
                    
                    # T-test if we have 2 gender categories
                    gender_categories = data[gender_var].unique()
                    if len(gender_categories) == 2:
                        group1 = data[data[gender_var] == gender_categories[0]][value_var].dropna()
                        group2 = data[data[gender_var] == gender_categories[1]][value_var].dropna()
                        t_stat, p_val = stats.ttest_ind(group1, group2)
                        
                        print(f"T-test for {value_var} between gender groups:")
                        print(f"t-statistic: {t_stat:.4f}")
                        print(f"p-value: {p_val:.4f}")
                        print(f"Significant difference: {p_val < 0.05}")
                        
                        results[f'{value_var}_by_gender'] = {
                            't_statistic': float(t_stat),
                            'p_value': float(p_val),
                            'significant': p_val < 0.05
                        }
                except Exception as e:
                    print(f"Gender-based analysis failed: {str(e)}")
            
            if age_var:
                try:
                    # Create age groups
                    data['age_group'] = pd.cut(data[age_var], bins=[0, 25, 35, 45, 55, 65, 100],
                                           labels=['18-24', '25-34', '35-44', '45-54', '55-64', '65+'])
                    
                    # Analysis by age group
                    plt.figure(figsize=(12, 6))
                    sns.boxplot(x='age_group', y=value_var, data=data)
                    plt.title(f'{value_var} by Age Group')
                    plt.savefig(f'{output_dir}/{value_var}_by_age.png')
                    plt.close()
                    
                    # ANOVA test for age groups
                    age_groups = data['age_group'].dropna().unique()
                    if len(age_groups) > 1:
                        groups = [data[data['age_group'] == group][value_var].dropna() for group in age_groups]
                        groups = [g for g in groups if len(g) > 0]
                        
                        if len(groups) > 1:
                            f_stat, p_value = stats.f_oneway(*groups)
                            print(f"\nANOVA test for {value_var} across age groups:")
                            print(f"F-statistic: {f_stat:.4f}")
                            print(f"p-value: {p_value:.4f}")
                            print(f"Significant differences: {p_value < 0.05}")
                            
                            results[f'{value_var}_by_age'] = {
                                'f_statistic': float(f_stat),
                                'p_value': float(p_value),
                                'significant': p_value < 0.05
                            }
                except Exception as e:
                    print(f"Age-based analysis failed: {str(e)}")
    
    return results


def french_specific_analysis(data, categories, var_docs=None, output_dir='evs_plots'):
    """Perform analysis specific to the French context"""
    os.makedirs(output_dir, exist_ok=True)
    results = {}
    
    print("\n--- FRENCH-SPECIFIC ANALYSIS ---")
    print("Analyzing patterns specific to the French EVS data")
    
    # Look for regional variables
    region_vars = [col for col in data.columns if 'region' in col.lower() or 'département' in col.lower()]
    
    # If we have regional data, analyze regional differences
    if region_vars and region_vars[0] in data.columns:
        region_var = region_vars[0]
        print(f"\nRegional analysis using variable: {region_var}")
        
        # Get distribution of regions
        region_counts = data[region_var].value_counts()
        
        # Visualize regional distribution
        plt.figure(figsize=(12, 8))
        sns.barplot(x=region_counts.index.astype(str), y=region_counts.values)
        plt.title('Distribution of French Regions in the Sample')
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/french_regions_distribution.png')
        plt.close()
        
        # If we have values variables, analyze regional differences in values
        if 'values' in categories and categories['values']:
            value_vars = [var for var in categories['values'] if var in data.columns 
                         and pd.api.types.is_numeric_dtype(data[var])]
            
            if value_vars:
                value_var = value_vars[0]
                print(f"\nAnalyzing regional differences in {value_var}")
                
                # ANOVA by region for this value
                regions = data[region_var].unique()
                if len(regions) > 1:
                    try:
                        plt.figure(figsize=(14, 8))
                        sns.boxplot(x=region_var, y=value_var, data=data)
                        plt.title(f'Regional Differences in {value_var} (France)')
                        plt.xticks(rotation=90)
                        plt.tight_layout()
                        plt.savefig(f'{output_dir}/france_regional_{value_var}.png')
                        plt.close()
                        
                        # Statistical test
                        groups = [data[data[region_var] == region][value_var].dropna() for region in regions]
                        groups = [g for g in groups if len(g) > 0]
                        
                        if len(groups) > 1:
                            f_stat, p_value = stats.f_oneway(*groups)
                            print(f"ANOVA test for {value_var} across French regions:")
                            print(f"F-statistic: {f_stat:.4f}")
                            print(f"p-value: {p_value:.4f}")
                            print(f"Significant differences: {p_value < 0.05}")
                    except Exception as e:
                        print(f"Regional analysis failed: {str(e)}")
    
    # Check for election/voting variables to analyze political attitudes
    political_vars = categories.get('political', [])
    voting_vars = [var for var in political_vars if 'vote' in var.lower() 
                  or 'election' in var.lower() or 'party' in var.lower()]
    
    if voting_vars:
        for var in voting_vars[:2]:  # Analyze up to 2 voting variables
            if var in data.columns:
                print(f"\nAnalyzing French political variable: {var}")
                if var_docs and var in var_docs:
                    print(f"Description: {var_docs[var]}")
                
                # Get distribution
                var_counts = data[var].value_counts()
                print(var_counts)
                
                # Visualize
                plt.figure(figsize=(12, 8))
                sns.barplot(x=var_counts.index.astype(str), y=var_counts.values)
                plt.title(f'Distribution of {var} (French Respondents)')
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                plt.savefig(f'{output_dir}/french_politics_{var}.png')
                plt.close()
    
    # Analyze French religiosity
    religion_vars = categories.get('religion', [])
    if religion_vars:
        print("\nAnalyzing French religiosity patterns")
        rel_vars = [var for var in religion_vars if var in data.columns][:3]  # Take up to 3 religion vars
        
        for var in rel_vars:
            if pd.api.types.is_numeric_dtype(data[var]):
                print(f"\n{var} statistics:")
                if var_docs and var in var_docs:
                    print(f"Description: {var_docs[var]}")
                print(data[var].describe())
            else:
                print(f"\n{var} distribution:")
                if var_docs and var in var_docs:
                    print(f"Description: {var_docs[var]}")
                print(data[var].value_counts(normalize=True) * 100)  # As percentages
    
    return results


def main():
    """Main function to run the EVS analysis"""
    parser = argparse.ArgumentParser(description='Analyze EVS (European Values Study) data')
    parser.add_argument('--file', '-f', help='Path to EVS data file (CSV, SPSS, Stata)')
    parser.add_argument('--docs', '-d', help='Path to variable documentation file (CSV, JSON)')
    parser.add_argument('--output', '-o', default='evs_plots_france', help='Directory for output plots')
    args = parser.parse_args()
    
    print("EVS Data Analysis - French Data Focus")
    print("-" * 40)
    
    # Get file path
    if args.file:
        file_path = args.file
    else:
        file_path = input("Enter the path to your EVS data file: ")
    
    # Get documentation path
    doc_path = args.docs if args.docs else None
    
    try:
        # Load data
        data = load_data(file_path)
        
        # Load variable documentation if provided
        var_docs = load_variable_documentation(doc_path)
        
        # Identify metadata columns (before 'cntry')
        metadata_cols = identify_metadata_columns(data)
        
        # Clean data
        data = clean_data(data)
        
        # Categorize variables
        categories = categorize_variables(data, var_docs, metadata_cols)
        
        # Enhanced exploratory analysis
        enhanced_exploratory_analysis(data, var_docs, categories)
        
        # Thematic analysis
        thematic_analysis(data, categories, var_docs, args.output)
        
        # French-specific analysis
        french_specific_analysis(data, categories, var_docs, args.output)
        
        # Demographic insights
        demographic_insights(data, categories, args.output)
        
        # Original visualizations
        visualize_data(data, args.output)
        
        # Statistical tests
        stats_results = statistical_analysis(data)
        
        # Print statistical results
        print("\nStatistical Analysis Results:")
        print("-" * 40)
        for test, result in stats_results.items():
            print(f"\n{test.upper()}:")
            for key, value in result.items():
                print(f"  {key}: {value}")
        
        # Save cleaned data
        output_path = os.path.join(os.path.dirname(file_path), 'evs_france_cleaned_data.csv')
        data.to_csv(output_path, index=False)
        print(f"\nCleaned French data saved to {output_path}")
        
        print("\nAnalysis complete!")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    main()
