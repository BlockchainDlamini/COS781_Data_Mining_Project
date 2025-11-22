# %% [markdown]
# #Association Rule Mining on Educational Data
# 
#  Mining association rules from the preprocessed KDD Cup 2010 Educational Data Mining Challenge dataset using weighted Apriori algorithm
# 

# %%
# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.frequent_patterns import apriori, association_rules, fpgrowth
from mlxtend.preprocessing import TransactionEncoder
import warnings
warnings.filterwarnings('ignore')

# Set display options
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
pd.set_option('display.width', None)

# Plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

print("Libraries imported successfully")

# %%
#load preprocessed dataset
file_path = './algebra_preprocessed.csv'
df = pd.read_csv(file_path)
print(f"Dataset shape: {df.shape}")
df.head()

# %% [markdown]
# ##1. Define item weights for weighted apriori

# %%
print("="*80)
print("DEFINING ITEM WEIGHTS FOR WEIGHTED APRIORI")
print("="*80)

# Weight assignment strategy (based on educational domain knowledge):
# 1. Struggle indicators (HIGH weight) - important to detect at-risk patterns
# 2. Success indicators (MEDIUM-HIGH weight) - important for best practices
# 3. Knowledge components (MEDIUM weight) - domain-specific importance
# 4. Engagement patterns (MEDIUM weight) - behavioral indicators
# 5. Neutral/contextual features (LOW weight) - supportive information

item_weights = {}

# ============================================================================
# 1. STRUGGLE INDICATORS (Weight: 2.5 - 3.0)
# ============================================================================
# High importance: patterns indicating student difficulty
print("\n1. Assigning weights to STRUGGLE INDICATORS...")

# Many errors/hints indicate struggle
item_weights['Incorrects_Binned=Many'] = 3.0
item_weights['Hints_Binned=Many'] = 2.5

# Long error durations suggest prolonged difficulty
item_weights['Error_Step_Duration_Binned=ESD_15'] = 2.0
item_weights['Error_Step_Duration_Binned=ESD_14'] = 1.8
item_weights['Error_Step_Duration_Binned=ESD_13'] = 1.6

# First attempt incorrect
item_weights['Correct First Attempt=0'] = 2.2

# Low engagement (early dropout risk)
item_weights['Engagement_Level_Binned=Light'] = 2.0

print(f"   Assigned {len([k for k in item_weights if 'Incorrect' in k or 'Hints' in k or 'Error' in k or 'Light' in k])} struggle-related weights")

# ============================================================================
# 2. SUCCESS INDICATORS (Weight: 1.5 - 2.0)
# ============================================================================
print("\n2. Assigning weights to SUCCESS INDICATORS...")

# Few errors/hints indicate mastery
item_weights['Incorrects_Binned=Few'] = 1.5
item_weights['Hints_Binned=Few'] = 1.5
item_weights['Corrects_Binned=Many'] = 1.8

# First attempt correct
item_weights['Correct First Attempt=1'] = 1.8

# High engagement (persistence)
item_weights['Engagement_Level_Binned=Very_Heavy'] = 1.7
item_weights['Engagement_Level_Binned=Heavy'] = 1.5

print(f"   Assigned {len([k for k in item_weights if 'Few' in k or 'Many' in k or 'Correct First Attempt=1' in k or 'Heavy' in k])} success-related weights")

# ============================================================================
# 3. KNOWLEDGE COMPONENTS (Weight: 1.3 - 2.0)
# ============================================================================
print("\n3. Assigning weights to KEY KNOWLEDGE COMPONENTS...")

# TODO: Alter, hard coding is not a good idea
# Core algebraic skills (high educational value)
important_kcs = {
    'KC_Entering_a_given': 1.8,
    'KC_Define_Variable': 1.7,
    'KC_Entering_the_slope': 1.6,
    'KC_Entering_the_y_intercept': 1.6,
    'KC_Setting_the_slope': 1.5,
    'KC_Setting_the_y_intercept': 1.5,
}

# SkillRule KCs (procedural knowledge - very important)
skillrule_weight = 1.8
kc_cols = [col for col in ['KC_[SkillRule:_Add/Subtract;_[Typein_Skill:_{Isolate_positive;_Isolate_negative;_Remove_constant;_Consolidate_vars,_no_coeff;_Consolidate_vars_with_coeff;_Consolidate_vars,_any}]]',
                            'KC_[SkillRule:_Multiply/Divide;_[Typein_Skill:_{Remove_coefficient;_Variable_in_denominator}]]',
                            'KC_[SkillRule:_Remove_constant;_{ax+b=c,_positive;_ax+b=c,_negative;_x+a=b,_positive;_x+a=b,_negative;_[var_expr]+[const_expr]=[const_expr],_positive;_[var_expr]+[const_expr]=[const_expr],_negative;_[var_expr]+[const_expr]=[const_expr],_all;_Combine_constants_to_right;_Combine_constants_to_left;_a_x=b,_positive;_a/x+b=c,_positive;_a/x+b=c,_negative}]']]

for kc in important_kcs:
    item_weights[f'{kc}=1'] = important_kcs[kc]

# All other KCs get default weight
item_weights['KC_default'] = 1.3

print(f"   Assigned {len(important_kcs)} high-priority KC weights")
print(f"   Other KCs will use default weight: 1.3")

# ============================================================================
# 4. TIME/DURATION PATTERNS (Weight: 1.2 - 1.5)
# ============================================================================
print("\n4. Assigning weights to DURATION PATTERNS...")

# Extreme durations (very quick or very slow) may indicate issues
item_weights['Step_Duration_Binned=SD_1'] = 1.3  # Very quick (possibly guessing)
item_weights['Step_Duration_Binned=SD_15'] = 1.4  # Very slow (struggling)

# Correct step durations
item_weights['Correct_Step_Duration_Binned=CSD_1'] = 1.2
item_weights['Correct_Step_Duration_Binned=CSD_15'] = 1.2

print(f"   Assigned duration-based weights")

# ============================================================================
# 5. CONTEXTUAL FEATURES (Weight: 1.0)
# ============================================================================
print("\n5. Setting DEFAULT weight for other items...")

# Default weight for all other items
item_weights['default'] = 1.0

print(f"   Default weight: 1.0")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*80)
print("WEIGHT ASSIGNMENT SUMMARY")
print("="*80)
print(f"Total explicit weights defined: {len(item_weights)}")
print(f"\nWeight distribution:")
print(f"   High priority (≥2.0):   {len([w for w in item_weights.values() if w >= 2.0])} items")
print(f"   Medium priority (1.5-2.0): {len([w for w in item_weights.values() if 1.5 <= w < 2.0])} items")
print(f"   Low priority (<1.5):    {len([w for w in item_weights.values() if w < 1.5])} items")

print("\n" + "="*80)
print("Sample weights:")
for item, weight in list(item_weights.items())[:15]:
    print(f"   {item:50s} → {weight}")
print("   ...")

# %% [markdown]
# ##2. Convert Categorical Binned Columns to One-Hot Encoded Binary Columns

# %%
print("="*80)
print("CONVERTING CATEGORICAL BINNED COLUMNS TO BINARY FORMAT")
print("="*80)

# Identify which columns need to be converted to binary
# These are the binned categorical columns from preprocessing
categorical_binned_cols = [
    'Problem_View_Binned',
    'Corrects_Binned',
    'Incorrects_Binned',
    'Hints_Binned',
    'Step_Duration_Binned',
    'Correct_Step_Duration_Binned',
    'Error_Step_Duration_Binned',
    'Engagement_Level_Binned'
]

print(f"\nCategorical columns to convert: {len(categorical_binned_cols)}")
for col in categorical_binned_cols:
    if col in df.columns:
        unique_vals = df[col].nunique()
        print(f"   {col:40s} → {unique_vals} unique values")

# Store original shape
original_shape = df.shape
print(f"\nOriginal dataframe shape: {original_shape}")

# Convert each categorical binned column to binary columns
# Format: ColumnName=Value (to match our weight dictionary)
print("\n" + "-"*80)
print("Converting to binary format...")

new_binary_cols = []

for col in categorical_binned_cols:
    if col not in df.columns:
        print(f"   Warning: {col} not found in dataframe, skipping...")
        continue

    # Get unique values for this column
    unique_values = df[col].unique()
    unique_values = [v for v in unique_values if pd.notna(v)]  # Remove NaN

    print(f"\n   {col}:")
    print(f"      Unique values: {unique_values}")

    # Create binary column for each unique value
    for value in unique_values:
        # Create column name in format: ColumnName=Value
        new_col_name = f"{col.replace('_Binned', '')}={value}"

        # Create binary column (1 if matches value, 0 otherwise)
        df[new_col_name] = (df[col] == value).astype(int)

        new_binary_cols.append(new_col_name)
        print(f"      Created: {new_col_name} (count={df[new_col_name].sum()})")

print("\n" + "-"*80)
print(f"Total new binary columns created: {len(new_binary_cols)}")

# Now drop the original categorical binned columns since we have binary versions
print("\nDropping original categorical columns...")
df.drop(columns=categorical_binned_cols, inplace=True, errors='ignore')

print(f"\nNew dataframe shape: {df.shape}")
print(f"Columns added: {df.shape[1] - original_shape[1]}")

# ============================================================================
# Verify binary format
# ============================================================================
print("\n" + "="*80)
print("VERIFICATION: Checking binary format")
print("="*80)

# Check all columns are now numeric (0/1) or categorical identifiers
numeric_cols = df.select_dtypes(include=[np.number]).columns
non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns

print(f"\nNumeric columns (binary/continuous): {len(numeric_cols)}")
print(f"Non-numeric columns (identifiers): {len(non_numeric_cols)}")
print(f"\nNon-numeric columns: {list(non_numeric_cols)}")

# Sample of new binary columns
print("\n" + "-"*80)
print("Sample of newly created binary columns:")
sample_new_cols = [col for col in new_binary_cols[:10]]
print(df[sample_new_cols].head())

# Check for any NaN values
missing_check = df.isna().sum()
if missing_check.sum() > 0:
    print(f"\n  Warning: Found {missing_check.sum()} missing values")
    print(missing_check[missing_check > 0])
else:
    print("\n No missing values detected")

print("\n" + "="*80)
print("CATEGORICAL TO BINARY CONVERSION COMPLETE")
print("="*80)

# %% [markdown]
# ##3. Dataset transformation
# 
# preparing the dataframe for apriori

# %%
print("="*80)
print("PREPARING DATA FOR ASSOCIATION RULE MINING")
print("="*80)

# ============================================================================
# 1. Identify columns to EXCLUDE from mining
# ============================================================================
print("\n1. Identifying columns to exclude...")

# Exclude identifier/contextual columns (not useful for rules)
exclude_cols = [
    'Anon Student Id',
    'Problem Hierarchy',
    'Problem Name',
    'Step Name',
    'KC(Default)',           # Original multi-valued KC (we have binary KC_* columns)
    'Opportunity(Default)',  # Original multi-valued Opp (we have binary Opp_* columns)
]

# Exclude continuous numeric columns (already binned)
continuous_cols = [
    'Problem View',
    'Step Duration (sec)',
    'Correct Step Duration (sec)',
    'Error Step Duration (sec)',
    'Incorrects',
    'Hints',
    'Corrects',
    'Engagement_Level'
]

all_exclude = exclude_cols + continuous_cols
print(f"   Excluding {len(all_exclude)} columns")

# ============================================================================
# 2. Select binary columns for mining
# ============================================================================
print("\n2. Selecting binary columns for mining...")

# Get all columns except excluded ones
mining_cols = [col for col in df.columns if col not in all_exclude]

print(f"   Total columns for mining: {len(mining_cols)}")

# Create mining dataframe
df_mining = df[mining_cols].copy()

print(f"   Mining dataframe shape: {df_mining.shape}")

# ============================================================================
# 3. Handle Correct First Attempt column specially
# ============================================================================
print("\n3. Processing 'Correct First Attempt' column...")

# This column should be binary (0 or 1)
# Convert to proper format for our weight dictionary
if 'Correct First Attempt' in df_mining.columns:
    # Create two binary columns: one for each value
    df_mining['Correct_First_Attempt=0'] = (df_mining['Correct First Attempt'] == 0).astype(int)
    df_mining['Correct_First_Attempt=1'] = (df_mining['Correct First Attempt'] == 1).astype(int)

    # Drop original column
    df_mining.drop(columns=['Correct First Attempt'], inplace=True)

    print(f"   Created Correct_First_Attempt=0 and Correct_First_Attempt=1")
    print(f"   Correct_First_Attempt=0 count: {df_mining['Correct_First_Attempt=0'].sum()}")
    print(f"   Correct_First_Attempt=1 count: {df_mining['Correct_First_Attempt=1'].sum()}")

# ============================================================================
# 4. Verify all columns are binary
# ============================================================================
print("\n4. Verifying all columns are binary (0 or 1)...")

non_binary_cols = []
for col in df_mining.columns:
    unique_vals = df_mining[col].unique()
    if not set(unique_vals).issubset({0, 1, 0.0, 1.0}):
        non_binary_cols.append(col)
        print(f"   ⚠️  {col} has non-binary values: {unique_vals[:10]}")

if len(non_binary_cols) == 0:
    print("    All columns are binary!")
else:
    print(f"   Found {len(non_binary_cols)} non-binary columns")

# ============================================================================
# 5. Summary statistics
# ============================================================================
print("\n" + "="*80)
print("DATA PREPARATION SUMMARY")
print("="*80)
print(f"Original dataframe: {df.shape}")
print(f"Mining dataframe: {df_mining.shape}")
print(f"Excluded columns: {len(all_exclude)}")
print(f"Binary feature columns: {df_mining.shape[1]}")

print("\nColumn type breakdown:")
kc_cols = [col for col in df_mining.columns if col.startswith('KC_')]
opp_cols = [col for col in df_mining.columns if col.startswith('Opp_')]
binned_cols = [col for col in df_mining.columns if '=' in col]
other_cols = [col for col in df_mining.columns if not (col.startswith('KC_') or col.startswith('Opp_') or '=' in col)]

print(f"   Knowledge Component (KC_*): {len(kc_cols)}")
print(f"   Opportunity (Opp_*): {len(opp_cols)}")
print(f"   Binned features (contain '='): {len(binned_cols)}")
print(f"   Other binary features: {len(other_cols)}")

print("\n" + "="*80)
print("READY FOR WEIGHTED APRIORI MINING")
print("="*80)

# %% [markdown]
# ##4. Implement Weighted Support Function
# 
# implement the weighted support calculation from the Wang et al. (2014) paper. This is the key innovation from the paper.

# %%
print("="*80)
print("IMPLEMENTING WEIGHTED APRIORI ALGORITHM")
print("="*80)

# ============================================================================
# Wang et al. (2014) Weighted Support Formula:
# wsup(X) = max{w1, w2, ..., wk} × sup(X)
#
# Where:
# - X is an itemset
# - w1, w2, ..., wk are weights of items in X
# - sup(X) is the traditional support (proportion of transactions containing X)
# ============================================================================

def get_item_weight(item, weight_dict):
    """
    Get weight for an item from the weight dictionary.
    Falls back to default weight if item not found.

    Args:
        item: Column name (e.g., 'Incorrects=Many', 'KC_Entering_a_given')
        weight_dict: Dictionary mapping item names to weights

    Returns:
        float: Weight for the item
    """
    # Direct match
    if item in weight_dict:
        return weight_dict[item]

    # Check for KC default
    if item.startswith('KC_'):
        return weight_dict.get('KC_default', 1.0)

    # Check for pattern matches (e.g., any Step_Duration bin)
    if 'Step_Duration=SD_' in item:
        # Already handled specific ones in weight_dict, use default for others
        return weight_dict.get('default', 1.0)

    if 'Correct_Step_Duration=CSD_' in item:
        return weight_dict.get('default', 1.0)

    if 'Error_Step_Duration=ESD_' in item:
        return weight_dict.get('default', 1.0)

    # Default weight
    return weight_dict.get('default', 1.0)


def calculate_weighted_support(frequent_itemsets, weight_dict):
    """
    Calculate weighted support for frequent itemsets according to Wang et al. (2014).

    wsup(X) = max{w1, w2, ..., wk} × sup(X)

    Args:
        frequent_itemsets: DataFrame from mlxtend.apriori with 'support' and 'itemsets' columns
        weight_dict: Dictionary mapping item names to weights

    Returns:
        DataFrame: Original itemsets with added 'weighted_support' column
    """
    weighted_supports = []

    for idx, row in frequent_itemsets.iterrows():
        itemset = row['itemsets']
        original_support = row['support']

        # Get weights for all items in the itemset
        weights = [get_item_weight(item, weight_dict) for item in itemset]

        # Weighted support = max weight × original support
        max_weight = max(weights)
        weighted_sup = max_weight * original_support

        weighted_supports.append(weighted_sup)

    # Add weighted support column
    result = frequent_itemsets.copy()
    result['weighted_support'] = weighted_supports
    result['max_weight_in_itemset'] = [
        max([get_item_weight(item, weight_dict) for item in itemset])
        for itemset in result['itemsets']
    ]

    return result


print("\n Weighted support functions implemented")
print("\nFormula: wsup(X) = max{w1, w2, ..., wk} × sup(X)")
print("\nThis means:")
print("  - Itemsets containing high-weight items (like 'Incorrects=Many') get boosted")
print("  - Even if an itemset has low frequency, if it contains important items,")
print("    it will have higher weighted support")
print("  - This allows us to discover educationally important patterns that might")
print("    be missed by traditional Apriori due to low frequency")

print("\n" + "="*80)

# %% [markdown]
# ##5. Filter Low-Frequency Items and Run FP-Growth
# 
# due to how large the dataset is filtering and sampling is need to balance pattern discovery and computational resources

# %%
print("="*80)
print("OPTIMIZING DATA FOR EFFICIENT MINING")
print("="*80)

# ============================================================================
# Step 1: Filter out extremely rare items to reduce dimensionality
# ============================================================================
print("\n1. Analyzing item frequencies...")

# Calculate frequency of each item (column)
item_frequencies = df_mining.sum() / len(df_mining)

print(f"   Total items (columns): {len(item_frequencies)}")
print(f"   Items appearing in >1% of transactions: {(item_frequencies > 0.01).sum()}")
print(f"   Items appearing in >0.5% of transactions: {(item_frequencies > 0.005).sum()}")
print(f"   Items appearing in <0.5% of transactions: {(item_frequencies < 0.005).sum()}")

# Filter: keep only items that appear in at least 0.5% of transactions
min_item_frequency = 0.005
frequent_items = item_frequencies[item_frequencies >= min_item_frequency].index.tolist()

print(f"\n   Filtering items with frequency < {min_item_frequency} ({min_item_frequency*100}%)")
print(f"   Keeping {len(frequent_items)} items")
print(f"   Removing {len(item_frequencies) - len(frequent_items)} rare items")

# Create filtered dataset
df_mining_filtered = df_mining[frequent_items].copy()

print(f"\n   New mining dataframe shape: {df_mining_filtered.shape}")
print(f"   Dimensionality reduction: {df_mining.shape[1]} → {df_mining_filtered.shape[1]} columns")

# ============================================================================
# Step 2: Run FP-Growth (much more memory efficient!)
# ============================================================================
print("\n" + "="*80)
print("RUNNING FP-GROWTH ALGORITHM")
print("="*80)

min_support = 0.01  # 1%
print(f"\nMinimum support threshold: {min_support} ({min_support*100}%)")
print("Using FP-Growth (memory-efficient alternative to Apriori)")

import time
start_time = time.time()

# Run FP-Growth
frequent_itemsets = fpgrowth(
    df_mining_filtered,
    min_support=min_support,
    use_colnames=True
)

elapsed = time.time() - start_time
print(f"\n✓ FP-Growth completed in {elapsed:.2f} seconds")
print(f"Found {len(frequent_itemsets)} frequent itemsets")

# ============================================================================
# Step 3: Calculate weighted support
# ============================================================================
print("\n" + "-"*80)
print("Calculating weighted support...")

frequent_itemsets_weighted = calculate_weighted_support(
    frequent_itemsets,
    item_weights
)

print(f"✓ Weighted support calculated")

# ============================================================================
# Step 4: Analyze results
# ============================================================================
print("\n" + "="*80)
print("FREQUENT ITEMSETS SUMMARY")
print("="*80)

print(f"\nTotal frequent itemsets: {len(frequent_itemsets_weighted)}")

# Breakdown by itemset size
itemset_sizes = frequent_itemsets_weighted['itemsets'].apply(len)
print(f"\nItemset size distribution:")
for size in sorted(itemset_sizes.unique()):
    count = (itemset_sizes == size).sum()
    print(f"   {size}-itemsets: {count}")

# Top itemsets by traditional support
print("\n" + "-"*80)
print("Top 10 itemsets by TRADITIONAL support:")
print("-"*80)
top_support = frequent_itemsets_weighted.nlargest(10, 'support')
for idx, row in top_support.iterrows():
    items = ', '.join(list(row['itemsets']))
    print(f"   sup={row['support']:.4f} | {items[:80]}...")

# Top itemsets by weighted support
print("\n" + "-"*80)
print("Top 10 itemsets by WEIGHTED support:")
print("-"*80)
top_weighted = frequent_itemsets_weighted.nlargest(10, 'weighted_support')
for idx, row in top_weighted.iterrows():
    items = ', '.join(list(row['itemsets']))
    print(f"   wsup={row['weighted_support']:.4f} (max_weight={row['max_weight_in_itemset']:.2f}) | {items[:70]}...")

# Compare: items that rank differently
print("\n" + "-"*80)
print("Items boosted by weighting (high weighted support, lower traditional support):")
print("-"*80)

# Add rank columns
frequent_itemsets_weighted['support_rank'] = frequent_itemsets_weighted['support'].rank(ascending=False)
frequent_itemsets_weighted['weighted_rank'] = frequent_itemsets_weighted['weighted_support'].rank(ascending=False)
frequent_itemsets_weighted['rank_boost'] = frequent_itemsets_weighted['support_rank'] - frequent_itemsets_weighted['weighted_rank']

# Show itemsets that got biggest boost from weighting
boosted = frequent_itemsets_weighted.nlargest(10, 'rank_boost')
for idx, row in boosted.iterrows():
    items = ', '.join(list(row['itemsets']))
    print(f"   Rank boost: {row['rank_boost']:.0f} | wsup={row['weighted_support']:.4f} | {items[:70]}...")

print("\n" + "="*80)
print("READY TO GENERATE ASSOCIATION RULES")
print("="*80)

# %% [markdown]
# ##6. Generate and Evaluate Association Rules

# %%
print("="*80)
print("GENERATING ASSOCIATION RULES (FIXED v2)")
print("="*80)

# ============================================================================
# Key insight: association_rules needs ALL itemsets including 1-itemsets
# to look up antecedent/consequent support
# ============================================================================

print("\n1. Preparing itemsets for rule generation...")

# Filter but KEEP 1-itemsets (they're needed for lookups)
filtered_original = frequent_itemsets[
    (frequent_itemsets['support'] >= 0.02) |  # 2% support
    (frequent_itemsets['itemsets'].apply(len) == 1)  # OR single items (needed for reference)
].copy()

print(f"   Total itemsets: {len(filtered_original)}")
print(f"   1-itemsets: {(filtered_original['itemsets'].apply(len) == 1).sum()}")
print(f"   2+ itemsets: {(filtered_original['itemsets'].apply(len) >= 2).sum()}")

# ============================================================================
# 2. Generate rules
# ============================================================================
print("\n2. Generating association rules...")

import time
start_time = time.time()

try:
    # Use support_only first to see if it helps
    rules = association_rules(
        filtered_original,
        metric="confidence",
        min_threshold=0.5,
        support_only=False  # We want all metrics
    )

    # Filter by lift
    rules = rules[rules['lift'] >= 1.2]

    elapsed = time.time() - start_time
    print(f"\n✓ Rules generated in {elapsed:.2f} seconds")
    print(f"Total rules: {len(rules)}")

except Exception as e:
    print(f"\n⚠️  Error with filtered itemsets: {e}")
    print("\nTrying with ALL itemsets (slower but should work)...")

    # Fall back to using ALL itemsets
    try:
        rules = association_rules(
            frequent_itemsets,  # Use ALL itemsets
            metric="confidence",
            min_threshold=0.6  # Higher threshold to reduce rules
        )

        rules = rules[
            (rules['lift'] >= 1.2) &
            (rules['support'] >= 0.02)  # Filter rules by support
        ]

        print(f"\n✓ Rules generated successfully")
        print(f"Total rules after filtering: {len(rules)}")

    except Exception as e2:
        print(f"\n❌ Still failed: {e2}")
        print("\nLet's try with even simpler approach...")

        # Last resort: higher support threshold for itemsets
        simple_itemsets = frequent_itemsets[
            frequent_itemsets['support'] >= 0.03  # 3% support
        ].copy()

        rules = association_rules(
            simple_itemsets,
            metric="lift",
            min_threshold=1.2
        )

        rules = rules[rules['confidence'] >= 0.5]

        print(f"\n✓ Rules generated with simpler approach")
        print(f"Total rules: {len(rules)}")

# ============================================================================
# 3. Add weight information and analyze
# ============================================================================
if len(rules) > 0:
    print("\n3. Adding weight information to rules...")

    rules['antecedent_max_weight'] = rules['antecedents'].apply(
        lambda items: max([get_item_weight(item, item_weights) for item in items])
    )
    rules['consequent_max_weight'] = rules['consequents'].apply(
        lambda items: max([get_item_weight(item, item_weights) for item in items])
    )
    rules['rule_max_weight'] = rules[['antecedent_max_weight', 'consequent_max_weight']].max(axis=1)

    # Importance score
    rules['importance_score'] = (
        rules['confidence'] *
        rules['lift'] *
        rules['rule_max_weight']
    )

    print("✓ Weight information added")

    # Save
    rules.to_pickle('./association_rules.pkl')
    print("✓ Saved to: association_rules.pkl")

    # ============================================================================
    # 4. Analysis
    # ============================================================================
    print("\n" + "="*80)
    print("ASSOCIATION RULES ANALYSIS")
    print("="*80)

    print(f"\nTotal rules: {len(rules)}")
    print(f"Avg confidence: {rules['confidence'].mean():.3f}")
    print(f"Avg lift: {rules['lift'].mean():.3f}")
    print(f"Avg weight: {rules['rule_max_weight'].mean():.3f}")

    print(f"\nWeight distribution:")
    print(f"   High-weight (≥2.0): {(rules['rule_max_weight'] >= 2.0).sum()}")
    print(f"   Medium-weight (1.5-2.0): {((rules['rule_max_weight'] >= 1.5) & (rules['rule_max_weight'] < 2.0)).sum()}")
    print(f"   Low-weight (<1.5): {(rules['rule_max_weight'] < 1.5).sum()}")

    # Show best rules
    print("\n" + "-"*80)
    print("TOP 20 RULES BY CONFIDENCE:")
    print("-"*80)

    for idx, row in rules.nlargest(20, 'confidence').iterrows():
        ant_list = sorted(list(row['antecedents']))
        cons_list = sorted(list(row['consequents']))

        ant = ', '.join(ant_list)[:70]
        cons = ', '.join(cons_list)[:70]

        print(f"\nIF {ant}")
        print(f"→ THEN {cons}")
        print(f"  [conf={row['confidence']:.3f}, lift={row['lift']:.2f}, sup={row['support']:.3f}, wt={row['rule_max_weight']:.2f}]")

    print("\n" + "-"*80)
    print("TOP 20 RULES BY IMPORTANCE SCORE:")
    print("-"*80)

    for idx, row in rules.nlargest(20, 'importance_score').iterrows():
        ant_list = sorted(list(row['antecedents']))
        cons_list = sorted(list(row['consequents']))

        ant = ', '.join(ant_list)[:70]
        cons = ', '.join(cons_list)[:70]

        print(f"\nIF {ant}")
        print(f"→ THEN {cons}")
        print(f"  [score={row['importance_score']:.2f}, conf={row['confidence']:.3f}, lift={row['lift']:.2f}, wt={row['rule_max_weight']:.2f}]")

    # High-weight rules
    high_wt = rules[rules['rule_max_weight'] >= 2.0]

    if len(high_wt) > 0:
        print("\n" + "-"*80)
        print(f"HIGH-WEIGHT RULES (Struggle/Success Indicators) - Total: {len(high_wt)}")
        print("-"*80)

        for idx, row in high_wt.nlargest(20, 'confidence').iterrows():
            ant_list = sorted(list(row['antecedents']))
            cons_list = sorted(list(row['consequents']))

            ant = ', '.join(ant_list)[:70]
            cons = ', '.join(cons_list)[:70]

            print(f"\nIF {ant}")
            print(f"→ THEN {cons}")
            print(f"  [conf={row['confidence']:.3f}, lift={row['lift']:.2f}, wt={row['rule_max_weight']:.2f}]")
    else:
        print("\n⚠️  No high-weight rules found.")

    print("\n" + "="*80)
    print("✓ RULE MINING COMPLETE!")
    print("="*80)

else:
    print("\n❌ No rules generated. Try lowering thresholds.")

print("\n" + "="*80)

# %% [markdown]
# ##7. Visualizations and Interpretation

# %%
print("="*80)
print("VISUALIZING ASSOCIATION RULES")
print("="*80)

# ============================================================================
# 1. Rule Distribution Analysis
# ============================================================================
print("\n1. Creating rule distribution plots...")

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Association Rules Distribution Analysis', fontsize=16, fontweight='bold')

# Support distribution
axes[0, 0].hist(rules['support'], bins=50, edgecolor='black', alpha=0.7)
axes[0, 0].set_xlabel('Support')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].set_title('Support Distribution')
axes[0, 0].axvline(rules['support'].mean(), color='red', linestyle='--', label=f'Mean: {rules["support"].mean():.3f}')
axes[0, 0].legend()

# Confidence distribution
axes[0, 1].hist(rules['confidence'], bins=50, edgecolor='black', alpha=0.7, color='orange')
axes[0, 1].set_xlabel('Confidence')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].set_title('Confidence Distribution')
axes[0, 1].axvline(rules['confidence'].mean(), color='red', linestyle='--', label=f'Mean: {rules["confidence"].mean():.3f}')
axes[0, 1].legend()

# Lift distribution
axes[0, 2].hist(rules['lift'], bins=50, edgecolor='black', alpha=0.7, color='green')
axes[0, 2].set_xlabel('Lift')
axes[0, 2].set_ylabel('Frequency')
axes[0, 2].set_title('Lift Distribution')
axes[0, 2].axvline(rules['lift'].mean(), color='red', linestyle='--', label=f'Mean: {rules["lift"].mean():.3f}')
axes[0, 2].legend()

# Weight distribution
axes[1, 0].hist(rules['rule_max_weight'], bins=30, edgecolor='black', alpha=0.7, color='purple')
axes[1, 0].set_xlabel('Max Weight in Rule')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].set_title('Rule Weight Distribution')
axes[1, 0].axvline(rules['rule_max_weight'].mean(), color='red', linestyle='--', label=f'Mean: {rules["rule_max_weight"].mean():.3f}')
axes[1, 0].legend()

# Importance score distribution
axes[1, 1].hist(rules['importance_score'], bins=50, edgecolor='black', alpha=0.7, color='red')
axes[1, 1].set_xlabel('Importance Score')
axes[1, 1].set_ylabel('Frequency')
axes[1, 1].set_title('Importance Score Distribution')
axes[1, 1].axvline(rules['importance_score'].mean(), color='blue', linestyle='--', label=f'Mean: {rules["importance_score"].mean():.3f}')
axes[1, 1].legend()

# Scatter: Confidence vs Lift
scatter = axes[1, 2].scatter(rules['confidence'], rules['lift'],
                             c=rules['rule_max_weight'], cmap='viridis',
                             alpha=0.5, s=10)
axes[1, 2].set_xlabel('Confidence')
axes[1, 2].set_ylabel('Lift')
axes[1, 2].set_title('Confidence vs Lift (colored by weight)')
plt.colorbar(scatter, ax=axes[1, 2], label='Weight')

plt.tight_layout()
plt.savefig('./rule_distributions.png', dpi=300, bbox_inches='tight')
print("   Saved: rule_distributions.png")
plt.show()

# ============================================================================
# 2. Top Rules Visualization
# ============================================================================
print("\n2. Creating top rules bar chart...")

# Top 15 by importance score
top_rules = rules.nlargest(15, 'importance_score').copy()

# Create readable labels
def create_rule_label(row):
    ant = list(row['antecedents'])
    cons = list(row['consequents'])
    # Shorten for readability
    ant_short = ', '.join([a[:25] for a in ant])[:40]
    cons_short = ', '.join([c[:25] for c in cons])[:40]
    return f"{ant_short} → {cons_short}"

top_rules['rule_label'] = top_rules.apply(create_rule_label, axis=1)

fig, ax = plt.subplots(figsize=(12, 8))
bars = ax.barh(range(len(top_rules)), top_rules['importance_score'],
               color=plt.cm.viridis(top_rules['rule_max_weight'] / top_rules['rule_max_weight'].max()))
ax.set_yticks(range(len(top_rules)))
ax.set_yticklabels(top_rules['rule_label'], fontsize=9)
ax.set_xlabel('Importance Score (Confidence × Lift × Weight)', fontsize=12)
ax.set_title('Top 15 Association Rules by Importance Score', fontsize=14, fontweight='bold')
ax.invert_yaxis()

# Add confidence and lift annotations
for i, (idx, row) in enumerate(top_rules.iterrows()):
    ax.text(row['importance_score'] + 0.5, i,
            f"conf:{row['confidence']:.2f}, lift:{row['lift']:.1f}",
            va='center', fontsize=8)

plt.tight_layout()
plt.savefig('./top_rules_importance.png', dpi=300, bbox_inches='tight')
print("   Saved: top_rules_importance.png")
plt.show()

# ============================================================================
# 3. Rule Length Analysis
# ============================================================================
print("\n3. Analyzing rule complexity...")

rules['antecedent_length'] = rules['antecedents'].apply(len)
rules['consequent_length'] = rules['consequents'].apply(len)
rules['total_items'] = rules['antecedent_length'] + rules['consequent_length']

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Antecedent length
ant_counts = rules['antecedent_length'].value_counts().sort_index()
axes[0].bar(ant_counts.index, ant_counts.values, edgecolor='black', alpha=0.7)
axes[0].set_xlabel('Number of Items in Antecedent')
axes[0].set_ylabel('Number of Rules')
axes[0].set_title('Antecedent Length Distribution')

# Consequent length
cons_counts = rules['consequent_length'].value_counts().sort_index()
axes[1].bar(cons_counts.index, cons_counts.values, edgecolor='black', alpha=0.7, color='orange')
axes[1].set_xlabel('Number of Items in Consequent')
axes[1].set_ylabel('Number of Rules')
axes[1].set_title('Consequent Length Distribution')

# Total items
total_counts = rules['total_items'].value_counts().sort_index()
axes[2].bar(total_counts.index, total_counts.values, edgecolor='black', alpha=0.7, color='green')
axes[2].set_xlabel('Total Items in Rule')
axes[2].set_ylabel('Number of Rules')
axes[2].set_title('Rule Complexity Distribution')

plt.tight_layout()
plt.savefig('./rule_complexity.png', dpi=300, bbox_inches='tight')
print("   Saved: rule_complexity.png")
plt.show()

# ============================================================================
# 4. Item Frequency in Rules
# ============================================================================
print("\n4. Analyzing most frequent items in rules...")

# Count item appearances
from collections import Counter

all_items = []
for itemset in rules['antecedents']:
    all_items.extend(list(itemset))
for itemset in rules['consequents']:
    all_items.extend(list(itemset))

item_counts = Counter(all_items)
top_items = pd.DataFrame(item_counts.most_common(20), columns=['Item', 'Frequency'])

fig, ax = plt.subplots(figsize=(12, 8))
bars = ax.barh(range(len(top_items)), top_items['Frequency'])
ax.set_yticks(range(len(top_items)))
ax.set_yticklabels([item[:50] for item in top_items['Item']], fontsize=10)
ax.set_xlabel('Frequency in Rules', fontsize=12)
ax.set_title('Top 20 Most Frequent Items in Association Rules', fontsize=14, fontweight='bold')
ax.invert_yaxis()

plt.tight_layout()
plt.savefig('./frequent_items_in_rules.png', dpi=300, bbox_inches='tight')
print("   Saved: frequent_items_in_rules.png")
plt.show()

print("\n" + "="*80)
print("ALL VISUALIZATIONS COMPLETE")
print("="*80)
print("\nGenerated files:")
print("   - rule_distributions.png")
print("   - top_rules_importance.png")
print("   - rule_complexity.png")
print("   - frequent_items_in_rules.png")

# %% [markdown]
# ##8. Intepretation of the results

# %% [markdown]
# 
# ## 1. **Rule Distributions (6-panel figure)**
# 
# ### **Support Distribution (Top Left)**
# - **What it shows:** How frequently itemsets appear together in transactions
# - **Interpretation:**
#   - **Heavily right-skewed** - most rules have very low support (0-0.1)
#   - **Mean: 0.041 (4.1%)** - average rule covers only 4% of students
#   - This is **expected** in educational data - specific learning patterns are diverse and individualized
#   - The spike near 0 indicates many rare but potentially important patterns
# 
# ### **Confidence Distribution (Top Middle)**
# - **What it shows:** How often the consequent occurs when antecedent is present
# - **Interpretation:**
#   - **Mean: 0.857 (85.7%)** - very high! Rules are highly reliable
#   - **Massive spike at 1.0** (perfect confidence) - many deterministic rules
#   - Example: "IF Error_Step_Duration=No_Error → THEN Correct_First_Attempt=1" (100% confidence)
#   - This suggests **strong causal relationships** in the data - certain conditions reliably predict outcomes
# 
# ### **Lift Distribution (Top Right)**
# - **What it shows:** How much more likely the consequent occurs with the antecedent vs. by chance
# - **Interpretation:**
#   - **Mean: 6.153** - items co-occur 6x more than random chance!
#   - **Right-skewed with long tail** - some rules have lift >20
#   - Values >1 indicate **positive correlation**, which all our rules have
#   - High lift rules (15-25) represent **strong unexpected associations** - these are the most interesting discoveries
# 
# ### **Weight Distribution (Bottom Left)**
# - **What it shows:** Educational importance (max weight of items in each rule)
# - **Interpretation:**
#   - **Bimodal distribution** - two peaks at 1.0 and 1.3
#   - **Mean: 1.128** - most rules involve low-weight items
#   - **Weight 1.0 dominates** (500k+ rules) - these are common success patterns
#   - **Weight 1.3 cluster** (360k rules) - rules involving KC items (default KC weight = 1.3)
#   - **No weights ≥2.0** - struggle indicators (`Incorrects=Many`, `Hints=Many`) are too rare in filtered data
# 
# ### **Importance Score Distribution (Bottom Middle)**
# - **What it shows:** Combined metric: confidence × lift × weight
# - **Interpretation:**
#   - **Heavily right-skewed** - most rules have low importance (0-5)
#   - **Mean: 5.824** - but some rules reach 30+!
#   - **Long tail** indicates a few **exceptionally important rules** worth investigating
#   - High-importance rules balance reliability, unexpectedness, and educational value
# 
# ### **Confidence vs Lift Scatter (Bottom Right)**
# - **What it shows:** Relationship between confidence and lift, colored by weight
# - **Interpretation:**
#   - **Positive correlation** - high confidence often means high lift
#   - **Horizontal bands** - many rules cluster at specific lift values (5, 10, 15, 20)
#   - **Yellow points (weight ~1.3)** scattered throughout - KC-involved rules at all confidence/lift levels
#   - **Dense purple region** (weight 1.0) - common patterns dominate
#   - Sweet spot: **top-right corner** (high confidence + high lift) = most valuable rules
# 
# ---
# 
# ## 2. **Top 15 Rules by Importance Score**
# 
# ### **Key Pattern: All involve KC_[SkillRule:_Remove_positive_coefficient]**
# - **What it shows:** Most important rules by combined metric
# - **Interpretation:**
#   - **Perfect confidence (1.00)** and **very high lift (24.5)** across all top rules
#   - **All involve specific KC**: `Remove_positive_coefficient` - a critical algebra skill
#   - **Common pattern:**
#     ```
#     IF {involving Remove_positive_coefficient KC}
#     → THEN {Engagement_Level=Light, Correct_First_Attempt=1, No_Error}
#     ```
#   - **Educational insight:** Students who work on this particular skill tend to:
#     - Have light engagement (early in course)
#     - Get correct first attempts
#     - Make no errors
#   - **Hypothesis:** This KC might be an **early/introductory skill** that students master easily, or it's well-taught in the curriculum
# 
# ---
# 
# ## 3. **Rule Complexity (3-panel figure)**
# 
# ### **Antecedent Length Distribution (Left)**
# - **What it shows:** Number of conditions (IF items) per rule
# - **Interpretation:**
#   - **Peak at 4 items** (~240k rules)
#   - **Range: 2-10 items** - most rules have 3-5 conditions
#   - **Bell-shaped** - balanced complexity
#   - Example: `IF Corrects=Few, Error_Duration=No_Error, Hints=Few, Opp_1`
# 
# ### **Consequent Length Distribution (Middle)**
# - **What it shows:** Number of predicted items (THEN items) per rule
# - **Interpretation:**
#   - **Peak at 4 items** (~250k rules) - similar to antecedent
#   - **More concentrated** than antecedent distribution
#   - Most rules predict **3-5 outcomes simultaneously**
#   - Example: `THEN Correct_First_Attempt=1, Incorrects=Few, Engagement_Level=Light`
# 
# ### **Total Rule Complexity (Right)**
# - **What it shows:** Total items in rule (antecedent + consequent)
# - **Interpretation:**
#   - **Peaks at 7-8 items** (220k+ rules each)
#   - **Range: 4-12 items** per rule
#   - **Balanced distribution** - neither too simple nor too complex
#   - This is ideal for interpretability - rules are detailed enough to be actionable but not overwhelming
# 
# ---
# 
# ## 4. **Most Frequent Items in Rules**
# 
# ### **Top Items (in order):**
# 
# 1. **Problem_View=Low** (~560k appearances)
#    - Students mostly see problems 1-2 times (not revisiting)
#    - Indicates first-time problem solving
# 
# 2-7. **Success Indicators** (all ~555k appearances):
#    - `Hints=Few`, `Incorrects=Few`, `Corrects=Few`
#    - `Error_Step_Duration=No_Error`
#    - `Correct_First_Attempt=1`
#    - `Engagement_Level=Light`
#    - **Pattern:** Most rules describe **successful student behavior**
# 
# 8. **Opp_1** (~510k)
#    - First opportunity to practice a KC
#    - Many rules involve initial learning experiences
# 
# 9. **KC_Entering_a_given** (~240k)
#    - Most common knowledge component
#    - Basic skill: entering given values from problem
# 
# 10. **Opp_11** (~220k)
#     - 11th practice opportunity
#     - Rules track progression over multiple attempts
# 
# 11-20. **Duration bins and other KCs**
#    - `Step_Duration=SD_1` (fastest completion)
#    - Various KC skills (`Using_small_numbers`, `Using_simple_numbers`)
# 
# ### **Key Insights:**
# - **Success patterns dominate** - most students succeed on most problems
# - **Early opportunities emphasized** - Opp_1 is critical
# - **Speed matters** - fastest durations (SD_1, CSD_1) appear frequently
# - **Basic skills prevalent** - `Entering_a_given` is foundational
# 
# ---
# 
# ## **Overall Interpretation:**
# 
# ### **What We Learned:**
# 1. **High-quality rules** - 85.7% average confidence with 6x lift shows strong patterns
# 2. **Specific KC importance** - `Remove_positive_coefficient` skill drives top rules
# 3. **Success-focused** - most patterns describe what successful students do
# 4. **Early learning critical** - first opportunities (Opp_1) heavily featured
# 5. **Balanced complexity** - 7-8 item rules are detailed yet interpretable
# 
# ### **Actionable Insights for Educators:**
# - Focus on **first opportunities** - they set the pattern
# - **Remove_positive_coefficient** skill deserves attention in curriculum design
# - Students who get **correct first attempts** tend to maintain success
# - **Light engagement** early on isn't necessarily bad - might indicate efficiency
# 
# ### **Limitations:**
# - **No struggle patterns** found (weight ≥2.0) - rare items filtered out
# - **Selection bias** - only top 20 problem hierarchies included
# - **Temporal aspects** not fully captured - rules are cross-sectional
# 
# 


