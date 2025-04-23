import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules, fpgrowth
import matplotlib.pyplot as plt

# Load dataset
try:
    df = pd.read_csv("assciation\Groceries_dataset.csv", header=None, skip_blank_lines=True)
    print("✅ Dataset loaded successfully!")
except Exception as e:
    print("❌ Error loading dataset:", e)
    exit()

# If the DataFrame is empty
if df.empty:
    print("❌ The dataset is empty. Check your file.")
    exit()

# Preprocess to get list of transactions
transactions = df[0].dropna().apply(lambda x: [item.strip() for item in x.split(',') if item.strip()]).tolist()

# One-hot encode the transactions
te = TransactionEncoder()
te_array = te.fit(transactions).transform(transactions)
df_encoded = pd.DataFrame(te_array, columns=te.columns_)

# Show basic stats
print("🔢 Total transactions:", len(df_encoded))
print("🛒 Total unique items:", len(df_encoded.columns))

# Try Apriori with safe support
min_support = 0.003
try:
    frequent_itemsets = apriori(df_encoded, min_support=min_support, use_colnames=True)
    if frequent_itemsets.empty:
        print("⚠️ No frequent itemsets found. Consider reducing min_support.")
except MemoryError:
    print("💥 MemoryError: Switching to FP-Growth algorithm...")
    frequent_itemsets = fpgrowth(df_encoded, min_support=min_support, use_colnames=True)

# Check results
if frequent_itemsets.empty:
    print("❌ Still no frequent itemsets found. Try lowering min_support slightly.")
    exit()
else:
    print("\n✅ Frequent Itemsets (Top 5):")
    print(frequent_itemsets.sort_values(by="support", ascending=False).head())

# Generate association rules
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.3)
rules_sorted = rules.sort_values(by='lift', ascending=False)

if not rules_sorted.empty:
    print("\n✅ Top 5 Association Rules by Lift:")
    print(rules_sorted[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head())
else:
    print("⚠️ No association rules generated.")

# Visualize item frequencies
item_freq = df_encoded.sum().sort_values(ascending=False).head(10)
plt.figure(figsize=(10, 6))
item_freq.plot(kind='bar', color='orchid')
plt.title("Top 10 Frequent Items")
plt.xlabel("Items")
plt.ylabel("Frequency")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Analyze strongest rule
if not rules_sorted.empty:
    top_rule = rules_sorted.iloc[0]
    print("\n📌 Strong Rule Analysis:")
    print(f"If a customer buys {set(top_rule['antecedents'])}, they are likely to also buy {set(top_rule['consequents'])}.")
    print(f"Confidence: {top_rule['confidence']:.2f}, Lift: {top_rule['lift']:.2f}")
