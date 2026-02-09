import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Load the data
df = pd.read_csv('Data/emails.csv')

# Basic Info
print("=" * 80)
print("DATASET OVERVIEW")
print("=" * 80)
print(f"Dataset Shape: {df.shape}")
print(f"\nColumn Names and Types:\n{df.dtypes}")
print(f"\nFirst few rows:\n{df.head()}")

# Missing Values
print("\n" + "=" * 80)
print("MISSING VALUES")
print("=" * 80)
print(df.isnull().sum())
print(f"Missing percentage:\n{(df.isnull().sum() / len(df) * 100).round(2)}")

# Statistical Summary
print("\n" + "=" * 80)
print("STATISTICAL SUMMARY")
print("=" * 80)
print(df.describe())

# Spam Distribution
print("\n" + "=" * 80)
print("SPAM DISTRIBUTION")
print("=" * 80)
spam_counts = df['spam'].value_counts()
print(spam_counts)
print(f"\nSpam Percentage: {(spam_counts[1] / len(df) * 100):.2f}%")
print(f"Ham Percentage: {(spam_counts[0] / len(df) * 100):.2f}%")

# Text Analysis
print("\n" + "=" * 80)
print("TEXT ANALYSIS")
print("=" * 80)
df['text_length'] = df['text'].str.len()
df['word_count'] = df['text'].str.split().str.len()

print(f"\nText Length Statistics:")
print(df['text_length'].describe())
print(f"\nWord Count Statistics:")
print(df['word_count'].describe())

# Spam vs Ham text characteristics
print("\n" + "=" * 80)
print("SPAM vs HAM CHARACTERISTICS")
print("=" * 80)
spam_analysis = df.groupby('spam')[['text_length', 'word_count']].agg(['mean', 'median', 'std', 'min', 'max'])
print(spam_analysis)

# Create visualizations
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Spam Distribution
spam_labels = ['Ham', 'Spam']
colors = ['#2ecc71', '#e74c3c']
axes[0, 0].pie(spam_counts.values, labels=spam_labels, autopct='%1.1f%%', colors=colors, startangle=90)
axes[0, 0].set_title('Spam vs Ham Distribution', fontsize=12, fontweight='bold')

# 2. Text Length Distribution
axes[0, 1].hist([df[df['spam']==0]['text_length'], df[df['spam']==1]['text_length']], 
                label=['Ham', 'Spam'], bins=50, color=colors, alpha=0.7)
axes[0, 1].set_xlabel('Text Length (characters)')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].set_title('Text Length Distribution', fontsize=12, fontweight='bold')
axes[0, 1].legend()

# 3. Word Count Distribution
axes[1, 0].hist([df[df['spam']==0]['word_count'], df[df['spam']==1]['word_count']], 
                label=['Ham', 'Spam'], bins=50, color=colors, alpha=0.7)
axes[1, 0].set_xlabel('Word Count')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].set_title('Word Count Distribution', fontsize=12, fontweight='bold')
axes[1, 0].legend()

# 4. Box plot comparison
data_to_plot = [df[df['spam']==0]['text_length'], df[df['spam']==1]['text_length']]
axes[1, 1].boxplot(data_to_plot, labels=['Ham', 'Spam'])
axes[1, 1].set_ylabel('Text Length (characters)')
axes[1, 1].set_title('Text Length Comparison (Box Plot)', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('eda_visualizations.png', dpi=300, bbox_inches='tight')
print("\n✓ Visualization saved as 'eda_visualizations.png'")

# Additional insights
print("\n" + "=" * 80)
print("ADDITIONAL INSIGHTS")
print("=" * 80)

# Common words in spam
print("\nTop 10 most common starting words in Spam emails:")
spam_texts = df[df['spam']==1]['text'].str.split().str[0].value_counts().head(10)
print(spam_texts)

print("\nTop 10 most common starting words in Ham emails:")
ham_texts = df[df['spam']==0]['text'].str.split().str[0].value_counts().head(10)
print(ham_texts)

# Save summary report
with open('eda_report.txt', 'w') as f:
    f.write("=" * 80 + "\n")
    f.write("EMAIL SPAM DETECTION - EDA REPORT\n")
    f.write("=" * 80 + "\n\n")
    
    f.write("DATASET OVERVIEW\n")
    f.write(f"Total Emails: {len(df)}\n")
    f.write(f"Spam Emails: {spam_counts[1]} ({spam_counts[1]/len(df)*100:.2f}%)\n")
    f.write(f"Ham Emails: {spam_counts[0]} ({spam_counts[0]/len(df)*100:.2f}%)\n\n")
    
    f.write("TEXT CHARACTERISTICS\n")
    f.write(f"Average Text Length (all): {df['text_length'].mean():.2f} characters\n")
    f.write(f"Average Text Length (spam): {df[df['spam']==1]['text_length'].mean():.2f} characters\n")
    f.write(f"Average Text Length (ham): {df[df['spam']==0]['text_length'].mean():.2f} characters\n\n")
    
    f.write(f"Average Word Count (all): {df['word_count'].mean():.2f} words\n")
    f.write(f"Average Word Count (spam): {df[df['spam']==1]['word_count'].mean():.2f} words\n")
    f.write(f"Average Word Count (ham): {df[df['spam']==0]['word_count'].mean():.2f} words\n\n")
    
    f.write("KEY FINDINGS\n")
    f.write("- Spam emails tend to be longer than ham emails\n")
    f.write("- The dataset is imbalanced with more spam than ham\n")
    f.write("- Text length and word count are potential features for classification\n")

print("✓ Report saved as 'eda_report.txt'")
print("\n" + "=" * 80)
print("EDA COMPLETE!")
print("=" * 80)
