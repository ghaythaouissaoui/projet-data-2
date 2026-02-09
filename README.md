# Email Spam Detection - Exploratory Data Analysis

A comprehensive exploratory data analysis (EDA) of email spam detection dataset. This project analyzes the characteristics of spam vs legitimate emails to identify patterns and features useful for classification.

## 📋 Project Overview

This project performs detailed EDA on an email dataset containing both spam and legitimate (ham) emails. The analysis includes:

- Dataset statistics and distribution analysis
- Text feature engineering (length, word count)
- Spam vs Ham comparative analysis
- Data visualizations
- Key insights and findings

## 📁 Project Structure

```
projet-data-2/
├── Data/
│   └── emails.csv              # Email dataset with text and spam labels
├── eda_emails.ipynb            # Jupyter notebook with interactive analysis
├── eda_emails.py               # Python script version of the analysis
├── eda_visualizations.png      # Generated visualization plots
└── README.md                   # This file
```

## 📊 Dataset

**File:** `Data/emails.csv`

**Columns:**
- `text`: Email content/body
- `spam`: Binary label (0 = Ham/Legitimate, 1 = Spam)

**Statistics:**
- Total emails: ~5,000+
- Spam emails: Majority of dataset
- Ham emails: Minority of dataset

## 🚀 Getting Started

### Prerequisites

- Python 3.7+
- pip (Python package manager)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/ghaythaouissaoui/projet-data-2.git
cd projet-data-2
```

2. Install required dependencies:
```bash
pip install pandas numpy matplotlib seaborn
```

### Running the Analysis

**Option 1: Jupyter Notebook (Interactive)**
```bash
jupyter notebook eda_emails.ipynb
```

**Option 2: Python Script**
```bash
python eda_emails.py
```

## 📈 Key Findings

- **Spam emails tend to be longer** than legitimate emails in terms of character count and word count
- **Dataset imbalance**: More spam emails than ham emails
- **Text length and word count** are strong indicators for spam classification
- **Common patterns**: Spam emails often start with promotional keywords

## 📊 Visualizations Generated

The analysis produces 4 key visualizations:

1. **Spam vs Ham Distribution** - Pie chart showing class distribution
2. **Text Length Distribution** - Histogram comparing email lengths
3. **Word Count Distribution** - Histogram comparing word counts
4. **Box Plot Comparison** - Statistical comparison of text lengths

Output: `eda_visualizations.png`

## 🔍 Analysis Sections

### 1. Data Loading & Exploration
- Load CSV data
- Display dataset shape and structure
- Show first few rows

### 2. Missing Values Check
- Identify missing data
- Calculate missing percentages

### 3. Statistical Summary
- Descriptive statistics for all columns
- Data type information

### 4. Spam Distribution
- Count spam vs ham emails
- Calculate percentages

### 5. Text Feature Engineering
- Calculate text length (characters)
- Calculate word count
- Generate statistics

### 6. Comparative Analysis
- Compare spam vs ham characteristics
- Mean, median, std, min, max values

### 7. Visualizations
- Create multi-panel plots
- Save high-resolution images

### 8. Additional Insights
- Most common starting words in spam
- Most common starting words in ham

## 📝 Requirements

```
pandas>=1.0.0
numpy>=1.18.0
matplotlib>=3.1.0
seaborn>=0.10.0
```

## 🛠️ Technologies Used

- **Python 3** - Programming language
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computing
- **Matplotlib** - Data visualization
- **Seaborn** - Statistical data visualization
- **Jupyter** - Interactive notebook environment

## 📌 Notes

- The dataset contains real email data with spam labels
- Text preprocessing is minimal to preserve original characteristics
- Analysis focuses on statistical patterns rather than content filtering
- Results can be used as baseline for machine learning models

## 👤 Author

- **Email:** raynerbruan5@gmail.com
- **Repository:** https://github.com/ghaythaouissaoui/projet-data-2

## 📄 License

This project is open source and available under the MIT License.

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report issues
- Suggest improvements
- Submit pull requests

## 📞 Contact

For questions or suggestions, please reach out via email or GitHub issues.
