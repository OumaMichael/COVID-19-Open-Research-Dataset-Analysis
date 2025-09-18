#!/usr/bin/env python3
"""
CORD-19: Basic Analysis & Streamlit-ready outputs
Copy-paste this into assignment.py or run in a Jupyter cell.
Adjust CSV path if your file location differs.

Requirements:
pip install pandas numpy matplotlib seaborn mlcroissant
(Optionally: pip install wordcloud)
"""

import os
import sys
import string
import warnings
from collections import Counter

import mlcroissant as mlc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display

warnings.filterwarnings("ignore", category=FutureWarning)
sns.set(style="whitegrid", palette="muted")

# ---------- CONFIG ----------
OUTPUT_DIR = "cord19_outputs"
SAMPLE_CLEANED_CSV = os.path.join(OUTPUT_DIR, "metadata_clean_sample.csv")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# Fetch the Croissant JSON-LD dataset using mlcroissant
croissant_dataset = mlc.Dataset('https://www.kaggle.com/datasets/allen-institute-for-ai/CORD-19-research-challenge/croissant/download')

# Check what record sets are in the dataset
record_sets = croissant_dataset.metadata.record_sets
print("Record sets available in the dataset:")
print(record_sets)

# Fetch the records and put them in a DataFrame
def fetch_croissant_data():
    record_set_df = pd.DataFrame(croissant_dataset.records(record_set=record_sets[0].uuid))
    return record_set_df


def initial_exploration(df):
    """Print head, tail, info, dtypes, shape and basic describe."""
    print("== Initial exploration ==")
    print("Shape:", df.shape)
    print("\nFirst 5 rows:")
    display(df.head())
    print("\nLast 5 rows:")
    display(df.tail())
    print("\nDataframe info:")
    df.info()
    print("\nData types:")
    print(df.dtypes)
    print("\nBasic describe (include='all') - transposed:")
    display(df.describe(include="all").T)
    print("\nNumber of unique values per column:")
    print(df.nunique())
    print("\nDuplicated rows present?:", df.duplicated().any())
    print("-" * 60)


def basic_cleaning_and_features(df):
    """Add derived columns: publish_time -> year, abstract_length, author_count"""
    print("== Cleaning & feature engineering ==")
    # Convert publish_time to datetime (coerce errors -> NaT)
    df["publish_time"] = pd.to_datetime(df["publish_time"], errors="coerce")

    # Extract year column
    df["year"] = df["publish_time"].dt.year

    # Compute abstract length (characters) - cast to str to avoid errors
    df["abstract_length"] = df["abstract"].astype(str).apply(len)

    # Author count: split by ';' when authors is present; treat nan gracefully
    def count_authors(x):
        if pd.isna(x):
            return 0
        try:
            # some entries might be empty string -> 0
            s = str(x).strip()
            if not s:
                return 0
            return len([a for a in s.split(";") if a.strip()])
        except Exception:
            return 0

    df["author_count"] = df["authors"].apply(count_authors)

    # Convert numeric-like columns to numeric types where possible
    for col in ["pubmed_id", "s2_id"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    print("Added columns: year, abstract_length, author_count")
    print("-" * 60)
    return df


def missing_value_analysis(df, output_dir=OUTPUT_DIR):
    """Compute missing values and plot missing percent bar chart."""
    print("== Missing value analysis ==")
    missing_values = df.isnull().sum().sort_values(ascending=False)
    missing_percent = (missing_values / len(df) * 100).round(3)
    missing_df = pd.DataFrame(
        {"Missing Values": missing_values, "Missing (%)": missing_percent}
    )
    display(missing_df.head(30))

    # Show columns with > 10% missing
    print("\nColumns with > 10% missing:")
    display(missing_df[missing_df["Missing (%)"] > 10])

    # Plot missing percentage
    plt.figure(figsize=(12, 6))
    missing_percent[missing_percent > 0].sort_values(ascending=False).plot(
        kind="bar", color="salmon"
    )
    plt.ylabel("Missing Value Percentage (%)")
    plt.title("Percentage of Missing Values by Column")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "missing_percent.png"), dpi=150)
    plt.show()
    print("-" * 60)


def outlier_and_numeric_summary(df, output_dir=OUTPUT_DIR):
    """Analyze numeric columns: skewness, std, IQR-based outliers; show boxplots."""
    print("== Numeric summary & outlier detection ==")
    numerical = df.select_dtypes(include=["number"]).columns.tolist()
    if not numerical:
        print("No numeric columns found.")
        return

    summary = df[numerical].describe().T
    display(summary)

    for col in numerical:
        try:
            ser = df[col].dropna()
            if ser.empty:
                print(f"{col}: no data")
                continue
            skew = ser.skew()
            std = ser.std()
            Q1 = ser.quantile(0.25)
            Q3 = ser.quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            outliers = ser[(ser < lower) | (ser > upper)]
            print(f"{col}: skew={skew:.3f}, std={std:.3f}, outliers={len(outliers)}")
        except Exception as e:
            print(f"Could not analyze {col}: {e}")

    # Plot boxplots for a few meaningful numeric cols if present
    for col in ["abstract_length", "author_count"]:
        if col in df.columns:
            plt.figure(figsize=(10, 4))
            sns.boxplot(x=df[col].dropna())
            plt.title(f"Boxplot of {col}")
            plt.xlabel(col)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"boxplot_{col}.png"), dpi=150)
            plt.show()
    print("-" * 60)


def correlation_and_heatmap(df, output_dir=OUTPUT_DIR):
    """Compute correlation matrix for numeric features and plot heatmap."""
    print("== Correlation matrix ==")
    numerical_cols = df.select_dtypes(include=["number"])
    if numerical_cols.shape[1] < 2:
        print("Not enough numeric columns for correlation.")
        return
    corr = numerical_cols.corr()
    plt.figure(figsize=(10, 6))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm", cbar_kws={"shrink": .8})
    plt.title("Correlation Matrix (numeric columns)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "correlation_heatmap.png"), dpi=150)
    plt.show()
    print("-" * 60)


def top_journals_plot(df, n=10, output_dir=OUTPUT_DIR):
    """Show top N journals by publication count."""
    print(f"== Top {n} journals by publication count ==")
    top_journals = df["journal"].value_counts().head(n)
    plt.figure(figsize=(10, 6))
    sns.barplot(y=top_journals.index, x=top_journals.values, palette="Blues_d")
    plt.title(f"Top {n} Journals by Number of Publications")
    plt.xlabel("Number of Publications")
    plt.ylabel("Journal")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"top_{n}_journals.png"), dpi=150)
    plt.show()
    print("-" * 60)


def publications_over_time(df, output_dir=OUTPUT_DIR):
    """Plot number of publications per year (filter to reasonable years)."""
    print("== Publications over time ==")
    df_filtered = df[df["year"].notnull()].copy()
    # Filter years to a reasonable range (e.g., 1900-2025) to remove bad dates
    df_filtered = df_filtered[(df_filtered["year"] >= 1900) & (df_filtered["year"] <= 2025)]
    yearly_counts = df_filtered["year"].value_counts().sort_index()
    plt.figure(figsize=(10, 5))
    plt.plot(yearly_counts.index, yearly_counts.values, marker="o")
    plt.title("Publications by Year")
    plt.xlabel("Year")
    plt.ylabel("Number of Publications")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "publications_by_year.png"), dpi=150)
    plt.show()
    print("-" * 60)


def abstract_length_histogram(df, output_dir=OUTPUT_DIR):
    """Histogram of abstract lengths with x-limit to exclude extreme outliers."""
    print("== Abstract length distribution ==")
    plt.figure(figsize=(10, 5))
    # Clip to 0-5000 to visualize main mass and avoid extreme tail
    plt.hist(df["abstract_length"].replace({np.nan: 0}), bins=100)
    plt.title("Distribution of Abstract Length (characters)")
    plt.xlabel("Abstract length (chars)")
    plt.ylabel("Count")
    plt.xlim(0, 5000)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "abstract_length_hist.png"), dpi=150)
    plt.show()
    print("-" * 60)


def source_and_license_charts(df, output_dir=OUTPUT_DIR):
    """Bar chart for source_x and pie chart for license distribution."""
    print("== Source and license distributions ==")
    # Top sources
    top_sources = df["source_x"].value_counts().head(20)
    plt.figure(figsize=(12, 6))
    sns.barplot(x=top_sources.index, y=top_sources.values, color="skyblue")
    plt.title("Top Publication Sources (source_x)")
    plt.xlabel("Source")
    plt.ylabel("Number of Publications")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "top_sources.png"), dpi=150)
    plt.show()

    # License pie chart
    license_counts = df["license"].value_counts().head(10)
    plt.figure(figsize=(8, 8))
    plt.pie(license_counts.values, labels=license_counts.index, autopct="%1.1f%%", startangle=140)
    plt.title("Top Licenses Distribution")
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "license_pie.png"), dpi=150)
    plt.show()
    print("-" * 60)


def title_word_frequency(df, top_n=30, output_dir=OUTPUT_DIR):
    """Compute word frequency from titles (basic tokenization, basic stopwords)."""
    print("== Title word frequency ==")
    titles = df["title"].dropna().astype(str).str.lower()
    # Basic stopwords (you can expand this list)
    stopwords = {
        "the", "and", "of", "in", "to", "a", "for", "on", "with", "is", "by", "an",
        "from", "study", "analysis", "case", "using", "novel", "review", "systematic"
    }
    translator = str.maketrans("", "", string.punctuation)

    counter = Counter()
    for t in titles:
        # remove punctuation, split whitespace
        cleaned = t.translate(translator)
        for w in cleaned.split():
            if not w or w.isnumeric() or w in stopwords:
                continue
            counter[w] += 1

    most_common = counter.most_common(top_n)
    if not most_common:
        print("No title tokens found.")
        return

    words, counts = zip(*most_common)
    plt.figure(figsize=(12, 6))
    sns.barplot(x=list(counts), y=list(words), palette="viridis")
    plt.title(f"Top {top_n} Words in Titles")
    plt.xlabel("Frequency")
    plt.ylabel("Word")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"title_top_{top_n}_words.png"), dpi=150)
    plt.show()
    print("-" * 60)


def save_clean_sample(df, path=SAMPLE_CLEANED_CSV, n=200000):
    """Save a sample of the cleaned dataframe to CSV (for sharing / Streamlit)."""
    try:
        # Save a sample (to limit size); if dataset smaller than n, save all
        sample_size = min(len(df), n)
        sample_df = df.head(sample_size)
        sample_df.to_csv(path, index=False)
        print(f"Saved cleaned sample (first {sample_size} rows) to: {path}")
    except Exception as e:
        print("Could not save sample CSV:", e)


def main():
    print("\nCORD-19 Basic Analysis Script\n")
    df = fetch_croissant_data()

    # Initial exploration
    initial_exploration(df)

    # Cleaning / Derived features
    df = basic_cleaning_and_features(df)

    # Missing value analysis
    missing_value_analysis(df)

    # Numeric summary & outliers
    outlier_and_numeric_summary(df)

    # Correlation heatmap
    correlation_and_heatmap(df)

    # Top journals
    top_journals_plot(df, n=10)

    # Publications over time
    publications_over_time(df)

    # Abstract length histogram
    abstract_length_histogram(df)

    # Source & license charts
    source_and_license_charts(df)

    # Title word frequency
    title_word_frequency(df, top_n=30)

    # Save a cleaned sample
    save_clean_sample(df)

    print("\nAnalysis complete. All plots saved to the folder:", OUTPUT_DIR)
    print("You can use the saved sample CSV for a Streamlit app or further analysis.")


if __name__ == "__main__":
    main()
