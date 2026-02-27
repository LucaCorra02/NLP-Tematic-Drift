from unittest.mock import inplace

import pandas as pd
import numpy as np
from pathlib import Path
import json

from pyparsing import empty


class CleanData:
    def __init__(self, input_parquet_path, output_parquet_path, report_path=None):
        self.input_path = Path(input_parquet_path)
        self.output_path = Path(output_parquet_path)
        self.report_path = Path(report_path) if report_path else None
        self.df = pd.read_parquet(self.input_path, engine='pyarrow')
        self.initial_count = len(self.df)
        self.removed_papers = {
            'duplicates': [],
            'invalid_year': [],
            'bad_abstract': [],
            'non_english': [],
            'invalid_issn': [],
            'no_authors': []
        }

    def clean(self, remove_issn_invalid=False, remove_no_authors=False):
        print(f"Initial dataset: {len(self.df):,} papers\n")

        # Step 1: Rimuovi duplicati
        self._remove_duplicates()

        # Step 2: Rimuovi anni invalidi
        self._remove_invalid_years()

        # Step 3: Rimuovi abstract problematici
        self._remove_bad_abstracts()

        # Step 4: Rimuovi non-English
        self._remove_non_english()

        # Step 5 (Opzionale): ISSN invalidi
        self._remove_invalid_issn()

        # Step 6 (Opzionale): Paper senza autori
        if remove_no_authors:
            self._remove_papers_without_authors()

        # Cleanup temporary columns
        self._cleanup_temp_columns()

        # Save
        self._save_cleaned_data()

        # Report
        self._generate_report()


        print(f"   Initial: {self.initial_count:,} papers")
        print(f"   Final:   {len(self.df):,} papers")
        print(
            f"   Removed: {self.initial_count - len(self.df):,} papers ({((self.initial_count - len(self.df)) / self.initial_count * 100):.2f}%)")
        return self.df

    def _remove_duplicates(self):
        id_dupes = self.df[self.df.duplicated(subset=['id'], keep='first')]
        if len(id_dupes) > 0:
            self.df.drop_duplicates(subset=['id'], keep='first', inplace=True)
        print("after_id:",len(self.df))

        if self.df["doi"].isna().sum() > 0:
            self.df.dropna(subset=["doi"], inplace=True)
        doi_dupes = self.df[self.df.duplicated(subset=['doi'], keep='first')]
        if len(doi_dupes) > 0:
            self.df.drop_duplicates(subset=['doi'], keep='first', inplace=True)
        print("after_doi:", len(self.df))

        tilte_dupes = self.df[self.df.duplicated(subset=['title'], keep='first')]
        if len(tilte_dupes) > 0:
            self.df.drop_duplicates(subset=['title'], keep='first', inplace=True)
        print("after_title:", len(self.df))

    def _remove_invalid_years(self, min_year=2008, max_year=2025):
        if self.df['publication_year'].isna().sum() > 0:
            self.df.dropna(subset=['publication_year'], inplace=True)

        valid_record = self.df['publication_year'].between(min_year, max_year, inclusive='both')
        self.df = self.df[valid_record]
        year_gap = np.arange(min_year, max_year + 1).tolist()
        assert year_gap == sorted(self.df['publication_year'].unique())
        print("after_year:", len(self.df))

    def _remove_bad_abstracts(self, min_length=100, max_length=5000):
        if self.df["abstract"].isna().sum() > 0:
            self.df.dropna(subset=["abstract"], inplace=True)

        self.df["abstract_len"] = self.df["abstract"].str.strip().str.replace(" ","").str.len()
        empty_abs = self.df[self.df["abstract_len"] == 0]
        if len(empty_abs) > 0:
            self.df.drop(empty_abs.index, inplace=True)

        low_char = self.df[self.df["abstract_len"] < min_length]
        if len(low_char) > 0:
            self.df.drop(low_char.index, inplace=True)

        print("after_abstract:", len(self.df))


    def _remove_non_english(self):
        if self.df["language"].isna().sum() > 0:
            self.df.dropna(subset=["language"], inplace=True)

        language_filter = self.df[self.df["language"] != "en"]
        if len(language_filter) > 0:
            self.df.drop(language_filter.index, inplace=True)
        print("after_language:", len(self.df))

    def _remove_invalid_issn(self, target_issn='1680-7324'):
        source_filter = self.df[self.df["primary_location"].str.get("source").isna()]
        if len(source_filter) > 0:
            self.df.drop(source_filter.index, inplace=True)

        def filter_valid_issn(loc_dict):
            if not isinstance(loc_dict, dict):
                return False
            source = loc_dict.get("source")
            if not isinstance(source, dict) or not source:
                return False
            issn = source.get("issn")
            if issn is None or len(issn) == 0:
                return False
            if target_issn not in issn:
                return False
            return True

        filter_mask = self.df["primary_location"].apply(filter_valid_issn)
        self.df = self.df[filter_mask]
        print("after_issn:", len(self.df))


    def _remove_papers_without_authors(self):
        pass
    def _cleanup_temp_columns(self):
        pass

    def _save_cleaned_data(self):
        pass

    def _generate_report(self):
        pass

if __name__ == '__main__':
    cleaner = CleanData(
        input_parquet_path="Raw/scraped_data.parquet",
        output_parquet_path="Raw/scraped_data_cleaned.parquet",
        report_path="Raw/cleaning_report.json"
    )

    df_clean = cleaner.clean(
        remove_issn_invalid=False,
        remove_no_authors=False
    )

    print(f"\n✅ Final dataset ready: {len(df_clean):,} papers")
    print(f"   Saved to: Raw/scraped_data_cleaned.parquet")