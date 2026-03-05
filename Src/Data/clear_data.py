import pandas as pd
import numpy as np
from pathlib import Path
import json
from pyparsing import empty


class CleanData:
    def __init__(self, input_parquet_path, output_path, report_path=None):
        assert input_parquet_path.endswith(".parquet"), "incorrect input format"
        assert output_path.endswith(".parquet"), "incorrect output format"
        assert report_path.endswith(".json"), "incorrect report format"
        self.input_path = Path(input_parquet_path)
        self.output_path = Path(output_path)
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

    def clean(self, remove_no_authors=False):
        print(f"Initial dataset: {len(self.df):,} papers\n")

        self._remove_duplicates()
        self._remove_invalid_years()
        self._remove_bad_abstracts()
        self._remove_non_english()
        self._remove_invalid_issn()
        if remove_no_authors:
            self._remove_papers_without_authors()
        self._save_cleaned_data()
        self._generate_report()

        print(f"   Initial: {self.initial_count:,} papers")
        print(f"   Final:   {len(self.df):,} papers")
        print(f"   Removed: {self.initial_count - len(self.df):,} papers ({((self.initial_count - len(self.df)) / self.initial_count * 100):.2f}%)")
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

    """
        check authors and affiliation
    """
    def _remove_papers_without_authors(self):
        def get_author_info(authorships_list):
            if not isinstance(authorships_list, (list, np.ndarray)) or len(authorships_list) == 0:
                return []
            authors_info = []
            for author_data in authorships_list:
                if isinstance(author_data, dict):
                    name = author_data.get("raw_author_name")
                    if pd.notna(name) and name.strip() != "":
                        affiliations = author_data.get("affiliations")
                        first_aff = None
                        if isinstance(affiliations, (list, np.ndarray)) and len(affiliations) > 0:
                            for affiliation in affiliations:
                                aff_name = affiliation.get("raw_affiliation_string")
                                if pd.notna(aff_name) and aff_name != "":
                                    first_aff = aff_name
                                    break

                        authors_info.append({
                            "name": name.strip(),
                            "first_affiliation": first_aff
                        })
            return authors_info

        self.df["authors_details"] = self.df["authorships"].apply(get_author_info)
        mask_affiliazioni = self.df["authors_details"].apply(
            lambda lista_autori: any(autore.get("first_affiliation") is not None for autore in lista_autori)
        )
        self.df = self.df[mask_affiliazioni]
        print("after_authors_details:",len(self.df))

    def _save_cleaned_data(self):
        #self.df.to_json(self.output_path, orient="records", lines=True)
        self.df.to_parquet(self.output_path, index=False)

    def _generate_report(self):
        pass

if __name__ == '__main__':
    cleaner = CleanData(
        input_parquet_path="Raw/scraped_data.parquet",
        output_path="Raw/scraped_data_cleaned.parquet",
        report_path="Raw/cleaning_report.json"
    )

    df_clean = cleaner.clean(
        remove_no_authors=True
    )
    print(f"Saved")