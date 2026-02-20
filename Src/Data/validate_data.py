from statistics import stdev, mean
import pandas as pd
from collections import Counter
import json
import numpy as np

class ConvertData:
    def __init__(self, input_json_path, output_parquet_path):
        assert input_json_path and output_parquet_path is not None
        self.input_json_path = input_json_path
        self.output_parquet_path = output_parquet_path

    def convert(self):
        df = pd.read_json(self.input_json_path, lines=True)
        column_nan_count = df.isna().sum()
        df.to_parquet(self.output_parquet_path, index=False)
        return len(df)

class ValidateData:
    def __init__(self, input_parquet_path, output_metrics_path):
        assert input_parquet_path and output_metrics_path is not None
        self.input_parquet_path = input_parquet_path
        self.output_metrics_path = output_metrics_path
        self.dataframe = pd.read_parquet(self.input_parquet_path, engine='pyarrow')

    def count_nan(self):
        df = self.dataframe
        ris = df.isna().sum()
        nan_dict = {}
        for key, val in ris.items(): nan_dict[key] = val

        df = self.dataframe
        def count_nans_in_value(val):
            if isinstance(val, (list, np.ndarray)):
                return sum(count_nans_in_value(item) for item in val)
            elif isinstance(val, dict):
                return sum(count_nans_in_value(v) for v in val.values())
            else:
                try:
                    if pd.isna(val):
                        return 1
                    return 0
                except ValueError:
                    return 0

        nan_dict = {}
        for col in df.columns:
            tot_nans = df[col].apply(count_nans_in_value).sum()
            nan_dict[col] = int(tot_nans)
        return nan_dict

    def check_duplicate(self):
        assert self.dataframe is not None
        df = self.dataframe
        duplicate_dict = {}

        duplicate_ids = df[df.duplicated(subset=["id"])]
        duplicate_dict["id"] = (0, [])
        if len(duplicate_ids) > 0:
            duplicate_dict["id"] = (len(duplicate_ids), duplicate_ids["id"].to_list())

        df_with_doi = df[df["doi"].notna()]
        duplicate_dois = df_with_doi[df_with_doi.duplicated(subset=["doi"])]
        duplicate_dict["doi"] = (0, [])
        if len(duplicate_dois) > 0:
            duplicate_dict["doi"] = (len(duplicate_dois), duplicate_dois["doi"].to_list())

        df['title_lower'] = df['title'].str.lower().str.strip()
        duplicate_titles = df[df.duplicated(subset=["title_lower"])]
        duplicate_dict["title"] = (0, [])
        if len(duplicate_titles) > 0:
            duplicate_dict["title"] = (len(duplicate_titles), duplicate_titles["title"].to_list())
        return  duplicate_dict

    def check_issn(self, issn = '1680-7324'):
        def check_issn_row(row):
            if row is None or not isinstance(row, dict): return False
            get_source_field = row.get('source')
            if get_source_field is None or not isinstance(get_source_field, dict): return False

            issn_list = get_source_field.get('issn',[])
            if issn_list is None: return False

            if issn not in issn_list: return False
            return True

        df = self.dataframe
        valid_issn = df["primary_location"].apply(check_issn_row)
        return {"invalid-issn": (len(df) - valid_issn.sum(), df[~valid_issn]["id"].to_list())}

    def abstract_metrics(self, num_bins = 30):
        df = self.dataframe
        df["abstract_length"] = df["abstract"].str.strip().str.replace(" ","").str.len()
        dict_abs = {
            "empty_abs": (df["abstract_length"] == 0).sum(), "abs_nan": df["abstract"].isna().sum(),
            "abs_distribution": {}, "num_abs": df["abstract"].notna().sum(),
            "too_short": (df["abstract_length"] <= 100).sum(),  # <= 100 chars
            "too_long": (df["abstract_length"] > 5000).sum(),  # > 5000 chars
            "mean_length": df["abstract_length"].mean(),
            "median_length": df["abstract_length"].median(),
            "median_std": df["abstract_length"].std(),
            "min": df.loc[df["abstract_length"] > 0, "abstract_length"].min(),
            "max": df["abstract_length"].max()
        }

        min_val = dict_abs["min"]
        max_val = dict_abs["max"]
        bins_edges = np.floor(np.linspace(min_val, max_val, num_bins + 1)).astype(int)
        bins_edges = np.unique(bins_edges)

        cats = pd.cut(
            df.loc[df["abstract_length"] > 0, "abstract_length"],
            bins=bins_edges,
            include_lowest=True,
            right=True
        )
        first_bins = cats.cat.categories[0]
        cats = cats.cat.rename_categories({cats.cat.categories[0]: pd.Interval(dict_abs["min"], first_bins.right)})
        counts = cats.value_counts(sort=False)
        dict_abs["abs_distribution"] = {str(interval): int(count) for interval, count in counts.items()}
        assert len(df["abstract"]) - (dict_abs["empty_abs"] + dict_abs["abs_nan"]) ==  sum(dict_abs["abs_distribution"].values())
        return dict_abs

    def cited_metrics(self, num_bins = 30):
        df = self.dataframe
        cited_dict = {
            "cited_nan":df["cited_by_count"].isna().sum(),
            "counts_year_nan":df["counts_by_year"].isna().sum(),
            "cited_distribution":{},
            "cited_per_year":{},
            "cited_min": df["cited_by_count"].min(),
            "cited_max": df["cited_by_count"].max(),
            "cited_avg": df["cited_by_count"].mean(),
            "cited_median": df["cited_by_count"].median(),
            "cited_std": df["cited_by_count"].std(),
            "low_citation_count": 0, # citation < 10
            "high_citation_count": 0, #citation > 100
            "zero_citation_count": 0
        }

        min_val = cited_dict["cited_min"]
        max_val = cited_dict["cited_max"]
        bins_edges = np.floor(np.linspace(min_val, max_val, num_bins + 1)).astype(int)
        bins_edges = np.unique(bins_edges)
        cats = pd.cut(
            df["cited_by_count"],
            bins=bins_edges,
            include_lowest=True,
            right=True
        )
        first_bins = cats.cat.categories[0]
        cats = cats.cat.rename_categories({cats.cat.categories[0]: pd.Interval(cited_dict["cited_min"], first_bins.right)})
        counts = cats.value_counts(sort=False)
        cited_dict["cited_distribution"] = {str(interval): int(count) for interval, count in counts.items()}

        for list_citation in df["counts_by_year"]:
            for value in list_citation:
                if value["year"] in cited_dict["cited_per_year"]:
                    cited_dict["cited_per_year"][value["year"]] += value["cited_by_count"]
                else:
                    cited_dict["cited_per_year"][value["year"]] = value["cited_by_count"]

        cited_dict["low_citation_count"] = len(df[df["cited_by_count"] < 10])
        cited_dict["high_citation_count"] = len(df[df["cited_by_count"] > 100])
        cited_dict["zero_citation_count"] = len(df[df["cited_by_count"] == 0])
        assert len(df["cited_by_count"]) - cited_dict["cited_nan"] == sum(cited_dict["cited_distribution"].values())
        assert  sum(cited_dict["cited_per_year"].values()) <= df["cited_by_count"].sum()
        return  cited_dict


    def language_metrics(self):
        df = self.dataframe
        language_dict = {"language_nan":df["language"].isna().sum(), "language_distribution":{}}
        for language in df["language"]:
            if language in language_dict["language_distribution"]:
                language_dict["language_distribution"][language]+=1
            else:
                language_dict["language_distribution"][language]=1
        assert len(df["language"]) - language_dict["language_nan"]  == sum(language_dict["language_distribution"].values())
        return language_dict

    def type_metrics(self):
        df = self.dataframe
        dict_concept = {"nan_concept":df["concepts"].isna().sum(), "empty_concepts":0 , "concept_distribution":{}}

        counter = Counter()
        for concept_list in df["concepts"]: #only take the highest score of level1 (macro area)
            level1_concepts = [c for c in concept_list if c.get('level') == 1]
            if len(level1_concepts) == 0: #empty concept or not have level1 concept
                dict_concept["empty_concepts"]+=1
                continue
            if level1_concepts:
                best = max(level1_concepts, key=lambda x: x.get('score', 0))
                counter[best['display_name']] += 1

        dict_concept["concept_distribution"] = dict(counter)
        assert len(df["concepts"]) - (dict_concept["nan_concept"] + dict_concept["empty_concepts"]) == sum(dict_concept["concept_distribution"].values())
        return dict_concept

    def year_metrics(self):
        df = self.dataframe
        dict_year = {
            "papers_nan_year": df["publication_year"].isna().sum(),
            "missing_year":[],
            "year_distribution": {},
            "avg_paper_per_year":0,
            "std_paper_per_year":0,
        }
        counter = Counter(df["publication_year"].to_list())
        dict_year["year_distribution"] = dict(sorted(dict(counter).items()))
        year_list = list(dict_year["year_distribution"].keys()) # needs to be sorted
        missing_year = []
        for i in range(0, len(year_list)-1, 2):
            if year_list[i]+1 != year_list[i+1]:
                gap = year_list[i+1] - year_list[i]
                missing_year+= [year_list[i]+index for index in range(1,gap)]

        dict_year["missing_year"] = missing_year
        values = list(dict_year["year_distribution"].values())
        dict_year["avg_paper_per_year"] = mean(values)
        dict_year["std_paper_per_year"] = stdev(values)

        assert len(df["publication_year"]) - dict_year["papers_nan_year"] == sum(dict_year["year_distribution"].values())
        return  dict_year

    def authors_metrics(self):
        df = self.dataframe
        auth_dict = {"authorships_nan": df['authorships'].isna().sum(), "empty_authorships": 0,
                     "papers_without_valid_author": 0, "author_count_distribution": {}, "avg_authors_per_paper": 0.0,
                     "max_authors": 0, "papers_with_valid_author": 0}

        author_counts = []
        papers_without_valid = 0

        for authorship_list in df["authorships"]:
            if pd.isna(authorship_list).sum() > 0:
                papers_without_valid += 1
                author_counts.append(0)
                continue

            if len(authorship_list) == 0:
                auth_dict["empty_authorships"] += 1
                papers_without_valid += 1
                author_counts.append(0)
                continue

            valid_authors = 0
            for author_entry in authorship_list:
                author_info = author_entry.get('author')
                display_name = author_info.get('display_name')
                if display_name and display_name.strip():
                    valid_authors += 1

            if valid_authors == 0:
                papers_without_valid += 1

            author_counts.append(valid_authors)

        count_distribution = Counter(author_counts)
        auth_dict["author_count_distribution"] = dict(sorted(count_distribution.items()))
        auth_dict["papers_without_valid_author"] = papers_without_valid
        auth_dict["papers_with_valid_author"] = len(df) - papers_without_valid

        valid_counts = [c for c in author_counts if c > 0]
        if valid_counts:
            auth_dict["avg_authors_per_paper"] = sum(valid_counts) / len(valid_counts)
            auth_dict["max_authors"] = max(valid_counts)

        assert len(df["authorships"]) - (auth_dict["authorships_nan"] + auth_dict["empty_authorships"]) == sum(auth_dict["author_count_distribution"].values())
        return auth_dict


    def run_all_check(self):
        full_report = {}
        full_report.update(self.check_duplicate())
        full_report.update(self.check_issn())
        full_report["nan_fileds"] = self.count_nan()
        full_report["abstract_metrics"] = self.abstract_metrics()
        full_report["cited_metrics"] = self.cited_metrics()
        full_report["language_metrics"] = self.language_metrics()
        full_report["type_metrics"] = self.type_metrics()
        full_report["year_metrics"] = self.year_metrics()
        full_report["authors_metrics"] = self.authors_metrics()
        try:
            with open(self.output_metrics_path, 'w', encoding='utf-8') as file:
                json.dump(full_report, file, indent=4, cls=NpEncoder)
            print(f"Report salvato in: {self.output_metrics_path}")
        except Exception as e:
            print(f"Errore durante il salvataggio: {e}")

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


if __name__ == '__main__':
    #conv = ConvertData("Raw/scraped_data_final.json", "Raw/scraped_data.parquet")
    #converted_record = conv.convert()
    #print("Converted record: ", converted_record)
    validate = ValidateData("Raw/scraped_data.parquet", "Raw/report.json")
    validate.run_all_check()