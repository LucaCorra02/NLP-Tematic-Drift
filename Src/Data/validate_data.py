import pandas as pd


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
        return  df.isna().sum()

    def check_duplicate(self):
        assert self.dataframe is not None
        df = self.dataframe
        duplicate_dict = {}

        duplicate_ids = df[df.duplicated(subset=["id"], keep=False)]
        duplicate_dict["id"] = (0, [])
        if len(duplicate_ids) > 0:
            duplicate_dict["id"] = (len(duplicate_ids), duplicate_dict["id"].to_list())

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
        print( df["primary_location"].isna().sum())
        valid_issn = df["primary_location"].apply(check_issn_row)
        return {"invalid-issn": (valid_issn.sum() - len(df), df[~valid_issn]["id"].to_list())}

    def abstract_metrics(self):
        df = self.dataframe
        dict_abs = {"empyt_abs": 0, "abs_nan": df["abstract"].isna().sum(), "abs_len": {}, "num_abs": df["abstract"].notna().sum()}
        for value in df["abstract"]:
            parsed = value.strip().replace(" ","")
            if parsed == "":
                dict_abs["empyt_abs"]+=1
                continue
            if len(parsed) in dict_abs["abs_len"]:
                dict_abs["abs_len"][len(parsed)]+=1
            else:
                dict_abs["abs_len"][len(parsed)]=1

        total_with_abs = sum(dict_abs["abs_len"].values())
        assert len(df["abstract"]) - (dict_abs["empyt_abs"] + dict_abs["abs_nan"]) == total_with_abs

        return dict_abs, total_with_abs

    def cited_metrics(self):
        pass

    def language_metrics(self):
        pass


if __name__ == '__main__':
    #conv = ConvertData("Raw/scraped_data_final.json", "Raw/scraped_data.parquet")
    #converted_record = conv.convert()
    #print("Converted record: ", converted_record)
    validate = ValidateData("Raw/scraped_data.parquet","ciao")
    #validate.count_nan()
    duplicate_dict = validate.check_duplicate()
    print(duplicate_dict)
    print(validate.check_issn())
    print(validate.abstract_metrics())