from transformers import AutoTokenizer
from adapters import AutoAdapterModel
import pandas as pd
import os
import pyarrow.parquet as pq
import regex as re
import html

class Embedder:
    BASE_MODEL   = "allenai/specter2_base"
    ADAPTER_TASK  = "allenai/specter2"
    ADAPTER_NAME = "proximity"

    def __init__(self, input_parquet, output_dir, batch_size):
        assert input_parquet.endswith(".parquet"), "input incorrect format"

        self.input_parquet = input_parquet
        self.output_dir    = output_dir
        self.batch_size    = batch_size
        os.makedirs(self.output_dir, exist_ok=True)

        """
        print(f"Load tokenizer {self.BASE_MODEL}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.BASE_MODEL)
        print(f"Load Model {self.BASE_MODEL}")
        self.model = AutoAdapterModel.from_pretrained(self.BASE_MODEL)
        self.model.load_adapter(
            self.ADAPTER_TASK,
            source="hf",
            load_as=self.ADAPTER_NAME,
            set_active=True
        )
        self.model.eval()
        self.model.active_adapters = self.ADAPTER_NAME
        """

    @staticmethod
    def _parse_abstract(batch_abs:list[str]):
        def parse_string(text:str):
            text = html.unescape(text)
            text = re.sub(r'\s*</?(?:sub|sup)[^>]*>\s+([^A-Z])', r' \1', text)
            text = re.sub(r'\s*</?(sub|sup)[^>]*>\s*', "", text, flags=re.IGNORECASE)
            text = re.sub(r'</?(i|b|em|strong)[^>]*>', "", text, flags=re.IGNORECASE)
            text = re.sub(r'<[^>]+>', "", text)
            text = re.sub(r'https?:\/\/.\S+', "", text)
            text = re.sub(r'www\.\S+', "", text)
            prefixes_to_remove = r'^(Abstract\.|Abstract|Summary\.|Background:|Introduction:|Methods:|Results:|Conclusion:|Conclusions:)\s*'
            text = re.sub(prefixes_to_remove, "", text, flags=re.IGNORECASE)
            text = re.sub(r'\s+'," ",text).strip()
            return text
        return [parse_string(text) for text in batch_abs]

    def load_data(self):
        parquet_file = pq.ParquetFile(self.input_parquet)
        for batch in parquet_file.iter_batches(self.batch_size):
            batch_df = batch.to_pandas()
            parsed_abs = Embedder._parse_abstract(batch_df["abstract"].tolist())
            titles =   Embedder._parse_abstract(batch_df["title"].tolist())
            assert len(titles) == len(parsed_abs)
            complete_text = [titles[i]+". "+parsed_abs[i] for i in range(0, len(parsed_abs))]


if __name__ == "__main__":
    model = Embedder("../Data/Raw/scraped_data_cleaned.parquet", "Emb", 100)
    model.load_data()
