import json

from transformers import AutoTokenizer
from adapters import AutoAdapterModel
import pandas as pd
import os
import pyarrow.parquet as pq
import re
import html
import torch
from pyarrow import Table
import torch.nn.functional as F

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
        self.model.active_adapters = self.ADAPTER_NAME
        self.model.eval()
        print(self.model.adapter_summary())
        print(f"Active adapter: {self.model.active_adapters}")
        weights = self.model.state_dict()['bert.encoder.layer.1.attention.self.query.weight'].detach().numpy()
        print(f"Mean: f{weights.mean().item()}")
        print(f"Std: f{weights.std().item()}")

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

    def load_data(self, output_file_name):
        assert output_file_name.endswith(".parquet"), "input incorrect format"
        parquet_file = pq.ParquetFile(self.input_parquet)
        output_file = os.path.join(self.output_dir, output_file_name)
        parquet_writer = None
        cont_batch = 0
        with torch.no_grad():
            for batch in parquet_file.iter_batches(self.batch_size):
                batch_df = batch.to_pandas()
                parsed_abs = Embedder._parse_abstract(batch_df["abstract"].tolist())
                titles =   Embedder._parse_abstract(batch_df["title"].tolist())
                ids = batch_df["id"].tolist()
                assert len(titles) == len(parsed_abs)
                complete_text = [titles[i]+ self.tokenizer.sep_token + parsed_abs[i] for i in range(0, len(parsed_abs))]

                inputs = self.tokenizer(complete_text, padding=True, truncation=True,
                                        return_tensors="pt", return_token_type_ids=False, max_length=512)

                if self.ADAPTER_NAME not in (self.model.active_adapters or []):
                    print(f"Not active adapters")
                    self.model.active_adapters = self.ADAPTER_NAME
                try:
                    output = self.model(**inputs)
                except Exception as e:
                    print(f"Error: {e}")
                    raise e

                embeddings_CLS = output.last_hidden_state[:, 0, :]
                embeddings= list(F.normalize(embeddings_CLS, p=2, dim=1).cpu().numpy())
                df = pd.DataFrame({
                    "id": ids,
                    "embedding":  embeddings
                })
                table = Table.from_pandas(df)
                if parquet_writer is None:
                    parquet_writer = pq.ParquetWriter(str(output_file), table.schema)
                parquet_writer.write_table(table)
                print(f"Batch: {cont_batch}, records {cont_batch*self.batch_size} ")
                cont_batch+=1

            if parquet_writer:
                parquet_writer.close()

    @staticmethod
    def __read_scope(input_file_path):
        with open(input_file_path) as f:
            ids = []
            title = []
            text = []
            fp = json.load(f)
            for key in fp:
                ids.append(key)
                title.append(fp[key]["title"])
                text.append(fp[key]["text"])
        return pd.DataFrame({
            'id': ids,
            'title': title,
            'text': text
        })

    def create_emb_scope(self, input_file_path ,output_file_name):
        assert output_file_name.endswith(".parquet"), "input incorrect format"
        df_scope = self.__read_scope(input_file_path)
        ids = []
        embeddings_ris = []
        with torch.no_grad():
            for index, row in df_scope.iterrows():
                parsed_text = Embedder._parse_abstract([row["text"]])
                complete_text = row["title"] + self.tokenizer.sep_token + parsed_text[0]
                inputs = self.tokenizer(complete_text, padding=True, truncation=True,
                                        return_tensors="pt", return_token_type_ids=False, max_length=512)

                if self.ADAPTER_NAME not in (self.model.active_adapters or []):
                    print(f"Not active adapters")
                    return

                output = self.model(**inputs)
                embeddings_CLS = output.last_hidden_state[:, 0, :]
                embeddings = list(F.normalize(embeddings_CLS, p=2, dim=1).cpu().numpy())
                ids.append(row["id"])
                embeddings_ris.append(embeddings[0])

        df_ris = pd.DataFrame({
            'id': ids,
            'embedding': embeddings_ris
        })
        output_file = os.path.join(self.output_dir, output_file_name)
        df_ris.to_parquet(str(output_file))
        return


if __name__ == "__main__":
    model = Embedder("../Data/Raw/scraped_data_cleaned.parquet", "Emb", 20)
    output_embeddings_path = "Emb/normalize_embedding.parquet"
    if not os.path.exists(output_embeddings_path):
        model.load_data("normalize_embedding.parquet")

    model.create_emb_scope("scope.json", "scope_embeddings.parquet")

    df = pd.read_parquet("Emb/normalize_embedding.parquet", engine='pyarrow')
    print(df)
    df = pd.read_parquet("Emb/scope_embeddings.parquet", engine='pyarrow')
    print(df)
