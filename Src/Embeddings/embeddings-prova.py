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

    def load_data(self):
        parquet_file = pq.ParquetFile(self.input_parquet)
        output_file = os.path.join(self.output_dir, "normalize_embedding.parquet")
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
                print("Batch: ", cont_batch * self.batch_size)
                cont_batch+=1

            if parquet_writer:
                parquet_writer.close()


if __name__ == "__main__":
    model = Embedder("../Data/Raw/scraped_data_cleaned.parquet", "Emb", 20)
    model.load_data()
    #df = pd.read_parquet("Emb/dataset_streaming.parquet", engine='pyarrow')
    #print(df)
