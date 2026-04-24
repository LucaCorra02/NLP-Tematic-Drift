import pandas as pd
from bertopic import BERTopic
import numpy as np
from tqdm import auto
from sklearn.feature_extraction.text import CountVectorizer
import html
import re
from umap import UMAP
from hdbscan import HDBSCAN


class TopicEvolution:
    def __init__(self, embeding_with_score_path, abstract_embedding_path):
        self.df_abstract = pd.read_parquet(abstract_embedding_path ,engine='pyarrow')
        self.df_score = pd.read_parquet(embeding_with_score_path, engine='pyarrow')
        self.df_merged = pd.merge(self.df_score, self.df_abstract[['id', 'abstract', 'publication_year']], on='id', how='left')
        assert self.df_merged.isna().sum().sum() == 0, "nan values"
        vectorizer_model = CountVectorizer(
            stop_words="english", min_df=2, ngram_range=(1, 2)
        )
        umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine', random_state=42)
        hdbscan_model = HDBSCAN(min_cluster_size=200, metric='euclidean', cluster_selection_method='eom',
                                prediction_data=True)
        self.model = BERTopic(
            language="english",
            verbose=True,
            top_n_words=10,
            vectorizer_model=vectorizer_model,
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
        )
        print(len(self.df_merged))

    @staticmethod
    def _parse_string(text: str):
        text = html.unescape(text)
        text = re.sub(r'\s*</?(?:sub|sup)[^>]*>\s+([^A-Z])', r' \1', text)
        text = re.sub(r'\s*</?(sub|sup)[^>]*>\s*', "", text, flags=re.IGNORECASE)
        text = re.sub(r'</?(i|b|em|strong)[^>]*>', "", text, flags=re.IGNORECASE)
        text = re.sub(r'<[^>]+>', "", text)
        text = re.sub(r'https?:\/\/.\S+', "", text)
        text = re.sub(r'www\.\S+', "", text)
        prefixes_to_remove = r'^(Abstract\.|Abstract|Summary\.|Background:|Introduction:|Methods:|Results:|Conclusion:|Conclusions:)\s*'
        text = re.sub(prefixes_to_remove, "", text, flags=re.IGNORECASE)
        text = re.sub(r'\s+', " ", text).strip()
        return text

    def evolution(self):
        df = self.df_merged
        abstract = [self._parse_string(abs) for abs in df["abstract"].tolist()]
        embeddings = np.vstack(df["embedding"].values)
        years = df["publication_year"].tolist()

        topics, probs = self.model.fit_transform(abstract, embeddings=embeddings)
        new_topics = self.model.reduce_outliers(abstract, topics)
        self.model.visualize_documents(abstract, embeddings=embeddings,
                                        hide_document_hover=True, hide_annotations=True)

        #topic_info = self.model.get_topic_info()
        #topics_over_time = self.model.topics_over_time(abstract, years, global_tuning=True)

if __name__ == "__main__":
    topicEvolution = TopicEvolution("../Similarity/similarity.parquet", "../../Data/Raw/scraped_data_cleaned.parquet")
    topicEvolution.evolution()
