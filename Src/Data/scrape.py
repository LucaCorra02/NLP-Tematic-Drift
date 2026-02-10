import requests
import pandas as pd
import os
import json
from dotenv import load_dotenv

class Scraper:
    def __init__(self, email=os.environ.get('API_EMAIL'), issn = "1680-7324" , output_file_path="scraped_data.json", base_url="https://api.openalex.org/works"):
        load_dotenv() # TODO: spostare in main
        assert os.environ.get('API_EMAIL') is not None
        assert output_file_path is not None
        self.base_url = base_url
        self.userEmail = email
        self.issn = issn
        self.output_file_path = output_file_path

    def scrapedata(self, from_year, to_year, size_paper_batch = 200):
        assert from_year <= to_year
        assert 0 < size_paper_batch <= 200

        start_date = "{year}-01-01".format(year = from_year)
        end_date = "{year}-12-31".format(year=to_year)

        headers = {
            "User-Agent": f"mailto:{self.userEmail}",
            "From": self.userEmail
        }
        filters = (
            f"primary_location.source.issn:{self.issn},"
            "has_abstract:true,"
            "type:article,"
            f"from_publication_date:{start_date}," 
            f"to_publication_date:{end_date}"
        )
        cursor = "*"
        params = {
            "filter": filters,
            "per-page": size_paper_batch,
            "cursor": cursor,
            "select": "id,doi,title,publication_year,publication_date,language,abstract_inverted_index,cited_by_count,counts_by_year,"
                      "referenced_works,related_works,"
                      "concepts,keywords,primary_location,authorships"
        }
        downloaded_records = 0
        print("Starting to scrape")
        while True:
            try:
                print("scraped: ", downloaded_records)
                r = requests.get(self.base_url, params=params, headers=headers)
                if r.status_code != 200:
                    print(f"Request Error: {r.status_code}")
                    break
                data = r.json()
                results = data.get('results', [])
                if not results: break
                for paper in results:
                    abstract_txt = Scraper.reconstruct_abstract(paper['abstract_inverted_index'])
                    paper['abstract'] = abstract_txt
                    if 'abstract_inverted_index' in paper: del paper['abstract_inverted_index']
                    self.save_record(paper)
                    downloaded_records+=1

                cursor = data['meta']['next_cursor']
                params['cursor'] = cursor
                if not cursor: break

            except Exception as e:
                print(f"Error: {e}")

        return downloaded_records

    @staticmethod
    def reconstruct_abstract(abstract):
        text = []
        for key, value in abstract.items():
            for index in value:
                text.append((index, key))

        sorted_text = sorted(text, key=lambda x: x[0])
        return " ".join([pair[1] for pair in sorted_text])

    def save_record(self, paper_record):
        assert paper_record is not None
        with open(self.output_file_path, "a") as f:
            f.write(json.dumps(paper_record) + '\n')


iss = "1680-7324"
output_path = "Raw/scraped_data_final.json"
s = Scraper(email=os.environ.get('API_EMAIL'), issn=iss, output_file_path=output_path)
downloaded_paper = s.scrapedata(from_year=1900, to_year=2025, size_paper_batch = 200)
print(downloaded_paper)