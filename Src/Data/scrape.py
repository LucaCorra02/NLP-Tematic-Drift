import requests
import pandas as pd
import time
import os

class Scraper:
    def __init__(self, email=os.environ.get('API_EMAIL'), issn = "1680-7324", base_url="https://api.openalex.org/works"):
        if os.environ.get('API_EMAIL') == 'None': print("Need Email")
        self.base_url = base_url
        self.userEmail = email
        self.issn = issn

    def scrapedata(self, from_year, to_year, max_papers = 10):
        assert from_year <= to_year
        assert max_papers > 0

        start_date = "{year}-01-01".format(year = from_year)
        end_date = "{year}-12-31".format(year=to_year)

        headers = {"User-Agent": self.userEmail}
        filters = {
            f"primary_location.source.issn:{self.issn},"
            "has_abstract:true,"
            "type:article,"
            f"from_publication_date:{start_date}," 
            f"to_publication_date:{end_date}"
        }
        params = {
            "filter": filters,
            "per-page": max_papers,
            "cursor": "*",
            "select": "id,doi,title,publication_year,abstract_inverted_index,cited_by_count"
        }

        try:
            r = requests.get(self.base_url, params=params, headers=headers)
            if r.status_code != 200:
                print(f"Request Error: {r.status_code}")
                return None

            data = r.json()
            results = data.get('results', [])
            return results

        except Exception as e:
            print(f"Error: {e}")


iss = "1680-7324"
s = Scraper(os.environ.get('API_EMAIL'), iss)
ris = Scraper.scrapedata(s,2023,2023)
print(ris)