from Src.Data.clear_data import *
from Src.Data.scrape import *
from Src.Data.validate_data import *
from Src.Data.plot_data import *
import os

"""
    input_file_path.json
    output_file_path.parquet
    report_file_path.json
"""
def validate_data(input_file_path, output_file_path, report_file_path):

    correct_input_path = input_file_path
    if input_file_path.endswith('.json'):
        conv = ConvertData(input_file_path,  output_file_path)
        converted_record = conv.convert()
        print("Converted record: ", converted_record)
        correct_input_path = output_file_path

    validate = ValidateData(correct_input_path, report_file_path)
    validate.run_all_check()

def plot_data(report_file_path, output_dir):
    plotter = PlotData(
        metrics_json_path=report_file_path,
        output_dir=output_dir
    )
    plotter.generate_all_plots()

if __name__ == '__main__':
    iss = "1680-7324"
    input_scraped_path = "Raw/scraped_data_final.json"
    output_scraped_data = "Raw/scraped_data.parquet"
    metrics_json_path = "Raw/report.json"

    if not os.path.isfile(input_scraped_path):
        s = Scraper(email=os.environ.get('API_EMAIL'), issn=iss, output_file_path=input_scraped_path)
        downloaded_paper = s.scrapedata(from_year=1900, to_year=2025, size_paper_batch=200)
        print(downloaded_paper)

    validate_data(input_scraped_path, output_scraped_data, metrics_json_path)
    plot_data(metrics_json_path, "plots_uncleaned")

    output_cleaned_data = "Raw/scraped_data_cleaned.parquet"
    output_cleaned_report = "Raw/cleaning_report.json"
    cleaner = CleanData(
        input_parquet_path=output_scraped_data,
        output_path=output_cleaned_data,
        report_path=output_cleaned_report
    )
    df_clean = cleaner.clean(
        remove_no_authors=True
    )
    print(f"Saved")

    output_cleaned_data_parquet = "Raw/data_cleaned.parquet"
    validate_data(output_cleaned_data,  output_cleaned_data_parquet, output_cleaned_report)
    plot_data(output_cleaned_report, "plots_cleaned")