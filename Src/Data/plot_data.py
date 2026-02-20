import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import numpy as np
from pathlib import Path
import re

class PlotData:
    def __init__(self, metrics_json_path, output_dir="plots"):
        with open(metrics_json_path, 'r') as f:
            self.metrics = json.load(f)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.total_paper = self.metrics["abstract_metrics"]["num_abs"]
        assert self.total_paper == sum(dict(self.metrics['year_metrics']['year_distribution']).values())
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 6)
        plt.rcParams['font.size'] = 10

    """
        Papers per year distribution + trend
    """
    def plot_papers_per_year(self):
        year_dit = self.metrics['year_metrics']['year_distribution']

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        years = list(year_dit.keys())
        counts = list(year_dit.values())
        ax1.bar(years, counts, color='steelblue', alpha=0.7, edgecolor='black')
        ax1.axhline(
            self.metrics['year_metrics']['avg_paper_per_year'],
            color='red', linestyle='--', linewidth=2,
            label=f'Mean: {self.metrics["year_metrics"]["avg_paper_per_year"]:.0f}'
        )
        ax1.set_xlabel('Year', fontsize=12)
        ax1.set_ylabel('Number of Papers', fontsize=12)
        ax1.set_title('Papers Published per Year', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)

        cumsum = np.cumsum(counts)
        window = 3
        rolling_avg = pd.Series(counts).rolling(window=window, center=True).mean()
        ax2_twin = ax2.twinx()
        ax2.plot(years, counts, marker='o', color='steelblue',
                 label='Papers per Year', linewidth=2)
        ax2.plot(years, rolling_avg, color='orange', linestyle='--',
                 linewidth=2, label=f'{window}-Year Rolling Avg')
        ax2_twin.plot(years, cumsum, color='green', linestyle='-',
                      linewidth=2, label='Cumulative', alpha=0.6)

        ax2.set_xlabel('Year', fontsize=12)
        ax2.set_ylabel('Papers per Year', fontsize=12)
        ax2_twin.set_ylabel('Cumulative Papers', fontsize=12, color='green')
        ax2.set_title('Publication Trends & Cumulative Growth', fontsize=14, fontweight='bold')

        lines1, labels1 = ax2.get_legend_handles_labels()
        lines2, labels2 = ax2_twin.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        ax2.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'papers_per_year.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Saved: papers_per_year.png")

    """
        Citations evolution over time
    """
    def plot_citations_evolution(self):
        cited_per_year = self.metrics['cited_metrics']['cited_per_year'] # Data from 2012 (API void)

        years = sorted(cited_per_year.keys())
        citations = [cited_per_year[str(y)] for y in years]
        fig, ax = plt.subplots(figsize=(14, 7))
        ax.bar(years, citations, color='blue', alpha=0.5, edgecolor='black')
        ax.set_xlabel('Year', fontsize=12)
        ax.set_ylabel('Total Citations Received', fontsize=12)
        ax.set_title('Citations Received per Year (Cumulative Impact)',
                     fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'citations_per_year.png', dpi=300, bbox_inches='tight')
        plt.close()

        print("Saved: citations_per_year.png")

    @staticmethod
    def __clean_label(label):
        nums = re.findall(r"\d+", label)
        if len(nums) >= 3:
            return f"{int(nums[0])}-{int(nums[2])}"
        return label

    """
        Distribution of citation
    """
    def plot_citation_distribution(self):
        cited_dist = self.metrics['cited_metrics']['cited_distribution']
        intervals = [PlotData.__clean_label(key) for key in cited_dist.keys()]
        counts = list(cited_dist.values())


        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))  # Aumentato larghezza per leggibilità

        bars = ax1.bar(intervals, counts, color='skyblue', edgecolor='black', alpha=0.7)
        ax1.set_yscale('log')
        ax1.set_title('Cited distribution (log scale)', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Citation interval')
        ax1.set_ylabel('Number of papers (Log Scale)')
        ax1.set_xticks(range(len(intervals)))
        ax1.set_xticklabels(intervals, rotation=45, ha='right', fontsize=8)
        ax1.grid(axis='y', linestyle='--', alpha=0.3)
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax1.text(
                    bar.get_x() + bar.get_width() / 2.,
                    height * 1.1,
                    f'{int(height)}',
                    ha='center', va='bottom', fontsize=8, fontweight='bold'
                )

        stats = self.metrics["cited_metrics"]
        metrics = {
            'Mean': stats["cited_avg"],
            'Median': stats["cited_median"],
            'Std Dev': stats["cited_std"],
            'Max': stats["cited_max"]
        }

        keys = list(metrics.keys())
        values = list(metrics.values())
        ax2.barh(keys, values,
                 color=['skyblue', 'lightcoral', 'lightgreen', 'gold'],
                 edgecolor='black', linewidth=1)

        ax2.set_xlabel('Value', fontsize=12)
        ax2.set_title('Citation Statistics Summary', fontsize=14, fontweight='bold')
        ax2.grid(axis='x', alpha=0.3, linestyle='--')
        max_val = max(values)
        for i, v in enumerate(values):
            ax2.text(
                v + (max_val * 0.02),
                i,
                f'{v:.1f}',
                va='center', fontweight='bold', fontsize=11
            )

        plt.tight_layout()
        output_path = self.output_dir / 'citation_distribution.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print("Saved: citation_distribution.png")

    """
        Abstract lenght distribution
    """
    def plot_abstract_distribution(self):
        abs_dist = self.metrics['abstract_metrics']['abs_distribution']

        intervals = [PlotData.__clean_label(key) for key in abs_dist.keys()]
        counts = [val for val in abs_dist.values()]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        x_pos = np.arange(len(intervals))
        ax1.bar(x_pos, counts, color='mediumpurple', alpha=0.7, edgecolor='black')
        ax1.set_xlabel('Abstract Length (interval)', fontsize=12)
        ax1.set_ylabel('Number of Papers', fontsize=12)
        ax1.set_title('Abstract Length Distribution', fontsize=12, fontweight='bold')
        ax1.set_xticks(x_pos[::3])
        ax1.set_xticklabels(
           [intervals[i] for i in range(0, len(intervals), 3)],
           rotation=45, ha='right',fontsize=9,
        )
        ax1.grid(axis='y', alpha=0.3)


        stats_data = {
            'Metric': ['Mean', 'Median', 'Std Dev', 'Min', 'Max'],
            'Value': [
                self.metrics['abstract_metrics']['mean_length'],
                self.metrics['abstract_metrics']['median_length'],
                self.metrics['abstract_metrics']['median_std'],
                self.metrics['abstract_metrics']['min'],
                self.metrics['abstract_metrics']['max']
            ]
        }
        ax2.barh(stats_data['Metric'], stats_data['Value'], color='skyblue', edgecolor='black')
        ax2.set_xlabel('Characters', fontsize=12)
        ax2.set_title('Abstract Length Statistics', fontsize=14, fontweight='bold')
        ax2.grid(axis='x', alpha=0.3)

        for i, v in enumerate(stats_data['Value']):
            ax2.text(v + 10, i, f'{v:.0f}', va='center', fontweight='bold')

        plt.tight_layout()
        plt.savefig(self.output_dir / 'abstract_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Saved: abstract_distribution.png")

    """
        TOP Level1 OpenAlex Ai concept
    """
    def plot_top_concepts(self, top_n=10):
        concept_dist = self.metrics['type_metrics']['concept_distribution']
        sorted_concepts = sorted(concept_dist.items(), key=lambda x: x[1], reverse=True)[:top_n]
        concepts, counts = zip(*sorted_concepts)

        fig, ax = plt.subplots(figsize=(12, 8))
        y_pos = np.arange(len(concepts))
        ax.barh(y_pos, counts, color=plt.cm.viridis(np.linspace(0, 1, len(concepts))),
                       edgecolor='black', linewidth=1)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(concepts, fontsize=11)
        ax.invert_yaxis()
        ax.set_xlabel('Number of Papers', fontsize=12)
        ax.set_title(f'Top {top_n} Research Concepts in ACP', fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)

        for i, v in enumerate(counts):
            ax.text(v + 50, i, f'{v:,}', va='center', fontweight='bold')

        plt.tight_layout()
        plt.savefig(self.output_dir / 'top_concepts.png', dpi=300, bbox_inches='tight')
        plt.close()

        print("Saved: top_concepts.png")

    """
        Pie: Chart: Concept Cover
    """
    def plot_concept_coverage(self):
        total_papers = self.total_paper
        empty_concepts = self.metrics['type_metrics']['empty_concepts']
        with_concepts = total_papers - empty_concepts

        fig, ax = plt.subplots(figsize=(8, 8))
        sizes = [with_concepts, empty_concepts]
        labels = [f'With Concepts\n({with_concepts:,})',
                  f'No Level-1 Concept\n({empty_concepts:,})']
        colors = ['lightgreen', 'lightcoral']
        explode = (0.05, 0.05)
        wedges, texts, autotexts = ax.pie(
            sizes, labels=labels, autopct='%1.1f%%',
            colors=colors, startangle=90, explode=explode,
            textprops={'fontsize': 11, 'fontweight': 'bold'}
        )

        for autotext in autotexts:
            autotext.set_fontsize(12)

        ax.set_title('Concept Coverage', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'concept_coverage.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Saved: concept_coverage.png")

    """
        Distribution Language: 
    """
    def plot_language_distribution(self):
        lang_dist = self.metrics['language_metrics']['language_distribution']

        fig, ax = plt.subplots(figsize=(10, 6))
        languages = list(lang_dist.keys())
        counts = list(lang_dist.values())
        colors = ['blue' if lang == 'en' else 'red' for lang in languages]

        ax.bar(languages, counts, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
        ax.set_xlabel('Language', fontsize=12)
        ax.set_ylabel('Number of Papers (log scale)', fontsize=12)
        ax.set_yscale('log')
        ax.set_title('Language Distribution',
                     fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3, which='both')

        for i, (lang, count) in enumerate(zip(languages, counts)):
            ax.text(i, count * 1.2, f'{count:,}', ha='center', fontweight='bold')

        plt.tight_layout()
        plt.savefig(self.output_dir / 'language_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()

        print("Saved: language_distribution.png")

    # TODO: Aggiunge abs quality (min, max, nan, empty)

    """
        Textual Report
    """
    def generate_text_report(self, output_path="plots/data_report.txt"):
        report = ["=" * 80, "ATMOSPHERIC CHEMISTRY & PHYSICS - DATA VALIDATION REPORT", "=" * 80, "",
                  "DATASET OVERVIEW", "-" * 80, f"Total Papers:          {self.total_paper:>10,}",
                  f"Date Range:            {min(self.metrics['year_metrics']['year_distribution'].keys())} - {max(self.metrics['year_metrics']['year_distribution'].keys())}",
                  f"Avg Papers/Year:       {self.metrics['year_metrics']['avg_paper_per_year']:>10,.0f}",
                  f"Std Papers/Year:       {self.metrics['year_metrics']['std_paper_per_year']:>10,.0f}", "",
                  "DATA QUALITY ISSUES", "-" * 80]

        # Duplicates
        total_dupes = self.metrics['id'][0] + self.metrics['doi'][0] + self.metrics['title'][0]
        report.append(f"Duplicates:")
        report.append(f"  - By ID:             {self.metrics['id'][0]:>10}")
        report.append(f"  - By DOI:            {self.metrics['doi'][0]:>10}")
        report.append(f"  - By Title:          {self.metrics['title'][0]:>10}")
        report.append(f"  TOTAL:               {total_dupes:>10}")
        report.append("")

        # Abstract Quality
        report.append(f"Abstract Quality:")
        report.append(f"  - Total Abstracts:   {self.metrics['abstract_metrics']['num_abs']:>10,}")
        report.append(f"  - Empty:             {self.metrics['abstract_metrics']['empty_abs']:>10}")
        report.append(f"  - Too Short (<100):  {self.metrics['abstract_metrics']['too_short']:>10}")
        report.append(f"  - Too Long (>5000):  {self.metrics['abstract_metrics']['too_long']:>10}")
        report.append(f"  - Mean Length:       {self.metrics['abstract_metrics']['mean_length']:>10,.0f} chars")
        report.append(f"  - Median Length:     {self.metrics['abstract_metrics']['median_length']:>10,.0f} chars")
        report.append("")

        # Language
        lang_dist = self.metrics['language_metrics']['language_distribution']
        non_english = sum(v for k, v in lang_dist.items() if k != 'en')
        report.append(f"Language Distribution:")
        for lang, count in lang_dist.items():
            report.append(f"  - {lang}: {count}")
        report.append(f"  Non-English Total:   {non_english:>10}")
        report.append("")

        # Year Outliers
        years_distribution = [key for key in self.metrics["year_metrics"]["year_distribution"].keys()]
        missing_years = self.metrics["year_metrics"]["missing_year"]
        report.append(f"Year Validity:")
        report.append(f"  - Years Range:       {years_distribution}")
        report.append(f"  - Missing Years:     {missing_years}")
        report.append("")

        # Citations
        report.append("CITATION STATISTICS")
        report.append("-" * 80)
        report.append(f"Mean Citations:        {self.metrics['cited_metrics']['cited_avg']:>10,.1f}")
        report.append(f"Median Citations:      {self.metrics['cited_metrics']['cited_median']:>10,.0f}")
        report.append(f"Std Dev:               {self.metrics['cited_metrics']['cited_std']:>10,.1f}")
        report.append(f"Min Citations:         {self.metrics['cited_metrics']['cited_min']:>10,}")
        report.append(f"Max Citations:         {self.metrics['cited_metrics']['cited_max']:>10,}")
        report.append("")

        # Concepts
        report.append("RESEARCH CONCEPTS")
        report.append("-" * 80)
        report.append(f"Papers with L1 Concepts:  {self.total_paper - self.metrics['type_metrics']['empty_concepts']}")
        report.append(f"Papers w/o L1 Concepts:   {self.metrics['type_metrics']['empty_concepts']}")
        report.append(f"\nTop 10 Concepts:")

        concept_dist = self.metrics['type_metrics']['concept_distribution']
        sorted_concepts = sorted(concept_dist.items(), key=lambda x: x[1], reverse=True)[:10]
        for i, (concept, count) in enumerate(sorted_concepts, 1):
            pct = (count / self.total_paper) * 100
            report.append(f"  {i}. {concept} {count} ({pct:>5.1f}%)")
        report.append("")

        report.append("=" * 80)
        output_path = Path(output_path)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))

        print(f"✅ Saved: {output_path}")

        # Print to console too
        print("\n" + '\n'.join(report))

    def generate_all_plots(self):
        print("GENERATING ALL VISUALIZATIONS")
        self.plot_papers_per_year()
        self.plot_citations_evolution()
        self.plot_abstract_distribution()
        self.plot_citation_distribution()
        self.plot_top_concepts()
        self.plot_concept_coverage()
        self.plot_language_distribution()
        self.generate_text_report()
        print("FINISHED")

if __name__ == '__main__':
    plotter = PlotData(
        metrics_json_path="Raw/report.json",
        output_dir="plots"
    )

    plotter.generate_all_plots()