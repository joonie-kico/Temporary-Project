import duckdb
import pandas as pd
from bs4 import BeautifulSoup
from rapidfuzz.fuzz import ratio

DATASET_PATH = (
    "hf://datasets/edwarddgao/open-apply-jobs/data/**/*.parquet"
)

START_DATE = "2026-07-20"
END_DATE = "2026-07-27"

TECH_KEYWORDS = [
    "software",
    "developer",
    "backend",
    "frontend",
    "full stack",
    "fullstack",
    "machine learning",
    "ml engineer",
    "ai engineer",
    "data engineer",
    "data scientist",
    "devops",
    "cloud engineer",
]


def html_to_text(html):
    if not html:
        return ""

    soup = BeautifulSoup(html, "html.parser")
    return " ".join(soup.stripped_strings)


def is_tech_job(title):
    if not title:
        return False

    title = title.lower()

    return any(keyword in title for keyword in TECH_KEYWORDS)


def load_jobs():
    print("Reading Open-Apply snapshots...")

    query = f"""
        SELECT
            date,
            id,
            source_slug AS company,
            title,
            description_html,
            locations,
            posted_at
        FROM read_parquet(
            '{DATASET_PATH}',
            hive_partitioning = 1
        )
        WHERE date BETWEEN '{START_DATE}' AND '{END_DATE}'
    """

    con = duckdb.connect()

    df = con.execute(query).fetchdf()

    print(f"Raw snapshot rows: {len(df):,}")

    return df


def preprocess(df):
    print("Filtering tech jobs...")

    df = df[df["title"].apply(is_tech_job)].copy()

    print(f"Tech snapshot rows: {len(df):,}")

    df["description"] = df["description_html"].apply(html_to_text)

    df["title_normalized"] = (
        df["title"]
        .fillna("")
        .str.lower()
        .str.replace(r"[^a-z0-9 ]", " ", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    return df


def basic_statistics(df):
    print("\n========== BASIC STATISTICS ==========")

    print(f"Snapshot rows:      {len(df):,}")
    print(f"Unique job IDs:     {df['id'].nunique():,}")
    print(f"Unique companies:   {df['company'].nunique():,}")
    print(f"Unique titles:      {df['title'].nunique():,}")

    print("\nJobs per snapshot:")

    daily = (
        df.groupby("date")["id"]
        .nunique()
        .sort_index()
    )

    print(daily)


def snapshot_diff(df):
    print("\n========== SNAPSHOT DIFF ==========")

    dates = sorted(df["date"].unique())

    results = []

    for i in range(1, len(dates)):
        previous_date = dates[i - 1]
        current_date = dates[i]

        previous_jobs = set(
            df[df["date"] == previous_date]["id"]
        )

        current_jobs = set(
            df[df["date"] == current_date]["id"]
        )

        added = current_jobs - previous_jobs
        removed = previous_jobs - current_jobs

        results.append(
            {
                "previous_date": previous_date,
                "current_date": current_date,
                "added": len(added),
                "removed": len(removed),
                "active": len(current_jobs),
            }
        )

    diff_df = pd.DataFrame(results)

    print(diff_df.to_string(index=False))

    return diff_df


def get_new_jobs(df):
    dates = sorted(df["date"].unique())

    all_seen = set()
    new_jobs = []

    for date in dates:
        daily = df[df["date"] == date]

        for _, row in daily.iterrows():
            if row["id"] not in all_seen:
                new_jobs.append(row)

        all_seen.update(daily["id"])

    result = pd.DataFrame(new_jobs)

    print(
        f"\nUnique jobs first observed during window: "
        f"{len(result):,}"
    )

    return result


def find_recurrent_candidates(new_jobs, threshold=85):
    """
    Simple baseline:
    Compare titles of postings from the same company.

    Later this will be replaced with:
    - BM25
    - embeddings
    - Elasticsearch hybrid retrieval
    """

    print("\n========== RECURRENT CANDIDATES ==========")

    candidates = []

    grouped = new_jobs.groupby("company")

    for company, jobs in grouped:
        if len(jobs) < 2:
            continue

        jobs = jobs.sort_values("date").reset_index(drop=True)

        for i in range(len(jobs)):
            for j in range(i):
                newer = jobs.iloc[i]
                older = jobs.iloc[j]

                similarity = ratio(
                    newer["title_normalized"],
                    older["title_normalized"],
                )

                if similarity >= threshold:
                    candidates.append(
                        {
                            "company": company,

                            "old_date": older["date"],
                            "old_id": older["id"],
                            "old_title": older["title"],

                            "new_date": newer["date"],
                            "new_id": newer["id"],
                            "new_title": newer["title"],

                            "title_similarity": similarity,
                        }
                    )

    result = pd.DataFrame(candidates)

    print(
        f"Potential recurrent pairs "
        f"(title similarity >= {threshold}): "
        f"{len(result):,}"
    )

    if not result.empty:
        result = result.sort_values(
            "title_similarity",
            ascending=False
        )

        print("\nTop candidates:\n")

        print(
            result[
                [
                    "company",
                    "old_date",
                    "old_title",
                    "new_date",
                    "new_title",
                    "title_similarity",
                ]
            ]
            .head(30)
            .to_string(index=False)
        )

    return result


def save_results(df, diff_df, new_jobs, candidates):
    print("\nSaving results...")

    df.to_parquet(
        "tech_jobs_snapshot.parquet",
        index=False
    )

    diff_df.to_csv(
        "snapshot_diff.csv",
        index=False
    )

    new_jobs.to_csv(
        "new_jobs.csv",
        index=False
    )

    candidates.to_csv(
        "recurrent_candidates.csv",
        index=False
    )

    print("Saved:")
    print("  tech_jobs_snapshot.parquet")
    print("  snapshot_diff.csv")
    print("  new_jobs.csv")
    print("  recurrent_candidates.csv")


def main():
    df = load_jobs()

    df = preprocess(df)

    basic_statistics(df)

    diff_df = snapshot_diff(df)

    new_jobs = get_new_jobs(df)

    candidates = find_recurrent_candidates(
        new_jobs,
        threshold=85
    )

    save_results(
        df,
        diff_df,
        new_jobs,
        candidates
    )


if __name__ == "__main__":
    main()