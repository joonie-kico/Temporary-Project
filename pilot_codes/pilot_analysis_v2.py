import duckdb
import pandas as pd
import re
from bs4 import BeautifulSoup
from rapidfuzz.fuzz import ratio
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATASET_PATH = (
    "hf://datasets/edwarddgao/open-apply-jobs/data/**/*.parquet"
)

# -----------------------------
# Analysis window
# -----------------------------
WARMUP_START = "2026-07-15"
WARMUP_END = "2026-07-18"

EVAL_START = "2026-07-19"
EVAL_END = "2026-07-21"

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
    "artificial intelligence",
    "data engineer",
    "data scientist",
    "devops",
    "cloud engineer",
]

# Candidate thresholds
TITLE_THRESHOLD = 65
DESCRIPTION_THRESHOLD = 0.65

# Avoid comparing extremely distant postings in the pilot.
MAX_LOOKBACK_DAYS = 90


def html_to_text(html):
    if not html:
        return ""

    soup = BeautifulSoup(str(html), "html.parser")
    return " ".join(soup.stripped_strings)


def normalize_text(text):
    if not text:
        return ""

    text = str(text).lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-z0-9+#./ -]", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def is_tech_job(title):
    if not title:
        return False

    title = title.lower()

    return any(keyword in title for keyword in TECH_KEYWORDS)


def load_jobs():
    print("Reading Open-Apply snapshots...")

    query = f"""
        SELECT
            CAST(date AS VARCHAR) AS date,
            CAST(id AS VARCHAR) AS id,
            CAST(source_slug AS VARCHAR) AS company,
            CAST(title AS VARCHAR) AS title,
            CAST(description_html AS VARCHAR) AS description_html,
            CAST(posted_at AS VARCHAR) AS posted_at
        FROM read_parquet(
            '{DATASET_PATH}',
            hive_partitioning = 1
        )
        WHERE date BETWEEN '{WARMUP_START}' AND '{EVAL_END}'
        AND (
            lower(title) LIKE '%software%'
            OR lower(title) LIKE '%developer%'
            OR lower(title) LIKE '%backend%'
            OR lower(title) LIKE '%frontend%'
            OR lower(title) LIKE '%full stack%'
            OR lower(title) LIKE '%fullstack%'
            OR lower(title) LIKE '%machine learning%'
            OR lower(title) LIKE '%ml engineer%'
            OR lower(title) LIKE '%ai engineer%'
            OR lower(title) LIKE '%artificial intelligence%'
            OR lower(title) LIKE '%data engineer%'
            OR lower(title) LIKE '%data scientist%'
            OR lower(title) LIKE '%devops%'
            OR lower(title) LIKE '%cloud engineer%'
        )
    """

    con = duckdb.connect()

    df = con.execute(query).fetchdf()

    print(f"Raw snapshot rows: {len(df):,}")

    return df


def preprocess(df):
    print("Filtering tech jobs...")

    print(f"Tech snapshot rows: {len(df):,}")

    df["date"] = pd.to_datetime(df["date"])

    df["description"] = (
        df["description_html"]
        .fillna("")
        .apply(html_to_text)
    )

    df["title_normalized"] = (
        df["title"]
        .fillna("")
        .apply(normalize_text)
    )

    df["description_normalized"] = (
        df["description"]
        .fillna("")
        .apply(normalize_text)
    )

    return df


def show_basic_statistics(df):
    print("\n========== BASIC STATISTICS ==========")

    print(f"Snapshot rows:    {len(df):,}")
    print(f"Unique job IDs:   {df['id'].nunique():,}")
    print(f"Unique companies: {df['company'].nunique():,}")

    daily = (
        df.groupby("date")["id"]
        .nunique()
        .sort_index()
    )

    print("\nJobs per day:")
    print(daily)


def build_first_seen_table(df):
    """
    Collapse repeated snapshot appearances of the same job ID.

    One row = one unique job ID.
    """

    print("\nBuilding first-seen table...")

    df = df.sort_values("date")

    first_seen = (
        df.groupby("id", as_index=False)
        .first()
    )

    first_seen["first_seen_date"] = first_seen["date"]

    print(
        f"Unique jobs in full analysis window: "
        f"{len(first_seen):,}"
    )

    return first_seen


def split_warmup_and_evaluation(first_seen):
    warmup_start = pd.Timestamp(WARMUP_START)
    warmup_end = pd.Timestamp(WARMUP_END)

    eval_start = pd.Timestamp(EVAL_START)
    eval_end = pd.Timestamp(EVAL_END)

    warmup_jobs = first_seen[
        (first_seen["first_seen_date"] >= warmup_start)
        & (first_seen["first_seen_date"] <= warmup_end)
    ].copy()

    eval_jobs = first_seen[
        (first_seen["first_seen_date"] >= eval_start)
        & (first_seen["first_seen_date"] <= eval_end)
    ].copy()

    print("\n========== WINDOW SPLIT ==========")

    print(f"Warm-up jobs:    {len(warmup_jobs):,}")
    print(f"Evaluation jobs: {len(eval_jobs):,}")

    return warmup_jobs, eval_jobs


def compute_description_similarity(text1, text2):
    """
    TF-IDF cosine similarity for a single pair.

    Returns 0.0 if one of the descriptions is empty.
    """

    if not text1 or not text2:
        return 0.0

    try:
        vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=5000
        )

        matrix = vectorizer.fit_transform(
            [text1, text2]
        )

        similarity = cosine_similarity(
            matrix[0:1],
            matrix[1:2]
        )[0][0]

        return float(similarity)

    except ValueError:
        return 0.0


def find_recurrent_candidates(first_seen, eval_jobs):
    print("\n========== RECURRENT CANDIDATES ==========")

    candidates = []

    first_seen = first_seen.sort_values(
        "first_seen_date"
    )

    total_eval = len(eval_jobs)

    for index, (_, new_job) in enumerate(
        eval_jobs.iterrows(),
        start=1
    ):
        if index % 100 == 0:
            print(
                f"Processing {index:,}/{total_eval:,} "
                f"evaluation jobs..."
            )

        company = new_job["company"]

        if pd.isna(company):
            continue

        new_date = new_job["first_seen_date"]

        # Only older jobs from same company.
        history = first_seen[
            (first_seen["company"] == company)
            & (
                first_seen["first_seen_date"]
                < new_date
            )
        ].copy()

        if history.empty:
            continue

        # Limit historical search window.
        lower_bound = (
            new_date
            - pd.Timedelta(days=MAX_LOOKBACK_DAYS)
        )

        history = history[
            history["first_seen_date"]
            >= lower_bound
        ]

        if history.empty:
            continue

        for _, old_job in history.iterrows():

            old_title = old_job[
                "title_normalized"
            ]

            new_title = new_job[
                "title_normalized"
            ]

            title_similarity = ratio(
                old_title,
                new_title
            )

            # Cheap title filter first.
            if title_similarity < TITLE_THRESHOLD:
                continue

            description_similarity = (
                compute_description_similarity(
                    old_job[
                        "description_normalized"
                    ],
                    new_job[
                        "description_normalized"
                    ]
                )
            )

            # Keep pairs that show meaningful
            # description similarity.
            if (
                description_similarity
                < DESCRIPTION_THRESHOLD
            ):
                continue

            days_between = (
                new_date
                - old_job["first_seen_date"]
            ).days

            candidates.append(
                {
                    "company": company,

                    "old_id": old_job["id"],
                    "old_date":
                        old_job["first_seen_date"],
                    "old_title":
                        old_job["title"],

                    "new_id": new_job["id"],
                    "new_date":
                        new_job["first_seen_date"],
                    "new_title":
                        new_job["title"],

                    "days_between":
                        days_between,

                    "title_similarity":
                        round(
                            title_similarity,
                            2
                        ),

                    "description_similarity":
                        round(
                            description_similarity,
                            4
                        ),
                }
            )

    result = pd.DataFrame(candidates)

    print(
        f"\nPotential recurrent pairs: "
        f"{len(result):,}"
    )

    if not result.empty:
        result["combined_score"] = (
            0.35
            * (
                result["title_similarity"]
                / 100
            )
            + 0.65
            * result[
                "description_similarity"
            ]
        )

        result = result.sort_values(
            "combined_score",
            ascending=False
        )

        print("\nTop 30 candidates:\n")

        columns = [
            "company",
            "old_date",
            "old_title",
            "new_date",
            "new_title",
            "days_between",
            "title_similarity",
            "description_similarity",
            "combined_score",
        ]

        print(
            result[columns]
            .head(30)
            .to_string(index=False)
        )

    return result


def save_results(
    first_seen,
    warmup_jobs,
    eval_jobs,
    candidates
):
    print("\nSaving results...")

    first_seen.to_parquet(
        "all_unique_jobs.parquet",
        index=False
    )

    warmup_jobs.to_csv(
        "warmup_jobs.csv",
        index=False
    )

    eval_jobs.to_csv(
        "evaluation_jobs.csv",
        index=False
    )

    candidates.to_csv(
        "recurrent_candidates_v2.csv",
        index=False
    )

    print("Saved:")
    print("  all_unique_jobs.parquet")
    print("  warmup_jobs.csv")
    print("  evaluation_jobs.csv")
    print("  recurrent_candidates_v2.csv")


def main():
    df = load_jobs()

    df = preprocess(df)

    show_basic_statistics(df)

    first_seen = build_first_seen_table(df)

    warmup_jobs, eval_jobs = (
        split_warmup_and_evaluation(
            first_seen
        )
    )

    candidates = find_recurrent_candidates(
        first_seen,
        eval_jobs
    )

    save_results(
        first_seen,
        warmup_jobs,
        eval_jobs,
        candidates
    )


if __name__ == "__main__":
    main()