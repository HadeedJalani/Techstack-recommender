from .config import DATA_PATH, DEFAULT_TOP_N
from .data_loader import load_tech_stacks
from .recommender import TechStackRecommender


def run_cli() -> None:
    catalog = load_tech_stacks(DATA_PATH)
    recommender = TechStackRecommender(catalog)

    print("\n=== CAPSTONE: TECH STACK RECOMMENDER ===")
    print("Content-based filtering | TF-IDF + cosine similarity\n")
    print("Enter at least three preferences.\n")

    inputs = []
    prompts = [
        "1) Target role/domain (e.g. data science, web development): ",
        "2) Project needs (e.g. dashboard, API, automation): ",
        "3) Skills/interests (e.g. Python, SQL, cloud): ",
    ]

    for prompt in prompts:
        value = input(prompt).strip()
        while not value:
            print("This input is required.")
            value = input(prompt).strip()
        inputs.append(value)

    extra = input("4) Optional extra interests: ").strip()
    if extra:
        inputs.append(extra)

    results = recommender.recommend(inputs, top_n=DEFAULT_TOP_N)

    print("\n--- TOP-N RECOMMENDATIONS ---")

    for item in results:
        print(f"\n#{item.rank} {item.name} | Match: {item.score:.2%}")
        print(f"   Category:   {item.category}")
        print(f"   Frontend:   {item.frontend}")
        print(f"   Backend:    {item.backend}")
        print(f"   Database:   {item.database}")
        print(f"   Deployment: {item.deployment}")
        print(f"   Why:        {item.description}")

    if results and results[0].fallback:
        print(
            "\nCold-start fallback: no input terms matched the catalog "
            "vocabulary; results were ranked by global popularity."
        )


if __name__ == "__main__":
    run_cli()
