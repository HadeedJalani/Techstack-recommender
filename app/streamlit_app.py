import streamlit as st

from .config import DATA_PATH, DEFAULT_TOP_N
from .data_loader import load_tech_stacks
from .recommender import TechStackRecommender


@st.cache_resource
def get_recommender():
    return TechStackRecommender(load_tech_stacks(DATA_PATH))


st.set_page_config(
    page_title="Tech Stack Recommender",
    page_icon="🧠",
    layout="centered",
)

st.title("🧠 Tech Stack Recommender")
st.caption(
    "Content-based filtering using TF-IDF weighting and cosine similarity"
)

recommender = get_recommender()

role = st.text_input(
    "1. Target role / domain",
    placeholder="e.g. data science",
)
needs = st.text_input(
    "2. Project needs",
    placeholder="e.g. analytics dashboard, API",
)
interests = st.text_input(
    "3. Skills / interests",
    placeholder="e.g. Python, SQL, cloud",
)
extra = st.text_input(
    "4. Optional extra interests",
    placeholder="e.g. automation",
)
top_n = st.slider("Number of recommendations", 1, 5, DEFAULT_TOP_N)

if st.button("Recommend tech stacks", type="primary"):
    inputs = [role, needs, interests]
    if extra.strip():
        inputs.append(extra)

    if any(not value.strip() for value in inputs[:3]):
        st.error("Please provide all three required inputs.")
    else:
        results = recommender.recommend(inputs, top_n=top_n)
        st.subheader("Top recommendations")

        for item in results:
            st.markdown(
                f"### {item.rank}. {item.name} — {item.score:.1%} match"
            )
            st.write(item.description)

            c1, c2 = st.columns(2)
            c1.write(f"**Frontend:** {item.frontend}")
            c1.write(f"**Backend:** {item.backend}")
            c2.write(f"**Database:** {item.database}")
            c2.write(f"**Deployment:** {item.deployment}")
            st.divider()

        if results and results[0].fallback:
            st.info(
                "Cold-start fallback was used: no input terms matched the "
                "catalog vocabulary, so results were ranked by global popularity."
            )
