import streamlit as st
import math
from collections import Counter


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Decision Tree AI",
    page_icon="🌳",
    layout="wide"
)


# ============================================================
# ENTROPY
# ============================================================

def entropy(labels):

    total = len(labels)

    if total == 0:
        return 0

    counts = Counter(labels)

    ent = 0.0

    for count in counts.values():

        p = count / total

        ent -= p * math.log2(p)

    return ent


# ============================================================
# INFORMATION GAIN
# ============================================================

def information_gain(data, attribute, target):

    total = len(data)

    # Entropy before splitting
    labels = [
        row[target]
        for row in data
    ]

    total_entropy = entropy(labels)

    # Create subsets
    subsets = {}

    for row in data:

        value = row[attribute]

        if value not in subsets:
            subsets[value] = []

        subsets[value].append(row)

    weighted_entropy = 0.0

    details = []

    for value, subset in subsets.items():

        subset_labels = [
            row[target]
            for row in subset
        ]

        subset_entropy = entropy(
            subset_labels
        )

        weight = len(subset) / total

        weighted_entropy += (
            weight * subset_entropy
        )

        details.append({
            "Value": value,
            "Count": len(subset),
            "Distribution": str(
                Counter(subset_labels)
            ),
            "Entropy": round(
                subset_entropy, 3
            ),
            "Weight": round(
                weight, 3
            )
        })

    gain = (
        total_entropy
        - weighted_entropy
    )

    return (
        gain,
        total_entropy,
        weighted_entropy,
        details
    )


# ============================================================
# LOAN DATASET
# ============================================================

dataset = [

    {
        "Credit Rating": "Excellent",
        "Income Level": "High",
        "Employment": "Stable",
        "Loan": "Yes"
    },

    {
        "Credit Rating": "Excellent",
        "Income Level": "High",
        "Employment": "Stable",
        "Loan": "Yes"
    },

    {
        "Credit Rating": "Excellent",
        "Income Level": "Medium",
        "Employment": "Stable",
        "Loan": "Yes"
    },

    {
        "Credit Rating": "Excellent",
        "Income Level": "Low",
        "Employment": "Stable",
        "Loan": "No"
    },

    {
        "Credit Rating": "Good",
        "Income Level": "High",
        "Employment": "Stable",
        "Loan": "Yes"
    },

    {
        "Credit Rating": "Good",
        "Income Level": "Low",
        "Employment": "Unstable",
        "Loan": "No"
    },

    {
        "Credit Rating": "Good",
        "Income Level": "Medium",
        "Employment": "Stable",
        "Loan": "Yes"
    },

    {
        "Credit Rating": "Good",
        "Income Level": "Low",
        "Employment": "Unstable",
        "Loan": "No"
    },

    {
        "Credit Rating": "Poor",
        "Income Level": "High",
        "Employment": "Unstable",
        "Loan": "No"
    },

    {
        "Credit Rating": "Poor",
        "Income Level": "Low",
        "Employment": "Unstable",
        "Loan": "No"
    },

    {
        "Credit Rating": "Poor",
        "Income Level": "Medium",
        "Employment": "Stable",
        "Loan": "Yes"
    },

    {
        "Credit Rating": "Poor",
        "Income Level": "Low",
        "Employment": "Unstable",
        "Loan": "No"
    }
]


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🌳 Decision Tree AI")

st.sidebar.markdown(
    """
    ### Navigation

    Select an option below.
    """
)

page = st.sidebar.radio(
    "Go to",
    [
        "🏠 Home",
        "📊 Dataset",
        "🧮 Entropy",
        "📈 Information Gain",
        "🔮 Prediction",
        "📚 About"
    ]
)


# ============================================================
# HOME
# ============================================================

if page == "🏠 Home":

    st.title("🌳 Decision Tree AI")

    st.subheader(
        "Decision Tree Classification from Scratch"
    )

    st.write(
        """
        This application demonstrates how a Decision Tree
        AI model can learn from data using Entropy and
        Information Gain.
        """
    )

    st.divider()

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Training Records",
            len(dataset)
        )

    with col2:

        labels = [
            row["Loan"]
            for row in dataset
        ]

        st.metric(
            "Classes",
            len(set(labels))
        )

    with col3:

        st.metric(
            "Features",
            3
        )

    st.divider()

    st.subheader(
        "How the AI works"
    )

    st.markdown(
        """
        ```text
        Training Dataset
                ↓
             Entropy
                ↓
        Information Gain
                ↓
        Best Attribute
                ↓
        Decision Tree
                ↓
            Prediction
        ```
        """
    )

    st.info(
        "The model uses Information Gain to determine "
        "which feature is most useful for classification."
    )


# ============================================================
# DATASET
# ============================================================

elif page == "📊 Dataset":

    st.title("📊 Loan Approval Dataset")

    st.write(
        "Training data used by the Decision Tree."
    )

    st.dataframe(
        dataset,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    st.subheader("Loan Distribution")

    labels = [
        row["Loan"]
        for row in dataset
    ]

    distribution = Counter(labels)

    st.bar_chart(
        distribution
    )


# ============================================================
# ENTROPY
# ============================================================

elif page == "🧮 Entropy":

    st.title("🧮 Entropy")

    st.write(
        """
        Entropy measures the amount of uncertainty
        or impurity in a dataset.
        """
    )

    st.latex(
        r"""
        Entropy(S)
        =
        -\sum_i p_i \log_2(p_i)
        """
    )

    labels = [
        row["Loan"]
        for row in dataset
    ]

    counts = Counter(labels)

    total = len(labels)

    st.subheader(
        "Step 1 — Class Distribution"
    )

    st.write(
        f"Total records: **{total}**"
    )

    for label, count in counts.items():

        probability = count / total

        st.write(
            f"**{label}**: {count} records "
            f"→ Probability = {probability:.3f}"
        )

    result = entropy(labels)

    st.subheader(
        "Step 2 — Entropy"
    )

    st.success(
        f"Overall Entropy = {result:.3f}"
    )


# ============================================================
# INFORMATION GAIN
# ============================================================

elif page == "📈 Information Gain":

    st.title("📈 Information Gain")

    st.write(
        """
        Information Gain tells us how useful an attribute
        is for splitting the dataset.
        """
    )

    st.latex(
        r"""
        IG(S,A)
        =
        Entropy(S)
        -
        Weighted\ Entropy
        """
    )

    attributes = [
        "Credit Rating",
        "Income Level",
        "Employment"
    ]

    selected_attribute = st.selectbox(
        "Select an attribute",
        attributes
    )

    (
        gain,
        total_entropy,
        weighted_entropy,
        details
    ) = information_gain(
        dataset,
        selected_attribute,
        "Loan"
    )

    st.divider()

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Total Entropy",
            f"{total_entropy:.3f}"
        )

    with col2:

        st.metric(
            "Weighted Entropy",
            f"{weighted_entropy:.3f}"
        )

    with col3:

        st.metric(
            "Information Gain",
            f"{gain:.3f}"
        )

    st.divider()

    st.subheader(
        "Detailed Calculation"
    )

    st.dataframe(
        details,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    st.info(
        f"Information Gain for "
        f"'{selected_attribute}' = "
        f"{gain:.3f}"
    )


# ============================================================
# PREDICTION
# ============================================================

elif page == "🔮 Prediction":

    st.title("🔮 Loan Approval Prediction")

    st.write(
        "Enter customer information and ask the AI "
        "to make a prediction."
    )

    st.divider()

    col1, col2 = st.columns(2)

    with col1:

        credit_rating = st.selectbox(
            "Credit Rating",
            [
                "Excellent",
                "Good",
                "Poor"
            ]
        )

        income_level = st.selectbox(
            "Income Level",
            [
                "High",
                "Medium",
                "Low"
            ]
        )

    with col2:

        employment = st.selectbox(
            "Employment Status",
            [
                "Stable",
                "Unstable"
            ]
        )

        loan_amount = st.number_input(
            "Loan Amount",
            min_value=10000,
            max_value=10000000,
            value=500000,
            step=10000
        )

    st.divider()

    if st.button(
        "🚀 Predict Loan Approval",
        use_container_width=True
    ):

        # Simple demonstration prediction logic.
        #
        # Later this section will call the actual
        # Decision Tree model.

        if credit_rating == "Excellent":

            prediction = "Yes"

        elif credit_rating == "Poor":

            prediction = "No"

        else:

            if (
                income_level == "High"
                and employment == "Stable"
            ):
                prediction = "Yes"

            else:
                prediction = "No"

        st.divider()

        st.subheader(
            "AI Prediction"
        )

        if prediction == "Yes":

            st.success(
                "✅ LOAN APPROVED"
            )

        else:

            st.error(
                "❌ LOAN NOT APPROVED"
            )

        st.write(
            f"""
            **Credit Rating:** {credit_rating}

            **Income Level:** {income_level}

            **Employment:** {employment}

            **Loan Amount:** ₹{loan_amount:,.0f}
            """
        )

        st.divider()

        st.subheader(
            "Decision Path"
        )

        st.code(
            f"""
Credit Rating
      ↓
{credit_rating}
      ↓
AI Decision
      ↓
{prediction}
            """
        )


# ============================================================
# ABOUT
# ============================================================
elif page == "📚 About":

    st.title("📚 About the AI")

    st.markdown(
        """
        ### What is this, really?

        Underneath the sidebar and the metrics, this is a
        **from-scratch implementation of ID3**, the algorithm
        that taught computers how to ask good questions.

        No `scikit-learn`. No black box. Every entropy value
        and every information gain score on this screen is
        computed by roughly 40 lines of Python you can read
        top to bottom.
        """
    )

    st.divider()

    col1, col2 = st.columns(2)

    with col1:

        st.markdown(
            """
            #### 🎯 The Problem

            A bank has **12 past loan decisions** on file.
            Each one is described by three things about the
            applicant — Credit Rating, Income Level, and
            Employment Status — and one outcome: was the
            loan approved?

            The question this app answers is the same one
            a loan officer silently asks:

            *"Of everything I know about this person,
            what matters most?"*
            """
        )

    with col2:

        st.markdown(
            """
            #### 🧠 The Method

            1. Measure how **mixed up** the outcomes are
               (Entropy)
            2. Test each feature to see how much splitting
               on it **reduces that mess** (Information Gain)
            3. Crown the feature with the highest gain as
               the best question to ask first
            4. Repeat on each branch until the data sorts
               itself out
            """
        )

    st.divider()

    st.subheader("Why entropy, specifically?")

    st.write(
        """
        Because a coin flip and a certainty shouldn't be
        scored the same way. A group of loans that's 50%
        approved is maximally *confusing* — entropy of 1.0.
        A group that's 100% approved (or 100% denied) is
        maximally *obvious* — entropy of 0. Information Gain
        just asks: **which question shrinks that confusion
        the fastest?**
        """
    )

    st.latex(
        r"""
        IG(S, A) = Entropy(S) - \sum_{v \in Values(A)}
        \frac{|S_v|}{|S|} \, Entropy(S_v)
        """
    )

    st.divider()

    st.subheader("What's real vs. what's a placeholder")

    col1, col2 = st.columns(2)

    with col1:
        st.success(
            """
            **✅ Fully working**
            - Entropy calculation
            - Information Gain per attribute
            - Weighted subset breakdown
            - Live dataset explorer
            """
        )

    with col2:
        st.warning(
            """
            **🚧 Simplified for now**
            - The Prediction page uses a small hand-coded
              rule, not a real tree traversal — a true
              recursive ID3 build (with branching on the
              winning attribute at each level) is the
              natural next step
            """
        )

    st.divider()

    st.caption(
        "Built with Python, Streamlit, and the same math "
        "Ross Quinlan formalized as ID3 back in 1986."
    )

