import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="ClearCheck Approval Dashboard", layout="wide")
st.title("ClearCheck Technologies — Approval Behavior Dashboard")
st.caption("Technician approval timing, fast approvals, and session patterns.")

COLOR_MAP = {"Arnold": "red", "Mendez": "blue", "Shawn": "green"}

# ---------------- LOAD DATA (ONLY ONCE) ----------------
@st.cache_data
def load_data():
    df = pd.read_csv("data_full.csv")
    df["APPROVAL_DATE"] = pd.to_datetime(df["APPROVAL_DATE"], errors="coerce")
    df = df[df["APPROVAL_DATE"].notna()].copy()

    # keep only valid durations (your CSV already has DURATION_SEC)
    df = df[df["DURATION_SEC"].notna() & (df["DURATION_SEC"] >= 0)].copy()

    # time-based features
    df["DATE"] = df["APPROVAL_DATE"].dt.date
    df["WEEKDAY"] = df["APPROVAL_DATE"].dt.day_name()
    df["MONTHNAME"] = df["APPROVAL_DATE"].dt.month_name()
    df["HOUR_OF_DAY"] = df["APPROVAL_DATE"].dt.hour

    return df

df = load_data()

# ---------------- Filters ----------------
with st.sidebar:
    tech = st.selectbox("Technician", ["Arnold", "Mendez", "Shawn"])
    min_d = df["APPROVAL_DATE"].min().date()
    max_d = df["APPROVAL_DATE"].max().date()
    date_range = st.date_input("Date Range", (min_d, max_d))
    fast_threshold = st.slider("Fast Approval Threshold (sec)", 1, 60, 10)
    clip_sec = st.slider("Histogram Clip (sec)", 60, 1800, 600)
    bins_n = st.slider("Histogram Bins", 20, 120, 50)

start = pd.to_datetime(date_range[0])
end = pd.to_datetime(date_range[1]) + pd.Timedelta(days=1)

# Filtered dataset (selected tech + date range)
d = df[(df["TECHNICIAN"] == tech) &
       (df["APPROVAL_DATE"] >= start) &
       (df["APPROVAL_DATE"] < end)].copy()

# Add derived fields
d["DURATION_MIN"] = d["DURATION_SEC"] / 60
d["FAST"] = d["DURATION_SEC"] < fast_threshold
d["SAME_SECOND"] = d["DURATION_SEC"] == 0
d["SAME_MINUTE"] = d["DURATION_SEC"] < 60

# ---------------- KPIs ----------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Approvals", f"{len(d):,}")
c2.metric("Median (sec)", round(d["DURATION_SEC"].median(), 2) if len(d) > 0 else "N/A")
c3.metric("% Fast", round(d["FAST"].mean() * 100, 2) if len(d) > 0 else "N/A")
c4.metric("% Same Minute", round(d["SAME_MINUTE"].mean() * 100, 2) if len(d) > 0 else "N/A")

st.divider()

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Histogram",
    "Fast Approvals",
    "Blocks / Sessions",
    "Time Patterns",
    "High-Risk Fast Approvals",
    "Policy Change"
])

# ---------------- TAB 1 Histogram ----------------
with tab1:
    st.subheader("Approval Duration Histogram")

    if len(d) == 0:
        st.warning("No data available for the selected filters.")
    else:
        d_plot = d.copy()
        d_plot["DUR_CLIPPED"] = d_plot["DURATION_SEC"].clip(upper=clip_sec)
        bins = np.linspace(0, clip_sec, bins_n)

        fig = plt.figure(figsize=(10, 4))
        counts, bin_edges, _ = plt.hist(
            d_plot["DUR_CLIPPED"],
            bins=bins,
            alpha=0.85,
            edgecolor="black",
            color=COLOR_MAP.get(tech, "gray")
        )

        plt.yscale("log")

        # Peak label
        if len(counts) > 0:
            max_count = counts.max()
            max_index = counts.argmax()
            x_peak = (bin_edges[max_index] + bin_edges[max_index + 1]) / 2

            plt.annotate(
                f"Peak: {int(max_count)} approvals",
                xy=(x_peak, max_count),
                xytext=(x_peak, max_count * 2),
                arrowprops=dict(arrowstyle="->", lw=1.5),
                ha="center",
                fontsize=10,
                weight="bold"
            )

        plt.xlabel("Approval Duration (seconds, clipped)")
        plt.ylabel("Number of Approvals (log scale)")
        plt.title(f"{tech} — Approval Duration Histogram")
        plt.grid(alpha=0.2)
        plt.tight_layout()
        st.pyplot(fig)

# ---------------- TAB 2 Fast Approvals ----------------
with tab2:
    st.subheader("Fast Approval Rate (Selected Technician)")
    st.caption("Higher percentages at low thresholds may indicate rushed approvals.")

    if len(d) == 0:
        st.warning("No data available.")
    else:
        thresholds = [2, 5, 10, 30, 60]
        rates = [(d["DURATION_SEC"] < t).mean() * 100 for t in thresholds]

        fig = plt.figure(figsize=(6, 4))
        plt.plot(thresholds, rates, marker="o", linewidth=3, color=COLOR_MAP.get(tech, "gray"))
        plt.xlabel("Threshold (seconds)")
        plt.ylabel("% Fast Approvals")
        plt.title(f"{tech} — Fast Approval Rate")
        plt.ylim(0, 100)
        plt.grid(alpha=0.25)
        plt.tight_layout()
        st.pyplot(fig)

# ---------------- TAB 3 Blocks / Sessions ----------------
with tab3:
    st.subheader("Block / Session Analysis")

    if len(d) == 0:
        st.warning("No approvals in this filter window.")
    else:
        d_block = d.sort_values("APPROVAL_DATE").copy()

        # Define block by >= 10 minute gap between approvals
        d_block["GAP_MIN"] = d_block["DURATION_SEC"] / 60
        d_block["NEW_BLOCK"] = d_block["GAP_MIN"] >= 10
        d_block["BLOCK_ID"] = d_block["NEW_BLOCK"].cumsum()

        block = d_block.groupby("BLOCK_ID").agg(
            cases=("DURATION_SEC", "count"),
            avg_gap=("DURATION_SEC", "mean")
        ).reset_index()

        st.subheader("Approvals per Block vs Average Gap (Bubble Insight)")
        fig = plt.figure(figsize=(10, 6))

        sizes = (block["cases"] / block["cases"].max()) * 800 + 50
        plt.scatter(
            block["cases"], block["avg_gap"],
            s=sizes,
            alpha=0.65,
            color=COLOR_MAP.get(tech, "gray"),
            edgecolors="black",
            linewidth=0.5
        )

        plt.xlabel("Number of Cases in Block")
        plt.ylabel("Average Duration Between Approvals (Seconds)")
        plt.title(f"{tech} — Review Sessions Bubble Chart")
        plt.grid(alpha=0.2)
        plt.tight_layout()
        st.pyplot(fig)

# ---------------- TAB 4 Time Patterns ----------------
with tab4:
    st.subheader("Approvals by Weekday")

    if len(d) == 0:
        st.warning("No data available.")
    else:
        weekday_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
        weekday_counts = d["WEEKDAY"].value_counts().reindex(weekday_order, fill_value=0)

        fig = plt.figure(figsize=(8, 4))
        plt.bar(weekday_counts.index, weekday_counts.values,
                color=COLOR_MAP.get(tech, "gray"), edgecolor="black")

        plt.xlabel("Weekday")
        plt.ylabel("Number of Approvals")
        plt.title(f"{tech} — Approvals by Weekday")
        plt.xticks(rotation=30)
        plt.grid(axis="y", alpha=0.25)
        plt.tight_layout()
        st.pyplot(fig)

# ---------------- TAB 5 High Risk ----------------
with tab5:
    st.subheader("High-Risk Fast Approvals (Worst Cases)")

    if len(d) == 0:
        st.warning("No data available.")
    else:
        worst = d.sort_values("DURATION_SEC").head(50)
        st.dataframe(worst[["APPROVAL_DATE", "CASE_NUMBER", "DURATION_SEC"]], use_container_width=True)

# ---------------- TAB 6 Policy Change (Arnold) ----------------
with tab6:
    st.subheader("Policy Change Behavior — Arnold")
    st.caption("Cutoff date: June 1, 2020 (payout reduced from $50 to $17)")
    cutoff = pd.Timestamp("2020-06-01")

    arn = df[df["TECHNICIAN"] == "Arnold"].copy()
    before = arn[arn["APPROVAL_DATE"] < cutoff]
    after  = arn[arn["APPROVAL_DATE"] >= cutoff]

    st.subheader("Monthly Fast-Approval Trend (<10 seconds)")
    arn["MONTH"] = arn["APPROVAL_DATE"].dt.to_period("M").dt.to_timestamp()
    arn["FAST_10S"] = arn["DURATION_SEC"] < 10

    monthly = arn.groupby("MONTH").agg(
        pct_fast=("FAST_10S", "mean")
    ).reset_index()

    fig = plt.figure(figsize=(10, 4))
    plt.plot(monthly["MONTH"], monthly["pct_fast"] * 100, marker="o", linewidth=3, color="red")
    plt.axvline(cutoff, linestyle="--", linewidth=2, color="black")
    plt.xlabel("Month")
    plt.ylabel("% Approvals < 10 sec")
    plt.title("Arnold Fast Approval Trend Over Time")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    st.pyplot(fig)

    st.subheader("Before vs After Summary")
    summary = pd.DataFrame({
        "Period": ["Before Policy", "After Policy"],
        "Total Approvals": [len(before), len(after)],
        "Mean Duration (sec)": [round(before["DURATION_SEC"].mean(), 2), round(after["DURATION_SEC"].mean(), 2)],
        "Median Duration (sec)": [round(before["DURATION_SEC"].median(), 2), round(after["DURATION_SEC"].median(), 2)],
        "% Fast (<10s)": [round((before["DURATION_SEC"] < 10).mean() * 100, 2),
                         round((after["DURATION_SEC"] < 10).mean() * 100, 2)]
    })
    st.dataframe(summary, use_container_width=True)
