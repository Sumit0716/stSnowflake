import streamlit as st
import pandas as pd
from io import BytesIO
from dotenv import load_dotenv
import os
from sqlalchemy import create_engine
from field_dictionary import FIELD_DICT

# Load environment variables
load_dotenv()

user = os.getenv("SNOWFLAKE_USER")
password = os.getenv("SNOWFLAKE_PASSWORD")
account = os.getenv("SNOWFLAKE_ACCOUNT")
warehouse = os.getenv("SNOWFLAKE_WAREHOUSE")
role = os.getenv("SNOWFLAKE_ROLE")
database = os.getenv("SNOWFLAKE_DATABASE")
schema = os.getenv("SNOWFLAKE_SCHEMA")

# Set wide layout
st.set_page_config(layout="wide")

# Apply custom CSS for tighter layout and styled download button
st.markdown("""
    <style>
        /* Main container padding */
        .main .block-container {
            padding-top: 1rem;
            padding-left: 2rem;
            padding-right: 2rem;
        }

        /* Sidebar width customization */
        section[data-testid="stSidebar"] > div:first-child {
            width: 300px;  /* Increase this value as needed */
        }

        /* Download button styling */
        .download-container {
            display: flex;
            justify-content: flex-end;
            margin-bottom: 1rem;
        }

        .download-btn {
            background-color: white;
            color: #262730;
            border: 1px solid #d3d3d3;
            padding: 6px 12px;
            border-radius: 8px;
            font-weight: 500;
            font-size: 14px;
            display: inline-flex;
            align-items: center;
            gap: 6px;
            text-decoration: none;
        }
    </style>
""", unsafe_allow_html=True)

# Snowflake connection engine
engine = create_engine(
    f"snowflake://{user}:{password}@{account}/{database}/{schema}?warehouse={warehouse}&role={role}"
)

table_name = 'DB_DEMO_DBT.PUBLIC.TARA_BASE_LAYER'

# App Title
st.markdown("## 📈 TARA Reports")
st.markdown("Use filters on the left to generate insights and download the report.")

# Sidebar - Configuration
st.sidebar.title("⚙️ Report Configuration")
report_type = st.sidebar.selectbox("📄 Report Template", ["Offer Accepts", "Adhoc Request"])

# Static Filters Section
st.sidebar.markdown("### 🔍 Basic Filters")

# Reporting Org Filter
ro_df = pd.read_sql(f"SELECT DISTINCT TARA_REPORTING_ORG FROM {table_name}", engine)
ro_options = sorted(ro_df.iloc[:, 0].dropna().unique().tolist())
selected_ro = st.sidebar.multiselect("Reporting Org(s)", options=ro_options)

# Gender Filter
gender_df = pd.read_sql(f"SELECT DISTINCT TARA_GENDER FROM {table_name}", engine)
gender_options = sorted(gender_df.iloc[:, 0].dropna().unique().tolist())
selected_gender = st.sidebar.multiselect("Gender(s)", options=gender_options)

# Year Filter for Offer Accepts
selected_years = None
if report_type == "Offer Accepts":
    year_df = pd.read_sql(
        f"SELECT DISTINCT YEAR(CAN_OFFER_ACCEPT_DATE) AS offer_year FROM {table_name} WHERE CAN_OFFER_ACCEPT_DATE IS NOT NULL",
        engine
    )
    year_options = sorted(year_df['offer_year'].dropna().astype(int).unique().tolist())
    selected_years = st.sidebar.multiselect("Offer Accept Year(s)", options=year_options)

# Validation
filters_ready = selected_ro and selected_gender
if report_type == "Offer Accepts":
    filters_ready = filters_ready and selected_years

# Query based on filters
df = pd.DataFrame()
if filters_ready:
    ro_str = ', '.join(f"'{ro}'" for ro in selected_ro)
    gender_str = ', '.join(f"'{g}'" for g in selected_gender)

    if report_type == "Offer Accepts":
        year_str = ', '.join(str(y) for y in selected_years)
        query = f"""
        SELECT
            CAN_APPLICATION_ID,
            CAN_ATS_ID,
            CAN_CONFIDENTIAL_INDICATOR,
            CAN_CURRENT_STAGE,
            CAN_GPA,
            CAN_CURRENT_STAGE TS,
            WD_POSITION_ID,
            CAN_JOB_CREATED_DATE,
            CAN_JOB_APP_SUB_TIME,
            CAN_OFFER_ACCEPT_DATE,
            YEAR(CAN_OFFER_ACCEPT_DATE) AS CAN_OFFER_ACCEPT_YEAR,
            CAN_EXPECTED_START_DATE,
            YEAR(CAN_EXPECTED_START_DATE) AS CAN_EXPECTED_START_YEAR,
            CAN_JOB_RECRUITER_EXTERNAL_ID,
            TARA_REPORTING_ORG,
            CAN_JOB_RECORD_FOLDER,
            TARA_GENDER
        FROM {table_name}
        WHERE TARA_REPORTING_ORG IN ({ro_str})
        AND TARA_GENDER IN ({gender_str})
        AND YEAR(CAN_OFFER_ACCEPT_DATE) IN ({year_str})
        """
    else:
        query = f"""
        SELECT *
        FROM {table_name}
        WHERE TARA_REPORTING_ORG IN ({ro_str})
        AND TARA_GENDER IN ({gender_str})
        """
    df = pd.read_sql(query, engine)

# Dynamic Filters in Sidebar (Advanced)
filtered_df = df.copy()
if not df.empty:
    with st.sidebar.expander("⚙️ Advanced View Filters"):
        all_cols = df.columns.tolist()
        all_cols_upper_sorted = sorted([col.upper() for col in all_cols])
        extra_cols = st.multiselect("Other Fields", options=all_cols_upper_sorted)

        for col in extra_cols:
            original_col = next((c for c in all_cols if c.upper() == col), col)
            unique_vals = df[original_col].dropna().unique().tolist()

            if len(unique_vals) > 100:
                selected_option = st.selectbox(
                    f"{original_col} (High Cardinality)", ["All", "Only Non-Null", "Only Null"], key=f"hc_{col}"
                )
                if selected_option == "Only Null":
                    filtered_df = filtered_df[filtered_df[original_col].isna()]
                elif selected_option == "Only Non-Null":
                    filtered_df = filtered_df[filtered_df[original_col].notna()]
            else:
                selected_vals = st.multiselect(
                    f"{original_col}", options=sorted(unique_vals), default=unique_vals, key=f"multi_{col}"
                )
                filtered_df = filtered_df[filtered_df[original_col].isin(selected_vals)]

# Result Section
st.markdown("---")
if not filtered_df.empty:
    st.subheader(f"📊 Filtered Data Preview ({len(filtered_df)} rows)")
    st.dataframe(filtered_df.head(10), use_container_width=True)

    # Download Area
    csv_data = filtered_df.to_csv(index=False)

    with st.expander("📥 Download Options", expanded=True):
        st.download_button(
            label="⬇️ Download CSV",
            data=csv_data,
            file_name="filtered_data.csv",
            mime="text/csv",
            key="download_csv_top"
        )

elif filters_ready:
    st.warning("⚠️ No data found for the selected filters.")
else:
    st.info("✅ Please complete all required filters to view data.")
