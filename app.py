import streamlit as st
from snowflake.snowpark import Session
from snowflake.snowpark.exceptions import SnowparkSQLException
import pandas as pd
import plotly.graph_objects as go
from typing import List
from io import BytesIO

# ----------------------- CONFIG -----------------------

# Load Snowflake connection info securely from Streamlit secrets
connection_parameters = {
    "account": st.secrets["SNOWFLAKE_ACCOUNT"],
    "user": st.secrets["SNOWFLAKE_USER"],
    "password": st.secrets["SNOWFLAKE_PASSWORD"],
    "role": st.secrets["SNOWFLAKE_ROLE"],
    "warehouse": st.secrets["SNOWFLAKE_WAREHOUSE"],
    "database": st.secrets["SNOWFLAKE_DATABASE"],
    "schema": st.secrets["SNOWFLAKE_SCHEMA"]
}

# Create Snowflake session
@st.cache_resource(show_spinner=False)
def get_session() -> Session:
    return Session.builder.configs(connection_parameters).create()

session = get_session()

# ----------------------- ROLES -----------------------

@st.cache_data(ttl=300)
def list_roles() -> List[str]:
    """List roles available to the current user"""
    rows = session.sql("SHOW ROLES").collect()
    return [row["name"] for row in rows]

roles = list_roles()
selected_role = st.sidebar.selectbox("Select Role", roles, index=roles.index(connection_parameters["role"]))

@st.cache_resource
def get_session_for_role(role: str) -> Session:
    """Create a new session for the selected role"""
    new_params = connection_parameters.copy()
    new_params["role"] = role
    return Session.builder.configs(new_params).create()

session = get_session_for_role(selected_role)

# ----------------------- TABLE LISTING -----------------------

@st.cache_data(ttl=600, show_spinner=False)
def list_tables(_session) -> List[str]:
    """List fully qualified table names (DB.SCHEMA.TABLE) visible to current role"""
    q = _session.sql("""
        SELECT table_catalog || '.' || table_schema || '.' || table_name AS fq
        FROM information_schema.tables
        WHERE table_type = 'BASE TABLE'
        ORDER BY 1
    """)
    return [r[0] for r in q.collect()]

all_tables = list_tables(session)

@st.cache_data(ttl=300, show_spinner=False)
def load_table(table_fq: str) -> pd.DataFrame:
    """Load Snowflake table to Pandas and uppercase column names"""
    df = session.table(table_fq).to_pandas()
    df.columns = [c.upper() for c in df.columns]
    return df

# ----------------------- UI -----------------------

st.set_page_config(page_title="Snowflake Explorer", layout="wide")
st.title("📊 Snowflake Explorer")

if not all_tables:
    st.error("No tables accessible to the current role.")
    st.stop()

# Table selection
table_fq = st.selectbox("Select a table", all_tables)
df = load_table(table_fq)

# Dimension and metric column selection
all_cols = df.columns.tolist()
with st.sidebar:
    st.header("Configure View")
    dims = st.multiselect("Dimension columns (categorical)", options=all_cols, default=[all_cols[0]])
    metrics = st.multiselect(
        "Metric columns (numeric)",
        options=[c for c in all_cols if pd.api.types.is_numeric_dtype(df[c])],
        default=[c for c in all_cols if pd.api.types.is_numeric_dtype(df[c])][:1]
    )

    # Null / Not Null filter per dimension
    st.markdown("---")
    st.subheader("Apply Filters")
    filters = {}
    for col in dims:
        filter_choice = st.radio(
            f"{col} filter:", 
            options=["All", "Not Null", "Null"], 
            index=1, 
            horizontal=True, 
            key=f"null_filter_{col}"
        )
        filters[col] = filter_choice

# ----------------------- FILTERING -----------------------

# ----------------------- FILTERING -----------------------

mask = pd.Series(True, index=df.index)
for col, choice in filters.items():
    if choice == "Not Null":
        mask &= df[col].notnull()
    elif choice == "Null":
        mask &= df[col].isnull()
filtered_df = df[mask]

# ✅ Show updated record count
st.success(f"🔢 {len(filtered_df):,} record(s) match the current filters.")

st.subheader("📄 Data Preview")
st.dataframe(filtered_df, height=400, use_container_width=True)

# ----------------------- SANKEY CHART -----------------------

if len(dims) >= 2 and metrics:
    st.subheader("🔀 Sankey Chart")
    dim1, dim2 = dims[:2]
    metric = metrics[0]

    sankey_df = (
        filtered_df[[dim1, dim2, metric]]
        .groupby([dim1, dim2], as_index=False)[metric]
        .sum()
    )

    if not sankey_df.empty:
        labels = list(pd.concat([sankey_df[dim1], sankey_df[dim2]]).unique())
        label_index = {l: i for i, l in enumerate(labels)}

        fig = go.Figure(data=[go.Sankey(
            node=dict(label=labels, pad=15, thickness=20, line=dict(color="black", width=0.5)),
            link=dict(
                source=[label_index[r] for r in sankey_df[dim1]],
                target=[label_index[t] for t in sankey_df[dim2]],
                value=sankey_df[metric]
            )
        )])
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No data available for Sankey chart with current filters.")

# ----------------------- SQL DISPLAY -----------------------

st.subheader("🧾 Generated SQL")
sql_conditions = []
for col, choice in filters.items():
    if choice == "Not Null":
        sql_conditions.append(f"{col} IS NOT NULL")
    elif choice == "Null":
        sql_conditions.append(f"{col} IS NULL")
sql_preview = f"SELECT * FROM {table_fq}"
if sql_conditions:
    sql_preview += " WHERE " + " AND ".join(sql_conditions)
st.code(sql_preview)

# ----------------------- DOWNLOAD -----------------------

st.subheader("⬇️ Download Filtered Data")
file_format = st.radio("Select file format:", ["CSV", "Excel"], horizontal=True)

if file_format == "CSV":
    csv = filtered_df.to_csv(index=False)
    st.download_button("Download CSV", data=csv, file_name="filtered_data.csv", mime="text/csv")
else:
    output = BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        filtered_df.to_excel(writer, index=False, sheet_name="Data")
    output.seek(0)
    st.download_button(
        "Download Excel",
        data=output,
        file_name="filtered_data.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
