import streamlit as st
from snowflake.snowpark import Session
from snowflake.snowpark.exceptions import SnowparkSQLException
import pandas as pd
import plotly.graph_objects as go
from typing import List
from io import BytesIO

# ----------------------- CONFIGURATION -----------------------

# Load connection credentials securely from Streamlit secrets
connection_parameters = {
    "account": st.secrets["SNOWFLAKE_ACCOUNT"],
    "user": st.secrets["SNOWFLAKE_USER"],
    "password": st.secrets["SNOWFLAKE_PASSWORD"],
    "role": st.secrets["SNOWFLAKE_ROLE"],
    "warehouse": st.secrets["SNOWFLAKE_WAREHOUSE"],
    "database": st.secrets["SNOWFLAKE_DATABASE"],
    "schema": st.secrets["SNOWFLAKE_SCHEMA"]
}

# Establish Snowpark session with default role
@st.cache_resource(show_spinner=False)
def get_session() -> Session:
    return Session.builder.configs(connection_parameters).create()

session = get_session()

# ----------------------- ROLE SWITCHING -----------------------

@st.cache_data(ttl=300)
def list_roles() -> List[str]:
    """Fetch all roles accessible to current user."""
    rows = session.sql("SHOW ROLES").collect()
    return [row["name"] for row in rows]

roles = list_roles()

# Sidebar UI for switching role
selected_role = st.sidebar.selectbox("Select Role", roles, index=roles.index(connection_parameters["role"]))

# Create session with selected role
@st.cache_resource
def get_session_for_role(role: str) -> Session:
    updated_params = connection_parameters.copy()
    updated_params["role"] = role
    return Session.builder.configs(updated_params).create()

session = get_session_for_role(selected_role)

# ----------------------- TABLE LISTING -----------------------

@st.cache_data(ttl=600, show_spinner=False)
def list_tables(_session) -> List[str]:
    """Return fully-qualified table names accessible by selected role."""
    query = _session.sql("""
        SELECT table_catalog || '.' || table_schema || '.' || table_name AS fq
        FROM information_schema.tables
        WHERE table_type = 'BASE TABLE'
        ORDER BY 1
    """)
    return [r[0] for r in query.collect()]

all_tables = list_tables(session)

# ----------------------- TABLE LOADER -----------------------

@st.cache_data(ttl=300, show_spinner=False)
def load_table(table_fq: str) -> pd.DataFrame:
    """Load selected table into a Pandas DataFrame with uppercased columns."""
    df = session.table(table_fq).to_pandas()
    df.columns = [c.upper() for c in df.columns]
    return df

# ----------------------- PAGE LAYOUT -----------------------

st.set_page_config(page_title="Dynamic Snowflake Explorer", layout="wide")
st.title("Dynamic Snowflake Explorer")

# Table selection dropdown
if not all_tables:
    st.error("No tables accessible to the selected role.")
    st.stop()

table_fq = st.selectbox("Select a table", all_tables)

# Load selected table
df = load_table(table_fq)
all_cols = df.columns.tolist()

# ----------------------- FILTER CONFIG -----------------------

with st.sidebar:
    st.header("Configure View")

    # Select dimension (categorical) columns
    dims = st.multiselect("Dimension columns (categorical)", options=all_cols, default=[all_cols[0]])

    # Select metric (numeric) columns
    metrics = st.multiselect(
        "Metric columns (numeric)",
        options=[c for c in all_cols if pd.api.types.is_numeric_dtype(df[c])],
        default=[c for c in all_cols if pd.api.types.is_numeric_dtype(df[c])][:1]
    )

    st.markdown("---")
    st.subheader("Apply Filters")

    # Apply filters dynamically for selected dimensions
    filters = {}
    for col in dims:
        unique_values = sorted(df[col].dropna().unique().tolist())
        selected_vals = st.multiselect(f"{col}", unique_values, default=unique_values, key=f"filter_{col}")
        filters[col] = selected_vals

# ----------------------- APPLY FILTERS -----------------------

mask = pd.Series(True, index=df.index)
for col, values in filters.items():
    mask &= df[col].isin(values)
filtered_df = df[mask]

# ----------------------- DATA PREVIEW -----------------------

st.subheader("Filtered Data")
st.dataframe(filtered_df, height=400, use_container_width=True)

# ----------------------- SANKEY CHART -----------------------

if len(dims) >= 2 and metrics:
    st.subheader("Sankey Chart")

    dim1, dim2 = dims[:2]
    metric = metrics[0]

    sankey_df = (
        filtered_df[[dim1, dim2, metric]]
        .groupby([dim1, dim2], as_index=False)[metric]
        .sum()
    )

    if not sankey_df.empty:
        labels = list(pd.concat([sankey_df[dim1], sankey_df[dim2]]).unique())
        label_index = {label: i for i, label in enumerate(labels)}

        fig = go.Figure(data=[go.Sankey(
            node=dict(label=labels, pad=15, thickness=20, line=dict(color="black", width=0.5)),
            link=dict(
                source=[label_index[src] for src in sankey_df[dim1]],
                target=[label_index[tgt] for tgt in sankey_df[dim2]],
                value=sankey_df[metric]
            )
        )])
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No data available for Sankey chart with current filters.")

# ----------------------- SQL DISPLAY -----------------------

st.subheader("Generated SQL")
filters_sql = " AND ".join([
    f"{col} IN ({', '.join([repr(v) for v in vals])})" for col, vals in filters.items() if vals
])
sql_preview = f"SELECT * FROM {table_fq}"
if filters_sql:
    sql_preview += f" WHERE {filters_sql}"
st.code(sql_preview)

# ----------------------- DOWNLOAD SECTION -----------------------

st.subheader("⬇Download Filtered Data")
download_format = st.radio("Choose format", ["CSV", "Excel"], horizontal=True)

if download_format == "CSV":
    csv = filtered_df.to_csv(index=False)
    st.download_button("Download CSV", data=csv, file_name="filtered_data.csv", mime="text/csv")
else:
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        filtered_df.to_excel(writer, sheet_name="FilteredData", index=False)
    output.seek(0)
    st.download_button(
        "Download Excel",
        data=output,
        file_name="filtered_data.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
