import streamlit as st
from snowflake.snowpark import Session
from snowflake.snowpark.exceptions import SnowparkSQLException
import pandas as pd
import plotly.graph_objects as go
from typing import List

# ----------------------- CONFIG -----------------------

# Use secrets for credentials
connection_parameters = {
    "account": st.secrets["SNOWFLAKE_ACCOUNT"],
    "user": st.secrets["SNOWFLAKE_USER"],
    "password": st.secrets["SNOWFLAKE_PASSWORD"],
    "role": st.secrets["SNOWFLAKE_ROLE"],
    "warehouse": st.secrets["SNOWFLAKE_WAREHOUSE"],
    "database": st.secrets["SNOWFLAKE_DATABASE"],
    "schema": st.secrets["SNOWFLAKE_SCHEMA"]
}

@st.cache_resource(show_spinner=False)
def get_session() -> Session:
    return Session.builder.configs(connection_parameters).create()

session = get_session()


@st.cache_data(ttl=300)
def list_roles() -> List[str]:
    """List all roles available to the current user"""
    rows = session.sql("SHOW ROLES").collect()
    return [row["name"] for row in rows]


roles = list_roles()
selected_role = st.sidebar.selectbox("Select Role", roles, index=roles.index(connection_parameters["role"]))

@st.cache_resource
def get_session_for_role(role: str) -> Session:
    new_params = connection_parameters.copy()
    new_params["role"] = role
    return Session.builder.configs(new_params).create()

session = get_session_for_role(selected_role)

@st.cache_data(ttl=600, show_spinner=False)
def list_tables(_session) -> List[str]:
    q = _session.sql("""
        SELECT table_catalog || '.' || table_schema || '.' || table_name AS fq
        FROM information_schema.tables
        WHERE table_type = 'BASE TABLE'
        ORDER BY 1
    """)
    return [r[0] for r in q.collect()]
# ----------------------- HELPERS -----------------------
all_tables = list_tables(session)

@st.cache_data(ttl=300, show_spinner=False)
def load_table(table_fq: str) -> pd.DataFrame:
    """Load table from Snowflake into Pandas and uppercase column names."""
    df = session.table(table_fq).to_pandas()
    df.columns = [c.upper() for c in df.columns]
    return df

# ----------------------- UI -----------------------
st.set_page_config(page_title="Dynamic Snowflake Explorer", layout="wide")
st.title("📊 Dynamic Snowflake Explorer (Snowpark + Streamlit)")

# 1️⃣ Select table
# all_tables = list_tables()
if not all_tables:
    st.error("No tables accessible to the current role.")
    st.stop()

table_fq = st.selectbox("Select a table", all_tables)

df = load_table(table_fq)

# 2️⃣ Select fields for dimensions & metrics
all_cols = df.columns.tolist()

with st.sidebar:
    st.header("🔧 Configure View")
    dims = st.multiselect("Dimension columns (categorical)", options=all_cols, default=[all_cols[0]])
    metrics = st.multiselect("Metric columns (numeric)", options=[c for c in all_cols if pd.api.types.is_numeric_dtype(df[c])],
                             default=[c for c in all_cols if pd.api.types.is_numeric_dtype(df[c])][:1])

    # Dynamic filters for each selected dimension
    st.markdown("---")
    st.subheader("Apply Filters")
    filters = {}
    for col in dims:
        uniques = sorted(df[col].dropna().unique().tolist())
        selected = st.multiselect(f"{col}", uniques, default=uniques, key=f"filter_{col}")
        filters[col] = selected

# 3️⃣ Apply filters
mask = pd.Series(True, index=df.index)
for col, values in filters.items():
    mask &= df[col].isin(values)
filtered_df = df[mask]

st.subheader("📄 Data Preview")
st.dataframe(filtered_df, height=400, use_container_width=True)

# 4️⃣ Optional Sankey (only if exactly 1 metric + 2 dimensions)
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
            node=dict(label=labels, pad=15, thickness=20,
                       line=dict(color="black", width=0.5)),
            link=dict(
                source=[label_index[r] for r in sankey_df[dim1]],
                target=[label_index[t] for t in sankey_df[dim2]],
                value=sankey_df[metric]
            )
        )])
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No data to display Sankey chart with current filters.")

# 5️⃣ Show generated SQL
st.subheader("🧾 Generated SQL")
# build simple SQL showcase
filters_sql = " AND ".join([f"{col} IN ({', '.join([repr(v) for v in vals])})" for col, vals in filters.items() if vals])
base_sql = f"SELECT * FROM {table_fq}"
if filters_sql:
    base_sql += f" WHERE {filters_sql}"
st.code(base_sql)
