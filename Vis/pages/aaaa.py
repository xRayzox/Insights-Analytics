import pandas as pd
import streamlit as st
from st_aggrid import GridOptionsBuilder, AgGrid, JsCode
from st_aggrid.shared import ColumnsAutoSizeMode

# Sample data
products = {
    "No.": [1, 2, 3],
    "Name": ['Chair', 'Cabinet', 'Table'],
    "Price": [4, 12, 10],
    "Stocks": [100, 50, 60],
    "Details": [
        "https://i.imgur.com/fH2LHvo.png",
        "https://i.imgur.com/bvHZX5j.png",
        "https://i.imgur.com/D7xDwT9.png"
    ]
}

# Create DataFrame
df = pd.DataFrame(products)

# Build grid options
ob = GridOptionsBuilder.from_dataframe(df)

# 1. Add an image next to the value under the Name column
image_cr = JsCode("""
    function(params) {
        var element = document.createElement("span");

        var imageElement = document.createElement("img");
        var anchorElement = document.createElement("a");

        // Update the image element. Use the value from Details column.
        imageElement.src = params.data.Details;
        imageElement.width="40"; // Smaller image size
        imageElement.height="40";
        element.appendChild(imageElement);

        // Add a link to the Name value. The link is from the Details column.
        anchorElement.href = params.data.Details;
        anchorElement.target = "_blank";
        anchorElement.innerHTML = params.value;
        element.appendChild(anchorElement);

        return element;
    }""")
ob.configure_column('Name', cellRenderer=image_cr)

# 2. Configure the Details column to add a link
image_url = JsCode("""
    function(params) {
        return `<a href=${params.value} target="_blank">${params.value}</a>`;
    }""")
ob.configure_column("Details", cellRenderer=image_url)

# 2.1. Style the Stocks column if stocks is below 60
low_supply = JsCode("""
    function(params) {
        if (params.value < 60) {
            return {
                'color': 'red',
                'font-size': '20px'
            };
        }
    }""")
ob.configure_column("Stocks", cellStyle=low_supply)

# 3. Update selection mode
ob.configure_selection(selection_mode="multiple", use_checkbox=True)

# 4. Update row height for clearer image display
ob.configure_grid_options(rowHeight=100)

# 5. Hide the Details column
ob.configure_column("Details", hide=True)

# 6. Build the options
grid_options = ob.build()

st.markdown('#### Streamlit-AgGrid')

# 7. Add custom CSS to center the values
grid_return = AgGrid(
    df,
    grid_options,
    allow_unsafe_jscode=True,
    enable_enterprise_modules=False,
    columns_auto_size_mode=ColumnsAutoSizeMode.FIT_ALL_COLUMNS_TO_VIEW,
    key='products',
    custom_css={'.ag-row .ag-cell': {'display': 'flex', 'justify-content': 'center', 'align-items': 'center'},
                '.ag-header-cell-label': {'justify-content': 'center'}}
) 

# Display selected rows if any
selected_rows = grid_return["selected_rows"]

if selected_rows:
    st.markdown('#### Selected Products')
    dfs = pd.DataFrame(selected_rows)

    # Drop the internal columns and the Details column
    dfsnet = dfs.drop(columns=['_selectedRowNodeInfo', 'Details'], errors='ignore')
    AgGrid(
        dfsnet,
        enable_enterprise_modules=False,
        columns_auto_size_mode=ColumnsAutoSizeMode.FIT_ALL_COLUMNS_TO_VIEW,
        reload_data=True,
        key='product_selected'
    )
