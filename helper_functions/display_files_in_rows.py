import streamlit as st

def display_files_in_rows(file_list, header):
    if file_list:
        st.subheader(header)
        for i in range(0, len(file_list), 3):
            cols = st.columns(3)  # Create 3 columns
            for j, file in enumerate(file_list[i:i+3]):  # Display up to 3 files per row
                with cols[j]:
                    st.write(file.name)