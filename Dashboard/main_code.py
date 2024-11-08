import streamlit as st
import pandas as pd
from Planning_visualization import plot_planning  
from Prognosis_data import plot_prognosis
from Battery_percentage import plot_battery_percentage
from Storage_data import plot_storage_data
from Speed_robot import plot_robot_speed
from Status_robot import plot_robot_status
from Scan_data import plot_scan_data
from datetime import datetime

'''
This Streamlit application displays a dashboard to visualize the progress of a robotic pilot project. 
The dashboard allows users to select between "Planning" and "Robot Information" visualizations based on uploaded data files.

Modules:
- Streamlit: For creating the interactive dashboard.
- Pandas: For data manipulation.
- Matplotlib and custom plotting functions (from imported modules): For creating visualizations.

Dashboard Components:
1. Planning Visualization: Allows users to upload a CSV file to display planning-related data.
2. Robot Information Visualization: Displays various robot data, including battery level, storage capacity, speed, status, 
   scan data, and prognosis based on an uploaded Excel file.

Streamlit Page Configuration:
- Title: "Progress Pilot Dashboard"
- Layout: Wide layout to fit multiple visualizations

User Interactions:
- Allows users to upload data files (CSV for planning, Excel for robot information).
- Provides visualizations for different robot metrics and tracks unsupported sheets.

'''

# Configuration of the Streamlit page
st.set_page_config(page_title="Dashboard", page_icon=":calendar:", layout="wide")

# Title of the dashboard
st.title(":calendar: Progress Pilot Dashboard")

# Visualization selection
selected_visualization = st.selectbox("Select Visualization", ("Planning", "Robot Information"))

# Planning visualization
if selected_visualization == "Planning":
    fl = st.file_uploader(":file_folder: Upload a CSV file", type="csv")
    if fl is not None:
        try:
            # Extracting the filename and parsing the date
            filename = fl.name
            date_str = filename.split('.')[0]
            current_date = pd.to_datetime(date_str, dayfirst=True)
            
            fig_storage = plot_storage_data()
            st.pyplot()
        except ValueError as e:
            st.error(f"Error in visualization: {e}")

# Robot information visualization
elif selected_visualization == "Robot Information":
    excel_file = st.file_uploader(":file_folder: Upload an Excel file", type="xlsx")
    if excel_file is not None:
        xls = pd.ExcelFile(excel_file)
        sheets = xls.sheet_names

        # Create two different columns with equal width
        col1, col2 = st.columns((1, 1))

        visualized_sheets = set()

        # Display the different visualizations in the robot information dashboard

        # Left column
        with col1:
            # Battery Percentage
            if "Battery percentage" in sheets:
                data_battery = pd.read_excel(xls, sheet_name="Battery percentage")
                st.subheader("Battery Level Over Time")
                try:
                    fig_battery = plot_battery_percentage(data_battery)
                    st.pyplot(fig_battery, use_container_width=True)
                    visualized_sheets.add("Battery percentage")
                except Exception as e:
                    st.error(f"Error visualizing battery level data: {e}")

            # Storage Data
            if "StorageData" in sheets:
                data_storage = pd.read_excel(xls, sheet_name="StorageData")
                st.subheader("Robot Storage Capacity")
                try:
                    fig_storage = plot_storage_data(data_storage)
                    st.pyplot(fig_storage, use_container_width=True)
                    visualized_sheets.add("StorageData")
                except Exception as e:
                    st.error(f"Error visualizing storage data: {e}")

            # Robot Speed
            if "Speed robot" in sheets:
                data_speed = pd.read_excel(xls, sheet_name="Speed robot")
                st.subheader("Robot Speed Over Time")
                try:
                    fig_speed = plot_robot_speed(data_speed)
                    st.pyplot(fig_speed, use_container_width=True)
                    visualized_sheets.add("Speed robot")
                except Exception as e:
                    st.error(f"Error visualizing robot speed data: {e}")

            # Robot Status
            if "Status robot" in sheets:
                data_status = pd.read_excel(xls, sheet_name="Status robot")
                st.subheader("Robot Status")
                try:
                    fig_status = plot_robot_status(data_status)
                    st.pyplot(fig_status, use_container_width=True)
                    visualized_sheets.add("Status robot")
                except Exception as e:
                    st.error(f"Error visualizing robot status data: {e}")

        # Right column
        with col2:
            for sheet_name in sheets:
                data = pd.read_excel(xls, sheet_name=sheet_name)

                # Scan Data
                if sheet_name == "Scan data":
                    st.subheader("Scan Data")
                    try:
                        fig_scan = plot_scan_data(data)
                        st.pyplot(fig_scan, use_container_width=True)
                        visualized_sheets.add(sheet_name)
                    except Exception as e:
                        st.error(f"Error visualizing scan data: {e}")

                # Prognosis Data
                elif sheet_name == "Prognosis":
                    st.subheader("Prognosis Data")
                    try:
                        fig_prognosis = plot_prognosis(data)
                        st.pyplot(fig_prognosis, use_container_width=True)
                        visualized_sheets.add(sheet_name)
                    except Exception as e:
                        st.error(f"Error visualizing prognosis data: {e}")

        # Warning for unsupported sheets, only if no visualization function is defined
        unvisualized_sheets = set(sheets) - visualized_sheets
        if unvisualized_sheets:
            st.warning(f"No visualization function defined for sheets: {', '.join(unvisualized_sheets)}")
