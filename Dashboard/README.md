---
To open the dashboard, start by running the main_code.py file. Once the script is running, open a Terminal window and enter the following command to launch the dashboard on your local server: streamlit run main_code.py This will create a localhost link, which you can open in your browser to view the dashboard. To ensure proper functionality in the dashboard, all CSV files used for scheduling should be uploaded with the filename format dd-mm-yyyy.csv, representing the current date.

```bash
streamlit run main_code.py
```

Due to the Mirte robot’s current lack of data export capability, simulated data has been created to represent what the robot might potentially export. This data is organized within robot_data.xlsx, where each sheet represents a different data type from the Mirte robot. The use of Excel allows users to import a single file into the dashboard, simplifying the setup process. Each visualization within the dashboard is managed by dedicated scripts, which are called by a main script to integrate all visualizations seamlessly. Once robot_data.xlsx is imported, all visualizations will appear on the dashboard at the localhost.

### Workflow:
![Dashboard](img/Dashboard_workflow.jpg)
