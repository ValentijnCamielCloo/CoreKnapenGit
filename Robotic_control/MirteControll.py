from mirte_robot import robot
from datetime import datetime
import time
import csv
import pandas as pd 

"""""
Initialize the robot
"""
mirte = robot.createRobot()


def stop_robot():
    """
    Function to stop the robot
    """
    print("Stopping robot")
    mirte.setMotorSpeed('left', 0)
    mirte.setMotorSpeed('right', 0)

def emergency_Stop():
    """
    Function to emergency stop the robot
    """
    print("Beep Beep! Emergency stop")
    mirte.setMotorSpeed('left', 0)
    mirte.setMotorSpeed('right', 0)
    time.sleep(1)
    mirte.setMotorSpeed('left', -28)
    mirte.setMotorSpeed('right', -28)
    time.sleep(2)


def celebration_spinning():
    """
    Function to show the robot is done scanning
    """
    print("All scans completed!")
    mirte.setMotorSpeed('left', -40)
    mirte.setMotorSpeed('right', 40)  # Wiggle
    time.sleep(0.3)  # Wiggle time
    mirte.setMotorSpeed('left', 40)
    mirte.setMotorSpeed('right', -40)  # Wiggle
    time.sleep(0.3)  # Wiggle time
    stop_robot()

def orient_wall(scan_log):
    """
    Function to orient to the wall and log the scan
    """
    print("Corner detected. Stopping for scan.")
    global scan_count
    mirte.setMotorSpeed('left', -30)
    mirte.setMotorSpeed('right', 30)
    time.sleep(turning_time) 
    distance = mirte.getDistance('right')
    timestamp = mirte.getTimestamp()  # Get time since robot initialized
    print(f"Scanning at distance: {distance} meters. Timestamp: {timestamp} seconds since initialization.")
    scan_log.append({
        'timestamp': timestamp,
        'distance': distance
    })
    stop_robot()
    time.sleep(scanning_time)  # Wait for manual scan
    mirte.setMotorSpeed('left', 30)
    mirte.setMotorSpeed('right', -30)
    time.sleep(turning_time)
    stop_robot()
    time.sleep(1)
    mirte.setMotorSpeed('left', 40)
    mirte.setMotorSpeed('right', 25)
    time.sleep(0.6)

    scan_count += 1
    print(f"Number of scans performed at this moment: {scan_count}")

def save_log_to_csv(log_data, file_path):
    """"
    Function to save the log to a CSV file
    """
    with open(file_path, "w", newline='') as csvfile:
        fieldnames = ['timestamp', 'distance']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()  # Write the header row
        writer.writerows(log_data)  # Write the scan log rows


def save_robot_data(file_path="robot_data.xlsx"):
    """
    Function to generate robot data and save to a single Excel file with multiple sheets
    """    
    # Tab 1: Battery Percentage
    battery_data = []
    for _ in range(10):  # Retrieve the most recent 10 battery readings
        timestamp = datetime.now()  # Example of real-time timestamp from the robot
        battery_percentage = mirte.getBatteryPercentage()  # Get actual battery percentage
        battery_data.append([timestamp.strftime("%Y-%m-%d %H:%M:%S"), battery_percentage])
    battery_df = pd.DataFrame(battery_data, columns=["Timestamp", "Battery Percentage"])

    # Tab 2: Storage Data
    total_storage = mirte.getTotalStorage()  # Retrieve actual total storage from the robot
    used_storage = mirte.getUsedStorage()  # Retrieve actual used storage
    storage_df = pd.DataFrame([[f"{total_storage} GB", f"{used_storage} GB"]],
                              columns=["Total Storage", "Used Storage"])

    # Tab 3: Status Robot
    speed = mirte.getSpeed()  # Retrieve current speed from the robot
    status = mirte.getStatus()  # "working" or "not working" based on actual robot status
    connected = mirte.getConnectionstatus()  # True or False, indicating connection status
    status_df = pd.DataFrame([[f"{speed} m/s", status, connected]],
                             columns=["Speed", "Status", "Connected"])

    # Tab 4: Robot Speed
    speed_data = []
    for _ in range(10):  # Retrieve the most recent 10 speed readings
        timestamp = datetime.now()  # Timestamp for each speed reading
        speed = mirte.getSpeed()  # Get actual speed from the robot
        speed_data.append([timestamp.strftime("%Y-%m-%d %H:%M:%S"), f"{speed} m/s"])
    speed_df = pd.DataFrame(speed_data, columns=["Timestamp", "Speed"])

    # Tab 5: Scan Data
    scan_data = []
    scan_count = mirte.getScanCount()  # Retrieve total scan count
    for scan_nr in range(1, scan_count + 1):
        scan_duration = mirte.getScanDuration(scan_nr)  # Duration for each scan
        scan_data.append([scan_nr, f"{scan_duration} seconds"])
    scan_df = pd.DataFrame(scan_data, columns=["Scan Number", "Scan Duration"])

    # Save all dataframes to a single Excel file with multiple sheets
    with pd.ExcelWriter(file_path) as writer:
        battery_df.to_excel(writer, sheet_name="Battery Percentage", index=False)
        storage_df.to_excel(writer, sheet_name="Storage Data", index=False)
        status_df.to_excel(writer, sheet_name="Status Robot", index=False)
        speed_df.to_excel(writer, sheet_name="Robot Speed", index=False)
        scan_df.to_excel(writer, sheet_name="Scan Data", index=False)

    print(f"Data saved successfully to {file_path}")

def follow_line():
    """
    Function for line-following control: adjust orientation based on IR sensor readings
    """
    right_intensity = mirte.getIntensity('right')
    distancetowall = mirte.getDistance('left')
    # Adjust motor speeds to stay on the line
    if right_intensity >= frontsensorthreshold:  # sensor is on the line
        print("On the line!")
        mirte.setMotorSpeed('left', 35)
        mirte.setMotorSpeed('right', 20)
        print(f"Distance to wall = {distancetowall}")
        time.sleep(0.08)
    elif right_intensity < frontsensorthreshold:  # sensor is off the line
        print("Off the line!")
        mirte.setMotorSpeed('left', -30)
        mirte.setMotorSpeed('right', 25)
        print(f"Distance to wall = {distancetowall}")
        time.sleep(0.08)

def main():
    """
    Main function to control the flow of the program
    """
    global scan_count
    scan_count = 0
    scan_log = []
    scan_goal = 4 #Change this to the amount of scans you want to make
    sidesensorthreshold = 2400 #Change this to alter the value the side IR sensor senses as black
    frontsensorthreshold = 2000 #Change this to alter the value the front IR sensor senses as black
    csv_file_path = r"scan_log.csv" #Change this to the wished directory to save the csv to
    scanning_time = 10 #Change this to alter how long the robot waits and scans 
    turning_time = 0.9 #Change this to alter how long the robot turns, dependent on the motor power
    stoppingthreshold = 4  #Change this to the distance at which the robot should perform an emergency stop
    
    while True:
        left_intensity = mirte.getIntensity('left') 
        right_intensity = mirte.getIntensity('right')
        distancefront = mirte.getDistance('right')
        time.sleep(0.3)
        
        # Continously update the robot data CSV
        save_robot_data()

        # Check if emergency stop is needed (robot too close to wall)
        if distancefront <= stoppingthreshold:
            emergency_Stop()
        
        # Check for corner (intensity threshold), then perform scan
        if left_intensity >= sidesensorthreshold:
            orient_wall(scan_log)
        
        # Check if the scanning goal is reached
        if scan_count == scan_goal:
            print(f"Number of required scans: {scan_count}, has been reached, shutting down!")
            celebration_spinning()

            # Save the scan log to a CSV file
            save_log_to_csv(scan_log, csv_file_path)

            print("\nScan Log:")
            for scan in scan_log:
                print(f"Scan at {scan['timestamp']} seconds: Distance = {scan['distance']} meters")

            print(f"Scan log saved to {csv_file_path}")
            break

        else:
            follow_line()  # Follow line when not in emergency or scan condition or completed

# Main
if __name__ == "__main__":
    main()
