import os
import json

def concatenate_json_files(folder_path: str, output_file_name: str):
    concatenated_data = []

    # For loop for looping through all files in the provided folder (supplied by folder path).
    for filename in os.listdir(folder_path):
        if filename.endswith(".json"):
            file_path = os.path.join(folder_path, filename)

            # Read the JSON file and append to the list (alternatively use a directory for storing), no good reason for using list.
            # it uses the "r" (read mode).
            with open(file_path, "r") as file:
                data = json.load(file)
                concatenated_data.append(data)

    # Write data to the specified file
    with open(output_file_name, "w") as output_file:
        json.dump(concatenated_data, output_file, indent=2)


def extract_schedule_from_json_single_file(json_file_path: str, schedule_name: str):
    with open(json_file_path, 'r') as file:
        data = json.load(file)

    # 'get' the specific schedule in 'data'
    schedule_data = data.get(schedule_name, {})

    # 'get' the specific 'schedule_data'
    weekDayRulesetDict = schedule_data.get("weekDayRulesetDict", {})
    add_noise = schedule_data.get("add_noise", False)
    saveSimulationResult = schedule_data.get("saveSimulationResult", False)
    id_value = schedule_data.get("id", "")

    # Create a new dictionary with the extracted data
    schedule_inputs = {
        "weekDayRulesetDict": weekDayRulesetDict,
        "add_noise": add_noise,
        "saveSimulationResult": saveSimulationResult,
        "id": id_value
    }

    return schedule_inputs


def read_json_file(json_file_path):
    with open(json_file_path, "r") as file:
        json_data = file.read()

    data = json.loads(json_data)

    result_list = []

    for instance in data:
        for schedule_type, schedule_data in instance.items():
            if isinstance(schedule_data, dict) and "id" in schedule_data:
                schedule_id = schedule_data["id"]
                result_list.append({
                    f"{schedule_type.capitalize()}": schedule_id,
                })

    return result_list

# Overload existing function
def extract_schedule_from_json(json_file_path: str, schedule_id: str, schedule_type: str):
    # Schedule_types:
    """occupancy
    indoor_temperature_setpoint
    co2_setpoint
    supply_water_temperature_setpoint"""

    with open(json_file_path, 'r') as file:
        data = json.load(file)

    matching_schedule = None
    for instance in data:
        current_schedule = instance.get(schedule_type + "_schedule")
        if current_schedule and current_schedule["id"] == schedule_id:
            matching_schedule = current_schedule
            break

    # Throw an error, if such schedule does not exist.
    if matching_schedule is None:
        raise ValueError(f"Schedule with ID {schedule_id} and type {schedule_type} not found in the JSON file.")

    # Do as in extract_schedule_from_json_single_file.
    weekDayRulesetDict = matching_schedule.get("weekDayRulesetDict", {})
    add_noise = matching_schedule.get("add_noise", False)
    saveSimulationResult = matching_schedule.get("saveSimulationResult", False)
    id_value = matching_schedule.get("id", "")

    schedule_inputs = {
        "weekDayRulesetDict": weekDayRulesetDict,
        "add_noise": add_noise,
        "saveSimulationResult": saveSimulationResult,
        "id": id_value
    }

    return schedule_inputs

def extract_schedule_names(schedules):
    # Fixed list names:
    occupancy_schedule_names = []
    indoor_temperature_setpoint_schedule_names = []
    co2_setpoint_schedule_names = []
    supply_water_temperature_setpoint_schedule_names = []

    # If statements for every instance:
    for schedule_info in schedules:
        if "Occupancy_schedule" in schedule_info:
            occupancy_schedule_names.append(schedule_info["Occupancy_schedule"])
        if "Indoor_temperature_setpoint_schedule" in schedule_info:
            indoor_temperature_setpoint_schedule_names.append(schedule_info["Indoor_temperature_setpoint_schedule"])
        if "Co2_setpoint_schedule" in schedule_info:
            co2_setpoint_schedule_names.append(schedule_info["Co2_setpoint_schedule"])
        if "Supply_water_temperature_setpoint_schedule" in schedule_info:
            supply_water_temperature_setpoint_schedule_names.append(
                schedule_info["Supply_water_temperature_setpoint_schedule"])

    # Return dictionary of list names:
    return {
        "occupancy_schedule_names": occupancy_schedule_names,
        "indoor_temperature_setpoint_schedule_names": indoor_temperature_setpoint_schedule_names,
        "co2_setpoint_schedule_names": co2_setpoint_schedule_names,
        "supply_water_temperature_setpoint_schedule_names": supply_water_temperature_setpoint_schedule_names
    }
