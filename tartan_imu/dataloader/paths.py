# Copyright 2026 Shibo Zhao
# Contact: shibowing@gmail.com, shiboz@andrew.cmu.edu
# Please keep the above information when modifying this file.

"""Dataset path discovery helpers, extracted from main_net.py."""

import logging
import os


def GetTartanAirDataPath(path: str, test_scene) -> list[str]:
    """Return sorted TartanAir trajectory dirs, excluding the test scenes."""
    names = os.listdir(path + "/")
    folders = []
    outdir = [".DS_Store", "dir_list.json"]
    for name in names:
        if name in test_scene or name in outdir:
            pass
        else:
            data_path = os.path.join(os.path.abspath(path), name, "Data")  # every scene
            traj_list = os.listdir(data_path + "/")
            for traj in traj_list:
                traj_path = os.path.join(data_path, traj)  # every traj
                if os.path.isdir(traj_path):
                    folders.append(traj_path)
    folders.sort()
    return folders


def get_category_path(path: str, cfg) -> tuple[str, str, str]:
    """Return (train, val, test) dir paths under a category root from config."""
    train_path = os.path.join(path, cfg["data"]["train_dir"])
    val_path = os.path.join(path, cfg["data"]["validation_dir"])
    test_path = os.path.join(path, cfg["data"]["test_dir"])

    return train_path, val_path, test_path


def GetTartanAirDataPath_test(path: str, test_scene) -> list[str]:
    """Return sorted TartanAir trajectory dirs for the given test scenes."""
    names = os.listdir(path + "/")
    folders = []
    for name in names:
        if name in test_scene:
            data_path = os.path.join(os.path.abspath(path), name, "Data")  # every scene
            traj_list = os.listdir(data_path + "/")
            for traj in traj_list:
                traj_path = os.path.join(data_path, traj)  # every traj
                if os.path.isdir(traj_path):
                    folders.append(traj_path)
    folders.sort()
    return folders


def GetTartanAirDataPath_val(path: str, val_scene) -> list[str]:
    """Return sorted TartanAir trajectory dirs for the given val scenes."""
    names = os.listdir(path + "/")
    folders = []
    for name in names:
        if name in val_scene:
            data_path = os.path.join(os.path.abspath(path), name, "Data")  # every scene
            traj_list = os.listdir(data_path + "/")
            for traj in traj_list:
                traj_path = os.path.join(data_path, traj)  # every traj
                if os.path.isdir(traj_path):
                    folders.append(traj_path)
    folders.sort()
    return folders


def GetDataPath(path: str) -> list[str]:
    """Return sorted immediate subdirectories of path (empty if missing)."""
    folders = []
    # Check if the path exists
    if not os.path.exists(path):
        logging.warning(f"Path does not exist: {path}")
        return folders

    try:
        names = os.listdir(path + "/")
        for name in names:
            data_path = os.path.join(os.path.abspath(path), name)
            if os.path.isdir(data_path):
                folders.append(data_path)
        folders.sort()
    except PermissionError:
        logging.warning(f"Permission denied accessing path: {path}")
    except Exception as e:
        logging.warning(f"Error accessing path {path}: {e}")
    return folders


def get_list_of_combined_dir(source_folder_list) -> list[str]:
    """Collect .npz file paths across several source folders (recursively)."""
    data_list = []
    for source_folder in source_folder_list:
        # Check if the source folder exists
        if not os.path.exists(source_folder):
            logging.warning(f"Source folder does not exist: {source_folder}")
            continue

        try:
            # First, check if there are .npz files directly in the source folder
            for file in os.listdir(source_folder):
                if file.endswith(".npz"):
                    data_list.append(os.path.join(source_folder, file))

            # If no .npz files found directly, look for subdirectories
            if not data_list:
                for folder_name in os.listdir(source_folder):
                    subfolder = os.path.join(source_folder, folder_name)
                    # Check if subfolder is a directory or a file
                    if os.path.isdir(subfolder):
                        # Check for .npz files in the subfolder
                        for file in os.listdir(subfolder):
                            if file.endswith(".npz"):
                                data_list.append(os.path.join(subfolder, file))

                        # Also check for nested subdirectories (for debug_dataset structure)
                        for nested_folder in os.listdir(subfolder):
                            nested_path = os.path.join(subfolder, nested_folder)
                            if os.path.isdir(nested_path):
                                for file in os.listdir(nested_path):
                                    if file.endswith(".npz"):
                                        data_list.append(
                                            os.path.join(nested_path, file)
                                        )
                    elif os.path.isfile(subfolder) and subfolder.endswith(".npz"):
                        # If the subfolder is actually a .npz file
                        data_list.append(subfolder)
        except PermissionError:
            logging.warning(f"Permission denied accessing folder: {source_folder}")
            continue
        except Exception as e:
            logging.warning(f"Error accessing folder {source_folder}: {e}")
            continue
    return data_list


def get_list_of_dir(source_folder: str) -> list[str]:
    """Collect .npz file paths under a single source folder (recursively)."""
    data_list = []
    # Check if the source folder exists
    if not os.path.exists(source_folder):
        logging.warning(f"Source folder does not exist: {source_folder}")
        return data_list

    try:
        # First, check if there are .npz files directly in the source folder
        for file in os.listdir(source_folder):
            if file.endswith(".npz"):
                data_list.append(os.path.join(source_folder, file))

        # If no .npz files found directly, look for subdirectories
        if not data_list:
            for folder_name in os.listdir(source_folder):
                subfolder = os.path.join(source_folder, folder_name)
                # Check if subfolder is a directory or a file
                if os.path.isdir(subfolder):
                    # Check for .npz files in the subfolder
                    for file in os.listdir(subfolder):
                        if file.endswith(".npz"):
                            data_list.append(os.path.join(subfolder, file))

                    # Also check for nested subdirectories (for debug_dataset structure)
                    for nested_folder in os.listdir(subfolder):
                        nested_path = os.path.join(subfolder, nested_folder)
                        if os.path.isdir(nested_path):
                            for file in os.listdir(nested_path):
                                if file.endswith(".npz"):
                                    data_list.append(os.path.join(nested_path, file))
                elif os.path.isfile(subfolder) and subfolder.endswith(".npz"):
                    # If the subfolder is actually a .npz file
                    data_list.append(subfolder)
    except PermissionError:
        logging.warning(f"Permission denied accessing folder: {source_folder}")
    except Exception as e:
        logging.warning(f"Error accessing folder {source_folder}: {e}")
    return data_list
