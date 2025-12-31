import pandas as pd
import os


class DataLoader:
    '''
    This class loads data from csv file.
    '''

    def __init__(self, file: str):
        '''
        Load a dataset in csv format and return a pandas dataframe.

        :param file: path to csv file.
        :return: pandas dataframe with all data.
        '''
        # check if the file exists
        if not os.path.isfile(file):
            raise FileNotFoundError(f"the file {file} does not exist")

        # if file exists load into pandas dataframe
        self.__data = pd.read_csv(file)

    def get_patient_data(self, patient_id: int) -> pd.DataFrame:
        '''
        Get the data of a patient with a specified patient_id.
        :param patient_id: the id of the patient to get data for.
        :return: pandas dataframe with patient data.
        '''
        if not patient_id in self.__data.index:
            raise KeyError(f"the patient {patient_id} does not exist")

        patient_data = self.__data[self.__data['Subject'] == patient_id].copy()
        return patient_data

    def patient_data_todict(self, patient_data: pd.DataFrame) -> dict:
        '''
        Convert patient data to a dictionary.
        :param patient_data: The patient data to be converted.
        :return: A dictionary with the patient data.
        '''
        if patient_data.empty:
            raise KeyError("the patient data does not contains data")

        patient_data_copy = patient_data.copy()
        patient_data_todict = {}

        # columns to process
        keys_to_process = patient_data_copy.columns.tolist()

        for key in keys_to_process:
            if key in patient_data_copy.columns:
                # if all values are equals we save only one scalar.
                if patient_data_copy[key].nunique(dropna=False) == 1:
                    patient_data_todict[key] = patient_data_copy[key].iloc[0]
                else:
                    patient_data_todict[key] = patient_data_copy[key].tolist()

        return patient_data_todict


