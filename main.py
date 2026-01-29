import matplotlib.pyplot as plt

from pharmacokinetic.CompartmentsModels import OneCompartmentModel, TwoCompartmentModel, \
    IV_OneCompartmentModel, IV_TwoCompartmentModel
from pharmacokinetic.ModelFitting import PKAnalyzer
from pharmacokinetic.DataLoader import DataLoader


if __name__ == "__main__":

    # loading data and selecting patient
    loader = DataLoader("datasets/Indomethacin.csv")
    patient_data = loader.get_patient_data(5)

    print(loader.patient_data_todict(patient_data))
    analyzer = PKAnalyzer(loader.patient_data_todict(patient_data))

    # fit
    analyzer.fit_model(IV_OneCompartmentModel())
    analyzer.fit_model(IV_TwoCompartmentModel())

    # textual report
    analyzer.print_terminal_report()

    # plots
    analyzer.plot_comparison()
    analyzer.plot_comparative_diagnostic_panel()