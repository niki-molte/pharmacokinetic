import matplotlib.pyplot as plt

from pharmacokinetic.CompartmentsModels import OneCompartmentModel, TwoCompartmentModel, \
    IV_OneCompartmentModel, IV_TwoCompartmentModel
from pharmacokinetic.ModelFitting import PKAnalyzer
from pharmacokinetic.DataLoader import DataLoader


if __name__ == "__main__":
    #loader = TheophDataLoader()
    loader = DataLoader("datasets/Indomethacin.csv")
    patient_data = loader.get_patient_data(5)

    print(loader.patient_data_todict(patient_data))



    # Seleziona il paziente (prova 1, 6 o 11)

    analyzer = PKAnalyzer(loader.patient_data_todict(patient_data))

    # Esegui i fit
    analyzer.fit_model(IV_OneCompartmentModel())
    analyzer.fit_model(IV_TwoCompartmentModel())

    # 1. Stampa il report testuale (quello che volevi)
    analyzer.print_terminal_report()

    # 2. Mostra il grafico pulito
    analyzer.plot_comparison()

    analyzer.plot_comparative_diagnostic_panel()

    #plt.show()