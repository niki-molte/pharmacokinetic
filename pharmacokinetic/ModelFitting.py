import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import traceback


class PKAnalyzer:
    '''
    Class responsible for fitting Pharmacokinetic models to patient data,
    calculating statistical metrics, and generating clinical reports.
    '''

    def __init__(self, data_dict):
        '''
        Initialize the analyzer with patient data.

        :param data_dict: dictionary containing 'time', 'conc', 'total_dose', 'weight', and 'id'.
        '''
        self.data = data_dict
        self.results = {}

    def fit_model(self, model_instance):
        '''
        Fit a specific PK model to the observed data and calculate performance metrics.

        This method performs the following steps:
        1. Optimization of model parameters (Curve Fitting).
        2. Calculation of Goodness-of-Fit statistics (R2, AIC).
        3. Derivation of secondary clinical metrics (Cmax, Tmax, AUC).

        :param model_instance: instance of a class inheriting from BaseModel.
        '''
        name = model_instance.get_name()

        # Wrapper function for the curve_fit optimization
        # It adapts the model's simulate method to the signature required by scipy
        def func_to_fit(t, *params):
            return model_instance.simulate(t, self.data['Dose'], *params)

        try:
            # 1. Curve Fitting
            p0 = model_instance.get_initial_guesses()

            # weights are useful because the error is non-linear.
            # higher at high drug concentration.
            sigma_weights = np.maximum(self.data['Conc'], 1e-6)

            # --- CURVE FITTING ---
            popt, pcov = curve_fit(
                func_to_fit,
                self.data['Time'],
                self.data['Conc'],
                p0=p0,
                bounds=(0, np.inf),
                sigma=sigma_weights,
                absolute_sigma=False,  # scale the covariance on chi-squared
                maxfev=10000
            )

            # evaluating RSE
            perr = np.sqrt(np.diag(pcov))
            rse_percent = (perr / popt) * 100

            # 2. Calculation of Error Statistics (R2, AIC)
            y_pred = func_to_fit(self.data['Time'], *popt)
            residuals = self.data['Conc'] - y_pred

            weights = 1.0 / (sigma_weights ** 2)

            # weighted sum of residuals and weighted residual evaluation
            ss_res = np.sum(weights * (residuals ** 2))
            weighted_residuals = residuals / sigma_weights


            # Sum of Squared Residuals (SS_res)
            #ss_res = np.sum(residuals ** 2)
            # Total Sum of Squares (SS_tot)
            #ss_tot = np.sum((self.data['Conc'] - np.mean(self.data['Conc'])) ** 2)

            n = len(self.data['Conc'])  # Number of data points
            k = len(popt)  # Number of parameters

            # Akaike Information Criterion (AIC) - Lower is better
            aic = n * np.log(ss_res / n) + 2 * k
            # Coefficient of Determination (R2) - Closer to 1 is better
            #r2 = 1 - (ss_res / ss_tot)

            # 3. Calculation of Secondary Metrics (AUC, Cmax, Tmax)
            # Generate a high-resolution curve to accurately find peaks and area
            t_fine = np.linspace(0, 50, 1000)  # Simulate up to 50 hours
            c_fine = model_instance.simulate(t_fine, self.data['Dose'], *popt)

            # Cmax (Maximum Concentration) and Tmax (Time at Maximum)
            c_max = np.max(c_fine)
            t_max = t_fine[np.argmax(c_fine)]

            # AUC (Area Under the Curve) using the trapezoidal rule
            auc_0_inf = np.trapezoid(c_fine, t_fine)

            # Store results in the dictionary
            self.results[name] = {
                'model': model_instance,
                'params_raw': popt,
                'params_se': perr,  # standard error
                'w_residuals': weighted_residuals,
                'params_rse': rse_percent,  # RSE
                'real_params': model_instance.get_real_parameters(popt),
                'aic': aic,
                'secondary': {
                    'Cmax': c_max,
                    'Tmax': t_max,
                    'AUC': auc_0_inf,
                    'AUC_ratio': np.trapezoid(self.data['Conc'], self.data['Time'])/auc_0_inf
                }
            }

        except Exception as e:
            print(f"Fitting error for {name}: {e}")
            traceback.print_exc()

    def print_terminal_report(self):
        '''
        Print a comprehensive clinical report to the terminal.
        Displays patient info, model statistics, and physiological parameters.
        '''
        d = self.data
        print("\n" + "=" * 70)
        print(f" CLINICAL PHARMACOKINETIC REPORT - PATIENT ID: {d.get('Subject', 'Unknown')}")
        print("=" * 70)

        # SECTION 1: PATIENT DATA
        print(f" PATIENT AND DOSING DATA")
        if 'Wt' in d:
            print(f" {'Body Weight':<35} : {d['Wt']:.2f} kg")
            print(f" {'Relative Dose':<35} : {d['Dose'] / d['Wt']:.2f} mg/kg")

        if 'Dose' in d:
            print(f" {'Total Administered Dose':<35} : {d['Dose']:.2f} mg")
            print("-" * 70)

        for name, res in self.results.items():
            print(f"\n >>> MODEL RESULTS: {name.upper()}")
            print("." * 70)

            # SECTION 2: FIT QUALITY
            print(f" [Model Quality / Goodness of Fit]")
            print(f" {'AIC':<35} : {res['aic']:.2f} (Lower is better)")
            # RSE rimosso da qui perché viene dettagliato nella tabella sotto

            # SECTION 3: PHYSIOLOGICAL PARAMETERS
            print(f"\n [Physiological Parameters]")
            print(f" {'PARAMETER':<40} | {'VALUE':<10} | {'UNIT':<10} | {'RSE (%)':<10}")
            print("-" * 70)

            # Recuperiamo l'array degli RSE calcolati
            rse_values = res['params_rse']
            idx = 0  # Indice per scorrere gli RSE

            for param_name, (val, unit) in res['real_params'].items():
                if val is None:
                    # Separator line
                    print(f"--- {param_name} ---")
                else:
                    # extracting RSE
                    current_rse = rse_values[idx] if idx < len(rse_values) else 0.0
                    idx += 1

                    print(f" {param_name:<40} | {val:<10.4f} | {unit:<10} | {current_rse:<10.2f}")

            # SECTION 4: DERIVED CLINICAL METRICS
            sec = res['secondary']
            print(f"\n [Derived Clinical Metrics]")
            print(f" {'Peak Concentration (Cmax)':<35} : {sec['Cmax']:.2f} mg/L")
            print(f" {'Time to Peak (Tmax)':<35} : {sec['Tmax']:.2f} hours post-dose")
            print(f" {'Total Exposure (AUC)':<35} : {sec['AUC']:.2f} mg*h/L")
            print(f" {'Exposure ratio (AUC)':<35} : {sec['AUC_ratio']:.2f}")

            print("=" * 70)

    def plot_comparison(self):
        '''
        Generate a Matplotlib plot comparing observed clinical data vs. model predictions.
        '''
        t_smooth = np.linspace(0, 25, 200)
        plt.figure(figsize=(10, 6))

        # Plot raw clinical data
        plt.scatter(self.data['Time'], self.data['Conc'], color='black', s=60, label='Clinical Data')

        colors = ['blue', 'red']
        styles = ['--', '-']

        # Plot each fitted model
        for i, (name, res) in enumerate(self.results.items()):
            model = res['model']
            popt = res['params_raw']

            # Simulate smooth curve using optimized parameters
            c_sim = model.simulate(t_smooth, self.data['Dose'], *popt)

            plt.plot(t_smooth, c_sim, color=colors[i % len(colors)],
                     linestyle=styles[i % len(styles)], linewidth=2,
                     label=f"{name}")

        plt.title(f"PK Curve Comparison - Patient {self.data['Subject']}")
        plt.xlabel("Time (hours)")
        plt.ylabel("Concentration (mg/L)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()