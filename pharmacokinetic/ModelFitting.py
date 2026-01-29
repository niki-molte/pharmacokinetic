import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import traceback
import matplotlib.gridspec as gridspec


class PKAnalyzer:
    '''
    Class for fitting Pharmacokinetic models to patient data,
    calculating statistical metrics and generating clinical reports.
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

        This method performs:
        1. optimization of model parameters (curve fitting).
        2. calculation of Goodness-of-Fit statistics (R2, AIC).
        3. derivation of secondary clinical metrics (Cmax, Tmax, AUC).

        :param model_instance: instance of a class inheriting from BaseModel.
        '''
        name = model_instance.get_name()

        # wrapper function for the curve_fit optimization
        # it adapts the model and simulate method to the signature required by scipy
        def func_to_fit(t, *params):
            return model_instance.simulate(t, self.data['Dose'], *params)

        try:
            # curve fitting
            p0 = model_instance.get_initial_guesses()

            # weights for error, it is non-linear and
            # higher at high drug concentration.
            sigma_weights = np.maximum(self.data['Conc'], 1e-6)

            # fitting
            popt, pcov = curve_fit(
                func_to_fit,
                self.data['Time'],
                self.data['Conc'],
                p0=p0,
                bounds=(0, np.inf),
                sigma=sigma_weights,
                absolute_sigma=False,
                maxfev=10000
            )

            # evaluating RSE
            perr = np.sqrt(np.diag(pcov))
            rse_percent = (perr / popt) * 100

            # error Statistics (RES)
            y_pred = func_to_fit(self.data['Time'], *popt)
            residuals = self.data['Conc'] - y_pred

            weights = 1.0 / (sigma_weights ** 2)

            # weighted sum of residuals and weighted residual evaluation
            ss_res = np.sum(weights * (residuals ** 2))
            weighted_residuals = residuals / sigma_weights

            n = len(self.data['Conc'])  # number of data points
            k = len(popt)  # number of parameters

            # Akaike Information Criterion (AIC)
            aic = n * np.log(ss_res / n) + 2 * k

            # Evaluating secondary metrics (AUC, Cmax, Tmax)
            t_fine = np.linspace(0, 50, 1000)
            c_fine = model_instance.simulate(t_fine, self.data['Dose'], *popt)

            # maximum concentration (cmax) time at maximum (tmax)
            c_max = np.max(c_fine)
            t_max = t_fine[np.argmax(c_fine)]

            # area under the curve (AUC)
            auc_0_inf = np.trapezoid(c_fine, t_fine)

            # result dict
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

        # patient data
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

            # fit quality data AIC, params RSE and params value.
            print(f" [Model Quality / Goodness of Fit]")
            print(f" {'AIC':<35} : {res['aic']:.2f} (Lower is better)")

            # physiological params
            print(f"\n [Physiological Parameters]")
            print(f" {'PARAMETER':<40} | {'VALUE':<10} | {'UNIT':<10} | {'RSE (%)':<10}")
            print("-" * 80)

            # RSE
            rse_values = res['params_rse']
            idx = 0  # rse index

            for param_name, (val, unit) in res['real_params'].items():
                if val is None:
                    print(f"--- {param_name} ---")
                else:
                    # if out of index RSE = 0, it is not evaluated.
                    current_rse = rse_values[idx] if idx < len(rse_values) else 0.0
                    idx += 1

                    print(f" {param_name:<40} | {val:<10.4f} | {unit:<10} | {current_rse:<10.2f}")

            # derived clinical params
            sec = res['secondary']
            print(f"\n [Derived Clinical Metrics]")
            print(f" {'Peak Concentration (Cmax)':<35} : {sec['Cmax']:.2f} mg/L")
            print(f" {'Time to Peak (Tmax)':<35} : {sec['Tmax']:.2f} hours post-dose")
            print(f" {'Total Exposure (AUC)':<35} : {sec['AUC']:.2f} mg*h/L")
            print(f" {'Exposure ratio (AUC)':<35} : {sec['AUC_ratio']:.2f}")

            print("=" * 80)

    def plot_comparison(self):
        '''
        Generate a plot that compare observed clinical data vs. model predictions.
        '''
        t_smooth = np.linspace(0, 25, 200)
        plt.figure(figsize=(10, 6))

        # raw clinical data from .csv
        plt.scatter(self.data['Time'], self.data['Conc'], color='black', s=60, label='Clinical Data')

        colors = ['blue', 'red']
        styles = ['--', '-']

        # plot fitted models
        for i, (name, res) in enumerate(self.results.items()):
            model = res['model']
            popt = res['params_raw']

            # curve generation
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

    def plot_diagnostic_panel(self):
        '''
        Generates a diagnostic panel.
        It uses Weighted Residuals (WRES) already evaluated and stored during fitting.
        '''

        # for each model
        for name, res in self.results.items():

            # numpy conversion
            obs = np.array(self.data['Conc'])
            time = np.array(self.data['Time'])

            model = res['model']
            popt = res['params_raw']

            # wres and numpy transform
            wres = res['w_residuals']
            wres = np.array(wres)

            # re-evaluating predictions
            pred = model.simulate(time, self.data['Dose'], *popt)

            fig = plt.figure(figsize=(14, 10))
            plt.suptitle(f"Diagnostic Panel: {name.upper()}", fontsize=16, weight='bold')
            gs = gridspec.GridSpec(2, 2, figure=fig)

            # observed vs predicted values (log-log)
            ax1 = fig.add_subplot(gs[0, 0])
            ax1.scatter(pred, obs, alpha=0.6, edgecolors='k')

            # plotting only values != 0
            valid_pred = pred[pred > 0]
            valid_obs = obs[obs > 0]

            if len(valid_pred) > 0 and len(valid_obs) > 0:
                max_val = max(np.max(valid_pred), np.max(valid_obs)) * 1.5
                min_val = min(np.min(valid_pred), np.min(valid_obs)) * 0.5
                ax1.plot([min_val, max_val], [min_val, max_val], 'k--', label='Identity')

            ax1.set_xscale('log')
            ax1.set_yscale('log')
            ax1.set_title('Observed concentration vs Predicted concentration(Log-Log)')
            ax1.set_xlabel('Predicted Concentration')
            ax1.set_ylabel('Observed Concentration')
            ax1.grid(True, which="both", ls="-", alpha=0.2)

            # wres vs time
            ax2 = fig.add_subplot(gs[0, 1])
            ax2.scatter(time, wres, alpha=0.6, edgecolors='k', color='blue')
            ax2.axhline(0, color='black', lw=2)
            ax2.axhline(2, color='red', ls='--', alpha=0.5, label='Limit (+2)')
            ax2.axhline(-2, color='red', ls='--', alpha=0.5, label='Limit (-2)')

            ax2.fill_between(time, -2, 2, color='red', alpha=0.05)

            ax2.set_title('Weighted residuals (WRES) vs Time')
            ax2.set_xlabel('Time (h)')
            ax2.set_ylabel('WRES (std. dev.)')
            ax2.set_ylim(-4, 4)
            ax2.grid(True, alpha=0.3)

            # wres vs predicted values
            ax3 = fig.add_subplot(gs[1, 0])
            ax3.scatter(pred, wres, alpha=0.6, edgecolors='k', color='green')
            ax3.axhline(0, color='black', lw=2)
            ax3.axhline(2, color='red', ls='--', alpha=0.5)
            ax3.axhline(-2, color='red', ls='--', alpha=0.5)

            # filling -2, +2 sigma
            if len(valid_pred) > 0:
                ax3.fill_between([min_val, max_val], -2, 2, color='red', alpha=0.05)

            ax3.set_xscale('log')
            ax3.set_title('Weighted Residuals (WRES) vs Predicted concentration')
            ax3.set_xlabel('Predicted Concentration (Log)')
            ax3.set_ylabel('WRES (std. dev.)')
            ax3.set_ylim(-4, 4)
            ax3.grid(True, alpha=0.3)

            ax4 = fig.add_subplot(gs[1, 1])
            ax4.set_visible(False)

            plt.tight_layout()
        plt.show()

    def plot_comparative_diagnostic_panel(self):
        '''
        Generates a comparative diagnostic panel with concentration and residuals.
        Plots both models on the same graphs:
        - 1-Compartment (or first model) in BLUE
        - 2-Compartment (or second model) in RED
        '''

        fig = plt.figure(figsize=(14, 10))
        plt.suptitle("Comparative Diagnostic Panel: 1-Comp vs 2-Comp", fontsize=16, weight='bold')
        gs = gridspec.GridSpec(2, 2, figure=fig)

        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[1, 0])
        ax4 = fig.add_subplot(gs[1, 1])

        # identity line
        global_min = float('inf')
        global_max = float('-inf')

        # for each models
        for i, (name, res) in enumerate(self.results.items()):

            lower_name = name.lower()
            if "1" in lower_name or "one" in lower_name:
                color = 'blue'
                label = "1-Comp"
            elif "2" in lower_name or "two" in lower_name:
                color = 'red'
                label = "2-Comp"
            else:
                # default if name do not contain information
                color = 'blue' if i == 0 else 'red'
                label = name

            # numpy conversion
            obs = np.array(self.data['Conc'])
            time = np.array(self.data['Time'])

            model = res['model']
            popt = res['params_raw']

            # wres
            wres = np.array(res['w_residuals'])

            # evaluating predictions
            pred = model.simulate(time, self.data['Dose'], *popt)

            # update global min and max values
            valid_vals = np.concatenate([pred[pred > 0], obs[obs > 0]])
            if len(valid_vals) > 0:
                global_min = min(global_min, np.min(valid_vals))
                global_max = max(global_max, np.max(valid_vals))

            # oberved vs predicted
            ax1.scatter(pred, obs, alpha=0.6, edgecolors='k', color=color, label=label)

            # wres vs time
            ax2.scatter(time, wres, alpha=0.6, edgecolors='k', color=color, label=label)

            # wres vs pred
            ax3.scatter(pred, wres, alpha=0.6, edgecolors='k', color=color, label=label)


        if global_max > global_min:
            # identity line
            ax1.plot([global_min * 0.5, global_max * 1.5], [global_min * 0.5, global_max * 1.5],
                     'k--', label='Identity', zorder=0)

        ax1.set_xscale('log')
        ax1.set_yscale('log')
        ax1.set_title('Observed concentration vs Predicted concentration(Log-Log)')
        ax1.set_xlabel('Predicted Concentration')
        ax1.set_ylabel('Observed Concentration')
        ax1.legend()
        ax1.grid(True, which="both", ls="-", alpha=0.2)

        ax2.axhline(0, color='black', lw=2, zorder=0)
        ax2.axhline(2, color='gray', ls='--', alpha=0.5)
        ax2.axhline(-2, color='gray', ls='--', alpha=0.5)

        # standard deviation lines
        ax2.fill_between([np.min(time), np.max(time)], -2, 2, color='gray', alpha=0.1)

        ax2.set_title('Weighted Residuals (WRES) vs Time')
        ax2.set_xlabel('Time (h)')
        ax2.set_ylabel('WRES (std. dev.)')
        ax2.set_ylim(-4, 4)
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        ax3.axhline(0, color='black', lw=2, zorder=0)
        ax3.axhline(2, color='gray', ls='--', alpha=0.5)
        ax3.axhline(-2, color='gray', ls='--', alpha=0.5)

        # standard deviation lines
        ax3.fill_between([global_min * 0.5, global_max * 1.5], -2, 2, color='gray', alpha=0.1)

        ax3.set_xscale('log')
        ax3.set_title('Weighted Residuals (WRES) vs Predicted concentration')
        ax3.set_xlabel('Predicted Concentration (Log)')
        ax3.set_ylabel('WRES (std. dev.)')
        ax3.set_ylim(-4, 4)
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        ax4.set_visible(False)

        plt.tight_layout()
        plt.show()