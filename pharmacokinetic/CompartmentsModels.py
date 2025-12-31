import numpy as np
from scipy.integrate import odeint

from abc import ABC, abstractmethod


class BaseModel(ABC):
    '''
    Abstract base class for Pharmacokinetic (PK) models.
    Enforces the structure for simulation and parameter translation.
    '''

    @abstractmethod
    def simulate(self, t, dose, *params):
        '''
        Simulate the drug concentration over time.

        :param t: array-like object containing time points.
        :param dose: float representing the administered dose.
        :param params: variable number of model specific parameters.
        :return: array-like object containing simulated concentration values.
        '''
        pass

    def get_name(self):
        '''
        Get the descriptive name of the model.

        :return: string representing the model name.
        '''
        return "BaseModel"

    def get_initial_guesses(self):
        '''
        Get initial parameter guesses for the optimization algorithm.

        :return: list of float values representing the initial guesses.
        '''
        return []

    @abstractmethod
    def get_real_parameters(self, popt):
        '''
        Translate raw optimization parameters into clinical parameters.

        :param popt: list or array of optimized parameters found by the fitter.
        :return: dictionary mapping parameter names to a tuple of (value, unit).
        '''
        pass


class OneCompartmentModel(BaseModel):
    '''
    Implementation of the One-Compartment PK model with first-order absorption.
    '''

    def get_name(self):
        '''
        Get the descriptive name of the model.

        :return: string representing the model name.
        '''
        return "One-Compartment Model"

    def ode_system(self, y, t, ka, ke):
        '''
        Define the system of Ordinary Differential Equations (ODEs).
        Describes the transfer of drug from the gut to the blood.

        :param y: list containing current amounts [Amount_gut, Amount_blood].
        :param t: current time point (required by odeint, but not used explicitly in equations).
        :param ka: absorption constant.
        :param ke: elimination constant.
        :return: list of derivatives [dAgut, dAblood].
        '''
        A_gut, A_blood = y
        dAgut = -ka * A_gut
        dAblood = ka * A_gut - ke * A_blood
        return [dAgut, dAblood]

    def simulate(self, t, dose, ka, ke, V):
        '''
        Simulate the drug concentration over time.

        :param t: array-like object containing time points.
        :param dose: float representing the administered dose.
        :param ka: absorption constant (1/h).
        :param ke: elimination constant (1/h).
        :param V: volume of distribution (L).
        :return: array-like object containing simulated concentration values (mg/L).
        '''
        y0 = [dose, 0.0]

        # solve the ODE system
        sol = odeint(self.ode_system, y0, t, args=(ka, ke))

        # return concentration (Amount in blood / Volume)
        return sol[:, 1] / V

    def get_initial_guesses(self):
        '''
        Get initial parameter guesses for the optimization algorithm.

        :return: list of float values [ka, ke, V].
        '''
        return [1.5, 0.2, 30.0]

    def get_real_parameters(self, popt):
        '''
        Translate raw optimization parameters into clinical parameters.

        :param popt: list containing [ka, ke, V].
        :return: dictionary mapping parameter names to a tuple of (value, unit).
        '''
        ka, ke, V = popt
        # Calculation of significant derived parameters
        half_life = np.log(2) / ke  # t1/2 = ln(2)/ke
        clearance = V * ke  # Cl = V * ke

        return {
            "Absorption Constant (ka)": (ka, "1/h"),
            "Elimination Constant (ke)": (ke, "1/h"),
            "Volume of Distribution (V)": (V, "L"),
            "--- CLINICAL PARAMETERS ---": (None, ""),
            "Half-life (t 1/2)": (half_life, "h"),
            "Body Clearance (Cl)": (clearance, "L/h")
        }


class IV_OneCompartmentModel(BaseModel):
    '''
    One-Compartment Model for IV Bolus administration.
    Equation: C(t) = (Dose / V) * exp(-ke * t)
    '''

    def get_name(self): return "IV Bolus (1-Comp)"

    def simulate(self, t, dose, ke, V):
        '''
        Analytical solution is faster and more precise than ODE for 1-Comp IV.
        '''
        t = np.asarray(t)

        if V <= 0:
            V = 1e-10

        return (dose / V) * np.exp(-ke * t)

    def get_initial_guesses(self):
        # Guess typical for Indomethacin: V ~10-15L, ke ~0.1-0.3
        return [0.2, 10.0]  # [ke, V]

    def get_real_parameters(self, popt):
        ke, V = popt
        half_life = np.log(2) / ke
        clearance = V * ke
        return {
            "Elimination Constant (ke)": (ke, "1/h"),
            "Volume of Distribution (V)": (V, "L"),
            "--- CLINICAL PARAMETERS ---": (None, ""),
            "Half-life (t 1/2)": (half_life, "h"),
            "Clearance (Cl)": (clearance, "L/h")
        }


class TwoCompartmentModel(BaseModel):
    '''
    Implementation of the Two-Compartment PK model.
    Includes a central compartment (blood) and a peripheral compartment (tissues).
    '''

    def get_name(self):
        '''
        Get the descriptive name of the model.

        :return: string representing the model name.
        '''
        return "Two-Compartment Model"

    def ode_system(self, y, t, ka, ke, kcp, kpc):
        '''
        Define the system of Ordinary Differential Equations (ODEs).
        Describes transfer between gut, central, and peripheral compartments.

        :param y: list containing current amounts [Amount_gut, Amount_central, Amount_peripheral].
        :param t: current time point.
        :param ka: absorption constant.
        :param ke: elimination constant (from central compartment).
        :param kcp: rate constant from central to peripheral.
        :param kpc: rate constant from peripheral to central.
        :return: list of derivatives [dAgut, dAcent, dAperi].
        '''
        A_gut, A_cent, A_peri = y
        dAgut = -ka * A_gut
        dAcent = ka * A_gut - ke * A_cent - kcp * A_cent + kpc * A_peri
        dAperi = kcp * A_cent - kpc * A_peri
        return [dAgut, dAcent, dAperi]

    def simulate(self, t, dose, ka, ke, kcp, kpc, V):
        '''
        Simulate the drug concentration over time.

        :param t: array-like object containing time points.
        :param dose: float representing the administered dose.
        :param ka: absorption constant.
        :param ke: elimination constant.
        :param kcp: rate constant central->peripheral.
        :param kpc: rate constant peripheral->central.
        :param V: volume of the central compartment.
        :return: array-like object containing simulated concentration values (mg/L).
        '''

        y0 = [dose, 0.0, 0.0]
        sol = odeint(self.ode_system, y0, t, args=(ka, ke, kcp, kpc))

        # concentration is amount in central compartment / Volume of central compartment
        return sol[:, 1] / V

    def get_initial_guesses(self):
        '''
        Get initial parameter guesses for the optimization algorithm.

        :return: list of float values [ka, ke, kcp, kpc, V].
        '''
        return [1.5, 0.2, 0.5, 0.5, 30.0]

    def get_real_parameters(self, popt):
        '''
        Translate raw optimization parameters into clinical parameters.

        :param popt: list containing [ka, ke, kcp, kpc, V].
        :return: dictionary mapping parameter names to a tuple of (value, unit).
        '''
        ka, ke, kcp, kpc, V = popt
        # Beta phase half-life (approximation for slow elimination phase)
        # We use central elimination half-life for simplicity here
        half_life_elim = np.log(2) / ke
        clearance = V * ke

        return {
            "Absorption Constant (ka)": (ka, "1/h"),
            "Elimination Constant (ke)": (ke, "1/h"),
            "Central->Peripheral Exchange (kcp)": (kcp, "1/h"),
            "Peripheral->Central Exchange (kpc)": (kpc, "1/h"),
            "Central Compartment Volume (V)": (V, "L"),
            "--- CLINICAL PARAMETERS ---": (None, ""),
            "Elimination Half-life (t 1/2)": (half_life_elim, "h"),
            "Body Clearance (Cl)": (clearance, "L/h")
        }


class IV_TwoCompartmentModel(BaseModel):
    '''
    Two-Compartment Model for IV Bolus administration.
    Drug starts in Central Comp and distributes to Peripheral.
    '''

    def get_name(self): return "IV Bolus (2-Comp)"

    def ode_system(self, y, t, ke, kcp, kpc):
        # y = [Amount_Central, Amount_Peripheral]
        A_cent, A_peri = y

        # Central loses to elimination (-ke) and to peripheral (-kcp)
        # Central gains from peripheral (+kpc)
        dAcent = -ke * A_cent - kcp * A_cent + kpc * A_peri

        # Peripheral gains from central (+kcp) and loses to central (-kpc)
        dAperi = kcp * A_cent - kpc * A_peri

        return [dAcent, dAperi]

    def simulate(self, t, dose, ke, kcp, kpc, V):
        # INITIAL CONDITION: All drug is in Central Compartment at t=0
        y0 = [dose, 0.0]

        sol = odeint(self.ode_system, y0, t, args=(ke, kcp, kpc))

        # Concentration = Amount_Central / Volume_Central
        return sol[:, 0] / V

    def get_initial_guesses(self):
        # ke, kcp, kpc, V
        # Indomethacin needs V around 10-15L.
        # kcp/kpc describe the fast distribution phase.
        return [0.2, 0.5, 0.5, 15.0]

    def get_real_parameters(self, popt):
        ke, kcp, kpc, V = popt

        # Complex derivation of alpha (fast) and beta (slow) half-lives
        k12 = kcp;
        k21 = kpc;
        k10 = ke
        sum_k = k12 + k21 + k10
        root = np.sqrt(sum_k ** 2 - 4 * k21 * k10)

        alpha = (sum_k + root) / 2
        beta = (sum_k - root) / 2

        t_half_alpha = np.log(2) / alpha
        t_half_beta = np.log(2) / beta
        clearance = V * ke

        return {
            "Elimination Constant (ke)": (ke, "1/h"),
            "Distr. Constant (kcp)": (kcp, "1/h"),
            "Redistr. Constant (kpc)": (kpc, "1/h"),
            "Volume Central (V)": (V, "L"),
            "--- CLINICAL PARAMETERS ---": (None, ""),
            "Alpha Half-life (Fast/Distr)": (t_half_alpha, "h"),
            "Beta Half-life (Slow/Elim)": (t_half_beta, "h"),
            "Total Clearance (Cl)": (clearance, "L/h")
        }
