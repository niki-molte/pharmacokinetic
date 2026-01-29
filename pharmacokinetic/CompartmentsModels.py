import numpy as np
from scipy.integrate import odeint

from abc import ABC, abstractmethod


class BaseModel(ABC):
    '''
    Abstract base class for Pharmacokinetic models.
    It defines the structure for simulation and parameter translation.
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

        :return: string containing the model name.
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
    Implementation of the One-Compartment model with absorption.
    '''

    def get_name(self):
        '''
        Get the descriptive name of the model.

        :return: string representing the model name.
        '''
        return "One-Compartment Model"

    def ode_system(self, y, t, ka, ke):
        '''
        Define the system of ordinary differential equations.
        It describes the transfer of drug from the gut to the blood.

        :param y: list containing current amounts [Amount_gut, Amount_blood].
        :param t: current time point, it is required by odeint, not used in equations.
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
        # calculating significant derived parameters
        half_life = np.log(2) / ke
        clearance = V * ke

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
    Implementation of the One-Compartment Model for IV Bolus administration.
    '''

    def get_name(self):
        '''
        Get the descriptive name of the model.

        :return: string containing the model name.
        '''
        return "IV Bolus (1-Comp)"

    def ode_system(self, y, t, ke):
        '''
        Define the system of ordinary differential equations.
        Describes the drug excretion from the blood.

        :param y: list containing current amounts [Amount_blood].
        :param t: current time point required by odeint not used in equations.
        :param ke: elimination constant.
        :return: list of derivatives [dAblood].
        '''
        A_blood = y

        dAdt = -ke * A_blood
        return np.ravel([dAdt])

    def simulate(self, t, dose, ke, V):
        '''
        Simulate the drug concentration over time.

        :param t: array-like object containing time points.
        :param dose: float representing the administered dose.
        :param ka: absorption constant (1/h).
        :param ke: elimination constant (1/h).
        :param V: volume of distribution (L).
        :return: array-like object containing simulated concentration values (mg/L).
        '''
        y0 = dose

        # solve the ODE system
        # args=(ke,) send ke as additional parameter
        sol = odeint(self.ode_system, y0, t, args=(ke,))

        # return concentration (Amount in blood / Volume)
        return sol.flatten() / V

    def get_initial_guesses(self):
        '''
         Get initial parameter guesses for the optimization algorithm.

         :return: list of float values [ke, V].
         '''
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
    Implementation of the Two-Compartment model.
    It includes a central compartment (blood) and a peripheral compartment (tissues).
    '''

    def get_name(self):
        '''
        Get the descriptive name of the model.

        :return: string containing the model name.
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

        # concentration = central compartment / Volume of central compartment
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
        # central elimination half life
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
    Drug starts directly in the Central Compartment and distributes to the Peripheral Compartment.
    '''

    def get_name(self):
        '''
        Get the name of the model.

        :return: string containing the model name.
        '''
        return "IV Bolus (2-Comp)"

    def ode_system(self, y, t, ke, kcp, kpc):
        '''
        Define the system of ordinary differential equations.
        It describes the transfer between central and peripheral compartments and elimination.

        :param y: list containing current amounts [Amount_Central, Amount_Peripheral].
        :param t: current time point (required by odeint).
        :param ke: elimination rate constant from the central compartment.
        :param kcp: distribution rate constant from central to peripheral.
        :param kpc: redistribution rate constant from peripheral to central.
        :return: list of derivatives [dAcent, dAperi].
        '''
        # y = [Amount_Central, Amount_Peripheral]
        A_cent, A_peri = y

        # central loses to elimination (-ke) and to peripheral (-kcp)
        # central gains from peripheral (+kpc)
        dAcent = -ke * A_cent - kcp * A_cent + kpc * A_peri

        # peripheral gains from central (+kcp) and loses to central (-kpc)
        dAperi = kcp * A_cent - kpc * A_peri

        return [dAcent, dAperi]

    def simulate(self, t, dose, ke, kcp, kpc, V):
        '''
        Simulate the drug concentration over time.

        :param t: array-like object containing time points.
        :param dose: float representing the administered IV bolus dose.
        :param ke: elimination rate constant (1/h).
        :param kcp: distribution rate constant central->peripheral (1/h).
        :param kpc: redistribution rate constant peripheral->central (1/h).
        :param V: volume of the central compartment (L).
        :return: array-like object containing simulated concentration values (mg/L).
        '''
        # initial condition: drug is all in central compartment at t=0
        y0 = [dose, 0.0]

        sol = odeint(self.ode_system, y0, t, args=(ke, kcp, kpc))

        # concentration = Amount_Central / Volume_Central
        return sol[:, 0] / V

    def get_initial_guesses(self):
        '''
        Get initial parameter guesses for the optimization algorithm.

        :return: list of float values [ke, kcp, kpc, V].
        '''
        # ke, kcp, kpc, V
        return [0.2, 0.5, 0.5, 15.0]

    def get_real_parameters(self, popt):
        '''
        Translate raw optimization parameters into clinical parameters.

        :param popt: list containing [ke, kcp, kpc, V].
        :return: dictionary mapping parameter names to a tuple of (value, unit).
        '''
        ke, kcp, kpc, V = popt

        # beta and alpha half life
        k12 = kcp
        k21 = kpc
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