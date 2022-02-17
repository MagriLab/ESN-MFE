To start, run Generating_data.py to create the data for the Reyndols, Re=400, or download from here. This is necessary as the resulting file is too large to be uploaded in github.
Once the data is available, it is possible to run the other scripts in the folder, including the Jupyeter Notebooks in ./Post_Processing/ which generate the plots from the paper.

The different files are:

- Generating_data.py: (i) solves the MFE (Moehlis et al., (2004)) equations with a fourth order Runge-Kutta method; (ii) computes the Lyapunov exponents spectrum using the QR algorithm by Ginelli et al. (2007).

- Lyapunov_expoent.py: Computes the dominant Lyapunov exponent as the average of the  perturbation evolution from different points along the traqjectory.

- Val.py: Trains and validates Echo State Networks with Recycle Validation and Single Shot Validation (Racca and Magri, (2021); https://github.com/MagriLab/Robust-Validation-ESN).

- Functions.py: Echo State Network implementation.
- Val_Functions.py: Recycle Validation and Single Shot Validation implementation.
- Functions_Test.py: Additional Echo State Network functions needed to test the network.

- Precision_Recall.py: Computes precision and recall in the real-time monitoring of extreme events for different prediction times in the test set.

- Extreme_Events_PH.py: Computes the Prediction Horizon of multiple extreme events in the test set.

- Stats_Runs.py: Generates long time series of the ESN starting from multiple points in the training set.

- Re_Control.py: Applies the control strategy based on increasing the Reynolds number when an event is predicted by the Echo State Network.
- Control_Functions.py: Functions to generate suppressed time series by reintegrating the governing equations with different Reynolds.

Post processing files are:

Time
