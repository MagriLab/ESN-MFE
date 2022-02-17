To start, run Generating_data.py to create the data for the Reyndols, Re=400. This is necessary as the resulting file is too large to be uploaded in github.
Once the data is saved, it is possible to run the other scripts in the folder, including the Jupyeter Notebooks in ./Post_Processing/ which generate the plots from the paper.

The different files are

- Generating_data.py: (i) solves the MFE (Moehlis et al., (2004)) equations with a fourth order Runge-Kutta method; (ii) computes the Lyapunov exponents spectrum using the QR algorithm by Ginelli et al. (2007).

- Lyapunov_expoent.py: Computes the dominant Lyapunov exponent as the average of the  perturbation evolution from different points along the traqjectory.
