\Agents has all the neural networks the RL stuff
\Enviroment has the rest
\Weights stores the weights of the SSA and CAM along with other files
\Policies is the policies on validation time
\Test Policies files related to test time

On this directory it is the train and validation functions


test.ipynb is the results on test time

the majority of files should be running without any adjustments.
On Enviroment\utils.py
there is the dir_path = os.path.dirname(os.path.realpath(__file__))
which reads the directory and it runs accordingly.
On Jupyter Notebooks, I have not used that, because I was running from another
directory. 