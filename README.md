# zurich_insurance
Github does not let me upload the final .csv files because they are too big, but they can be generated with the code and stored locally.

If something does not work for you, I can send you my python environment in a .venv zip file.

Also, the order of execution of the scripts is: 
1) preprocess_inputs.py
2) prepare_x.py
3) preprocess_outputs.py

It is not a very proffesional organization, but a simple one. After executing the 3 files, you should have a df with the data ready to train a model (maybe a neural network), for the first of the two problems: "Given the hazard and the singe location account datasets provided by Zurich, define a proxy cat model that is 
able to predict the financial losses (average annual loss, quantiles) for a single location."  For the Y (or expected outputs, we have a lot of columns, but I don´t know if we have to predict all of them. Maybe we can start with AAL_GU and AAL_GR.

I now will try some models, and then I can try to prepare the Y for the multiple locations outputs, but I have to think a bit how to approach it.

Feel free to ask me if there is any question. Also note me if there is some bug, or I missed some data in the preprocessing.
