# NLP_App
This application aims to classify german texts. The dataset used for the training was downloaded from https://tblock.github.io/10kGNAD/.
In this project you find:
* A **Jupyter Notebook file**, in which you find all the steps for the text classification (pre-processing, training, testing, evaluation, saving the model and upload it for the classification of new texts). A *REST API* is added in the same file, to predict the class of a given text.
* A **streamlit application** aims to predict the class for an input text. To run this file use this command: "streamlit run streamlit_app.py"
* To deploy the streamlit_app.py on **Docker**: 
	* docker build -t streamlitapp:latest .
	* docker run -p 8501:8501 streamlitapp:latest
