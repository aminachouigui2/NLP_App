FROM python:3.8-slim-buster
COPY . /app
WORKDIR /app
RUN pip install matplotlib==3.2.1
RUN pip install seaborn==0.11.2
RUN pip install scikit-learn>=0.22
RUN pip install -r requirements.txt
EXPOSE 8501
ENTRYPOINT ["streamlit","run"]
CMD ["streamlit_app.py"]
