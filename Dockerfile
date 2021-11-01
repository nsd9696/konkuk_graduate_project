# lightweight python
FROM ubuntu:20.04

RUN apt-get update && apt-get install python3 python3-pip -y

# Copy local code to the container image.
ENV APP_HOME /app
WORKDIR $APP_HOME
COPY . ./

RUN ls -la $APP_HOME/

# Install dependencies
RUN pip install -r requirements.txt

# Run the streamlit on container startup
CMD [ "streamlit run app.py" ]