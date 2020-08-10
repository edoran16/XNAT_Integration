FROM python:3.7
MAINTAINER Emma

ADD SNR_UNI_measurement.py /
COPY sfuncs.py /
COPY ufuncs.py /

RUN apt-get update
RUN apt-get install -y build-essential apt-utils
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y\
        python3-pip\
        python3-opencv\
        python3-scipy\
        python3-matplotlib\
        python3-skimage\
        python3-pandas\
        python3-regex

RUN pip3 install numpy pydicom opencv-python scikit-image pandas regex

CMD [ "python", "./SNR_UNI_measurement.py" ]

