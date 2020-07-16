FROM python:3.7
MAINTAINER Emma

ADD SNR_measurement.py /

RUN apt-get update
RUN apt-get install -y build-essential apt-utils
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y\
        python3-pip\
        python3-opencv\
        python3-scipy\
        python3-matplotlib\
        python3-skimage

RUN pip3 install numpy pydicom opencv-python scikit-image

CMD [ "python", "./SNR_measurement.py" ]

