FROM ubuntu:latest

WORKDIR /home/BioGPT

COPY . .

RUN chmod +x install.sh
RUN ./install.sh
