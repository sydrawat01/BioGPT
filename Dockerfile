FROM ubuntu:latest

WORKDIR /home/BioGPT

COPY . .

RUN ./install.sh
