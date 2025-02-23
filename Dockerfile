FROM arm64v8/ubuntu:24.04

RUN apt-get update && apt-get install -y python3.12 python3-pip