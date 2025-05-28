FROM python:3.10.12-slim-bullseye

WORKDIR /app

RUN apt-get update && apt-get install -y git wget

RUN pip install mlip "jax[cuda12]==0.4.33" huggingface_hub git+https://github.com/jax-md/jax-md.git notebook

RUN wget https://raw.githubusercontent.com/instadeepai/mlip/refs/heads/main/tutorials/simulation_tutorial.ipynb \
         https://raw.githubusercontent.com/instadeepai/mlip/refs/heads/main/tutorials/model_training_tutorial.ipynb \
         https://raw.githubusercontent.com/instadeepai/mlip/refs/heads/main/tutorials/model_addition_tutorial.ipynb

EXPOSE 8888

CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
