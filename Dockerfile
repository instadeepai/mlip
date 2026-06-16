FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y git wget

RUN pip install "mlip[cuda13]" huggingface_hub notebook

RUN wget https://raw.githubusercontent.com/instadeepai/mlip/refs/heads/main/tutorials/simulation_tutorial.ipynb \
         https://raw.githubusercontent.com/instadeepai/mlip/refs/heads/main/tutorials/model_training_tutorial.ipynb \
         https://raw.githubusercontent.com/instadeepai/mlip/refs/heads/main/tutorials/model_addition_tutorial.ipynb \
         https://raw.githubusercontent.com/instadeepai/mlip/refs/heads/main/tutorials/advanced_simulation_tutorial.ipynb \
         https://raw.githubusercontent.com/instadeepai/mlip/refs/heads/main/tutorials/moe_training_and_inference_tutorial.ipynb \
         https://raw.githubusercontent.com/instadeepai/mlip/refs/heads/main/tutorials/hessian_model_training_tutorial.ipynb

EXPOSE 8888

CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
