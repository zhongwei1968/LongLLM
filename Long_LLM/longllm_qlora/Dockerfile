FROM nvcr.io/nvidia/pytorch:24.01-py3

WORKDIR /app

COPY requirements.txt /app/
RUN pip install -r requirements.txt

COPY . /app/

# RUN pip install --upgrade --force-reinstall --no-cache-dir torch==2.2.0 triton --index-url https://download.pytorch.org/whl/cu121
RUN pip install "unsloth[cu121-ampere-torch220] @ git+https://github.com/unslothai/unsloth.git"

RUN pip install -e .[deepspeed,accelerate,metrics,bitsandbytes,peft]
RUN pip install flash-attn==2.3.3 --no-build-isolation
RUN pip uninstall transformer_engine -y

COPY llama.py  /usr/local/lib/python3.10/dist-packages/unsloth/models/llama.py

VOLUME [ "/root/.cache/huggingface/", "/app/data", "/app/output" ]
EXPOSE 7860

CMD [ "./train.sh" ]
