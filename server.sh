# adapte le chemin du .gguf si besoin
llama-server \
  -m llama_models/mistral-7b-instruct-v0.2.Q4_K_M.gguf \
  --host 127.0.0.1 --port 8000 \
  --ctx-size 4096 --batch-size 512 --n-gpu-layers 99