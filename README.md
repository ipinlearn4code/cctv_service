pip install torch==2.5.1 torchvision --index-url https://download.pytorch.org/whl/cu118 
uvicorn app.main:app --host 0.0.0.0 --port 18888
