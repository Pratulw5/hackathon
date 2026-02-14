Build commands:
Server:
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port $PORT
Locally:
py -3.10 -m venv venv 
venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload