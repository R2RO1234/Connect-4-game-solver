# MAIS202 - Connect4 AI Agent
Final team project for MAIS 202 Bootcamp.

## Prerequisites
- Python 3.12 is required (python 3.13+ is not compatible with pyarrow library)
- Node.js

## Installing dependencies
Python: 
```
pip install -r requirements.txt
```
Node.js:
```
cd frontend
npm i -D daisyui@latest
npm install tailwindcss@latest @tailwindcss/vite@latest daisyui@latest
```


## Running the application
Open two separate terminals, one for frontend, one for backend.

Backend:
```
cd backend
start_server.cmd
```
If start_server.cmd does not work run ./start_server.cmd instead.

start_server builds the frontend and starts the server. If your frontend is already built (npm run dev), just use
```
fastapi dev server.py
```

Frontend:
```
cd frontend
npm run dev
```

When server is live, go to http://127.0.0.1:8000.

