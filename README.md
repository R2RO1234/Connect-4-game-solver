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
Open a cmd

Backend:
```
cd backend
start_server.cmd
```
If start_server.cmd does not work run ./start_server.cmd instead.

start_server builds the frontend and starts the server. If your frontend is already built (npm run build), just use
```
fastapi dev server.py
```
When server is live, go to http://127.0.0.1:8000.

## running frontend for development:
```
cd frontend
npm run dev
```

## hosted website
This project is also hosted on this [website](https://connect-4-game-solver.onrender.com/)
however, the ai model is very slow on it.


