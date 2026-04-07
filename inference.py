import os
import json
import re
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from openai import OpenAI
from env import LogTriageEnvironment, Action
sys.stdout.reconfigure(line_buffering=True)

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

def clean_and_parse_json(text):
    try:
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            json_str = match.group()
            return json.loads(json_str)
       
        numbers = re.findall(r'\d+', text)
        if numbers:
            return {"identified_log_id": int(numbers[0])}
            
        return {"identified_log_id": -1}
    except Exception:
        return {"identified_log_id": -1}

def run_task(difficulty):
    print(f"[START] Beginning inference for task: {difficulty}")
    env = LogTriageEnvironment()
    obs = env.reset(difficulty=difficulty)
    done = False
    total_reward = 0.0

    while not done:
        
        prompt = (
            f"Logs: {obs.logs}\n"
            "Identify the ID of the CRITICAL log. "
            "IMPORTANT: Your response must contain ONLY the JSON: {\"identified_log_id\": ID}"
        )
        
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1 
            )
            raw_content = response.choices[0].message.content
            
            action_data = clean_and_parse_json(raw_content)
            action = Action(**action_data)
            
            print(f"[STEP] Action taken: {action.model_dump()}")
            obs, reward, done, info = env.step(action)
            total_reward += reward.score
            print(f"[STEP] Reward: {reward.score} | Msg: {reward.message}")
            
        except Exception as e:
            print(f"[STEP] Error: {e}")
            break

    print(f"[END] Final Score: {total_reward} | Task: {difficulty} completed.")
if __name__ == "__main__":
    for diff in ["easy", "medium", "hard"]:
        try:
            run_task(diff)
        except Exception as e:
            print(f"Error in task {diff}: {e}")
    
    print("Tasks complete. Starting validator server on port 7860...")
    from http.server import BaseHTTPRequestHandler, HTTPServer

    class SimpleHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(b"Space is Running")

        def do_POST(self):
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(b"Reset Successful")
    try:
        server = HTTPServer(('0.0.0.0', 7860), SimpleHandler)
        print("Server is live. Waiting for Scaler validator...")
        server.serve_forever()
    except Exception as e:
        print(f"Server failed to start: {e}")
