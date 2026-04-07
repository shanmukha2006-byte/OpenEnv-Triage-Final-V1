from pydantic import BaseModel
from typing import List, Dict, Any
class Observation(BaseModel):
    logs: List[str]
    remaining_attempts: int

class Action(BaseModel):
    identified_log_id: int

class Reward(BaseModel):
    score: float
    message: str

class LogTriageEnvironment:
    def __init__(self):
        self.state_data = {}

    def _generate_logs(self, difficulty: str):
        if difficulty == "easy":
            return ["[ID: 0] INFO: OK", "[ID: 1] CRITICAL: Disk Full", "[ID: 2] INFO: OK"], 1
        elif difficulty == "medium":
            logs = [f"[ID: {i}] INFO: User Access" for i in range(10)]
            logs.insert(5, "[ID: 5] CRITICAL: Auth Bypass Attempt")
            return logs, 5
        else: 
            logs = [f"[ID: {i}] INFO: System Heartbeat" for i in range(20)]
            logs.insert(12, "[ID: 12] CRITICAL: Kernel Panic")
            return logs, 12

    def reset(self, difficulty="easy") -> Observation:
        logs, target_id = self._generate_logs(difficulty)
        self.state_data = {
            "logs": logs,
            "target_id": target_id,
            "attempts": 3,
            "done": False,
            "difficulty": difficulty
        }
        return Observation(logs=logs, remaining_attempts=3)

    def state(self) -> Observation:
        return Observation(
            logs=self.state_data.get("logs", []),
            remaining_attempts=self.state_data.get("attempts", 0)
        )

    def step(self, action: Action) -> tuple[Observation, Reward, bool, Dict[str, Any]]:
        if self.state_data.get("done", False):
            return self.state(), Reward(score=0.0, message="Finished"), True, {}

        self.state_data["attempts"] -= 1
        
        if action.identified_log_id == self.state_data["target_id"]:
            score = 1.0
            msg = "Success: Correct ID!"
            self.state_data["done"] = True
        else:
            score = 0.0
            msg = f"Wrong ID. Attempts left: {self.state_data['attempts']}"
            if self.state_data["attempts"] <= 0:
                self.state_data["done"] = True

        obs = self.state()
        reward = Reward(score=score, message=msg)
        return obs, reward, self.state_data["done"], {}