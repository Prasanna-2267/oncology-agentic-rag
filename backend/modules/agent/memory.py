class AgentMemory:
    def __init__(self):
        self.history = []

    def add(self, step):
        self.history.append(step)

    def get_history(self):
        return self.history