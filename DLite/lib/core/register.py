class Register:
    def __init__(self, name):
        self.name = name
        self._registry = {}
    
    def register(self, name):
        def decorator(obj):
            if name in self._registry:
                print(f"[{self.name} Register] Warning: Overwriting registration for '{name}'")
            self._registry[name] = obj
            return obj
        return decorator
    
    def get(self, name):
        if name not in self._registry:
            raise KeyError(f"{name} not found in {self.name} registry. "
                          f"Available: {list(self._registry.keys())}")
        return self._registry[name]
    
    def list_available(self):
        return list(self._registry.keys())
    
ARCH_REGISTER = Register('Architecture')
DATASET_REGISTER = Register('Dataset')
LOSS_REGISTER = Register('Loss')
OPTIMIZER_REGISTER = Register('Optimizer')
SCHEDULER_REGISTER = Register('Scheduler')
TRAINER_REGISTER = Register('Trainer')
INFERENCER_REGISTER = Register('Inferencer')