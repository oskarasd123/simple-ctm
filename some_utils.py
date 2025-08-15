





class ExpDecay:
    def __init__(self, initial_value : float, a : float, recursion_steps : int = 0):
        self.a = a
        self.value = initial_value
        self.sub_decay = None
        if recursion_steps > 0:
            self.sub_decay = ExpDecay(initial_value, a, recursion_steps-1)
    
    def update(self, value):
        if self.sub_decay:
            self.sub_decay.update(value)
            self.value = self.value * (1-self.a) + self.sub_decay.value * self.a
        else:
            self.value = self.value * (1-self.a) + value * self.a
        return self.value