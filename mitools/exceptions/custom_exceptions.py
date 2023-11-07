

class ArgumentTypeError(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)

class ArgumentValueError(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)

class ArgumentStructureError(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)
