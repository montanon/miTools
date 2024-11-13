class ArgumentTypeError(TypeError):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


class ArgumentKeyError(KeyError):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


class ArgumentValueError(ValueError):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


class ArgumentStructureError(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


class ColumnValidationError(ArgumentValueError):
    pass


class IndexValidationError(ArgumentValueError):
    pass


class ColumnTypeError(ArgumentTypeError):
    pass


class IndexTypeError(ArgumentTypeError):
    pass


class ValuesTypeError(ArgumentTypeError):
    pass
