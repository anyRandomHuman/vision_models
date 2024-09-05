class a:
    def __init__(self, par=0) -> None:
        self.par = par
    
class b(a):
    def __init__(self, par=0) -> None:
        super().__init__(par)