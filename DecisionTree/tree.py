class tree:
    #op = node label
    #cls = classifications (e.g. 0 or 1)
    #kids = children nodes (sub trees)
    def __init__(self, op = None, cls = None):
        self.op = op # what attribiute it's evaluation
        self.kids = [None, None]
        self.cls = cls # class i.e. classification it evaluates to
