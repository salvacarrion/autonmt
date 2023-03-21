class Score:

    def __init__(self, engine, model_name, eval_name, train_ds=None, eval_ds=None, config=None):
        self.engine = engine
        self.model_name = model_name
        self.eval_name = eval_name
        self.translations = {}
        self.config = config

        self.train_ds = train_ds
        self.eval_ds = eval_ds
