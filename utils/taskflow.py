from node_perturbation_classification_task import NodePerturbationClassificationTask

class TaskFlow:
    def __init__(self, args):
        self.args = args
        if args.task == 'node_perturbation_classification':
            self.task = NodePerturbationClassificationTask(args)

    def run(self):
        self.task.run()

