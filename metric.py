from collections import defaultdict

from data import TAG_MAP, TAG_MAP_REVERSED, BEGIN_TAGS, OUT_TAG

class NER_Metric:
    def __init__(self):

        self.num_total = defaultdict(int)
        self.num_correct = defaultdict(int)
        self.num_predict = defaultdict(int)

    def update(self, tags, preds):
        in_entity = False
        current_entity = None

        for tag, pred in zip(tags, preds):
            # one entity begain
            if tag in BEGIN_TAGS:
                self.num_total[tag] += 1
            
            if pred in BEGIN_TAGS:
                self.num_predict[pred] += 1

            if tag != pred:
                in_entity = False
            elif tag in BEGIN_TAGS:
                if in_entity:
                    self.num_correct[current_entity] += 1
                in_entity = True
                current_entity = tag
            elif tag == OUT_TAG:
                if in_entity:
                    self.num_correct[current_entity] += 1
                in_entity = False

        # B-PER I-PER <EOS>
        if in_entity:
            self.num_correct[current_entity] += 1

    @property
    def recall(self):
        return {
            tag: self.num_correct[tag] / self.num_predict[tag]
                for tag in BEGIN_TAGS
        }

    @property
    def global_recall(self):
        correct = sum(self.num_correct.values())
        predict = sum(self.num_predict.values())
        return correct / predict

    @property
    def precision(self):
        return {
            tag: self.num_correct[tag] / self.num_total[tag]
                for tag in BEGIN_TAGS
        }

    @property
    def global_precision(self):
        correct = sum(self.num_correct.values())
        total = sum(self.num_total.values())
        return correct / total

    @property
    def f1(self):
        recall = self.recall
        precision = self.precision
        return {
            tag: 2 * recall[tag] * precision[tag] / (recall[tag] + precision[tag])
                for tag in BEGIN_TAGS
        }

    @property
    def global_f1(self):
        recall = self.global_recall
        precision = self.global_precision
        return 2 * recall * precision / (recall + precision)

    def __repr__(self):
        return self.report()

    def report(self):
        tags = list(BEGIN_TAGS)
        tags_name = [TAG_MAP_REVERSED[tag].split('-')[1] for tag in tags]
        
        recall = self.recall
        precision = self.precision
        f1 = self.f1

        s = ('{:<12}' + '{:<12}' * len(tags)).format('', *tags_name) + '\n'

        s += ('{:<12}' + '{:<12.2f}' * len(tags)).format('precision', *[precision[tag] for tag in tags]) + '\n'

        s += ('{:<12}' + '{:<12.2f}' * len(tags)).format('recall', *[recall[tag] for tag in tags]) + '\n'

        s += ('{:<12}' + '{:<12.2f}' * len(tags)).format('f1', *[f1[tag] for tag in tags]) + '\n'

        s += '-' * (12 * (1 + len(tags))) + '\n'
        
        s += '{:<12}{:.2f}'.format('precision', self.global_precision) + '\n'

        s += '{:<12}{:.2f}'.format('recall', self.global_recall) + '\n'

        s += '{:<12}{:.2f}'.format('f1', self.global_f1)

        return s