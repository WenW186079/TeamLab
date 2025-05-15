'''

 This file contains the evaulation functions. eg: f-score function

'''

class Evaluation:
    def __init__(self, predictions, labels, class_label=None):
        self.predictions = predictions
        self.labels = labels
        self.class_label = class_label

    def true_positive(self):
        tp = 0
        for pred, act in zip(self.predictions, self.labels):
            if pred == self.class_label and act == self.class_label:
                tp += 1
        return tp

    def false_positive(self):
        fp = 0
        for pred, act in zip(self.predictions, self.labels):
            if pred == self.class_label and act != self.class_label:
                fp += 1
        return fp

    def true_negative(self):
        tn = 0
        for pred, act in zip(self.predictions, self.labels):
            if pred != self.class_label and act != self.class_label:
                tn += 1
        return tn
    
    def false_negative(self):
        fn = 0
        for pred, act in zip(self.predictions, self.labels):
            if pred != self.class_label and act == self.class_label:
                fn += 1
        return fn
    
    def recall(self):
        tp = self.true_positive()
        fn = self.false_negative()
        if tp + fn == 0:
            return 0.0
        return round(tp / (tp + fn), 4)
    
    def precision(self):
        tp = self.true_positive()
        fp = self.false_positive()
        if tp + fp == 0:
            return 0.0
        return round(tp / (tp + fp), 4)
    
    def f1_score(self):
        precision_val = self.precision()
        recall_val = self.recall()
        if precision_val + recall_val == 0:
            return 0.0
        return round(2.0 * (precision_val * recall_val) / (precision_val + recall_val), 4)
    
    def accuracy(self):
        correct_predictions = sum(1 for pred, true_label in zip(self.predictions, self.labels) if pred == true_label)
        return round(correct_predictions / len(self.labels), 4)
    
    def evaluate(self):
        precision = self.precision()
        recall = self.recall()
        f1 = self.f1_score()
        accuracy = self.accuracy()
        return precision, recall, f1, accuracy

    def evaluate_multiclass_metrics(self, model_name):
        unique_labels = set(self.labels)
        precision_scores = {}
        recall_scores = {}
        f1_scores = {}
        for label in unique_labels:
            evaluator = Evaluation(self.predictions, self.labels, class_label=label)
            precision_scores[label] = evaluator.precision()
            recall_scores[label] = evaluator.recall()
            f1_scores[label] = evaluator.f1_score()
        
        accuracy_score = self.accuracy()
    
        print(f'Performance for {model_name}:')
        print("Precision Scores:", precision_scores)
        print("Recall Scores:", recall_scores)
        print("F1 Scores:", f1_scores)
        print("Accuracy Score:", accuracy_score)
    
        overall_precision = round(sum(precision_scores.values()) / len(precision_scores), 4)
        overall_recall = round(sum(recall_scores.values()) / len(recall_scores), 4)
        overall_f1 = round(sum(f1_scores.values()) / len(f1_scores), 4)
        
        print("Overall Precision:", overall_precision)
        print("Overall Recall:", overall_recall)
        print("Overall F1 Score:", overall_f1)
        print("Overall Accuracy:", accuracy_score)
        
        return precision_scores, recall_scores, f1_scores, accuracy_score, overall_precision, overall_recall, overall_f1


        
'''

####### Example use ######
forest_evaluation = Evaluation(forest_predictions, y_test)
forest_evaluation.evaluate_multiclass_metrics('random forest')


'''
