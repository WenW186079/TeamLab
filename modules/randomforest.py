import numpy as np
from decisiontree import DecisionTree 

class RandomForest():
    def __init__(self, n_estimators=50,random_seed = 42, **tree_params):
        self.n_estimators = n_estimators
        self.tree_params = tree_params
        self.random_seed = random_seed
        self.forest = []

    def train(self, X_train, Y_train):
        '''
        
        Forest Creation:
        Create a decision tree by bootstrapping a sample from the training data, 
        train the decision tree on these samples, 
        and then add the trained tree to the forest.
        
        '''
        
        np.random.seed(self.random_seed)
        for _ in range(self.n_estimators):
            
            # Create a decision tree
            tree = DecisionTree(max_depth=self.tree_params.get('max_depth', 30),
                                min_samples_leaf=self.tree_params.get('min_samples_leaf', 1),
                                min_information_gain=self.tree_params.get('min_information_gain', 0.0),)
                                
            # Randomly selects elements from a given array, with replacement
            elements = np.random.choice(len(X_train), len(X_train), replace=True)
            
            # Train the decision tree on the sampled data
            tree.train(X_train[elements], Y_train[elements])

            # Add the trained tree to the forest
            self.forest.append(tree)

    def predict_proba(self, X_set):
        '''

        Using a majority voting mechanism 
        by accumulating votes for each class from each tree 
        and normalizing them to probabilities. 
        
        '''
        # Initialize
        pred_probs = np.zeros((len(X_set), len(self.forest[0].train_labels)))
        # Accumulate predictions from each tree in the forest
        for tree in self.forest:
            # Predict labels for the input features
            preds = tree.predict(X_set)
            # Update the accumulated predictions
            for i, pred in enumerate(preds):
                pred_probs[i, pred] += 1
        # Normalize
        pred_probs /= self.n_estimators
        return pred_probs

    def predict(self, X_set):
        '''
        The final prediction for each sample is the class with the highest probability.
        
        '''
        # Get predicted probabilities for each sample
        pred_probs = self.predict_proba(X_set)

        # Choose the highest one
        return np.argmax(pred_probs, axis=1)
