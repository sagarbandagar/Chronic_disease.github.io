import pandas as pd
from joblib import load


class DiseasePredictor:
    def __init__(self, model_path='./saved_model/random_forest.joblib'):
        self.model_path = model_path
        self.symptoms = {
            'itching': 0, 'skin_rash': 0, 'nodal_skin_eruptions': 0, 'continuous_sneezing': 0,
            # Add all other symptoms here...
        }

    def set_symptom(self, symptom, value):
        if symptom in self.symptoms:
            self.symptoms[symptom] = value
        else:
            print(f"Symptom '{symptom}' not found in the list of symptoms.")

    def predict_disease(self):
        # Prepare Test Data
        df_test = pd.DataFrame([list(self.symptoms.values())], columns=list(self.symptoms.keys()))

        # Load pre-trained model
        try:
            clf = load(self.model_path)
            result = clf.predict(df_test)
            return result[0]
        except Exception as e:
            print(f"Error loading or making predictions: {e}")
            return None


if __name__ == '__main__':
    predictor = DiseasePredictor()
    
    # Set symptoms (1 for present, 0 for absent)
    predictor.set_symptom('itching', 1)
    predictor.set_symptom('skin_rash', 1)
    # Add other symptoms as needed
    
    # Predict disease
    predicted_disease = predictor.predict_disease()
    
    if predicted_disease:
        print(f"Predicted Disease: {predicted_disease}")
    else:
        print("Failed to make a prediction.")
