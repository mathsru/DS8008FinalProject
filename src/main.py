from TrainAndEvaluateClassifier import TrainAndEvaluateClassifer
from TrainAndEvaluateLSTM import TrainAndEvaluateLSTM

print("Loading dataset...")

# Step 1: Get training data
data = TrainAndEvaluateClassifer(
    NumberOfArtificialArticlesTouse=50,
    UseAll=1
)

print("Training LSTM...")

# Step 2: Train + evaluate
accuracy = TrainAndEvaluateLSTM(data)

print("Final Accuracy:", accuracy)