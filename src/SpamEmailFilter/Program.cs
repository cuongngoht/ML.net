using Microsoft.ML;
using Microsoft.ML.Data;

namespace ConsoleApp2
{
    internal class Program
    {
        static void Main(string[] args)
        {
            try
            {
                MLContext context = new();

                // Load the data
                IDataView data = context.Data.LoadFromTextFile<SpamInput>("./email.csv", separatorChar: ',');

                // Define the pipeline
                EstimatorChain<Microsoft.ML.Transforms.ColumnCopyingTransformer> pipeline = context.Transforms.Text.FeaturizeText("Features", "Message")
                          .Append(context.BinaryClassification.Trainers.SdcaLogisticRegression())
                          .Append(context.Transforms.CopyColumns(outputColumnName: "Features", inputColumnName: "Message"));

                // Train the model
                TransformerChain<Microsoft.ML.Transforms.ColumnCopyingTransformer> model = pipeline.Fit(data);

                // Create a prediction engine
                PredictionEngine<SpamInput, SpamPrediction> predictionEngine = context.Model.CreatePredictionEngine<SpamInput, SpamPrediction>(model);

                // Test the model
                SpamInput testEmail = new() { Message = "How are you there?" };
                SpamPrediction prediction = predictionEngine.Predict(testEmail);

                Console.WriteLine($"The message '{testEmail.Message}' is {(prediction.Prediction ? "spam" : "not spam")}");
            }
            catch (Exception)
            {
                // Just rethrow the exception without changing the stack trace
                throw;
            }
        }
    }

    public class SpamInput
    {
        [LoadColumn(0)]
        public string Message { get; set; }

        [LoadColumn(1)]
        public bool Label { get; set; }
    }

    public class SpamPrediction
    {
        [ColumnName("PredictedLabel")]
        public bool Prediction { get; set; }
    }
}
