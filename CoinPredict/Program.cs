using Microsoft.ML;
using Microsoft.ML.Data;
using Newtonsoft.Json.Linq;

namespace CoinPredict
{
    internal class Program
    {
        private static readonly DateTime epoch = new(1970, 1, 1, 0, 0, 0, DateTimeKind.Utc);


        public class ModelInput
        {
            [LoadColumn(0)] public string Time { get; set; }
            [LoadColumn(1)] public float Price { get; set; }
            [LoadColumn(2)] public float Volume { get; set; }
        }

        public class ModelOutput
        {
            [ColumnName("Score")]
            public float[] Score; // Change Score type to float array
        }

        public static void Predict()
        {
            // Create new MLContext
            MLContext mlContext = new();

            // Load data
            var dataView = mlContext.Data.LoadFromTextFile<ModelInput>("dogecoin_data.csv", separatorChar: ',');

            // Define pipeline
            var pipeline = mlContext.Transforms.CopyColumns(outputColumnName: "Label", inputColumnName: "Price")
    .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "TimeEncoded", inputColumnName: "Time"))
    .Append(mlContext.Transforms.Concatenate("Features", "TimeEncoded", "Volume"))
    .Append(mlContext.Transforms.NormalizeMinMax("Features"))
    .Append(mlContext.Regression.Trainers.Sdca(labelColumnName: "Label", featureColumnName: "Features"))
    .Append(mlContext.Transforms.CopyColumns(outputColumnName: "Score", inputColumnName: "Features"))
    .Append(mlContext.Transforms.NormalizeMinMax(outputColumnName: "Score"));

            // Train model
            var model = pipeline.Fit(dataView);

            // Make a prediction
            var predictionEngine = mlContext.Model.CreatePredictionEngine<ModelInput, ModelOutput>(model);
            var sampleData = new ModelInput { Time = "1710219064", Volume = 1000000 };
            var prediction = predictionEngine.Predict(sampleData);

            Console.WriteLine($"Predicted price for {sampleData.Time} with volume {sampleData.Volume}: {prediction.Score}");
        }

        static void Main(string[] args)
        {
            try
            {
                //GetData().Wait();
                Predict();
            }
            catch (Exception ex)
            {
                // Just rethrow the exception without changing the stack trace
                throw ex;
            }
        }
        async static Task GetData()
        {
            string coin = "dogecoin";
            string url = $"https://api.coingecko.com/api/v3/coins/{coin}/market_chart?vs_currency=usd&days=max&interval=daily";

            using (HttpClient client = new())
            {
                var response = await client.GetStringAsync(url);
                var data = JObject.Parse(response);

                using (StreamWriter file = new($"{coin}_data.csv"))
                {
                    file.WriteLine("Time,Price,Volume");

                    var prices = (JArray)data["prices"];
                    var volumes = (JArray)data["total_volumes"];

                    for (int i = 0; i < prices.Count; i++)
                    {
                        var unixTime = (long)prices[i][0];
                        var time = unixTime;
                        var price = (decimal)prices[i][1];
                        var volume = (decimal)volumes[i][1];

                        file.WriteLine($"{time},{price},{volume}");
                    }
                }
            }
        }

        public static DateTime UnixTimeStampToDateTime(double unixTimeStamp)
        {
            // Unix timestamp is seconds past epoch
            DateTime dateTime = new(1970, 1, 1, 0, 0, 0, 0, DateTimeKind.Utc);
            dateTime = dateTime.AddSeconds(unixTimeStamp).ToLocalTime();
            return dateTime;
        }
    }
}
