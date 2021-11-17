using System;
using System.IO;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace Name_Entity_Recognition_Test_Program
{
    class Program
    {
        /// <summary>
        /// 처리 컨텍스트
        /// </summary>
        private static MLContext _mlContext;

        /// <summary>
        /// <InputData, Prediction> 형식의 예측 엔진
        /// </summary>
        private static PredictionEngine<InputData, Prediction> _predEngine;

        /// <summary>
        /// 학습 관련 변수들
        /// </summary>
        private static ITransformer _trainedModel;
        static IDataView _trainingDataView;

        /// <summary>
        /// 파일 관련 변수들
        /// </summary>
        private static string _appPath => Path.GetDirectoryName(Environment.GetCommandLineArgs()[0]);
        private static string _trainDataPath => Path.Combine(_appPath, "Data", "InputData.tsv");
        private static string _testDataPath => Path.Combine(_appPath, "Data", "Test.csv");
        private static string _modelPath => Path.Combine(_appPath, "Models", "Model.zip");

        static void Main(string[] args)
        {
            // 시드는 변경 가능
            _mlContext = new MLContext(seed: 0);

            ReadFile();

            var pipeline = ProcessData();

            var trainingPipeline = BuildAndTrainModel(_trainingDataView, pipeline);

            _trainedModel = trainingPipeline.Fit(_trainingDataView);

            _predEngine = _mlContext.Model.CreatePredictionEngine<InputData, Prediction>(_trainedModel);
            InputData data = new InputData()
            {
                Name = "gold fish"
            };

            var prediction = _predEngine.Predict(data);

            Console.WriteLine($"데이터 : {data.Name} | 분류 결과 : {prediction.Entity}");
        }

        /// <summary>
        /// 지정한 파일을 읽어오기
        /// </summary>
        static void ReadFile()
        {
            if (!File.Exists(_trainDataPath)) throw new Exception("파일이 존재하지 않습니다..");

            try { _trainingDataView = _mlContext.Data.LoadFromTextFile<InputData>(_trainDataPath, hasHeader: true); }
            catch (Exception ex) { Console.WriteLine(ex.ToString()); }
        }

        /// <summary>
        /// 데이터 처리
        /// </summary>
        /// <returns></returns>
        static IEstimator<ITransformer> ProcessData()
        {
            var pipeline = _mlContext.Transforms.Conversion.MapValueToKey(inputColumnName: "Entity", outputColumnName: "Label")
                .Append(_mlContext.Transforms.Text.FeaturizeText(inputColumnName: "Name", outputColumnName: "NameFeaturized"))
                // Features 에 데이터를 연결
                .Append(_mlContext.Transforms.Concatenate("Features", "NameFeaturized"))
                // DataView 캐쉬
                .AppendCacheCheckpoint(_mlContext);

            return pipeline;
        }

        /// <summary>
        /// 모델 빌드
        /// </summary>
        /// <param name="trainingDataView"></param>
        /// <param name="pipeline"></param>
        /// <returns></returns>
        static IEstimator<ITransformer> BuildAndTrainModel(IDataView trainingDataView, IEstimator<ITransformer> pipeline)
        {
            try
            {
                var trainingPipeline = pipeline.Append(_mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy("Label", "Features"))
                .Append(_mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

                return trainingPipeline;
            }
            catch(Exception e)
            {
                Console.WriteLine(e.ToString());
                return null;
            }
        }
    }
}
