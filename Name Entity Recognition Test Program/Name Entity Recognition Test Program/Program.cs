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
        /// <InputData, Prediction_Category> 형식의 예측 엔진
        /// </summary>
        private static PredictionEngine<InputData, Prediction_Category> C_predEngine;

        /// <summary>
        /// <InputData, Prediction_Grammar> 형식의 예측 엔진
        /// </summary>
        private static PredictionEngine<InputData, Prediction_Grammar> G_predEngine;

        /// <summary>
        /// 학습 관련 변수들
        /// </summary>
        private static ITransformer C_trainedModel;
        private static ITransformer G_trainedModel;
        static IDataView _trainingDataView;

        /// <summary>
        /// 파일 관련 변수들
        /// </summary>
        private static string _appPath => Path.GetDirectoryName(Environment.GetCommandLineArgs()[0]);
        private static string _trainDataPath => Path.Combine(_appPath, "Data", "InputData.tsv");
        private static string _trainSubjectDataPath => Path.Combine(_appPath, "Data", "TextData.tsv");
        private static string _testDataPath => Path.Combine(_appPath, "Data", "Test.csv");
        private static string _modelPath => Path.Combine(_appPath, "Models", "Model.zip");

        /// <summary>
        /// 불용어 리스트
        /// </summary>
        private static string[] StopWord = { "be", "is", "are", "was", "were", "have", "has", "will",
                                                                "the", "as", "an", "a", "at", "of", "to", "in", "for", "'s",
                                                                "with", "both", "and", "or",
                                                                "it", "its", "by", "under", "on",
                                                                "here", "there", "this", "that"};

        /// <summary>
        /// 불용어 리스트
        /// </summary>
        private static string[] Symbol = { "(", ")", ",", "!", "|", "/", "\\", "&", "#", "@", "*", "^", "&" };


        static void Main(string[] args)
        {
            Test_Grammar();
        }

        static void Test_Grammar()
        {
            // 시드는 변경 가능
            _mlContext = new MLContext(seed: 0);

            Grammar_ReadFile();

            // 카테고리 분류 모델
            var Category_pipeline = Category_ProcessData();
            var CtrainingPipeline = BuildAndTrainModel(_trainingDataView, Category_pipeline);
            C_trainedModel = CtrainingPipeline.Fit(_trainingDataView);

            C_predEngine = _mlContext.Model.CreatePredictionEngine<InputData, Prediction_Category>(C_trainedModel);

            // 문법 분류 모델
            var Grammar_pipeline = Grammar_ProcessData();
            var GtrainingPipeline = BuildAndTrainModel(_trainingDataView, Grammar_pipeline);
            G_trainedModel = GtrainingPipeline.Fit(_trainingDataView);

            G_predEngine = _mlContext.Model.CreatePredictionEngine<InputData, Prediction_Grammar>(G_trainedModel);

            string Data = "Subchronic or Prechronic Exposure/ Four-week-old male Hartley guinea pigs were exposed to the test substance (500 ppb) for 4 wk (6 hr/d, 5 d/wk). At the end of exposure, GPs were challenged with an OA aerosol (0.1% in NaCl) and pulmonary functions were measured. The day after, guinea pigs were anesthetized and several endpoints related to inflammatory and allergic responses were assessed in blood, whole-lung histology, and bronchoalveolar lavage (BAL). Exposure to the test substance did not lead to an allergic reaction after the challenge with OA on any of the parameters investigated. In the nasal cavities slight irritation (metaplasia/hyperplasia) of the respiratory epithelium was observed. Histological examination of the trachea and lungs showed a slight irritation of respiratory epithelium for nonsensitized GPs.";

            Data = Remove_StopWord(Data);
            InputData[] datas = Tokenize(Data);

            foreach (var data in datas)
            {
                var predictionC = C_predEngine.Predict(data);
                var predictionG = G_predEngine.Predict(data);

                Console.WriteLine($"단어 : {data.Name,-30} | 카테고리 : {predictionC.Category,-20} | 문법 : {predictionG.Grammar,-10}");
            }
        }

        /// <summary>
        /// 지정한 파일을 읽어오기
        /// </summary>
        static void Grammar_ReadFile()
        {
            if (!File.Exists(_trainDataPath)) throw new Exception("파일이 존재하지 않습니다..");

            try { _trainingDataView = _mlContext.Data.LoadFromTextFile<InputData>(_trainDataPath, hasHeader: true); }
            catch (Exception ex) { Console.WriteLine(ex.ToString()); }
        }

        /// <summary>
        /// 데이터 처리
        /// </summary>
        /// <returns></returns>
        static IEstimator<ITransformer> Category_ProcessData()
        {
            var CategoryModel = _mlContext.Transforms.Conversion.MapValueToKey(inputColumnName: "Category", outputColumnName: "Label")
                .Append(_mlContext.Transforms.Text.FeaturizeText(inputColumnName: "Name", outputColumnName: "NameFeaturized"))
                .Append(_mlContext.Transforms.Text.FeaturizeText(inputColumnName: "Grammar", outputColumnName: "GrammarFeaturized"))
                // Features 에 데이터를 연결
                .Append(_mlContext.Transforms.Concatenate("Features", "NameFeaturized", "GrammarFeaturized"))
                // DataView 캐쉬
                .AppendCacheCheckpoint(_mlContext);

            return CategoryModel;
        }

        /// <summary>
        /// 문법 데이터 처리
        /// </summary>
        /// <returns></returns>
        static IEstimator<ITransformer> Grammar_ProcessData()
        {
            var GrammarModel = _mlContext.Transforms.Conversion.MapValueToKey(inputColumnName: "Grammar", outputColumnName: "Label")
               .Append(_mlContext.Transforms.Text.FeaturizeText(inputColumnName: "Name", outputColumnName: "NameFeaturized"))
               .Append(_mlContext.Transforms.Text.FeaturizeText(inputColumnName: "Category", outputColumnName: "CategoryFeaturized"))
               // Features 에 데이터를 연결
               .Append(_mlContext.Transforms.Concatenate("Features", "NameFeaturized", "CategoryFeaturized"))
               // DataView 캐쉬
               .AppendCacheCheckpoint(_mlContext);

            return GrammarModel;
        }

        /// <summary>
        /// 분류 모델 빌드
        /// </summary>
        /// <param name="trainingDataView"></param>
        /// <param name="pipeline"></param>
        /// <returns></returns>
        static IEstimator<ITransformer> BuildAndTrainModel(IDataView trainingDataView, IEstimator<ITransformer> pipeline)
        {
            try
            {
                var Pipeline = pipeline.Append(_mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy("Label", "Features"))
                .Append(_mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

                return Pipeline;
            }
            catch (Exception e)
            {
                Console.WriteLine(e.ToString());
                return null;
            }
        }

        /// <summary>
        /// 텍스트 토큰화
        /// </summary>
        /// <param name="Text"></param>
        /// <returns></returns>
        static InputData[] Tokenize(string Text)
        {
            string[] Token_Text = Text.Replace("  ", " ").Split(' ');
            
            InputData[] inputDatas = new InputData[Token_Text.Length];
            for (int i = 0; i < Token_Text.Length; i++)
            {
                inputDatas[i] = new InputData { Name = Token_Text[i] };
            }

            return inputDatas;
        }

        /// <summary>
        /// 텍스트에서 불용어 제거
        /// </summary>
        /// <param name="Text"></param>
        /// <returns></returns>
        static string Remove_StopWord(string Text)
        {
            string result = "";

            foreach (var symbol in Symbol)
            {
                Text = Text.Replace(symbol, "");
            }

            string[] Tokens = Text.Split(' ');

            foreach(var Token in Tokens)
            {
                bool Contain = false;
                foreach(var sw in StopWord)
                {
                    if (Token.ToLower() == sw) 
                    { 
                        Contain = true;
                        break;
                    }
                }
                if (!Contain) result += Token + " ";
            }

            result = result.Substring(0, result.Length - 1);
            return result;
        }
    }
}
