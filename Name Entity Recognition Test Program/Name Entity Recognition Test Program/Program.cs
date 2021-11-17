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
        private static string _testDataPath => Path.Combine(_appPath, "Data", "Test.csv");
        private static string _modelPath => Path.Combine(_appPath, "Models", "Model.zip");

        /// <summary>
        /// 불용어 리스트
        /// </summary>
        private static string[] StopWord = { "be", "is", "are", "was", "were", "have", "has", "will",
                                                                "the", "as", "an", "a", "at", "of", "to", "in", "for", "'s",
                                                                "with", "both", "and", "or",
                                                                "it", "its"};

        /// <summary>
        /// 불용어 리스트
        /// </summary>
        private static string[] Symbol = { "(", ")", ",", "!", "|", "/", "\\", "&", "#", "@", "*", "^", "&" };


        static void Main(string[] args)
        {
            // 시드는 변경 가능
            _mlContext = new MLContext(seed: 0);

            ReadFile();

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

            string Data = "Exposure to formaldehyde, a known air toxic, is associated with cancer and lung disease. Despite the adverse health effects of formaldehyde, the mechanisms underlying formaldehyde-induced disease remain largely unknown. Research has uncovered microRNAs (miRNAs) as key posttranscriptional regulators of gene expression that may influence cellular disease state. Although studies have compared different miRNA expression patterns between diseased and healthy tissue, this is the first study to examine perturbations in global miRNA levels resulting from formaldehyde exposure. We investigated whether cellular miRNA expression profiles are modified by formaldehyde exposure to test the hypothesis that formaldehyde exposure disrupts miRNA expression levels within lung cells, representing a novel epigenetic mechanism through which formaldehyde may induce disease. Human lung epithelial cells were grown at air-liquid interface and exposed to gaseous formaldehyde at 1 ppm for 4 hr. Small RNAs and protein were collected and analyzed for miRNA expression using microarray analysis and for interleukin (IL-8) protein levels by enzyme-linked immunosorbent assay (ELISA). RESULTS: Gaseous formaldehyde exposure altered the miRNA expression profiles in human lung cells. Specifically, 89 miRNAs were significantly down-regulated in formaldehyde-exposed samples versus controls. Functional and molecular network analysis of the predicted miRNA transcript targets revealed that formaldehyde exposure potentially alters signaling pathways associated with cancer, inflammatory response, and endocrine system regulation. IL-8 release increased in cells exposed to formaldehyde, and results were confirmed by real-time polymerase chain reaction. Formaldehyde alters miRNA patterns that regulate gene expression, potentially leading to the initiation of a variety of diseases.";

            Data = Remove_StopWord(Data);
            InputData[] datas = Tokenize(Data);

            foreach(var data in datas)
            {
                var predictionC = C_predEngine.Predict(data);
                var predictionG = G_predEngine.Predict(data);

                Console.WriteLine($"데이터 : {data.Name, -30} | 카테고리 : {predictionC.Category, -20} | 문법 : {predictionG.Grammar, -10}");
            }
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
