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
                                                                "it", "its", "by", "under", "on",
                                                                "here", "there", "this", "that"};

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

            string Data = "3,3'-Dimethoxybenzidine's production and use as a chemical intermediate in the production of azo dyes may result in its release to the environment through various waste streams(SRC). Although 3,3'-dimethoxybenzidine is apparently not produced in the United States any longer, it may still be imported into the country. If released to air, an estimated vapor pressure of 1.2X10-7 mm Hg at 25 °C indicates 3,3'-dimethoxybenzidine will exist in both the vapor and particulate phases in the ambient atmosphere. Vapor-phase 3,3'-dimethoxybenzidine will be degraded in the atmosphere by reaction with photochemically-produced hydroxyl radicals; the half-life for this reaction in air is estimated to be 3 hours. 3,3'-Dimethoxybenzidine absorbs light greater than 290 nm, and it may be susceptible to direct photolysis in the environment; however, the rate of this potential reaction is unknown. Particulate-phase 3,3'-dimethoxybenzidine will be removed from the atmosphere by wet and dry deposition. If released to soil, 3,3'-dimethoxybenzidine is expected to have moderate mobility based upon an estimated Koc of 230. The first pKa of 3,3'-dimethoxybenzidine is estimated as 4.2, which indicates that 3,3'-dimethoxybenzidine will partially exist in the protonated form under acidic conditions and cations have greater adsorption to soils than neutral molecules. Furthermore, 3,3'-dimethoxybenzidine is an aromatic amine which may form covalent bonds with humic materials resulting in relatively immobile quinone-like complexes. Volatilization from moist soil surfaces is not expected to be an important fate process for 3,3'-dimethoxybenzidine because cations do not volatilize, and the estimated Henry's Law constant of the neutral species is 4.7X10-11 atm-cu m/mole. 3,3'-Dimethoxybenzidine is not expected to volatilize from dry soil surfaces based upon its estimated vapor pressure. No data regarding the biodegradation of 3,3'-dimethoxybenzidine in soil or natural water were found. However, screening studies using sewage sludge inoculum suggest biodegradation will occur slowly in the environment. Benzidine and its derivatives such as 3,3'-dimethoxybenzidine are known to be to be rapidly oxidized by Fe(III) and other cations which are frequently found in soil and environmental waters. If released into water, 3,3'-dimethoxybenzidine is expected to adsorb to suspended solids and sediment based upon the estimated Koc. Volatilization from water surfaces is not expected to be an important fate process for either the free base or its conjugate acid based upon this compound's estimated Henry's Law constant and the fact that cations are non-volatile. An estimated BCF of 5 suggests the potential for bioconcentration in aquatic organisms is low. 3,3'-Dimethoxybenzidine is not expected to undergo hydrolysis due to a lack of hydrolyzable functional groups. Occupational exposure to 3,3'-dimethoxybenzidine may occur through inhalation and dermal contact with this compound at workplaces where 3,3'-dimethoxybenzidine is used. (SRC)";

           Data = Remove_StopWord(Data);
            InputData[] datas = Tokenize(Data);

            foreach(var data in datas)
            {
                var predictionC = C_predEngine.Predict(data);
                var predictionG = G_predEngine.Predict(data);

                Console.WriteLine($"단어 : {data.Name, -30} | 카테고리 : {predictionC.Category, -20} | 문법 : {predictionG.Grammar, -10}");
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
