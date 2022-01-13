using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;
using MySql.Data.MySqlClient;

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

        private static PredictionEngine<SectionData.InputData, SectionData.Prediction_Title> S_predEngine;

        /// <summary>
        /// 학습 관련 변수들
        /// </summary>
        private static ITransformer C_trainedModel;
        private static ITransformer G_trainedModel;
        private static ITransformer S_trainedModel;
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
        private static string[] Symbol = { "(", ")", ",", "!", "|", "/", "\\", "&", "#", "@", "*", "^", "&", "[", "]", ":", "%", "...", ";"};

        /// <summary>
        /// 동시 출현 빈도
        /// </summary>
        private static int[][] FrequencyMetrics;
        private static List<int> NotContainsRow = new List<int>();
        /// <summary>
        /// 전체 출현 빈도
        /// </summary>
        private static Dictionary<string, int> WordFrequency = new Dictionary<string, int>();
        private static Dictionary<string, int> OWordFrequency = new Dictionary<string, int>();
        /// <summary>
        /// 단어 - 숫자 임베딩
        /// </summary>
        private static Dictionary<string, int> WordEmbbeding = new Dictionary<string, int>();
        private static Dictionary<string, int> OWordEmbbeding = new Dictionary<string, int>();

        static void Main(string[] args)
        {
            string FilePath = @"C:\Users\82105\Desktop\회사자료\UNCIENT\CasData\";

            /*
            int Count = 0, DocumentNumber = 1;
            foreach (var file in new DirectoryInfo(FilePath).EnumerateFiles("*.txt", SearchOption.AllDirectories))
            {
                Console.WriteLine(DocumentNumber + "번 째 문서의 단어 모음을 시작합니다...");
                DocumentNumber++;

                WordCount(file.FullName);
                Console.WriteLine("완료...\n");
            }
            Count = WordEmbbeding.Count;

            Console.WriteLine($"단어 {Count}개를 위한 매트릭스 생성");
            FrequencyMetrics = new int[Count][];

            for (int i = 0; i < Count; i++)
                FrequencyMetrics[i] = new int[Count];

            Console.WriteLine("생성 완료\n");

            Console.WriteLine("정렬 시작");
            ListOrder();
            Console.WriteLine("정렬 완료\n");

            DocumentNumber = 1;
            foreach (var file in new DirectoryInfo(FilePath).EnumerateFiles("*.txt", SearchOption.AllDirectories))
            {
                Console.WriteLine(DocumentNumber + "번 째 문서의 문장 분리를 시작합니다...");
                List<string> Documents = MakeDoucument(File.ReadAllText(file.FullName));
                Console.WriteLine("완료...\n");
            }
            Optimzing();
            Console.WriteLine($"최적화를 위해 {NotContainsRow.Count} 개의 데이터를 삭제하였습니다.\n");

            Console.WriteLine("정렬 시작");
            ListOrder();
            Console.WriteLine("정렬 완료\n");

            int NCount = OWordEmbbeding.Count;

            MakeExistMetrics(NCount);
            MakeFrequencyMetrics(Count);
            TF(NCount);
            
            /*
            int Page = 1;
            
            foreach (var Doc in Documents)
            {
                Test_Grammar(Doc, Page);
                Page++;
            }
            */


            string text = File.ReadAllText(@"C:\Users\82105\Desktop\회사자료\Spark\HDSB_Data\92-87-5.txt");
            Test_Grammar(text.Replace("\r", " ").Replace("\n", " ").Replace("\r\n", " ").Replace("  ", " "), 1);
        }

        // 전체 단어 개수 구하기
        static int WordCount(string Path)
        {
            string AllText = File.ReadAllText(Path).Replace("\n", "").Replace("\r", "").Replace("\r\n", "").ToLower();

            string Data = Remove_StopWord(AllText.Replace("\r\n", ""));
            InputData[] datas = Tokenize(Data);

            int KeyNumber = 0;
            foreach (var data in datas)
            {
                if (!ThatsNumber(data.Name))
                {
                    if (!WordEmbbeding.ContainsKey(data.Name.Replace("\n", " ")))
                    {
                        WordEmbbeding.Add(data.Name.Replace("\n", " "), KeyNumber);
                        WordFrequency.Add(data.Name.Replace("\n", " "), 1);
                        KeyNumber++;
                    }
                }
            }

            return WordEmbbeding.Count;
        }

        /// <summary>
        /// 리스트 정렬하기
        /// </summary>
        static void ListOrder()
        {
            OWordEmbbeding.Clear();
            OWordFrequency.Clear();

            // 임베딩 리스트 정렬
            var EmbList = WordEmbbeding.OrderBy(Label => Label.Key);

            int CurrentPoint = 0;
            for (int Line = 0; Line < EmbList.Count(); Line++)
            {
                OWordEmbbeding.Add(EmbList.ElementAt(Line).Key, CurrentPoint);
                CurrentPoint++;
            }

            // 빈도수 리스트 정렬
            var FreList = WordFrequency.OrderBy(Label => Label.Key);
            for (int Line = 0; Line < FreList.Count(); Line++)
            {
                OWordFrequency.Add(FreList.ElementAt(Line).Key, 0);
            }
        }

        /// <summary>
        /// 예측
        /// </summary>
        /// <param name="Data"></param>
        /// <param name="CurrentPageNumber"></param>
        static void Test_Grammar(string Data, int CurrentPageNumber)
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

            Data.Replace("\r", "").Replace("\n", "");
            Data = Remove_StopWord(Data);
            InputData[] datas = Tokenize(Data);

            int Last = datas.Length - 1;
            int Count = 0;
            string Content = "";
            foreach (var data in datas)
            {

                var predictionC = C_predEngine.Predict(data);
                var predictionG = G_predEngine.Predict(data);

                if (Count < Last)
                {
                    if (predictionC.Category != "None")
                    {
                        Console.WriteLine($"단어 : {data.Name,-30} | 카테고리 : {predictionC.Category,-20} | 문법 : {predictionG.Grammar,-10}");
                        Content += $"{data.Name}\t{predictionC.Category}\t{predictionG.Grammar}\n";
                        Count++;
                    }
                }
                else
                {
                    Console.WriteLine($"###############################################\t{CurrentPageNumber}\t###############################################");
                    //File.WriteAllText(@"C:\Users\82105\Desktop\회사자료\UNCIENT\Result.tsv", $"###############################################\t{CurrentPageNumber}\t###############################################\n");
                }
            }

            File.WriteAllText(@$"C:\Users\82105\Desktop\회사자료\UNCIENT\Page{CurrentPageNumber}.tsv", Content);
        }

        static void Test_Section(string Data, int CurrentPageNumber)
        {
            // 시드는 변경 가능
            _mlContext = new MLContext(seed: 0);

            Section_DBFile();

            var Section_pipeline = Section_ProcessData();

            var trainingPipeline = BuildAndTrainModel(_trainingDataView, Section_pipeline);
            S_trainedModel = trainingPipeline.Fit(_trainingDataView);
            S_predEngine = _mlContext.Model.CreatePredictionEngine<SectionData.InputData, SectionData.Prediction_Title>(S_trainedModel);
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

        static void Section_DBFile()
        {
            string ConnectString = string.Format("Server={0};Database={1};Uid={2};Pwd={3};Port={4}",
                                                 "14.192.80.155", "cecdb", "cecdb_user", "_!cecdb//user2020", "3306");

            try
            {
                using(MySqlConnection conn = new MySqlConnection(ConnectString))
                {
                    conn.Open();

                    string SelectSQL = "SELECT * FROM test_hsdb_datawarehouse1";

                    MySqlCommand cmd = new MySqlCommand(SelectSQL, conn);
                    MySqlDataReader dr = cmd.ExecuteReader();

                    List<SectionData.InputData> Data = new List<SectionData.InputData>();
                    while (dr.Read())
                    {
                        SectionData.InputData data = new SectionData.InputData();

                        data.Title = dr["SectionTitle"].ToString();
                        data.Context = dr["SectionContent"].ToString();

                        if(data.Title != null && data.Context != null)
                        {
                            Data.Add(data);
                        }
                    }
                    dr.Close();

                    try { _trainingDataView = _mlContext.Data.LoadFromEnumerable<SectionData.InputData>(Data); }
                    catch (Exception ex) { Console.WriteLine(ex.ToString()); }
                }
            }
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

        static IEstimator<ITransformer> Section_ProcessData()
        {
            var SectionModel = _mlContext.Transforms.Conversion.MapValueToKey(inputColumnName: "Title", outputColumnName: "Label")
                .Append(_mlContext.Transforms.Text.FeaturizeText(inputColumnName: "Context", outputColumnName: "ContextFeaurized"))
                .Append(_mlContext.Transforms.Concatenate("Features", "ContextFeaurized"))
                .AppendCacheCheckpoint(_mlContext);

            return SectionModel;
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
            string[] Token_Text = Text.Replace("\n", "").Replace("  ", " ").Split(' ');
            
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

        /// <summary>
        /// 한 문장을 문서 하나로 침
        /// </summary>
        /// <param name="AllText"></param>
        /// <returns></returns>
        static List<string> MakeDoucument(string AllText)
        {
            // 한 문장씩 하나의 문서
            List<string> Documents = new List<string>();

            // 문장을 나누기 위한 전처리
            string[] Texts = AllText.Split('\n');

            foreach(var Text in Texts)
            {
                string PreText = Remove_StopWord(Text.Replace("\n", ""));
                InputData[] Datas = Tokenize(PreText);

                if(Datas.Length > 0)
                {
                    // 출현 빈도
                    foreach (var Data in Datas)
                    {
                        if ((Data.Name == "" && Data.Name == " ") || ThatsNumber(Data.Name)) { }
                        else
                        {
                            if (OWordFrequency.ContainsKey($"{Data.Name}")) OWordFrequency[$"{Data.Name}"]++;
                            else { }
                            //Console.WriteLine($"{Data.Name} Key 가 WordFequency에 포함되어 있지 않습니다.");
                        }
                    }

                    // 동시출현 빈도
                    for (int p = 1; p < Datas.Length; p++)
                    {
                        if (OWordEmbbeding.ContainsKey(Datas[0].Name) && OWordEmbbeding.ContainsKey($"{Datas[p].Name}"))
                        {
                            FrequencyMetrics[OWordEmbbeding[Datas[0].Name]][OWordEmbbeding[Datas[p].Name]]++;
                            FrequencyMetrics[OWordEmbbeding[Datas[p].Name]][OWordEmbbeding[Datas[0].Name]]++;
                        }
                    }
                }
            }

            Documents.AddRange(Texts);
     
            return Documents;
        }

        static void Optimzing()
        {
            int WordCount = OWordEmbbeding.Count;
            Console.WriteLine(WordCount.ToString());
            for (int y = 0; y < WordCount; y++)
            {
                int Sum = 0;
                for (int x = 0; x < WordCount; x++)
                {
                    Sum += FrequencyMetrics[y][x];
                }
                if (Sum == 0) NotContainsRow.Add(y);
            }
            Console.WriteLine(NotContainsRow.Count.ToString());
            if (NotContainsRow.Count == 0) return;
            else
            {
                Console.WriteLine($"{NotContainsRow.Count} 개 데이터 살펴보는 중...");
                for (int i = 0; i < NotContainsRow.Count; i++)
                {
                    string Key = WordEmbbeding.ElementAt(NotContainsRow[i]).Key;
                    WordFrequency.Remove(Key);
                }

                WordEmbbeding.Clear();

                int p = 0;
                foreach (var Key in WordFrequency.Keys)
                {
                    WordEmbbeding.Add(Key, p);
                    p++;
                }
            }
        }

        static bool EndParagraph(string Paragraph)
        {
            bool check = false;
            if(       Paragraph.Contains("Chemical and Physical Properties")
               && Paragraph.Contains(""))
            {

            }

            return check;
        }

        /// <summary>
        /// 매트릭스 만들기
        /// </summary>
        /// <param name="Size"></param>
        static void MakeExistMetrics(int Size)
        {

            string Label = "시작\t";
            for (int i = 0; i < OWordEmbbeding.Count; i++)
            {
                Label += OWordEmbbeding.ElementAt(i).Key + "\t";
            }
            Label += "\n";

            File.AppendAllText(@$"C:\Users\82105\Desktop\회사자료\UNCIENT\Metrics.tsv", Label);
            
            for (int y = 0; y < Size; y++)
            {
                string Metric = OWordEmbbeding.ElementAt(y).Key + "\t";
                for (int x = 0; x < Size; x++)
                {
                    if (x == y) Metric += "-\t";
                    else
                    {
                        int Val = FrequencyMetrics[y][x] > 0 ? 1 : 0;
                        Metric += Val +"\t";
                    }//Console.Write(FrequencyMetrics[y][x].ToString() + "\t");
                }
                //Console.WriteLine("\n\n");
                Console.WriteLine(y + " : Calculating...");
                Metric += "\n";
                File.AppendAllText(@$"C:\Users\82105\Desktop\회사자료\UNCIENT\Metrics.tsv", Metric);
            }
        }

        static void MakeFrequencyMetrics(int Size)
        {

            string Label = "시작\t";
            for (int i = 0; i < OWordEmbbeding.Count; i++)
            {
                Label += OWordEmbbeding.ElementAt(i).Key + "\t";
            }
            Label += "\n";

            File.AppendAllText(@$"C:\Users\82105\Desktop\회사자료\UNCIENT\FrequencyMetrics.tsv", Label);

            for (int y = 0; y < Size; y++)
            {
                if (!NotContainsRow.Contains(y))
                {
                    string Metric = OWordEmbbeding.ElementAt(y).Key + "\t";
                    for (int x = 0; x < Size; x++)
                    {
                        if (x == y) Metric += "-\t";
                        else
                        {
                            int Val = FrequencyMetrics[y][x];
                            Metric += Val + "\t";
                        }//Console.Write(FrequencyMetrics[y][x].ToString() + "\t");
                    }
                    //Console.WriteLine("\n\n");
                    Console.WriteLine(y + " : Calculating...");
                    Metric += "\n";
                    File.AppendAllText(@$"C:\Users\82105\Desktop\회사자료\UNCIENT\FrequencyMetrics.tsv", Metric);
                }
            }
        }

        static void TF(int Size)
        {
            string Label = "단어\t빈도수\n";
            File.AppendAllText(@$"C:\Users\82105\Desktop\회사자료\UNCIENT\Metrics_TF.tsv", Label);

            for (int y = 0; y < Size; y++)
            {
                string Line = "";
                Line = OWordFrequency.ElementAt(y).Key + "\t" + OWordFrequency.ElementAt(y).Value + "\n";
                File.AppendAllText(@$"C:\Users\82105\Desktop\회사자료\UNCIENT\Metrics_TF.tsv", Line);
            }
        }

        static bool ThatsNumber(string Text)
        {
            try
            {
                if (Text.Contains("e+") || Text.Contains("e-")) return true;

                float B = float.Parse(Text);
                return true;
            }
            catch
            {
                return false;
            }
        }
    }
}
