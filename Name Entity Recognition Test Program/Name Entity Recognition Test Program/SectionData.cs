using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Text;

namespace Name_Entity_Recognition_Test_Program
{
    internal class SectionData
    {
        public class InputData
        {
            [LoadColumn(0)]
            public string Title { get; set; }

            [LoadColumn(1)]
            public string Context { get; set; }
        }

        public class Prediction_Title
        {
            [ColumnName("PredictedLabel")]
            public string PTitle;
        }
    }
}
