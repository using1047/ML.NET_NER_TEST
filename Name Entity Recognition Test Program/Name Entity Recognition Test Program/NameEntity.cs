using System;
using System.Collections.Generic;
using System.Text;
using Microsoft.ML.Data;

namespace Name_Entity_Recognition_Test_Program
{
    public class InputData
    {
        [LoadColumn(0)]
        public string Name { get; set; }

        [LoadColumn(1)]
        public string Category { get; set; }

        [LoadColumn(2)]
        public string Grammar { get; set; }
    }

    public class Prediction_Category
    {
        [ColumnName("PredictedLabel")]
        public string Category;
    }

    public class Prediction_Grammar
    {
        [ColumnName("PredictedLabel")]
        public string Grammar;
    }
}
