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
        public string Entity { get; set; }
    }

    public class Prediction
    {
        [ColumnName("PredictedLabel")]
        public string Entity;
    }
}
