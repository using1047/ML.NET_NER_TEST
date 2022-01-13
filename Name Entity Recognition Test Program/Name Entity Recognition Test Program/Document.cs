using System;
using System.Collections.Generic;
using System.Text;
using System.Text.RegularExpressions;

namespace Name_Entity_Recognition_Test_Program
{
    public class Document
    {
        private int _DocumentNumber;
        private string _FilePath;
        private int _WordsCount;

        public int DocumentNumber
        {
            get
            {
                return _DocumentNumber;
            }
            set
            {
                _DocumentNumber = value;
            }
        }
        public string FilePath
        {
            get
            {
                if (_FilePath != null)
                    return _FilePath;
                else return "지정된 파일이 없습니다.";
            }
            set
            {
                _FilePath = value;
            }
        }
        public int WordsCount
        {
            get
            {
                return _WordsCount;
            }
        }
    }
}
