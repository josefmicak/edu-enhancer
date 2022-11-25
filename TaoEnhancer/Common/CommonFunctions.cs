using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static Common.EnumTypes;

namespace Common
{
    /// <summary>
    /// Functions located in this class are called from multiple projects that already have a dependency link between them
    /// </summary>
    public class CommonFunctions
    {
        public static double CalculateCorrectChoicePoints(double subquestionPoints, string[] correctChoiceArray, SubquestionType subquestionType)
        {
            double correctChoicePoints = 0;
            switch (subquestionType)
            {
                case SubquestionType n when (n == SubquestionType.OrderingElements || n == SubquestionType.FreeAnswer || n == SubquestionType.MultiChoiceSingleCorrectAnswer
                || n == SubquestionType.MultiChoiceTextFill || n == SubquestionType.FreeAnswerWithDeterminedCorrectAnswer || n == SubquestionType.Slider):
                    correctChoicePoints = subquestionPoints;
                    break;
                case SubquestionType.MultiChoiceMultipleCorrectAnswers:
                    correctChoicePoints = (double)subquestionPoints / (double)correctChoiceArray.Length;
                    break;
                case SubquestionType n when (n == SubquestionType.MatchingElements || n == SubquestionType.GapMatch || n == SubquestionType.MultipleQuestions):
                    correctChoicePoints = (double)subquestionPoints / ((double)correctChoiceArray.Length / 2) / 2;
                    break;
            }
            if(correctChoicePoints == double.NegativeInfinity || correctChoicePoints == double.PositiveInfinity)
            {
                int a = 0;
            }
            return Math.Round(correctChoicePoints, 2);
        }
    }
}
