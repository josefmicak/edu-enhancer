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
            return Math.Round(correctChoicePoints, 2);
        }

        public static (double, double, EnumTypes.AnswerStatus) CalculateStudentsAnswerAttributes(SubquestionType subquestionType, string[] possibleAnswersArray, 
            string[] correctAnswersArray, double subquestionPoints, double wrongChoicePoints, string[] studentsAnswers)
        {
            double defaultStudentPoints = 0;
            double answerCorrectness = 0;
            int studentsCorrectAnswers = 0;
            EnumTypes.AnswerStatus answerStatus = AnswerStatus.NotDetermined;

            switch (subquestionType)
            {
                case SubquestionType n when (n == SubquestionType.MultiChoiceSingleCorrectAnswer ||
                n == SubquestionType.MultiChoiceTextFill || n == SubquestionType.FreeAnswerWithDeterminedCorrectAnswer):
                    bool areStudentsAnswersCorrect = Enumerable.SequenceEqual(correctAnswersArray, studentsAnswers);
                    if (areStudentsAnswersCorrect)
                    {
                        defaultStudentPoints = subquestionPoints;
                        answerStatus = AnswerStatus.Correct;
                        answerCorrectness = 1;
                    }
                    else
                    {
                        defaultStudentPoints -= wrongChoicePoints * (-1);
                        answerStatus = AnswerStatus.Incorrect;
                        answerCorrectness = -1;
                    }
                    break;
                case SubquestionType.MultiChoiceMultipleCorrectAnswers:
                    for (int i = 0; i < studentsAnswers.Length; i++)
                    {
                        for (int j = 0; j < correctAnswersArray.Length; j++)
                        {
                            if (studentsAnswers[i] == correctAnswersArray[j])
                            {
                                studentsCorrectAnswers++;
                                defaultStudentPoints += ((double)subquestionPoints / (double)correctAnswersArray.Length);
                            }
                        }
                    }

                    defaultStudentPoints -= Math.Abs(Math.Abs(studentsAnswers.Length - studentsCorrectAnswers) * (wrongChoicePoints));

                    if (studentsCorrectAnswers == correctAnswersArray.Length && correctAnswersArray.Length == studentsAnswers.Length)
                    {
                        answerStatus = AnswerStatus.Correct;
                    }
                    else if (studentsAnswers.Length > 1 && studentsCorrectAnswers == 0)
                    {
                        answerStatus = AnswerStatus.Incorrect;
                    }
                    else
                    {
                        answerStatus = AnswerStatus.PartiallyCorrect;
                    }

                    answerCorrectness += (double)studentsCorrectAnswers / correctAnswersArray.Length;
                    answerCorrectness -= (double)(studentsAnswers.Length - studentsCorrectAnswers) / correctAnswersArray.Length;
                    break;
                case SubquestionType n when (n == SubquestionType.MultipleQuestions || n == SubquestionType.GapMatch):
                    for (int i = 0; i < studentsAnswers.Length; i++)
                    {
                        for (int j = 0; j < correctAnswersArray.Length; j++)
                        {
                            if(i != j)
                            {
                                continue;
                            }
                            //student hasn't answered
                            if ((studentsAnswers[i] == "X" && n == SubquestionType.MultipleQuestions) ||
                                (studentsAnswers[i] == "|" && n == SubquestionType.GapMatch))
                            {
                                continue;
                            }
                            //student answered correctly
                            if (studentsAnswers[i] == correctAnswersArray[j])
                            {
                                studentsCorrectAnswers++;
                                defaultStudentPoints += ((double)subquestionPoints / (double)correctAnswersArray.Length);
                            }
                            //student answered incorrectly
                            else
                            {
                                defaultStudentPoints -= wrongChoicePoints * (-1);
                            }
                        }
                    }

                    if (answerStatus != AnswerStatus.NotAnswered)
                    {
                        if (studentsCorrectAnswers == correctAnswersArray.Length)
                        {
                            answerStatus = AnswerStatus.Correct;
                        }
                        else if (studentsCorrectAnswers == 0 && studentsAnswers.Length > 0)
                        {
                            answerStatus = AnswerStatus.Incorrect;
                        }
                        else
                        {
                            answerStatus = AnswerStatus.PartiallyCorrect;
                        }
                    }

                    answerCorrectness += (double)studentsCorrectAnswers / correctAnswersArray.Length;
                    answerCorrectness -= (double)(studentsAnswers.Length - studentsCorrectAnswers) / correctAnswersArray.Length;
                    break;
                /*case SubquestionType n when (n == SubquestionType.MultipleQuestions || n == SubquestionType.GapMatch):
                    string separator = "";
                    switch (n)
                    {
                        case SubquestionType.MultipleQuestions:
                            separator = " -> ";
                            break;
                        case SubquestionType.GapMatch:
                            separator = " - ";
                            break;
                    }
                    for (int i = 0; i < studentsAnswers.Length; i++)
                    {
                        for (int j = 0; j < correctAnswersArray.Length; j++)
                        {
                            string[] studentsAnswerSplit = studentsAnswers[i].Split(separator);
                            string[] correctAnswerSplit = correctAnswersArray[j].Split(separator);
                            if (studentsAnswerSplit[0] == correctAnswerSplit[0])
                            {
                                //student answered correctly
                                if (studentsAnswerSplit[1] == correctAnswerSplit[1])
                                {
                                    studentsCorrectAnswers++;
                                    defaultStudentPoints += ((double)subquestionPoints / (double)correctAnswersArray.Length);
                                }
                                //student answered incorrectly
                                else
                                {
                                    defaultStudentPoints -= wrongChoicePoints * (-1);
                                }
                            }
                        }
                    }

                    if (answerStatus != AnswerStatus.NotAnswered)
                    {
                        if (studentsCorrectAnswers == correctAnswersArray.Length)
                        {
                            answerStatus = AnswerStatus.Correct;
                        }
                        else if (studentsCorrectAnswers == 0 && studentsAnswers.Length > 0)
                        {
                            answerStatus = AnswerStatus.Incorrect;
                        }
                        else
                        {
                            answerStatus = AnswerStatus.PartiallyCorrect;
                        }
                    }

                    answerCorrectness += (double)studentsCorrectAnswers / correctAnswersArray.Length;
                    answerCorrectness -= (double)(studentsAnswers.Length - studentsCorrectAnswers) / correctAnswersArray.Length;
                    break;*/
                case SubquestionType.MatchingElements:
                    studentsCorrectAnswers = 0;

                    for (int i = 0; i < studentsAnswers.Length; i++)
                    {
                        for (int j = 0; j < correctAnswersArray.Length; j++)
                        {
                            string[] studentsAnswerSplit = studentsAnswers[i].Split("|");
                            string[] correctAnswerSplit = correctAnswersArray[j].Split("|");
                            //for this type of subquestion, the order of the elements contained in the answer is not always the same
                            if ((studentsAnswerSplit[0] == correctAnswerSplit[0] && studentsAnswerSplit[1] == correctAnswerSplit[1]) ||
                                (studentsAnswerSplit[0] == correctAnswerSplit[1] && studentsAnswerSplit[1] == correctAnswerSplit[0]))
                            {
                                studentsCorrectAnswers++;
                            }
                        }
                    }

                    if (answerStatus != AnswerStatus.NotAnswered)
                    {
                        //increase points for every correct answer
                        defaultStudentPoints += studentsCorrectAnswers * ((double)subquestionPoints / (double)correctAnswersArray.Length);
                        //decrease points for every incorrect answer
                        defaultStudentPoints -= (studentsAnswers.Length - studentsCorrectAnswers) * Math.Abs(wrongChoicePoints);
                    }

                    if (studentsAnswers.Length > 0 && studentsCorrectAnswers == correctAnswersArray.Length)
                    {
                        answerStatus = AnswerStatus.Correct;
                    }
                    else if (studentsAnswers.Length > 0 && studentsCorrectAnswers == 0)
                    {
                        answerStatus = AnswerStatus.Incorrect;
                    }
                    else
                    {
                        answerStatus = AnswerStatus.PartiallyCorrect;
                    }

                    answerCorrectness += (double)studentsCorrectAnswers / correctAnswersArray.Length;
                    answerCorrectness -= (double)(studentsAnswers.Length - studentsCorrectAnswers) / correctAnswersArray.Length;
                    break;
                case SubquestionType.Slider:
                    areStudentsAnswersCorrect = Enumerable.SequenceEqual(correctAnswersArray, studentsAnswers);
                    if (studentsAnswers.Length == 0)
                    {
                        answerStatus = AnswerStatus.NotAnswered;
                    }
                    else
                    {
                        if (areStudentsAnswersCorrect)
                        {
                            defaultStudentPoints = subquestionPoints;
                            answerStatus = AnswerStatus.Correct;
                            answerCorrectness = 1;
                        }
                        else
                        {
                            defaultStudentPoints -= wrongChoicePoints * (-1);
                            answerStatus = AnswerStatus.Incorrect;
                        }
                    }

                    if (!areStudentsAnswersCorrect && studentsAnswers.Length > 0)
                    {
                        //the closer the student's answer is to the actual correct answer, the more correct his answer is
                        int lowerBound = int.Parse(possibleAnswersArray[0]);
                        int upperBound = int.Parse(possibleAnswersArray[1]);
                        int correctAnswer = int.Parse(correctAnswersArray[0]);
                        int maxDifference = Math.Max(Math.Abs(lowerBound - correctAnswer), Math.Abs(upperBound - correctAnswer));
                        int studentsAnswer = int.Parse(studentsAnswers[0]);
                        int studentsAnswerDifference = Math.Abs(correctAnswer - studentsAnswer);
                        double studentsAnswerDifferenceVar = ((double)maxDifference / 2) - studentsAnswerDifference;
                        answerCorrectness = studentsAnswerDifferenceVar / ((double)maxDifference / 2);
                    }
                    break;
                case SubquestionType.OrderingElements:
                    areStudentsAnswersCorrect = Enumerable.SequenceEqual(correctAnswersArray, studentsAnswers);
                    if (areStudentsAnswersCorrect)
                    {
                        defaultStudentPoints = subquestionPoints;
                        answerStatus = AnswerStatus.Correct;
                    }
                    else
                    {
                        defaultStudentPoints -= wrongChoicePoints * (-1);
                        answerStatus = AnswerStatus.Incorrect;
                    }

                    if (!areStudentsAnswersCorrect && studentsAnswers.Length > 0)
                    {
                        for (int i = 0; i < studentsAnswers.Length; i++)
                        {
                            int studentsAnswerIndex = i;
                            int correctAnswerIndex = Array.IndexOf(correctAnswersArray, studentsAnswers[i]);
                            int studentsAnswerDifference = Math.Abs(studentsAnswerIndex - correctAnswerIndex);
                            int maxDifference = Math.Max(correctAnswerIndex, correctAnswersArray.Length - correctAnswerIndex);
                            double studentsAnswerDifferenceVar = ((double)maxDifference / 2) - studentsAnswerDifference;
                            answerCorrectness += studentsAnswerDifferenceVar / ((double)maxDifference / 2);
                        }
                        answerCorrectness = answerCorrectness / (double)studentsAnswers.Length;
                    }
                    break;
            }

            if (subquestionType == SubquestionType.FreeAnswer)
            {
                answerStatus = AnswerStatus.CannotBeDetermined;
            }
            else if (subquestionType != SubquestionType.FreeAnswer && studentsAnswers.Length == 0)
            {
                answerStatus = AnswerStatus.NotAnswered;
            }

            //with some rare combinations of possible and correct answers, answerCorrectness could potentially be set as lower than -1 or higher than 1
            if (answerCorrectness > 1)
            {
                answerCorrectness = 1;
            }
            if (answerCorrectness < -1)
            {
                answerCorrectness = -1;
            }

            defaultStudentPoints = Math.Round(defaultStudentPoints, 2);
            answerCorrectness = Math.Round(answerCorrectness, 2);
            return (defaultStudentPoints, answerCorrectness, answerStatus);
        }

        public static double? RoundDecimal(double? number)
        {
            if (number.HasValue)
            {
                return (double?)Math.Round(number.Value, 2);
            }
            return number;
        }
    }
}
