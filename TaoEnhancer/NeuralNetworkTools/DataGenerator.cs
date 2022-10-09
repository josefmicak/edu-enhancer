﻿using DomainModel;
using Common;
using CsvHelper;
using System.Globalization;
using System.Diagnostics;
using DataLayer;
using System.Reflection;
using System;

namespace NeuralNetworkTools
{
    public class DataGenerator
    {
        //static DataFunctions dataFunctions = new DataFunctions();
        private readonly CourseContext _context;
        /// <summary>
        /// Generates .csv file of a number of subquestion templates with parameters that are used by the neural network
        /// </summary>
        /// <param name="dataColleration">Decides whether the generated data will be randomized or if there are going to be collerations between the templates</param>
        public static void GenerateTemplatesFile(string dataColleration)
        {
            List<TestTemplate> testTemplates = new List<TestTemplate>();
            if(dataColleration == "none")
            {
                testTemplates = GenerateRandomTestTemplates();
            }
            else if(dataColleration == "on")
            {
                testTemplates = GenerateCorrelationalTestTemplates();
            }
            var subquestionTemplateRecords = GetSubquestionTemplateRecords(testTemplates);

            string filePath = "";
            if (dataColleration == "none")
            {
                filePath = "D:\\Users\\granders\\Desktop\\RandomTemplatesFile.csv";
            }
            else if(dataColleration == "on")
            {
                filePath = "D:\\Users\\granders\\Desktop\\CorrelationalTemplatesFile.csv";
            }

            File.Delete(filePath);
            using (var writer = new StreamWriter(filePath))
            using (var csv = new CsvWriter(writer, CultureInfo.InvariantCulture))
            {
                csv.WriteRecords(subquestionTemplateRecords);
            }

        }

        /// <summary>
        /// Creates a list of subquestion templates with parameters that are used by the neural network
        /// </summary>
        public static List<SubquestionTemplateRecord> GetSubquestionTemplateRecords(List<TestTemplate> testTemplates)
        {
            List<SubquestionTemplateRecord> subquestionTemplateRecords = new List<SubquestionTemplateRecord>();
            string[] subjectsArray = { "Chemie", "Zeměpis", "Matematika", "Dějepis", "Informatika" };
            double[] subquestionTypeAveragePoints = GetSubquestionTypeAveragePoints(testTemplates);
            double[] subjectAveragePoints = GetSubjectAveragePoints(testTemplates);
            int subquestionTemplateRecordId = 0;

            for (int i = 0; i < testTemplates.Count; i++)
            {
                TestTemplate testTemplate = testTemplates[i];
                double? minimumPointsShare = GetMinimumPointsShare(testTemplate);

                for (int j = 0; j < testTemplate.QuestionTemplateList.Count; j++)
                {
                    QuestionTemplate questionTemplate = testTemplate.QuestionTemplateList.ElementAt(j);

                    for (int k = 0; k < questionTemplate.SubquestionTemplateList.Count; k++)
                    {
                        SubquestionTemplate subquestionTemplate = questionTemplate.SubquestionTemplateList.ElementAt(k);
                        SubquestionTemplateRecord subquestionTemplateRecord = new SubquestionTemplateRecord();
                        subquestionTemplateRecord.Id = "record-" + subquestionTemplateRecordId;
                        int subquestionType = subquestionTemplate.SubquestionType;
                        subquestionTemplateRecord.SubquestionTypeAveragePoints = Math.Round(subquestionTypeAveragePoints[subquestionType - 1], 2);
                        int possibleAnswersCount = 0;
                        int correctAnswersCount = 0;
                        if (subquestionTemplate.PossibleAnswerList != null)
                        {
                            possibleAnswersCount = subquestionTemplate.PossibleAnswerList.Count();
                        }
                        if (subquestionTemplate.CorrectAnswerList != null)
                        {
                            correctAnswersCount = subquestionTemplate.CorrectAnswerList.Count();
                        }

                        if (possibleAnswersCount != 0)
                        {
                            //This type of subquestion typically contains 2 possible answers and many correct answers, so we set CorrectAnswersShare manually instead
                            if (subquestionType == 4)
                            {
                                subquestionTemplateRecord.CorrectAnswersShare = 0.5;
                            }
                            //These type of subquestion are about sorting elements - the more elements there are, the harder it is to answer correctly
                            else if (subquestionType == 1 || subquestionType == 9)
                            {
                                subquestionTemplateRecord.CorrectAnswersShare = 1 / (double)possibleAnswersCount;
                            }
                            //This type of subquestion uses slider - typically tens to hunders of possible answers
                            else if (subquestionType == 10)
                            {
                                subquestionTemplateRecord.CorrectAnswersShare = 0;
                            }
                            else
                            {
                                subquestionTemplateRecord.CorrectAnswersShare = (double)correctAnswersCount / (double)possibleAnswersCount;
                            }
                            subquestionTemplateRecord.CorrectAnswersShare = Math.Round(subquestionTemplateRecord.CorrectAnswersShare, 2);
                        }

                        string? subject = testTemplate.Subject;
                        int subjectId = Array.FindIndex(subjectsArray, x => x.Contains(subject));
                        subquestionTemplateRecord.SubjectAveragePoints = Math.Round(subquestionTypeAveragePoints[subjectId], 2);
                        subquestionTemplateRecord.ContainsImage = Convert.ToInt32((subquestionTemplate.ImageSource == "") ? false : true);
                        subquestionTemplateRecord.NegativePoints = Convert.ToInt32(testTemplate.NegativePoints);
                        subquestionTemplateRecord.MinimumPointsShare = minimumPointsShare;
                        if (subquestionTemplate.SubquestionPoints != null)
                        {
                            subquestionTemplateRecord.SubquestionPoints = Math.Round((double)subquestionTemplate.SubquestionPoints, 2);
                        }
                        subquestionTemplateRecords.Add(subquestionTemplateRecord);
                        subquestionTemplateRecordId++;
                    }
                }
            }

            return subquestionTemplateRecords;
        }

        /// <summary>
        /// Returns minimum points share of the test template (percentage of minimum points in regards to the sum of test template's points)
        /// </summary>
        public static double? GetMinimumPointsShare(TestTemplate testTemplate)
        {
            double? totalSubquestionPoints = 0;
            for(int j = 0; j < testTemplate.QuestionTemplateList.Count; j++)
            {
                QuestionTemplate questionTemplate = testTemplate.QuestionTemplateList.ElementAt(j);

                for(int k = 0; k < questionTemplate.SubquestionTemplateList.Count; k++)
                {
                    totalSubquestionPoints += questionTemplate.SubquestionTemplateList.ElementAt(k).SubquestionPoints;
                }
            }

            return testTemplate.MinimumPoints / totalSubquestionPoints;
        }

        /// <summary>
        /// Generates random test templates - their subquestion points are randomized, not dependent on any variables
        /// </summary>
        public static List<TestTemplate> GenerateRandomTestTemplates()
        {
            List<TestTemplate> testTemplates = new List<TestTemplate>();
            Random random = new Random();
            User owner = new User() { Login = "login", Email = "email", FirstName = "name", LastName = "surname", Role = (EnumTypes.Role)3, IsTestingData = true };
            int subquestionCount = 0;

            for (int i = 0; i < 10; i++)
            {
                string testId = i.ToString();
                TestTemplate testTemplate = new TestTemplate();
                testTemplate.TestNameIdentifier = "TestNameIdentifier_" + testId;
                testTemplate.TestNumberIdentifier = "TestNumberIdentifier_" + testId;
                testTemplate.Title = "Title_" + testId;
                testTemplate.NegativePoints = (EnumTypes.NegativePoints)random.Next(1, 4);
                double? totalSubquestionPoints = 0;
                int minimumPointsShare = random.Next(0, 52);
                testTemplate.Subject = "Subject_" + testId;
                testTemplate.OwnerLogin = owner.Login;
                testTemplate.Owner = owner;
                testTemplate.IsTestingData = true;
                List<QuestionTemplate> questionTemplates = new List<QuestionTemplate>();

                for (int j = 0; j < 10; j++)
                {
                    string questionId = j.ToString();
                    QuestionTemplate questionTemplate = new QuestionTemplate();
                    questionTemplate.QuestionNameIdentifier = "QuestionNameIdentifier_" + testId + "_" + questionId;
                    questionTemplate.QuestionNumberIdentifier = "QuestionNumberIdentifier_" + testId + "_" + questionId;
                    questionTemplate.Title = "Title_" + testId + "_" + questionId;
                    questionTemplate.Label = "Label_" + testId + "_" + questionId;
                    questionTemplate.OwnerLogin = owner.Login;
                    questionTemplate.TestTemplate = testTemplate;
                    List<SubquestionTemplate> subquestionTemplates = new List<SubquestionTemplate>();

                    for (int k = 0; k < 5; k++)
                    {
                        string subquestionId = k.ToString();
                        SubquestionTemplate subquestionTemplate = new SubquestionTemplate();
                        subquestionTemplate.SubquestionIdentifier = "SubquestionIdentifier_" + testId + "_" + questionId + "_" + subquestionId;
                        int subquestionType = random.Next(1, 11);
                        subquestionTemplate.SubquestionType = subquestionType;
                        subquestionTemplate.SubquestionText = "SubquestionText" + testId + "_" + questionId + "_" + subquestionId;
                        bool containsImage = random.NextDouble() < 0.25;
                        if (containsImage)
                        {
                            subquestionTemplate.ImageSource = "ImageSource_" + testId + "_" + questionId + "_" + subquestionId;
                        }
                        else
                        {
                            subquestionTemplate.ImageSource = "";
                        }

                        int possibleAnswersCount = random.Next(2, 11);
                        if (subquestionType == 4)
                        {
                            if (possibleAnswersCount % 2 == 1)
                            {
                                possibleAnswersCount = 2;
                            }
                        }
                        string[] possibleAnswers = new string[possibleAnswersCount];
                        for (int l = 0; l < possibleAnswersCount; l++)
                        {
                            possibleAnswers[l] = "PossibleAnswer_" + testId + "_" + questionId + "_" + subquestionId + "_" + l.ToString();
                        }
                        if (subquestionType == 1 || subquestionType == 2 || subquestionType == 3 || subquestionType == 4 ||
                            subquestionType == 6 || subquestionType == 7 || subquestionType == 9)
                        {
                            subquestionTemplate.PossibleAnswerList = possibleAnswers;
                        }
                        else
                        {
                            subquestionTemplate.PossibleAnswerList = new string[0];
                        }

                        int correctAnswersCount = 0;
                        if (subquestionType == 1 || subquestionType == 9)
                        {
                            correctAnswersCount = possibleAnswersCount;
                        }
                        else if (subquestionType == 2)
                        {
                            correctAnswersCount = random.Next(1, possibleAnswersCount + 1);
                        }
                        else if (subquestionType == 3)
                        {
                            correctAnswersCount = possibleAnswersCount / 2;
                        }
                        else if (subquestionType == 4)
                        {
                            correctAnswersCount = random.Next(2, 11);
                            if (correctAnswersCount % 2 == 1)
                            {
                                correctAnswersCount++;
                            }
                        }
                        else if (subquestionType == 6 || subquestionType == 7 || subquestionType == 8)
                        {
                            correctAnswersCount = 1;
                        }
                        string[] correctAnswers = new string[correctAnswersCount];
                        for (int l = 0; l < correctAnswersCount; l++)
                        {
                            correctAnswers[l] = "CorrectAnswer_" + testId + "_" + questionId + "_" + subquestionId + "_" + l.ToString();
                        }
                        subquestionTemplate.CorrectAnswerList = correctAnswers;

                        subquestionTemplate.SubquestionPoints = random.Next(1, 16);
                        totalSubquestionPoints += subquestionTemplate.SubquestionPoints;
                        subquestionTemplate.QuestionNumberIdentifier = questionTemplate.QuestionNumberIdentifier;
                        subquestionTemplate.OwnerLogin = owner.Login;
                        subquestionTemplate.QuestionTemplate = questionTemplate;
                        subquestionTemplates.Add(subquestionTemplate);
                        subquestionCount++;
                    }
                    questionTemplate.SubquestionTemplateList = subquestionTemplates;
                    questionTemplates.Add(questionTemplate);
                }

                testTemplate.MinimumPoints = totalSubquestionPoints * ((double)minimumPointsShare / 100);
                //testTemplate.MinimumPoints = random.Next(0, (int)totalSubquestionPoints);
                testTemplate.QuestionTemplateList = questionTemplates;
                testTemplates.Add(testTemplate);
            }

            return testTemplates;
        }

        /// <summary>
        /// Generates correlational test templates - their subquestion points are dependent on variables such as subquestion type, subject etc.
        /// </summary>
        public static List<TestTemplate> GenerateCorrelationalTestTemplates()
        {
            List<TestTemplate> testTemplates = new List<TestTemplate>();
            Random random = new Random();
            User owner = new User() { Login = "login", Email = "email", FirstName = "name", LastName = "surname", Role = (EnumTypes.Role)3, IsTestingData = true };
            int subquestionCount = 0;
            string[] subjectsArray = { "Chemie", "Zeměpis", "Matematika", "Dějepis", "Informatika" };
            int[] subquestionPointsByTypeArray = { 0, 2, -4, -3, -4, 5, -1, -1, 2, -3, 0 };
            int[] subquestionPointsBySubjectArray = { 3, -1, 3, 1, 1 };
            int[] negativePointsArray = { -4, -2, 0 };

            for (int i = 0; i < 10; i++)
            {
                string testId = i.ToString();
                TestTemplate testTemplate = new TestTemplate();
                testTemplate.TestNameIdentifier = "TestNameIdentifier_" + testId;
                testTemplate.TestNumberIdentifier = "TestNumberIdentifier_" + testId;
                testTemplate.Title = "Title_" + testId;
                int negativePoints = random.Next(1, 4);
                testTemplate.NegativePoints = (EnumTypes.NegativePoints)negativePoints;
                double? totalSubquestionPoints = 0;
                int minimumPointsShare = random.Next(0, 52);
                int subject = random.Next(0, 5);
                testTemplate.Subject = subjectsArray[subject];
                testTemplate.OwnerLogin = owner.Login;
                testTemplate.Owner = owner;
                testTemplate.IsTestingData = true;
                List<QuestionTemplate> questionTemplates = new List<QuestionTemplate>();

                for (int j = 0; j < 10; j++)
                {
                    string questionId = j.ToString();
                    QuestionTemplate questionTemplate = new QuestionTemplate();
                    questionTemplate.QuestionNameIdentifier = "QuestionNameIdentifier_" + testId + "_" + questionId;
                    questionTemplate.QuestionNumberIdentifier = "QuestionNumberIdentifier_" + testId + "_" + questionId;
                    questionTemplate.Title = "Title_" + testId + "_" + questionId;
                    questionTemplate.Label = "Label_" + testId + "_" + questionId;
                    questionTemplate.OwnerLogin = owner.Login;
                    questionTemplate.TestTemplate = testTemplate;
                    List<SubquestionTemplate> subquestionTemplates = new List<SubquestionTemplate>();

                    for (int k = 0; k < 5; k++)
                    {
                        string subquestionId = k.ToString();
                        SubquestionTemplate subquestionTemplate = new SubquestionTemplate();
                        subquestionTemplate.SubquestionIdentifier = "SubquestionIdentifier_" + testId + "_" + questionId + "_" + subquestionId;
                        int subquestionType = random.Next(1, 11); 
                        subquestionTemplate.SubquestionType = subquestionType;
                        subquestionTemplate.SubquestionText = "SubquestionText" + testId + "_" + questionId + "_" + subquestionId;
                        bool containsImage = random.Next(0, 2) > 0; 
                        if (containsImage)
                        {
                            subquestionTemplate.ImageSource = "ImageSource_" + testId + "_" + questionId + "_" + subquestionId;
                        }
                        else
                        {
                            subquestionTemplate.ImageSource = "";
                        }

                        int possibleAnswersCount = random.Next(2, 11);
                        if (subquestionType == 4)
                        {
                            if (possibleAnswersCount % 2 == 1)
                            {
                                possibleAnswersCount = 2;
                            }
                        }
                        else if(subquestionType == 10)
                        {
                            possibleAnswersCount = 1;
                        }
                        string[] possibleAnswers = new string[possibleAnswersCount];
                        for (int l = 0; l < possibleAnswersCount; l++)
                        {
                            possibleAnswers[l] = "PossibleAnswer_" + testId + "_" + questionId + "_" + subquestionId + "_" + l.ToString();
                        }
                        if (subquestionType == 1 || subquestionType == 2 || subquestionType == 3 || subquestionType == 4 ||
                            subquestionType == 6 || subquestionType == 7 || subquestionType == 9)
                        {
                            subquestionTemplate.PossibleAnswerList = possibleAnswers;
                        }
                        else
                        {
                            subquestionTemplate.PossibleAnswerList = new string[0];
                        }

                        int correctAnswersCount = 0;
                        if (subquestionType == 1 || subquestionType == 9)
                        {
                            correctAnswersCount = possibleAnswersCount;
                        }
                        else if (subquestionType == 2)
                        {
                            correctAnswersCount = random.Next(1, possibleAnswersCount + 1);
                        }
                        else if (subquestionType == 3)
                        {
                            correctAnswersCount = possibleAnswersCount / 2;
                        }
                        else if (subquestionType == 4)
                        {
                            correctAnswersCount = random.Next(2, 11);
                            if (correctAnswersCount % 2 == 1)
                            {
                                correctAnswersCount++;
                            }
                        }
                        else if (subquestionType == 6 || subquestionType == 7 || subquestionType == 8 || subquestionType == 10)
                        {
                            correctAnswersCount = 1;
                        }
                        string[] correctAnswers = new string[correctAnswersCount];
                        for (int l = 0; l < correctAnswersCount; l++)
                        {
                            correctAnswers[l] = "CorrectAnswer_" + testId + "_" + questionId + "_" + subquestionId + "_" + l.ToString();
                        }
                        subquestionTemplate.CorrectAnswerList = correctAnswers;
                        double correctAnswersShare = 0;
                        if (subquestionTemplate.PossibleAnswerList != null && subquestionTemplate.CorrectAnswerList != null)
                        {
                            //This type of subquestion typically contains 2 possible answers and many correct answers, so we set CorrectAnswersShare manually instead
                            if (subquestionType == 4)
                            {
                                correctAnswersShare = 0.5;
                            }
                            //These type of subquestion are about sorting elements - the more elements there are, the harder it is to answer correctly
                            else if (subquestionType == 1 || subquestionType == 9)
                            {
                                correctAnswersShare = 1 / (double)possibleAnswersCount;
                            }
                            //This type of subquestion uses slider - typically tens to hunders of possible answers
                            else if (subquestionType == 10)
                            {
                                correctAnswersShare = 0;
                            }
                            else
                            {
                                correctAnswersShare = (double)correctAnswersCount / (double)possibleAnswersCount;
                            }
                        }

                        double subquestionPoints = random.Next(8, 13);
                        subquestionPoints += subquestionPointsByTypeArray[subquestionType];
                        subquestionPoints += correctAnswersShare * (-2);//correctAnswersShare modifier
                        subquestionPoints += subquestionPointsBySubjectArray[subject];//subjectAveragePoints modifier
                        //containsImage variable is not correlational
                        subquestionPoints += negativePointsArray[negativePoints-1];//negativePoints modifier
                        subquestionPoints += ((double)minimumPointsShare / 100) * 4;//minimumPointsShare modifier

                        subquestionTemplate.SubquestionPoints = subquestionPoints;

                        totalSubquestionPoints += subquestionTemplate.SubquestionPoints;
                        subquestionTemplate.QuestionNumberIdentifier = questionTemplate.QuestionNumberIdentifier;
                        subquestionTemplate.OwnerLogin = owner.Login;
                        subquestionTemplate.QuestionTemplate = questionTemplate;
                        subquestionTemplates.Add(subquestionTemplate);
                        subquestionCount++;
                    }
                    questionTemplate.SubquestionTemplateList = subquestionTemplates;
                    questionTemplates.Add(questionTemplate);
                }

                testTemplate.MinimumPoints = totalSubquestionPoints * ((double)minimumPointsShare / 100);
                testTemplate.QuestionTemplateList = questionTemplates;
                testTemplates.Add(testTemplate);
            }

            return testTemplates;
        }

        /// <summary>
        /// Generates correlational test templates - their subquestion points are dependent on variables such as subquestion type, subject etc.
        /// </summary>
        public static double[] GetSubquestionTypeAveragePoints(List<TestTemplate> testTemplates)
        {
            int subquestionTypeCount = 10;//todo: po predelani subquestionType na enum predelat tuto promennou
            double[] subquestionTypePointsShare = new double[subquestionTypeCount];
            double[] subquestionCountByType = new double[subquestionTypeCount];

            for (int i = 0; i < testTemplates.Count; i++)
            {
                TestTemplate testTemplate = testTemplates[i];

                for (int j = 0; j < testTemplate.QuestionTemplateList.Count; j++)
                {
                    QuestionTemplate questionTemplate = testTemplate.QuestionTemplateList.ElementAt(j);

                    for (int k = 0; k < questionTemplate.SubquestionTemplateList.Count; k++)
                    {
                        SubquestionTemplate subquestionTemplate = questionTemplate.SubquestionTemplateList.ElementAt(k);
                        int subquestionType = subquestionTemplate.SubquestionType;
                        if(subquestionTemplate.SubquestionPoints != null)
                        {
                            subquestionTypePointsShare[subquestionType - 1] += (double)subquestionTemplate.SubquestionPoints;
                        }
                        else
                        {
                            continue;
                        }
                        subquestionCountByType[subquestionType-1] += 1;
                    }
                }
            }

            for(int i = 0; i < subquestionTypePointsShare.Length; i++)
            {
                subquestionTypePointsShare[i] = subquestionTypePointsShare[i] / subquestionCountByType[i];
            }

            return subquestionTypePointsShare;
        }

        /// <summary>
        /// Returns array of average points for each subject
        /// </summary>
        public static double[] GetSubjectAveragePoints(List<TestTemplate> testTemplates)
        {
            int subjectsCount = 5;//todo: predelat
            string[] subjectsArray = { "Chemie", "Zeměpis", "Matematika", "Dějepis", "Informatika" };
            double[] subjectPointsShare = new double[subjectsCount];
            double[] subquestionCountBySubject = new double[subjectsCount];

            for (int i = 0; i < testTemplates.Count; i++)
            {
                TestTemplate testTemplate = testTemplates[i];

                for (int j = 0; j < testTemplate.QuestionTemplateList.Count; j++)
                {
                    QuestionTemplate questionTemplate = testTemplate.QuestionTemplateList.ElementAt(j);

                    for (int k = 0; k < questionTemplate.SubquestionTemplateList.Count; k++)
                    {
                        SubquestionTemplate subquestionTemplate = questionTemplate.SubquestionTemplateList.ElementAt(k);
                        string? subject = testTemplate.Subject;
                        int subjectId = Array.FindIndex(subjectsArray, x => x.Contains(subject));
                        if(subquestionTemplate.SubquestionPoints != null)
                        {
                            subjectPointsShare[subjectId] += (double)subquestionTemplate.SubquestionPoints;
                        }
                        else
                        {
                            continue;
                        }
                        subquestionCountBySubject[subjectId] += 1;
                    }
                }
            }

            for (int i = 0; i < subjectPointsShare.Length; i++)
            {
                subjectPointsShare[i] = subjectPointsShare[i] / subquestionCountBySubject[i];
            }

            return subjectPointsShare;
        }

        /// <summary>
        /// Runs selected python file
        /// </summary>
        public static string GetSubquestionTemplateSuggestedPoints(string function, SubquestionTemplateRecord? subquestionTemplateRecord)
        {
            string[] arguments = new string[] { subquestionTemplateRecord.SubquestionTypeAveragePoints.ToString().Replace(",", "."), subquestionTemplateRecord.CorrectAnswersShare.ToString().Replace(",", "."),
            subquestionTemplateRecord.SubjectAveragePoints.ToString().Replace(",", "."), subquestionTemplateRecord.ContainsImage.ToString().Replace(",", "."), subquestionTemplateRecord.NegativePoints.ToString().Replace(",", "."), subquestionTemplateRecord.MinimumPointsShare.ToString().Replace(",", ".")};

            ProcessStartInfo start = new ProcessStartInfo();
            start.FileName = Config.GetPythonPath();
            start.Arguments = string.Format("{0} {1} {2} {3} {4} {5} {6} {7}", 
                Path.GetDirectoryName(Environment.CurrentDirectory) + "\\NeuralNetworkTools\\TemplateNeuralNetwork.py", function, arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5]); 
            start.UseShellExecute = false;
            start.CreateNoWindow = true; 
            start.RedirectStandardOutput = true;
            start.RedirectStandardError = true; 
            using (Process process = Process.Start(start))
            {
                using (StreamReader reader = process.StandardOutput)
                {
                    string stderr = process.StandardError.ReadToEnd(); 
                    string result = reader.ReadToEnd(); 
                    return result;
                }
            }
        }
    }
}