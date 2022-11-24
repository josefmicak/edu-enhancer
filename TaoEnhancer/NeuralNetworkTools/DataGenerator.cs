﻿using DomainModel;
using Common;
using CsvHelper;
using System.Globalization;

namespace NeuralNetworkTools
{
    public class DataGenerator
    {
        /// <summary>
        /// Generates .csv file of a number of subquestion templates with parameters that are used by the neural network
        /// </summary>
        /// <param name="dataColleration">Decides whether the generated data will be randomized or if there are going to be collerations between the templates</param>
        public static void GenerateTemplatesFile(string dataColleration)
        {
            List<TestTemplate> testTemplates = new List<TestTemplate>();
            if (dataColleration == "none")
            {
                testTemplates = GenerateRandomTestTemplates(testTemplates, 500);
            }
            else if (dataColleration == "on")
            {
                testTemplates = GenerateCorrelationalTestTemplates(testTemplates, 500);
            }
            var subquestionTemplateRecords = GetSubquestionTemplateRecords(testTemplates);

            string filePath = "";
            if (dataColleration == "none")
            {
                filePath = "D:\\Users\\granders\\Desktop\\RandomTemplatesFile.csv";
            }
            else if (dataColleration == "on")
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

        public static void GenerateResultsFile(List<TestTemplate> testTemplates, string dataColleration)
        {
            List<TestResult> testResults = new List<TestResult>();
            if (dataColleration == "none")
            {
                testResults = GenerateRandomTestResults(testTemplates, 0, 500);
            }
            else if (dataColleration == "on")
            {
                testResults = GenerateCorrelationalTestResults(testTemplates, 0, 500);
            }
            var subquestionResultRecords = GetSubquestionResultRecords(testResults);
            string filePath = "";
            if (dataColleration == "none")
            {
                filePath = "D:\\Users\\granders\\Desktop\\RandomResultsFile.csv";
            }
            else if (dataColleration == "on")
            {
                filePath = "D:\\Users\\granders\\Desktop\\CorrelationalResultsFile.csv";
            }
            File.Delete(filePath);
            using (var writer = new StreamWriter(filePath))
            using (var csv = new CsvWriter(writer, CultureInfo.InvariantCulture))
            {
                csv.WriteRecords(subquestionResultRecords);
            }
        }

        /// <summary>
        /// Creates a list of subquestion templates with parameters that are used by the neural network
        /// </summary>
        public static List<SubquestionTemplateRecord> GetSubquestionTemplateRecords(List<TestTemplate> testTemplates)
        {
            List<SubquestionTemplateRecord> subquestionTemplateRecords = new List<SubquestionTemplateRecord>();
            string[] subjectsArray = { "Chemie", "Zeměpis", "Matematika", "Dějepis", "Informatika" };
            double[] subquestionTypeAveragePoints = GetSubquestionTypeAverageTemplatePoints(testTemplates);
            double[] subjectAveragePoints = GetSubjectAverageTemplatePoints(testTemplates);
            int subquestionTemplateRecordId = 0;
            User owner = testTemplates[0].Owner;

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
                        subquestionTemplateRecord.SubquestionTemplate = subquestionTemplate;
                        subquestionTemplateRecord.SubquestionIdentifier = subquestionTemplate.SubquestionIdentifier;
                        subquestionTemplateRecord.QuestionNumberIdentifier = subquestionTemplate.QuestionNumberIdentifier;
                        subquestionTemplateRecord.Owner = owner;
                        subquestionTemplateRecord.OwnerLogin = owner.Login;
                        EnumTypes.SubquestionType subquestionType = subquestionTemplate.SubquestionType;
                        subquestionTemplateRecord.SubquestionTypeAveragePoints = Math.Round(subquestionTypeAveragePoints[Convert.ToInt32(subquestionType) - 1], 2);
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
                            if (subquestionType == EnumTypes.SubquestionType.MultipleQuestions)
                            {
                                subquestionTemplateRecord.CorrectAnswersShare = 0.5;
                            }
                            //These type of subquestion are about sorting elements - the more elements there are, the harder it is to answer correctly
                            else if (subquestionType == EnumTypes.SubquestionType.OrderingElements || subquestionType == EnumTypes.SubquestionType.GapMatch)
                            {
                                subquestionTemplateRecord.CorrectAnswersShare = 1 / (double)possibleAnswersCount;
                            }
                            //This type of subquestion uses slider - typically tens to hunders of possible answers
                            else if (subquestionType == EnumTypes.SubquestionType.Slider)
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
                        double? minimumPointsShareRound = minimumPointsShare.HasValue
                        ? (double?)Math.Round(minimumPointsShare.Value, 2)
                        : null;
                        subquestionTemplateRecord.MinimumPointsShare = minimumPointsShareRound;
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
            for (int j = 0; j < testTemplate.QuestionTemplateList.Count; j++)
            {
                QuestionTemplate questionTemplate = testTemplate.QuestionTemplateList.ElementAt(j);

                for (int k = 0; k < questionTemplate.SubquestionTemplateList.Count; k++)
                {
                    totalSubquestionPoints += questionTemplate.SubquestionTemplateList.ElementAt(k).SubquestionPoints;
                }
            }

            return testTemplate.MinimumPoints / totalSubquestionPoints;
        }

        /// <summary>
        /// Generates random test templates - their subquestion points are randomized, not dependent on any variables
        /// <param name="existingTestTemplates">Already existing test templates owned by the testing user</param>
        /// <param name="amountOfSubquestionTemplatesToBeGenerated">Amount of subquestion templates to be generated</param>
        /// </summary>
        public static List<TestTemplate> GenerateRandomTestTemplates(List<TestTemplate> existingTestTemplates, int amountOfSubquestionTemplatesToBeGenerated)
        {
            List<TestTemplate> testTemplates = existingTestTemplates;
            int existingTestTemplatesCount = existingTestTemplates.Count;
            Random random = new Random();
            User owner = new User() { Login = "login", Email = "adminemail", FirstName = "name", LastName = "surname", Role = (EnumTypes.Role)3, IsTestingData = true };
            int subquestionCount = 0;
            bool stopDataGeneration = false;
            string[] subjectsArray = { "Chemie", "Zeměpis", "Matematika", "Dějepis", "Informatika" };

            for (int i = existingTestTemplates.Count; ; i++)
            {
                if (stopDataGeneration)
                {
                    break;
                }

                string testId = i.ToString();
                TestTemplate testTemplate = new TestTemplate();
                testTemplate.TestNameIdentifier = "TestNameIdentifier_" + testId;
                testTemplate.TestNumberIdentifier = "TestNumberIdentifier_" + testId;
                testTemplate.Title = "Title_" + testId;
                testTemplate.NegativePoints = (EnumTypes.NegativePoints)random.Next(1, 4);
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
                    if (stopDataGeneration)
                    {
                        break;
                    }

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
                        if (stopDataGeneration)
                        {
                            break;
                        }

                        string subquestionId = k.ToString();
                        SubquestionTemplate subquestionTemplate = new SubquestionTemplate();
                        subquestionTemplate.SubquestionIdentifier = "SubquestionIdentifier_" + testId + "_" + questionId + "_" + subquestionId;
                        int subquestionType = random.Next(1, 11);
                        subquestionTemplate.SubquestionType = (EnumTypes.SubquestionType)subquestionType;
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

                        if (subquestionCount >= amountOfSubquestionTemplatesToBeGenerated)//we have reached the desired amount of generated subquestion templates
                        {
                            stopDataGeneration = true;
                        }
                    }
                    questionTemplate.SubquestionTemplateList = subquestionTemplates;
                    questionTemplates.Add(questionTemplate);
                }

                double? minimumPoints = totalSubquestionPoints * ((double)minimumPointsShare / 100);
                double? minimumPointsRound = minimumPoints.HasValue
                ? (double?)Math.Round(minimumPoints.Value, 2)
                : null;
                testTemplate.MinimumPoints = minimumPointsRound;
                testTemplate.QuestionTemplateList = questionTemplates;
                testTemplates.Add(testTemplate);
            }

            testTemplates.RemoveRange(0, existingTestTemplatesCount);
            return testTemplates;
        }

        /// <summary>
        /// Generates correlational test templates - their subquestion points are dependent on variables such as subquestion type, subject etc.
        /// <param name="existingTestTemplates">Already existing test templates owned by the testing user</param>
        /// <param name="amountOfSubquestionTemplatesToBeGenerated">Amount of subquestion templates to be generated</param>
        /// </summary>
        public static List<TestTemplate> GenerateCorrelationalTestTemplates(List<TestTemplate> existingTestTemplates, int amountOfSubquestionTemplatesToBeGenerated)
        {
            //existing test templates have to be added here because otherwise newly created test templates would not be related to the old ones
            List<TestTemplate> testTemplates = existingTestTemplates;
            int existingTestTemplatesCount = existingTestTemplates.Count;
            Random random = new Random();
            User owner = new User() { Login = "login", Email = "adminemail", FirstName = "name", LastName = "surname", Role = (EnumTypes.Role)3, IsTestingData = true };
            int subquestionCount = 0;
            bool stopDataGeneration = false;
            string[] subjectsArray = { "Chemie", "Zeměpis", "Matematika", "Dějepis", "Informatika" };
            //int[] subquestionPointsByTypeArray = { 0, 2, -4, -3, -4, 5, -1, -1, 2, -3, 0 };
            int[] subquestionPointsByTypeArray = { 0, 4, -2, -1, -2, 7, 1, 1, 4, -1, 2 };
            int[] subquestionPointsBySubjectArray = { 3, -1, 3, 1, 1 };
            int[] negativePointsArray = { -4, -2, 0 };

            for (int i = existingTestTemplates.Count; ; i++)
            {
                if (stopDataGeneration)
                {
                    break;
                }

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
                    if (stopDataGeneration)
                    {
                        break;
                    }

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
                        if (stopDataGeneration)
                        {
                            break;
                        }

                        string subquestionId = k.ToString();
                        SubquestionTemplate subquestionTemplate = new SubquestionTemplate();
                        subquestionTemplate.SubquestionIdentifier = "SubquestionIdentifier_" + testId + "_" + questionId + "_" + subquestionId;
                        int subquestionType = random.Next(1, 11);
                        subquestionTemplate.SubquestionType = (EnumTypes.SubquestionType)subquestionType;
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
                        else if (subquestionType == 10)
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
                        subquestionPoints += negativePointsArray[negativePoints - 1];//negativePoints modifier
                        subquestionPoints += ((double)minimumPointsShare / 100) * 4;//minimumPointsShare modifier

                        subquestionTemplate.SubquestionPoints = Math.Round(subquestionPoints, 2);

                        totalSubquestionPoints += subquestionTemplate.SubquestionPoints;
                        subquestionTemplate.QuestionNumberIdentifier = questionTemplate.QuestionNumberIdentifier;
                        subquestionTemplate.OwnerLogin = owner.Login;
                        subquestionTemplate.QuestionTemplate = questionTemplate;
                        subquestionTemplate.CorrectChoicePoints = CalculateCorrectChoicePoints(
                            Math.Round(Convert.ToDouble(subquestionPoints), 2), subquestionTemplate.CorrectAnswerList, subquestionTemplate.SubquestionType);
                        subquestionTemplates.Add(subquestionTemplate);
                        subquestionCount++;

                        if (subquestionCount >= amountOfSubquestionTemplatesToBeGenerated)//we have reached the desired amount of generated subquestion templates
                        {
                            stopDataGeneration = true;
                        }
                    }
                    questionTemplate.SubquestionTemplateList = subquestionTemplates;
                    questionTemplates.Add(questionTemplate);
                }

                double? minimumPoints = totalSubquestionPoints * ((double)minimumPointsShare / 100);
                double? minimumPointsRound = minimumPoints.HasValue
                ? (double?)Math.Round(minimumPoints.Value, 2)
                : null;
                testTemplate.MinimumPoints = minimumPointsRound; testTemplate.QuestionTemplateList = questionTemplates;
                testTemplates.Add(testTemplate);
            }

            testTemplates.RemoveRange(0, existingTestTemplatesCount);
            return testTemplates;
        }

        /// <summary>
        /// Generates random test results - student's points are randomized, not dependent on any variables
        /// <param name="existingTestTemplates">Already existing test templates owned by the testing user</param>
        /// <param name="testingDataTestResultsCount">Amount of existing test results that are marked as testing data</param>
        /// <param name="amountOfSubquestionResultsToBeGenerated">Amount of subquestion results to be generated</param>
        /// </summary>
        public static List<TestResult> GenerateRandomTestResults(List<TestTemplate> existingTestTemplates, int testingDataTestResultsCount, int amountOfSubquestionResultsToBeGenerated)
        {
            List<TestResult> testResults = new List<TestResult>();
            Student student = new Student() { Login = "testingstudent", Email = "studentemail", StudentIdentifier = "testingstudent", FirstName = "name", LastName = "surname", IsTestingData = true };
            Random random = new Random();
            int subquestionCount = 0;
            bool stopDataGeneration = false;

            for (int i = 0; i < existingTestTemplates.Count; i++)
            {
                if (stopDataGeneration)
                {
                    break;
                }

                string testId = testingDataTestResultsCount.ToString();
                TestTemplate testTemplate = existingTestTemplates[i];
                TestResult testResult = new TestResult();
                testResult.TestTemplate = testTemplate;
                testResult.TestResultIdentifier = "TestResultIdentifier_" + testId;
                testResult.OwnerLogin = testTemplate.OwnerLogin;
                testResult.TestNameIdentifier = testTemplate.TestNameIdentifier;
                testResult.TestNumberIdentifier = testTemplate.TestNumberIdentifier;
                testResult.TimeStamp = DateTime.Now;
                testResult.Student = student;
                testResult.StudentLogin = student.Login;
                testResult.IsTestingData = true;
                List<QuestionResult> questionResults = new List<QuestionResult>();

                for (int j = 0; j < testTemplate.QuestionTemplateList.Count; j++)
                {
                    if (stopDataGeneration)
                    {
                        break;
                    }

                    string questionId = j.ToString();
                    QuestionTemplate questionTemplate = testTemplate.QuestionTemplateList.ElementAt(j);
                    QuestionResult questionResult = new QuestionResult();
                    questionResult.QuestionTemplate = questionTemplate;
                    questionResult.TestResultIdentifier = testResult.TestResultIdentifier;
                    questionResult.TestResult = testResult;
                    questionResult.QuestionNumberIdentifier = questionTemplate.QuestionNumberIdentifier;
                    questionResult.OwnerLogin = testResult.OwnerLogin;
                    List<SubquestionResult> subquestionResults = new List<SubquestionResult>();

                    for (int k = 0; k < questionTemplate.SubquestionTemplateList.Count; k++)
                    {
                        if (stopDataGeneration)
                        {
                            break;

                        }
                        string subquestionId = k.ToString();
                        SubquestionTemplate subquestionTemplate = questionTemplate.SubquestionTemplateList.ElementAt(k);
                        SubquestionResult subquestionResult = new SubquestionResult();
                        subquestionResult.SubquestionTemplate = subquestionTemplate;
                        subquestionResult.QuestionResult = questionResult;
                        subquestionResult.TestResultIdentifier = questionResult.TestResultIdentifier;
                        subquestionResult.QuestionNumberIdentifier = questionResult.QuestionNumberIdentifier;
                        subquestionResult.SubquestionIdentifier = subquestionTemplate.SubquestionIdentifier;
                        subquestionResult.OwnerLogin = questionResult.OwnerLogin;
                        int answerCount = random.Next(0, 5);
                        string[] studentsAnswerList = new string[answerCount];
                        for (int l = 0; l < answerCount; l++)
                        {
                            studentsAnswerList[l] = "Answer_" + testId + "_" + questionId + "_" + subquestionId + "_" + l;
                        }
                        subquestionResult.StudentsAnswerList = studentsAnswerList;
                        int points = random.Next(0, (int)subquestionTemplate.SubquestionPoints.Value);
                        subquestionResult.StudentsPoints = points;
                        subquestionResult.DefaultStudentsPoints = points;
                        int answerCorrect = random.Next(0, 6);
                        subquestionResult.AnswerCorrect = (EnumTypes.AnswerCorrect)answerCorrect;
                        subquestionResults.Add(subquestionResult);
                        subquestionCount++;

                        if (subquestionCount >= amountOfSubquestionResultsToBeGenerated)//we have reached the desired amount of generated subquestion results
                        {
                            stopDataGeneration = true;
                        }
                    }

                    questionResult.SubquestionResultList = subquestionResults;
                    questionResults.Add(questionResult);
                }

                testResult.QuestionResultList = questionResults;
                testResults.Add(testResult);
                testingDataTestResultsCount++;
            }

            return testResults;
        }

        /// <summary>
        /// Generates random test results - student's points are randomized, not dependent on any variables
        /// <param name="existingTestTemplates">Already existing test templates owned by the testing user</param>
        /// <param name="testingDataTestResultsCount">Amount of existing test results that are marked as testing data</param>
        /// <param name="amountOfSubquestionResultsToBeGenerated">Amount of subquestion results to be generated</param>
        /// </summary>
        public static List<TestResult> GenerateCorrelationalTestResults(List<TestTemplate> existingTestTemplates, int testingDataTestResultsCount, int amountOfSubquestionResultsToBeGenerated)
        {
            List<TestResult> testResults = new List<TestResult>();
            Student student = new Student() { Login = "testingstudent", Email = "studentemail", StudentIdentifier = "testingstudent", FirstName = "name", LastName = "surname", IsTestingData = true };
            Random random = new Random();
            string[] subjectsArray = { "Chemie", "Zeměpis", "Matematika", "Dějepis", "Informatika" };
            int[] subquestionPointsByTypeArray = { 0, 4, -2, -1, -2, 7, 1, 1, 4, -1, 2 };
            int[] subquestionPointsBySubjectArray = { 3, -1, 3, 1, 1 };
            int[] negativePointsArray = { -4, -2, 0 };
            int subquestionCount = 0;
            bool stopDataGeneration = false;

            for (int i = 0;; i++)
            {
                if (stopDataGeneration)
                {
                    break;
                }

                //by doing this, we ensure that there can be more testing data results than testing data templates in the system
                if(i == existingTestTemplates.Count)
                {
                    i = 0;
                }

                string testId = testingDataTestResultsCount.ToString();
                TestTemplate testTemplate = existingTestTemplates[i];
                TestResult testResult = new TestResult();
                testResult.TestTemplate = testTemplate;
                testResult.TestResultIdentifier = "TestResultIdentifier_" + testId;
                testResult.OwnerLogin = testTemplate.OwnerLogin;
                testResult.TestNameIdentifier = testTemplate.TestNameIdentifier;
                testResult.TestNumberIdentifier = testTemplate.TestNumberIdentifier;
                testResult.TimeStamp = DateTime.Now;
                testResult.Student = student;
                testResult.StudentLogin = student.Login;
                testResult.IsTestingData = true;
                List<QuestionResult> questionResults = new List<QuestionResult>();

                for (int j = 0; j < testTemplate.QuestionTemplateList.Count; j++)
                {
                    if (stopDataGeneration)
                    {
                        break;
                    }

                    string questionId = j.ToString();
                    QuestionTemplate questionTemplate = testTemplate.QuestionTemplateList.ElementAt(j);
                    QuestionResult questionResult = new QuestionResult();
                    questionResult.QuestionTemplate = questionTemplate;
                    questionResult.TestResultIdentifier = testResult.TestResultIdentifier;
                    questionResult.TestResult = testResult;
                    questionResult.QuestionNumberIdentifier = questionTemplate.QuestionNumberIdentifier;
                    questionResult.OwnerLogin = testResult.OwnerLogin;
                    List<SubquestionResult> subquestionResults = new List<SubquestionResult>();

                    for (int k = 0; k < questionTemplate.SubquestionTemplateList.Count; k++)
                    {
                        if (stopDataGeneration)
                        {
                            break;
                        }

                        string subquestionId = k.ToString();
                        SubquestionTemplate subquestionTemplate = questionTemplate.SubquestionTemplateList.ElementAt(k);
                        SubquestionResult subquestionResult = new SubquestionResult();
                        subquestionResult.SubquestionTemplate = subquestionTemplate;
                        subquestionResult.QuestionResult = questionResult;
                        subquestionResult.TestResultIdentifier = questionResult.TestResultIdentifier;
                        subquestionResult.QuestionNumberIdentifier = questionResult.QuestionNumberIdentifier;
                        subquestionResult.SubquestionIdentifier = subquestionTemplate.SubquestionIdentifier;
                        subquestionResult.OwnerLogin = questionResult.OwnerLogin;
                        int answerCount = random.Next(0, 5);
                        string[] studentsAnswerList = new string[answerCount];
                        for (int l = 0; l < answerCount; l++)
                        {
                            studentsAnswerList[l] = "Answer_" + testId + "_" + questionId + "_" + subquestionId + "_" + l;
                        }
                        subquestionResult.StudentsAnswerList = studentsAnswerList;

                        double answerCorectness = (double)(random.Next(0, 101)) / 100;
                        int subquestionType = (int)subquestionTemplate.SubquestionType;
                        double? studentsPoints;
                        //for some subquestion types the answer is either entirely correct or entirely incorrect
                        if (subquestionType == 6 || subquestionType == 7 || subquestionType == 8)
                        {
                            if(answerCorectness > 0.25)
                            {
                                answerCorectness = 1;
                                subquestionResult.DefaultStudentsPoints = subquestionTemplate.SubquestionPoints;
                                studentsPoints = subquestionTemplate.SubquestionPoints;
                            }
                            else
                            {
                                answerCorectness = 0;
                                subquestionResult.DefaultStudentsPoints = 0;
                                studentsPoints = 0;
                            }
                        }
                        else
                        {
                            subquestionResult.DefaultStudentsPoints = subquestionTemplate.SubquestionPoints * answerCorectness;
                            studentsPoints = subquestionResult.DefaultStudentsPoints;
                            studentsPoints += subquestionPointsByTypeArray[(int)subquestionTemplate.SubquestionType];//subquestionTypeAveragePoints modifier
                            int subject = Array.IndexOf(subjectsArray, testTemplate.Subject);//subjectAveragePoints modifier
                            studentsPoints += subquestionPointsBySubjectArray[subject];//subjectAveragePoints modifier
                            //containsImage variable is not correlational
                            studentsPoints += negativePointsArray[((int)testTemplate.NegativePoints) - 1];//negativePoints modifier
                            studentsPoints += ((double)testTemplate.MinimumPoints / 100) * 4;//minimumPointsShare modifier
                        }

                        subquestionResult.AnswerCorrectness = answerCorectness;

                        //for certain combinations of variables, student could potentially get awarded more points than the original subquestion points
                        if(studentsPoints > subquestionTemplate.SubquestionPoints)
                        {
                            studentsPoints = subquestionTemplate.SubquestionPoints;
                        }

                        subquestionResult.StudentsPoints = Math.Round(studentsPoints.Value, 2);
                        subquestionResults.Add(subquestionResult);
                        subquestionCount++;

                        if (subquestionCount >= amountOfSubquestionResultsToBeGenerated)//we have reached the desired amount of generated subquestion results
                        {
                            stopDataGeneration = true;
                        }
                    }

                    questionResult.SubquestionResultList = subquestionResults;
                    questionResults.Add(questionResult);
                }

                testResult.QuestionResultList = questionResults;
                testResults.Add(testResult);
                testingDataTestResultsCount++;
            }

            return testResults;
        }

        public static List<SubquestionResultRecord> GetSubquestionResultRecords(List<TestResult> testResults)
        {
            List<SubquestionResultRecord> subquestionResultRecords = new List<SubquestionResultRecord>();
            string[] subjectsArray = { "Chemie", "Zeměpis", "Matematika", "Dějepis", "Informatika" };
            double[] subquestionTypeAveragePoints = GetSubquestionTypeAverageStudentsPoints(testResults);
            double[] subjectAveragePoints = GetSubjectAverageStudentsPoints(testResults);
            int subquestionResultRecordId = 0;
            User owner = testResults[0].TestTemplate.Owner;

            for (int i = 0; i < testResults.Count; i++)
            {
                TestResult testResult = testResults[i];
                TestTemplate testTemplate = testResult.TestTemplate;
                double? minimumPointsShare = GetMinimumPointsShare(testTemplate);

                for (int j = 0; j < testResult.QuestionResultList.Count; j++)
                {
                    QuestionResult questionResult = testResult.QuestionResultList.ElementAt(j);

                    for (int k = 0; k < questionResult.SubquestionResultList.Count; k++)
                    {
                        SubquestionResult subquestionResult = questionResult.SubquestionResultList.ElementAt(k);
                        SubquestionTemplate subquestionTemplate = subquestionResult.SubquestionTemplate;
                        SubquestionResultRecord subquestionResultRecord = new SubquestionResultRecord();
                        subquestionResultRecord.SubquestionResult = subquestionResult;
                        subquestionResultRecord.SubquestionIdentifier = subquestionTemplate.SubquestionIdentifier;
                        subquestionResultRecord.QuestionNumberIdentifier = subquestionTemplate.QuestionNumberIdentifier;
                        subquestionResultRecord.TestResultIdentifier = subquestionResult.TestResultIdentifier;
                        subquestionResultRecord.Owner = owner;
                        subquestionResultRecord.OwnerLogin = owner.Login;
                        EnumTypes.SubquestionType subquestionType = subquestionTemplate.SubquestionType;
                        subquestionResultRecord.SubquestionTypeAveragePoints = Math.Round(subquestionTypeAveragePoints[Convert.ToInt32(subquestionType) - 1], 2);
                        string? subject = testTemplate.Subject;
                        int subjectId = Array.FindIndex(subjectsArray, x => x.Contains(subject));
                        subquestionResultRecord.SubjectAveragePoints = Math.Round(subquestionTypeAveragePoints[subjectId], 2);
                        subquestionResultRecord.ContainsImage = Convert.ToInt32((subquestionTemplate.ImageSource == "") ? false : true);
                        subquestionResultRecord.NegativePoints = Convert.ToInt32(testTemplate.NegativePoints);
                        double? minimumPointsShareRound = minimumPointsShare.HasValue
                            ? (double?)Math.Round(minimumPointsShare.Value, 2)
                            : null;
                        subquestionResultRecord.MinimumPointsShare = minimumPointsShareRound;
                        subquestionResultRecord.AnswerCorrectness = subquestionResult.AnswerCorrectness;
                        subquestionResultRecord.StudentsPoints = subquestionResult.StudentsPoints;
                        subquestionResultRecords.Add(subquestionResultRecord);
                    }
                }
            }

            return subquestionResultRecords;
        }

        /// <summary>
        /// Generates correlational test templates - their subquestion points are dependent on variables such as subquestion type, subject etc.
        /// </summary>
        public static double[] GetSubquestionTypeAverageTemplatePoints(List<TestTemplate> testTemplates)
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
                        EnumTypes.SubquestionType subquestionType = subquestionTemplate.SubquestionType;
                        if (subquestionTemplate.SubquestionPoints != null)
                        {
                            subquestionTypePointsShare[Convert.ToInt32(subquestionType) - 1] += (double)subquestionTemplate.SubquestionPoints;
                        }
                        else
                        {
                            continue;
                        }
                        subquestionCountByType[Convert.ToInt32(subquestionType) - 1] += 1;
                    }
                }
            }

            for (int i = 0; i < subquestionTypePointsShare.Length; i++)
            {
                subquestionTypePointsShare[i] = subquestionTypePointsShare[i] / subquestionCountByType[i];
            }

            return subquestionTypePointsShare;
        }

        public static double[] GetSubquestionTypeAverageStudentsPoints(List<TestResult> testResults)
        {
            int subquestionTypeCount = 10;//todo: po predelani subquestionType na enum predelat tuto promennou
            double[] subquestionTypePointsShare = new double[subquestionTypeCount];
            double[] subquestionCountByType = new double[subquestionTypeCount];

            for (int i = 0; i < testResults.Count; i++)
            {
                TestResult testResult = testResults[i];

                for (int j = 0; j < testResult.QuestionResultList.Count; j++)
                {
                    QuestionResult questionResult = testResult.QuestionResultList.ElementAt(j);

                    for (int k = 0; k < questionResult.SubquestionResultList.Count; k++)
                    {
                        SubquestionResult subquestionResult = questionResult.SubquestionResultList.ElementAt(k);
                        EnumTypes.SubquestionType subquestionType = subquestionResult.SubquestionTemplate.SubquestionType;
                        if (subquestionResult.SubquestionTemplate.SubquestionPoints != null)
                        {
                            subquestionTypePointsShare[Convert.ToInt32(subquestionType) - 1] += (double)subquestionResult.StudentsPoints / (double)subquestionResult.SubquestionTemplate.SubquestionPoints;
                        }
                        else
                        {
                            continue;
                        }
                        subquestionCountByType[Convert.ToInt32(subquestionType) - 1] += 1;
                    }
                }
            }

            for (int i = 0; i < subquestionTypePointsShare.Length; i++)
            {
                subquestionTypePointsShare[i] = subquestionTypePointsShare[i] / subquestionCountByType[i];
            }

            return subquestionTypePointsShare;
        }

        /// <summary>
        /// Returns array of average points for each subject
        /// </summary>
        public static double[] GetSubjectAverageTemplatePoints(List<TestTemplate> testTemplates)
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
                        if (subquestionTemplate.SubquestionPoints != null)
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

        public static double[] GetSubjectAverageStudentsPoints(List<TestResult> testResults)
        {
            int subjectsCount = 5;//todo: predelat
            string[] subjectsArray = { "Chemie", "Zeměpis", "Matematika", "Dějepis", "Informatika" };
            double[] subjectPointsShare = new double[subjectsCount];
            double[] subquestionCountBySubject = new double[subjectsCount];

            for (int i = 0; i < testResults.Count; i++)
            {
                TestResult testResult = testResults[i];

                for (int j = 0; j < testResult.QuestionResultList.Count; j++)
                {
                    QuestionResult questionResult = testResult.QuestionResultList.ElementAt(j);

                    for (int k = 0; k < questionResult.SubquestionResultList.Count; k++)
                    {
                        SubquestionResult subquestionResult = questionResult.SubquestionResultList.ElementAt(k);
                        string? subject = testResult.TestTemplate.Subject;
                        int subjectId = Array.FindIndex(subjectsArray, x => x.Contains(subject));
                        if (subquestionResult.SubquestionTemplate.SubquestionPoints != null)
                        {
                            //when it comes to subquestion results, we have to consider the difference between subquestion's and student's points
                            subjectPointsShare[subjectId] += (double)subquestionResult.StudentsPoints / (double)subquestionResult.SubquestionTemplate.SubquestionPoints;
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
    }
}