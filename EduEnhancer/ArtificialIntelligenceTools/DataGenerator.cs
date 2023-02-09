using DomainModel;
using Common;
using CsvHelper;
using System.Globalization;
using static Common.EnumTypes;

namespace ArtificialIntelligenceTools
{
    public class DataGenerator
    {
        /// <summary>
        /// Generates .csv file of a number of subquestion templates with parameters that are used by the neural network
        /// </summary>
        /// <param name="dataColleration">Decides whether the generated data will be randomized or if there are going to be collerations between the templates</param>
        /// <param name="testingDataSubjects">Existing testing data subjects (subjects where IsTestingData == true)</param>
        public static void GenerateTemplatesFile(bool dataColleration, List<Subject> testingDataSubjects)
        {
            List<TestTemplate> testTemplates = new List<TestTemplate>();
            if (!dataColleration)
            {
                testTemplates = GenerateRandomTestTemplates(testTemplates, 500, testingDataSubjects);
            }
            else
            {
                testTemplates = GenerateCorrelationalTestTemplates(testTemplates, 500, testingDataSubjects);
            }
            var subquestionTemplateRecords = CreateSubquestionTemplateRecords(testTemplates);

            string filePath = "";
            if (!dataColleration)
            {
                filePath = Directory.GetParent(Environment.CurrentDirectory) + "\\ArtificialIntelligenceTools\\RandomTemplatesFile.csv";
            }
            else if (dataColleration)
            {
                filePath = Directory.GetParent(Environment.CurrentDirectory) + "\\ArtificialIntelligenceTools\\CorrelationalTemplatesFile.csv";
            }

            File.Delete(filePath);
            using (var writer = new StreamWriter(filePath))
            using (var csv = new CsvWriter(writer, CultureInfo.InvariantCulture))
            {
                csv.WriteRecords(subquestionTemplateRecords);
            }

        }

        /// <summary>
        /// Generates .csv file of a number of subquestion results with parameters that are used by the neural network
        /// </summary>
        /// <param name="testTemplates">Existing testing data templates (test templates where IsTestingData == true)</param>
        /// <param name="dataColleration">Decides whether the generated data will be randomized or if there are going to be collerations between the templates</param>
        public static void GenerateResultsFile(List<TestTemplate> testTemplates, bool dataColleration)
        {
            List<TestResult> testResults;
            if (!dataColleration)
            {
                testResults = GenerateRandomTestResults(testTemplates, 0, 500);
            }
            else
            {
                testResults = GenerateCorrelationalTestResults(testTemplates, 0, 500);
            }
            var subquestionResultRecords = CreateSubquestionResultRecords(testResults);
            string filePath = "";
            if (!dataColleration)
            {
                filePath = Directory.GetParent(Environment.CurrentDirectory) + "\\ArtificialIntelligenceTools\\RandomResultsFile.csv";
            }
            else if (dataColleration)
            {
                filePath = Directory.GetParent(Environment.CurrentDirectory) + "\\ArtificialIntelligenceTools\\CorrelationalResultsFile.csv";
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
        public static List<SubquestionTemplateRecord> CreateSubquestionTemplateRecords(List<TestTemplate> testTemplates)
        {
            List<SubquestionTemplateRecord> subquestionTemplateRecords = new List<SubquestionTemplateRecord>();
            double[] subquestionTypeAveragePoints = GetSubquestionTypeAverageTemplatePoints(testTemplates);
            List<(Subject, double)> subjectAveragePointsTuple = GetSubjectAverageTemplatePoints(testTemplates);
            int subquestionTemplateRecordId = 0;
            User owner = testTemplates[0].Owner;

            for (int i = 0; i < testTemplates.Count; i++)
            {
                TestTemplate testTemplate = testTemplates[i];
                double minimumPointsShare = GetMinimumPointsShare(testTemplate);

                for (int j = 0; j < testTemplate.QuestionTemplates.Count; j++)
                {
                    QuestionTemplate questionTemplate = testTemplate.QuestionTemplates.ElementAt(j);

                    for (int k = 0; k < questionTemplate.SubquestionTemplates.Count; k++)
                    {
                        SubquestionTemplate subquestionTemplate = questionTemplate.SubquestionTemplates.ElementAt(k);
                        SubquestionTemplateRecord subquestionTemplateRecord = CreateSubquestionTemplateRecord(subquestionTemplate, owner,
                            subjectAveragePointsTuple, subquestionTypeAveragePoints, minimumPointsShare);
                        subquestionTemplateRecords.Add(subquestionTemplateRecord);
                        subquestionTemplateRecordId++;
                    }
                }
            }

            return subquestionTemplateRecords;
        }

        /// <summary>
        /// Creates a subquestion template with parameters that are used by the neural network
        /// </summary>
        public static SubquestionTemplateRecord CreateSubquestionTemplateRecord(SubquestionTemplate subquestionTemplate, User owner, 
            List<(Subject, double)> subjectAveragePointsTuple, double[] subquestionTypeAveragePoints, double minimumPointsShare)
        {
            SubquestionTemplateRecord subquestionTemplateRecord = new SubquestionTemplateRecord();
            TestTemplate testTemplate = subquestionTemplate.QuestionTemplate.TestTemplate;
            subquestionTemplateRecord.SubquestionTemplate = subquestionTemplate;
            subquestionTemplateRecord.SubquestionTemplateId = subquestionTemplate.SubquestionTemplateId;
            subquestionTemplateRecord.Owner = owner;
            subquestionTemplateRecord.OwnerLogin = owner.Login;
            EnumTypes.SubquestionType subquestionType = subquestionTemplate.SubquestionType;
            subquestionTemplateRecord.SubquestionTypeAveragePoints = Math.Round(subquestionTypeAveragePoints[Convert.ToInt32(subquestionType) - 1], 2);
            int possibleAnswersCount = 0;
            int correctAnswersCount = 0;
            if (subquestionTemplate.PossibleAnswers != null)
            {
                possibleAnswersCount = subquestionTemplate.PossibleAnswers.Length;
            }
            if (subquestionTemplate.CorrectAnswers != null)
            {
                correctAnswersCount = subquestionTemplate.CorrectAnswers.Length;
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

            for(int i = 0; i < subjectAveragePointsTuple.Count; i++)
            {
                if (subjectAveragePointsTuple[i].Item1 == testTemplate.Subject)
                {
                    subquestionTemplateRecord.SubjectAveragePoints = Math.Round(subjectAveragePointsTuple[i].Item2, 2);
                    break;
                }
            }
            subquestionTemplateRecord.WrongChoicePointsShare = Math.Round((double)(subquestionTemplate.WrongChoicePoints / subquestionTemplate.DefaultWrongChoicePoints), 2);
            subquestionTemplateRecord.NegativePoints = Convert.ToInt32(testTemplate.NegativePoints);
            subquestionTemplateRecord.MinimumPointsShare = Math.Round(minimumPointsShare, 2);
            subquestionTemplateRecord.SubquestionPoints = Math.Round(subquestionTemplate.SubquestionPoints, 2);

            return subquestionTemplateRecord;
        }

        /// <summary>
        /// Returns minimum points share of the test template (percentage of minimum points in regards to the sum of test template's points)
        /// </summary>
        public static double GetMinimumPointsShare(TestTemplate testTemplate)
        {
            double totalSubquestionPoints = 0;
            for (int j = 0; j < testTemplate.QuestionTemplates.Count; j++)
            {
                QuestionTemplate questionTemplate = testTemplate.QuestionTemplates.ElementAt(j);

                for (int k = 0; k < questionTemplate.SubquestionTemplates.Count; k++)
                {
                    totalSubquestionPoints += questionTemplate.SubquestionTemplates.ElementAt(k).SubquestionPoints;
                }
            }

            return testTemplate.MinimumPoints / totalSubquestionPoints;
        }

        /// <summary>
        /// Generates random test templates - their subquestion points are randomized, not dependent on any variables
        /// <param name="existingTestTemplates">Already existing test templates owned by the testing user</param>
        /// <param name="amountOfSubquestionTemplatesToBeGenerated">Amount of subquestion templates to be generated</param>
        /// <param name="testingDataSubjects">Existing testing data subjects (subjects where IsTestingData == true)</param>
        /// </summary>
        public static List<TestTemplate> GenerateRandomTestTemplates(List<TestTemplate> existingTestTemplates, int amountOfSubquestionTemplatesToBeGenerated,
            List<Subject> testingDataSubjects)
        {
            List<TestTemplate> testTemplates = existingTestTemplates;
            int existingTestTemplatesCount = existingTestTemplates.Count;
            Random random = new Random();
            User owner = new User() { Login = "login", Email = "adminemail", FirstName = "name", LastName = "surname", Role = (EnumTypes.Role)3, IsTestingData = true };
            int subquestionCount = 0;
            bool stopDataGeneration = false;

            for (int i = existingTestTemplates.Count; ; i++)
            {
                if (stopDataGeneration)
                {
                    break;
                }

                string testId = i.ToString();
                TestTemplate testTemplate = new TestTemplate();
                testTemplate.Title = "Title_" + testId;
                testTemplate.NegativePoints = (EnumTypes.NegativePoints)random.Next(1, 4);
                double totalSubquestionPoints = 0;
                int minimumPointsShare = random.Next(0, 52);
                int subject = random.Next(0, testingDataSubjects.Count);
                testTemplate.Subject = testingDataSubjects[subject];
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
                    questionTemplate.Title = "Title_" + testId + "_" + questionId;
                    questionTemplate.OwnerLogin = owner.Login;
                    questionTemplate.TestTemplate = testTemplate;
                    List<SubquestionTemplate> subquestionTemplates = new List<SubquestionTemplate>();

                    for (int k = 0; k < 5; k++)
                    {
                        if (stopDataGeneration)
                        {
                            break;
                        }

                        SubquestionTemplate subquestionTemplate = new SubquestionTemplate();
                        int subquestionType = random.Next(1, 11);
                        subquestionTemplate.SubquestionType = (EnumTypes.SubquestionType)subquestionType;
                        (string subquestionText, _, _, string[] possibleAnswers, string[] correctAnswers) =
                            CreateSubquestionAnswers(subquestionType, random);
                        subquestionTemplate.SubquestionText = subquestionText;
                        bool containsImage = random.NextDouble() < 0.25;
                        if (containsImage)
                        {
                            subquestionTemplate.ImageSource = "TestingImage.png";
                        }
                        else
                        {
                            subquestionTemplate.ImageSource = null;
                        }

                        subquestionTemplate.PossibleAnswers = possibleAnswers;
                        subquestionTemplate.CorrectAnswers = correctAnswers;

                        int subquestionPoints = random.Next(1, 16);
                        subquestionTemplate.SubquestionPoints = subquestionPoints;
                        totalSubquestionPoints += subquestionTemplate.SubquestionPoints;
                        subquestionTemplate.OwnerLogin = owner.Login;
                        subquestionTemplate.QuestionTemplate = questionTemplate;
                        subquestionTemplate.CorrectChoicePoints = CommonFunctions.CalculateCorrectChoicePoints(
                            Math.Round(Convert.ToDouble(subquestionPoints), 2), subquestionTemplate.CorrectAnswers, subquestionTemplate.SubquestionType);
                        double defaultWrongChoicePoints = subquestionTemplate.CorrectChoicePoints * (-1);
                        subquestionTemplate.DefaultWrongChoicePoints = defaultWrongChoicePoints;
                        subquestionTemplate.WrongChoicePoints = Math.Round(random.NextDouble() * defaultWrongChoicePoints, 2);
                        subquestionTemplates.Add(subquestionTemplate);
                        subquestionCount++;

                        if (subquestionCount >= amountOfSubquestionTemplatesToBeGenerated)//we have reached the desired amount of generated subquestion templates
                        {
                            stopDataGeneration = true;
                        }
                    }
                    questionTemplate.SubquestionTemplates = subquestionTemplates;
                    questionTemplates.Add(questionTemplate);
                }

                double minimumPoints = totalSubquestionPoints * ((double)minimumPointsShare / 100);
                minimumPoints = Math.Round(minimumPoints, 2);
                testTemplate.MinimumPoints = minimumPoints;
                testTemplate.QuestionTemplates = questionTemplates;
                testTemplates.Add(testTemplate);
            }

            testTemplates.RemoveRange(0, existingTestTemplatesCount);
            return testTemplates;
        }

        /// <summary>
        /// Generates correlational test templates - their subquestion points are dependent on variables such as subquestion type, subject etc.
        /// <param name="existingTestTemplates">Already existing test templates owned by the testing user</param>
        /// <param name="amountOfSubquestionTemplatesToBeGenerated">Amount of subquestion templates to be generated</param>
        /// <param name="testingDataSubjects">Existing testing data subjects (subjects where IsTestingData == true)</param>
        /// </summary>
        public static List<TestTemplate> GenerateCorrelationalTestTemplates(List<TestTemplate> existingTestTemplates, int amountOfSubquestionTemplatesToBeGenerated,
            List<Subject> testingDataSubjects)
        {
            //existing test templates have to be added here because otherwise newly created test templates would not be related to the old ones
            List<TestTemplate> testTemplates = existingTestTemplates;
            int existingTestTemplatesCount = existingTestTemplates.Count;
            Random random = new Random();
            User owner = new User() { Login = "login", Email = "adminemail", FirstName = "name", LastName = "surname", Role = (EnumTypes.Role)3, IsTestingData = true };
            int subquestionCount = 0;
            bool stopDataGeneration = false;
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
                testTemplate.Title = "Title_" + testId;
                int negativePoints = random.Next(1, 4);
                testTemplate.NegativePoints = (EnumTypes.NegativePoints)negativePoints;
                double totalSubquestionPoints = 0;
                int minimumPointsShare = random.Next(0, 52);
                int subject = random.Next(0, testingDataSubjects.Count);
                testTemplate.Subject = testingDataSubjects[subject];
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
                    questionTemplate.Title = "Title_" + testId + "_" + questionId;
                    questionTemplate.OwnerLogin = owner.Login;
                    questionTemplate.TestTemplate = testTemplate;
                    List<SubquestionTemplate> subquestionTemplates = new List<SubquestionTemplate>();

                    for (int k = 0; k < 5; k++)
                    {
                        if (stopDataGeneration)
                        {
                            break;
                        }

                        SubquestionTemplate subquestionTemplate = new SubquestionTemplate();
                        int subquestionType = random.Next(1, 11);
                        subquestionTemplate.SubquestionType = (EnumTypes.SubquestionType)subquestionType;
                        bool containsImage = random.Next(0, 2) > 0;
                        if (containsImage)
                        {
                            subquestionTemplate.ImageSource = "TestingImage.png";
                        }
                        else
                        {
                            subquestionTemplate.ImageSource = null;
                        }
                        (string subquestionText, int possibleAnswersCount, int correctAnswersCount, string[] possibleAnswers, string[] correctAnswers) =
                            CreateSubquestionAnswers(subquestionType, random);
                        subquestionTemplate.PossibleAnswers = possibleAnswers;
                        subquestionTemplate.CorrectAnswers = correctAnswers;
                        subquestionTemplate.SubquestionText = subquestionText;

                        double correctAnswersShare = 0;
                        if (possibleAnswersCount > 0 && correctAnswersCount > 0)
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
                        //wrongChoicePoints variable is not correlational
                        subquestionPoints += negativePointsArray[negativePoints - 1];//negativePoints modifier
                        subquestionPoints += ((double)minimumPointsShare / 100) * 4;//minimumPointsShare modifier

                        subquestionTemplate.SubquestionPoints = Math.Round(subquestionPoints, 2);

                        totalSubquestionPoints += subquestionTemplate.SubquestionPoints;
                        subquestionTemplate.QuestionTemplateId = questionTemplate.QuestionTemplateId;
                        subquestionTemplate.OwnerLogin = owner.Login;
                        subquestionTemplate.QuestionTemplate = questionTemplate;
                        subquestionTemplate.CorrectChoicePoints = CommonFunctions.CalculateCorrectChoicePoints(
                            Math.Round(Convert.ToDouble(subquestionPoints), 2), subquestionTemplate.CorrectAnswers, subquestionTemplate.SubquestionType);
                        double defaultWrongChoicePoints = subquestionTemplate.CorrectChoicePoints * (-1);
                        subquestionTemplate.DefaultWrongChoicePoints = defaultWrongChoicePoints;
                        subquestionTemplate.WrongChoicePoints = Math.Round(random.NextDouble() * defaultWrongChoicePoints, 2);
                        subquestionTemplates.Add(subquestionTemplate);
                        subquestionCount++;

                        if (subquestionCount >= amountOfSubquestionTemplatesToBeGenerated)//we have reached the desired amount of generated subquestion templates
                        {
                            stopDataGeneration = true;
                        }
                    }
                    questionTemplate.SubquestionTemplates = subquestionTemplates;
                    questionTemplates.Add(questionTemplate);
                }

                double minimumPoints = totalSubquestionPoints * ((double)minimumPointsShare / 100);
                minimumPoints = Math.Round(minimumPoints, 2);
                testTemplate.MinimumPoints = minimumPoints;
                testTemplate.QuestionTemplates = questionTemplates;
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
            Student student = new Student() { Login = "testingstudent", Email = "studentemail", FirstName = "name", LastName = "surname", IsTestingData = true };
            Random random = new Random();
            int subquestionCount = 0;
            bool stopDataGeneration = false;

            for (int i = 0; i < existingTestTemplates.Count; i++)
            {
                if (stopDataGeneration)
                {
                    break;
                }

                TestTemplate testTemplate = existingTestTemplates[i];
                TestResult testResult = new TestResult();
                testResult.TestTemplate = testTemplate;
                testResult.OwnerLogin = testTemplate.OwnerLogin;
                testResult.TestTemplateId = testTemplate.TestTemplateId;
                testResult.TimeStamp = DateTime.Now;
                testResult.Student = student;
                testResult.StudentLogin = student.Login;
                testResult.IsTurnedIn = true;
                testResult.IsTestingData = true;
                List<QuestionResult> questionResults = new List<QuestionResult>();

                for (int j = 0; j < testTemplate.QuestionTemplates.Count; j++)
                {
                    if (stopDataGeneration)
                    {
                        break;
                    }

                    QuestionTemplate questionTemplate = testTemplate.QuestionTemplates.ElementAt(j);
                    QuestionResult questionResult = new QuestionResult();
                    questionResult.QuestionTemplate = questionTemplate;
                    questionResult.TestResultId = testResult.TestResultId;
                    questionResult.TestResult = testResult;
                    questionResult.OwnerLogin = testResult.OwnerLogin;
                    List<SubquestionResult> subquestionResults = new List<SubquestionResult>();

                    for (int k = 0; k < questionTemplate.SubquestionTemplates.Count; k++)
                    {
                        if (stopDataGeneration)
                        {
                            break;

                        }
                        SubquestionTemplate subquestionTemplate = questionTemplate.SubquestionTemplates.ElementAt(k);
                        SubquestionResult subquestionResult = new SubquestionResult();
                        subquestionResult.SubquestionTemplate = subquestionTemplate;
                        subquestionResult.QuestionResult = questionResult;
                        subquestionResult.TestResultId = questionResult.TestResultId;
                        subquestionResult.SubquestionTemplateId = subquestionTemplate.SubquestionTemplateId;
                        subquestionResult.OwnerLogin = questionResult.OwnerLogin;
                        subquestionResult.StudentsAnswers = CreateStudentsAnswers(subquestionTemplate, random);
                        (double defaultStudentsPoints, double answerCorrectness, EnumTypes.AnswerStatus answerStatus) = CommonFunctions.CalculateStudentsAnswerAttributes(
                            subquestionTemplate.SubquestionType, subquestionTemplate.PossibleAnswers, subquestionTemplate.CorrectAnswers,
                            subquestionTemplate.SubquestionPoints, subquestionTemplate.WrongChoicePoints, subquestionResult.StudentsAnswers);
                        int studentsPoints = random.Next(0, (int)subquestionTemplate.SubquestionPoints);
                        subquestionResult.StudentsPoints = studentsPoints;
                        subquestionResult.AnswerCorrectness = answerCorrectness;
                        subquestionResult.DefaultStudentsPoints = defaultStudentsPoints;
                        subquestionResult.AnswerStatus = answerStatus;
                        subquestionResults.Add(subquestionResult);
                        subquestionCount++;

                        if (subquestionCount >= amountOfSubquestionResultsToBeGenerated)//we have reached the desired amount of generated subquestion results
                        {
                            stopDataGeneration = true;
                        }
                    }

                    questionResult.SubquestionResults = subquestionResults;
                    questionResults.Add(questionResult);
                }

                testResult.QuestionResults = questionResults;
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
            Student student = new Student() { Login = "testingstudent", Email = "studentemail", FirstName = "name", LastName = "surname", IsTestingData = true };
            Random random = new Random();
            List<Subject> subjectList = existingTestTemplates.Select(t => t.Subject).Distinct().ToList();
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
                testResult.OwnerLogin = testTemplate.OwnerLogin;
                testResult.TestTemplateId = testTemplate.TestTemplateId;
                testResult.TimeStamp = DateTime.Now;
                testResult.Student = student;
                testResult.StudentLogin = student.Login;
                testResult.IsTurnedIn = true;
                testResult.IsTestingData = true;
                List<QuestionResult> questionResults = new List<QuestionResult>();

                for (int j = 0; j < testTemplate.QuestionTemplates.Count; j++)
                {
                    if (stopDataGeneration)
                    {
                        break;
                    }

                    QuestionTemplate questionTemplate = testTemplate.QuestionTemplates.ElementAt(j);
                    QuestionResult questionResult = new QuestionResult();
                    questionResult.QuestionTemplate = questionTemplate;
                    questionResult.TestResultId = testResult.TestResultId;
                    questionResult.TestResult = testResult;
                    questionResult.QuestionTemplateId = questionTemplate.QuestionTemplateId;
                    questionResult.OwnerLogin = testResult.OwnerLogin;
                    List<SubquestionResult> subquestionResults = new List<SubquestionResult>();

                    for (int k = 0; k < questionTemplate.SubquestionTemplates.Count; k++)
                    {
                        if (stopDataGeneration)
                        {
                            break;
                        }

                        SubquestionTemplate subquestionTemplate = questionTemplate.SubquestionTemplates.ElementAt(k);
                        SubquestionResult subquestionResult = new SubquestionResult();
                        subquestionResult.SubquestionTemplate = subquestionTemplate;
                        subquestionResult.QuestionResult = questionResult;
                        subquestionResult.TestResultId = questionResult.TestResultId;
                        subquestionResult.OwnerLogin = questionResult.OwnerLogin;
                        subquestionResult.StudentsAnswers = CreateStudentsAnswers(subquestionTemplate, random);
                        (double defaultStudentsPoints, double answerCorrectness, EnumTypes.AnswerStatus answerCorrect) = CommonFunctions.CalculateStudentsAnswerAttributes(
                            subquestionTemplate.SubquestionType, subquestionTemplate.PossibleAnswers, subquestionTemplate.CorrectAnswers,
                            subquestionTemplate.SubquestionPoints, subquestionTemplate.WrongChoicePoints, subquestionResult.StudentsAnswers);

                        subquestionResult.DefaultStudentsPoints = defaultStudentsPoints;
                        double studentsPoints = subquestionTemplate.SubquestionPoints * answerCorrectness;
                        studentsPoints += subquestionPointsByTypeArray[(int)subquestionTemplate.SubquestionType];//subquestionTypeAveragePoints modifier
                        int subjectIndex = subjectList.FindIndex(s => s.SubjectId == testTemplate.Subject.SubjectId);//subjectAveragePoints modifier
                        studentsPoints += subquestionPointsBySubjectArray[subjectIndex];//subjectAveragePoints modifier
                        //wrongChoicePointsShare variable is not correlational
                        studentsPoints += negativePointsArray[((int)testTemplate.NegativePoints) - 1];//negativePoints modifier
                        studentsPoints += ((double)testTemplate.MinimumPoints / 100) * 4;//minimumPointsShare modifier

                        //for certain combinations of variables, student could potentially get awarded more points than the original subquestion points
                        if (studentsPoints > subquestionTemplate.SubquestionPoints)
                        {
                            studentsPoints = subquestionTemplate.SubquestionPoints;
                        }
                        else if (studentsPoints < subquestionTemplate.SubquestionPoints * (-1))
                        {
                            studentsPoints = subquestionTemplate.SubquestionPoints * (-1);
                        }

                        //for certain combinations of variables, students may not get full points despite his answer being entirely correct
                        if (answerCorrectness == 1)
                        {
                            studentsPoints = subquestionTemplate.SubquestionPoints;
                        }
                        else if (answerCorrectness == -1)
                        {
                            studentsPoints = subquestionTemplate.SubquestionPoints * (-1);
                        }
                        else if(answerCorrectness < 0 && studentsPoints > 0)
                        {
                            studentsPoints = 0;
                        }
                        else if(answerCorrectness > 0 && studentsPoints < 0)
                        {
                            studentsPoints = 0;
                        }

                        subquestionResult.StudentsPoints = Math.Round(studentsPoints, 2);
                        subquestionResult.AnswerCorrectness = answerCorrectness;
                        subquestionResult.AnswerStatus = answerCorrect;
                        subquestionResults.Add(subquestionResult);
                        subquestionCount++;

                        if (subquestionCount >= amountOfSubquestionResultsToBeGenerated)//we have reached the desired amount of generated subquestion results
                        {
                            stopDataGeneration = true;
                        }
                    }

                    questionResult.SubquestionResults = subquestionResults;
                    questionResults.Add(questionResult);
                }

                testResult.QuestionResults = questionResults;
                testResults.Add(testResult);
                testingDataTestResultsCount++;
            }

            return testResults;
        }

        /// <summary>
        /// Creates a list of subquestion results with parameters that are used by the neural network
        /// </summary>
        public static List<SubquestionResultRecord> CreateSubquestionResultRecords(List<TestResult> testResults)
        {
            List<SubquestionResultRecord> subquestionResultRecords = new List<SubquestionResultRecord>();
            double[] subquestionTypeAveragePoints = GetSubquestionTypeAverageStudentsPoints(testResults);
            List<(Subject, double)> subjectAveragePointsTuple = GetSubjectAverageStudentsPoints(testResults);
            User owner = testResults[0].TestTemplate.Owner;

            for (int i = 0; i < testResults.Count; i++)
            {
                TestResult testResult = testResults[i];
                TestTemplate testTemplate = testResult.TestTemplate;
                double minimumPointsShare = GetMinimumPointsShare(testTemplate);

                for (int j = 0; j < testResult.QuestionResults.Count; j++)
                {
                    QuestionResult questionResult = testResult.QuestionResults.ElementAt(j);

                    for (int k = 0; k < questionResult.SubquestionResults.Count; k++)
                    {
                        SubquestionResult subquestionResult = questionResult.SubquestionResults.ElementAt(k);
                        SubquestionResultRecord subquestionResultRecord = CreateSubquestionResultRecord(subquestionResult, owner, subjectAveragePointsTuple,
                            subquestionTypeAveragePoints, minimumPointsShare);
                        subquestionResultRecords.Add(subquestionResultRecord);
                    }
                }
            }

            return subquestionResultRecords;
        }

        /// <summary>
        /// Creates a subquestion result with parameters that are used by the neural network
        /// </summary>
        public static SubquestionResultRecord CreateSubquestionResultRecord(SubquestionResult subquestionResult, User owner, 
            List<(Subject, double)> subjectAveragePointsTuple, double[] subquestionTypeAveragePoints, double minimumPointsShare)
        {
            SubquestionTemplate subquestionTemplate = subquestionResult.SubquestionTemplate;
            TestTemplate testTemplate = subquestionTemplate.QuestionTemplate.TestTemplate;

            SubquestionResultRecord subquestionResultRecord = new SubquestionResultRecord();
            subquestionResultRecord.SubquestionResult = subquestionResult;
            subquestionResultRecord.Owner = owner;
            subquestionResultRecord.OwnerLogin = owner.Login;
            EnumTypes.SubquestionType subquestionType = subquestionTemplate.SubquestionType;
            subquestionResultRecord.SubquestionTypeAveragePoints = Math.Round(subquestionTypeAveragePoints[Convert.ToInt32(subquestionType) - 1], 2);
            for (int i = 0; i < subjectAveragePointsTuple.Count; i++)
            {
                if (subjectAveragePointsTuple[i].Item1 == testTemplate.Subject)
                {
                    subquestionResultRecord.SubjectAveragePoints = Math.Round(subjectAveragePointsTuple[i].Item2, 2);
                    break;
                }
            }
            subquestionResultRecord.WrongChoicePointsShare = Math.Round((double)(subquestionTemplate.WrongChoicePoints / subquestionTemplate.DefaultWrongChoicePoints), 2);
            subquestionResultRecord.NegativePoints = Convert.ToInt32(testTemplate.NegativePoints);
            subquestionResultRecord.MinimumPointsShare = Math.Round(minimumPointsShare, 2);
            subquestionResultRecord.AnswerCorrectness = subquestionResult.AnswerCorrectness;
            subquestionResultRecord.StudentsPoints = subquestionResult.StudentsPoints;
            return subquestionResultRecord;
        }

        /// <summary>
        /// Generates subquestion text, possible answers and correct answers
        /// <param name="subquestionType">Subquestion type, based on which the text and possible/correct answers will be generated</param>
        /// <param name="random">Instance of the Random class</param>
        /// </summary>
        public static (string, int, int, string[], string[]) CreateSubquestionAnswers(int subquestionType, Random random)
        {
            string subquestionText = "(TESTOVACÍ OTÁZKA): ";
            int possibleAnswersCount = 0;
            int correctAnswersCount = 0;
            string[] possibleAnswers;
            string[] correctAnswers;
            switch (subquestionType)
            {
                case 1:
                    possibleAnswersCount = random.Next(2, 11);

                    possibleAnswers = new string[possibleAnswersCount];
                    correctAnswers = new string[possibleAnswersCount];

                    for (int l = 0; l < possibleAnswersCount; l++)
                    {
                        possibleAnswers[l] = "možnost " + (l + 1);
                        correctAnswers[l] = "možnost " + (l + 1);
                    }

                    subquestionText += "Seřaďte následující možnosti.";
                    break;
                case 2:
                    possibleAnswersCount = random.Next(2, 11);
                    correctAnswersCount = random.Next(2, possibleAnswersCount);

                    possibleAnswers = new string[possibleAnswersCount];
                    correctAnswers = new string[correctAnswersCount];

                    for (int l = 0; l < possibleAnswersCount; l++)
                    {
                        possibleAnswers[l] = "možnost " + (l + 1);
                    }
                    for (int l = 0; l < correctAnswersCount; l++)
                    {
                        correctAnswers[l] = "možnost " + (l + 1);
                    }

                    subquestionText += "Zvolte z možností správné odpovědi.";
                    break;
                case 3:
                    possibleAnswersCount = random.Next(4, 11);
                    //for this type of subquestion, there has to be an even number of possible answers
                    if (possibleAnswersCount % 2 == 1)
                    {
                        possibleAnswersCount++;
                    }
                    correctAnswersCount = possibleAnswersCount / 2;

                    possibleAnswers = new string[possibleAnswersCount];
                    correctAnswers = new string[correctAnswersCount];

                    for (int l = 0; l < possibleAnswersCount; l++)
                    {
                        possibleAnswers[l] = "možnost " + (l + 1);
                    }
                    int choiceCounter = 0;
                    for (int l = 0; l < correctAnswersCount; l++)
                    {
                        correctAnswers[l] = "možnost " + (choiceCounter + 1) + "|" + "možnost " + (choiceCounter + 2);
                        choiceCounter += 2;
                    }

                    subquestionText += "Přiřaďte k sobě následující možnosti.";
                    break;
                case 4:
                    possibleAnswersCount = random.Next(1, 11);
                    correctAnswersCount = possibleAnswersCount;

                    possibleAnswers = new string[possibleAnswersCount];
                    correctAnswers = new string[correctAnswersCount];

                    string answersToText = string.Empty;
                    for (int l = 0; l < possibleAnswersCount; l++)
                    {
                        answersToText += "možnost " + (l + 1) + ", ";
                        possibleAnswers[l] = "možnost " + (l + 1);
                    }
                    for (int l = 0; l < correctAnswersCount; l++)
                    {
                        correctAnswers[l] = random.Next(0, 2).ToString();
                    }

                    answersToText = answersToText.Substring(0, answersToText.Length - 2);
                    subquestionText += "Je u těchto možností odpověď ano nebo ne? (" + answersToText + ")";
                    break;
                case 5:
                    possibleAnswersCount = 0;
                    correctAnswersCount = 0;

                    possibleAnswers = new string[possibleAnswersCount];
                    correctAnswers = new string[correctAnswersCount];

                    subquestionText += "Volná odpověď - text.";
                    break;
                case 6:
                    possibleAnswersCount = random.Next(2, 11);
                    correctAnswersCount = 1;

                    possibleAnswers = new string[possibleAnswersCount];
                    correctAnswers = new string[correctAnswersCount];

                    for (int l = 0; l < possibleAnswersCount; l++)
                    {
                        possibleAnswers[l] = "možnost " + (l + 1);
                    }
                    correctAnswers[0] = possibleAnswers[random.Next(0, possibleAnswersCount)];
                    subquestionText += "Která z možností je správná?";
                    break;
                case 7:
                    possibleAnswersCount = random.Next(2, 11);
                    correctAnswersCount = 1;

                    possibleAnswers = new string[possibleAnswersCount];
                    correctAnswers = new string[correctAnswersCount];

                    for (int l = 0; l < possibleAnswersCount; l++)
                    {
                        possibleAnswers[l] = "možnost " + (l + 1);
                    }
                    correctAnswers[0] = possibleAnswers[random.Next(0, possibleAnswersCount)];
                    subquestionText += "(Text)|(text)";
                    break;
                case 8:
                    possibleAnswersCount = 0;
                    correctAnswersCount = 1;

                    possibleAnswers = new string[possibleAnswersCount];
                    correctAnswers = new string[correctAnswersCount];

                    correctAnswers[0] = "možnost - správná odpověď";
                    subquestionText += "(Text)|(text)";
                    break;
                case 9:
                    possibleAnswersCount = 0;
                    correctAnswersCount = random.Next(2, 11);

                    possibleAnswers = new string[possibleAnswersCount];
                    correctAnswers = new string[correctAnswersCount];

                    for (int l = 0; l < correctAnswersCount; l++)
                    {
                        correctAnswers[l] = "možnost " + (l + 1);
                    }
                    string gapsToText = string.Empty;
                    for (int l = 0; l < correctAnswersCount; l++)
                    {
                        gapsToText += "|(text)";
                    }
                    subquestionText += "(Text)" + gapsToText;
                    break;
                case 10:
                    int range = random.Next(1, 500);
                    possibleAnswersCount = 2;
                    correctAnswersCount = 1;

                    possibleAnswers = new string[possibleAnswersCount];
                    correctAnswers = new string[correctAnswersCount];

                    possibleAnswers[0] = "0";
                    possibleAnswers[1] = range.ToString();
                    correctAnswers[0] = random.Next(1, range).ToString();

                    subquestionText += "Vyberte hodnotu od 0 do " + range.ToString();
                    break;
                default:
                    possibleAnswers = new string[0];
                    correctAnswers = new string[0];
                    subquestionText += "Chyba: neznámý typ otázky.";
                    break;
            }
            return (subquestionText, possibleAnswersCount, correctAnswersCount, possibleAnswers, correctAnswers);
        }

        /// <summary>
        /// Generates student's answers
        /// <param name="subquestionTemplate">Subquestion template containing subquestion type, based on which the text and possible/correct answers will be generated</param>
        /// <param name="random">Instance of the Random class</param>
        /// </summary>
        public static string[] CreateStudentsAnswers(SubquestionTemplate subquestionTemplate, Random random)
        {
            string[] possibleAnswerArray = subquestionTemplate.PossibleAnswers;
            string[] correctAnswerArray = subquestionTemplate.CorrectAnswers;
            List<string> studentsAnswers = new List<string>();
            switch (subquestionTemplate.SubquestionType)
            {
                case SubquestionType n when (n == SubquestionType.MultiChoiceSingleCorrectAnswer ||
                    n == SubquestionType.MultiChoiceTextFill || n == SubquestionType.FreeAnswerWithDeterminedCorrectAnswer):
                    //student will have answered correctly in at least 50% of cases
                    int correctAnswerProbability = random.Next(0, 101);
                    if (correctAnswerProbability >= 50)
                    {
                        studentsAnswers.Add(correctAnswerArray[0]);
                    }
                    else
                    {
                        if (n == SubquestionType.FreeAnswerWithDeterminedCorrectAnswer)
                        {
                            studentsAnswers.Add("možnost - správná odpověď");
                        }
                        else
                        {
                            studentsAnswers.Add(possibleAnswerArray[random.Next(0, possibleAnswerArray.Length)]);
                        }
                    }
                    break;
                case SubquestionType.MultiChoiceMultipleCorrectAnswers:
                    int studentsAnswersCount = random.Next(1, possibleAnswerArray.Length + 1);
                    IEnumerable<string> studentsAnswersList = possibleAnswerArray.ToList().OrderBy(x => random.Next()).Take(studentsAnswersCount);
                    studentsAnswers = studentsAnswersList.ToList();
                    break;
                case SubquestionType.MultipleQuestions:
                    for (int l = 0; l < correctAnswerArray.Length; l++)
                    {
                        correctAnswerProbability = random.Next(0, 101);
                        if (correctAnswerProbability >= 50)
                        {
                            studentsAnswers.Add(correctAnswerArray[0]);
                        }
                        else
                        {
                            studentsAnswers.Add(random.Next(0, 2).ToString());
                        }
                    }
                    break;
                case SubquestionType.GapMatch:
                    for (int l = 0; l < correctAnswerArray.Length / 2; l++)
                    {
                        studentsAnswers.Add(correctAnswerArray[l]);
                    }
                    break;
                case SubquestionType.MatchingElements:
                    int answerIndex = 0;
                    while (true)
                    {
                        if (answerIndex > possibleAnswerArray.Length / 2)
                        {
                            break;
                        }
                        studentsAnswers.Add(possibleAnswerArray[answerIndex] + "|" + possibleAnswerArray[answerIndex + 1]);
                        answerIndex += 2;
                    }
                    break;
                case SubquestionType.Slider:
                    studentsAnswers.Add(random.Next(int.Parse(possibleAnswerArray[0]), int.Parse(possibleAnswerArray[1]) + 1).ToString());
                    break;
                case SubquestionType.OrderingElements:
                    answerIndex = 0;
                    while (true)
                    {
                        if (answerIndex == possibleAnswerArray.Length || answerIndex + 1 == possibleAnswerArray.Length)
                        {
                            break;
                        }
                        studentsAnswers.Add(possibleAnswerArray[answerIndex + 1]);
                        studentsAnswers.Add(possibleAnswerArray[answerIndex]);
                        answerIndex += 2;
                    }
                    break;
            }
            return studentsAnswers.ToArray();
        }

        /// <summary>
        /// Generates correlational test templates - their subquestion points are dependent on variables such as subquestion type, subject etc.
        /// </summary>
        public static double[] GetSubquestionTypeAverageTemplatePoints(List<TestTemplate> testTemplates)
        {
            int subquestionTypeCount = 10;
            double[] subquestionTypePointsShare = new double[subquestionTypeCount];
            double[] subquestionCountByType = new double[subquestionTypeCount];

            for (int i = 0; i < testTemplates.Count; i++)
            {
                TestTemplate testTemplate = testTemplates[i];

                for (int j = 0; j < testTemplate.QuestionTemplates.Count; j++)
                {
                    QuestionTemplate questionTemplate = testTemplate.QuestionTemplates.ElementAt(j);

                    for (int k = 0; k < questionTemplate.SubquestionTemplates.Count; k++)
                    {
                        SubquestionTemplate subquestionTemplate = questionTemplate.SubquestionTemplates.ElementAt(k);
                        EnumTypes.SubquestionType subquestionType = subquestionTemplate.SubquestionType;
                        subquestionTypePointsShare[Convert.ToInt32(subquestionType) - 1] += (double)subquestionTemplate.SubquestionPoints;
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
        /// Generates a double[10] array of average student's points based on subquestion type
        /// </summary>
        public static double[] GetSubquestionTypeAverageStudentsPoints(List<TestResult> testResults)
        {
            int subquestionTypeCount = 10;
            double[] subquestionTypePointsShare = new double[subquestionTypeCount];
            double[] subquestionCountByType = new double[subquestionTypeCount];

            for (int i = 0; i < testResults.Count; i++)
            {
                TestResult testResult = testResults[i];

                for (int j = 0; j < testResult.QuestionResults.Count; j++)
                {
                    QuestionResult questionResult = testResult.QuestionResults.ElementAt(j);

                    for (int k = 0; k < questionResult.SubquestionResults.Count; k++)
                    {
                        SubquestionResult subquestionResult = questionResult.SubquestionResults.ElementAt(k);
                        EnumTypes.SubquestionType subquestionType = subquestionResult.SubquestionTemplate.SubquestionType;
                        subquestionTypePointsShare[Convert.ToInt32(subquestionType) - 1] += (double)subquestionResult.StudentsPoints / (double)subquestionResult.SubquestionTemplate.SubquestionPoints;
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
        /// Generates a double[10] array of average answer correctness based on subquestion type
        /// </summary>
        public static double[] GetSubquestionTypeAverageAnswerCorrectness(List<TestResult> testResults)
        {
            int subquestionTypeCount = 10;
            double[] subquestionTypeAnswerCorrectness = new double[subquestionTypeCount];
            double[] subquestionCountByType = new double[subquestionTypeCount];

            for (int i = 0; i < testResults.Count; i++)
            {
                TestResult testResult = testResults[i];

                for (int j = 0; j < testResult.QuestionResults.Count; j++)
                {
                    QuestionResult questionResult = testResult.QuestionResults.ElementAt(j);

                    for (int k = 0; k < questionResult.SubquestionResults.Count; k++)
                    {
                        SubquestionResult subquestionResult = questionResult.SubquestionResults.ElementAt(k);
                        EnumTypes.SubquestionType subquestionType = subquestionResult.SubquestionTemplate.SubquestionType;
                        subquestionTypeAnswerCorrectness[Convert.ToInt32(subquestionType) - 1] += (double)subquestionResult.AnswerCorrectness;
                        subquestionCountByType[Convert.ToInt32(subquestionType) - 1] += 1;
                    }
                }
            }

            for (int i = 0; i < subquestionTypeAnswerCorrectness.Length; i++)
            {
                subquestionTypeAnswerCorrectness[i] = subquestionTypeAnswerCorrectness[i] / subquestionCountByType[i];
            }

            return subquestionTypeAnswerCorrectness;
        }

        /// <summary>
        /// Returns a list of tuples containing a subject and an average amount of points that the subquestion template of this subject has
        /// </summary>
        public static List<(Subject, double)> GetSubjectAverageTemplatePoints(List<TestTemplate> testTemplates)
        {
            List<Subject> subjectList = testTemplates.Select(t => t.Subject).Distinct().ToList();
            double[] subjectPointsShare = new double[subjectList.Count];
            double[] subquestionCountBySubject = new double[subjectList.Count];

            for (int i = 0; i < testTemplates.Count; i++)
            {
                TestTemplate testTemplate = testTemplates[i];

                for (int j = 0; j < testTemplate.QuestionTemplates.Count; j++)
                {
                    QuestionTemplate questionTemplate = testTemplate.QuestionTemplates.ElementAt(j);

                    for (int k = 0; k < questionTemplate.SubquestionTemplates.Count; k++)
                    {
                        SubquestionTemplate subquestionTemplate = questionTemplate.SubquestionTemplates.ElementAt(k);
                        int subjectIndex = subjectList.FindIndex(s => s.SubjectId == testTemplate.Subject.SubjectId);
                        subjectPointsShare[subjectIndex] += (double)subquestionTemplate.SubquestionPoints;
                        subquestionCountBySubject[subjectIndex] += 1;
                    }
                }
            }

            for (int i = 0; i < subjectPointsShare.Length; i++)
            {
                subjectPointsShare[i] = subjectPointsShare[i] / subquestionCountBySubject[i];
            }

            List<(Subject, double)> subjectAveragePointsTuple = new List<(Subject, double)>();
            for(int i = 0; i < subjectList.Count; i++)
            {
                subjectAveragePointsTuple.Add((subjectList[i], subjectPointsShare[i]));
            }

            return subjectAveragePointsTuple;
        }

        /// <summary>
        /// Returns a list of tuples containing a subject and an average amount of points that a student has received for his
        /// answer for a subquestion of this subject
        /// </summary>
        public static List<(Subject, double)> GetSubjectAverageStudentsPoints(List<TestResult> testResults)
        {
            List<Subject> subjectList = testResults.Select(t => t.TestTemplate.Subject).Distinct().ToList();
            double[] subjectPointsShare = new double[subjectList.Count];
            double[] subquestionCountBySubject = new double[subjectList.Count];

            for (int i = 0; i < testResults.Count; i++)
            {
                TestResult testResult = testResults[i];

                for (int j = 0; j < testResult.QuestionResults.Count; j++)
                {
                    QuestionResult questionResult = testResult.QuestionResults.ElementAt(j);

                    for (int k = 0; k < questionResult.SubquestionResults.Count; k++)
                    {
                        SubquestionResult subquestionResult = questionResult.SubquestionResults.ElementAt(k);
                        int subjectIndex = subjectList.FindIndex(s => s.SubjectId == testResult.TestTemplate.Subject.SubjectId);
                        //when it comes to subquestion results, we have to consider the difference between subquestion's and student's points
                        subjectPointsShare[subjectIndex] += (double)subquestionResult.StudentsPoints / (double)subquestionResult.SubquestionTemplate.SubquestionPoints;
                        subquestionCountBySubject[subjectIndex] += 1;
                    }
                }
            }

            for (int i = 0; i < subjectPointsShare.Length; i++)
            {
                subjectPointsShare[i] = subjectPointsShare[i] / subquestionCountBySubject[i];
            }

            List<(Subject, double)> subjectAveragePointsTuple = new List<(Subject, double)>();
            for (int i = 0; i < subjectList.Count; i++)
            {
                subjectAveragePointsTuple.Add((subjectList[i], subjectPointsShare[i]));
            }

            return subjectAveragePointsTuple;
        }
    }
}