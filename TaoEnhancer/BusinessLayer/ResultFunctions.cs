using Common;
using DataLayer;
using DomainModel;
using Microsoft.EntityFrameworkCore;
using ArtificialIntelligenceTools;
using System.Diagnostics;
using System.Text.RegularExpressions;
using System.Xml;
using static Common.EnumTypes;

namespace BusinessLayer
{
    /// <summary>
    /// Functions related to test results (entities: TestResult, QuestionResult, SubquestionResult)
    /// </summary>
    public class ResultFunctions
    {
        private DataFunctions dataFunctions;

        public ResultFunctions(CourseContext context)
        {
            dataFunctions = new DataFunctions(context);
        }

        public DbSet<TestResult> GetTestResultDbSet()
        {
            return dataFunctions.GetTestResultDbSet();
        }

        public DbSet<QuestionResult> GetQuestionResultDbSet()
        {
            return dataFunctions.GetQuestionResultDbSet();
        }

        public DbSet<SubquestionResult> GetSubquestionResultDbSet()
        {
            return dataFunctions.GetSubquestionResultDbSet();
        }

        public IQueryable<TestResult> GetTestResultsByOwnerLogin(string login)
        {
            return GetTestResultDbSet().
                Include(s => s.Student).
                Include(t => t.TestTemplate).
                Where(t => t.OwnerLogin == login);
        }

        public List<TestResult> GetTestResultList(string login)
        {
            return GetTestResultDbSet()
                .Include(s => s.Student)
                .Include(t => t.TestTemplate)
                .ThenInclude(q => q.QuestionTemplateList)
                .ThenInclude(q => q.SubquestionTemplateList)
                .Include(t => t.QuestionResultList)
                .ThenInclude(q => q.SubquestionResultList)
                .Where(t => t.OwnerLogin == login).ToList();
        }

        public IQueryable<TestResult> GetTestResultsByStudentLogin(string login)
        {
            return GetTestResultDbSet()
                .Include(t => t.Student)
                .Include(t => t.TestTemplate)
                .Include(t => t.TestTemplate.Owner)
                .Where(t => t.Student.Login == login);
        }

        public async Task<string> AddTestResults(string login)
        {
            List<TestResult> testResults = LoadTestResults(login);
            return await dataFunctions.AddTestResults(testResults);
        }

        public async Task<string> DeleteTestResults(string login)
        {
            return await dataFunctions.DeleteTestResults(login);
        }

        public async Task<string> DeleteTestResult(string login, string testResultIdentifier)
        {
            TestResult testResult = GetTestResultDbSet().First(t => t.OwnerLogin == login && t.TestResultIdentifier == testResultIdentifier);
            return await dataFunctions.DeleteTestResult(testResult);
        }

        public IQueryable<QuestionResult> GetQuestionResultsByOwnerLogin(string login, string testResultIdentifier)
        {
            return GetQuestionResultDbSet()
                .Include(t => t.TestResult)
                .Include(q => q.QuestionTemplate)
                .Include(q => q.QuestionTemplate.TestTemplate)
                .Include(q => q.QuestionTemplate.SubquestionTemplateList)
                .Include(s => s.TestResult.Student)
                .Include(q => q.SubquestionResultList)
                .Where(t => t.TestResultIdentifier == testResultIdentifier && t.OwnerLogin == login);
        }

        public IQueryable<QuestionResult> GetQuestionResultsByStudentLogin(string studentLogin, string ownerLogin, string testResultIdentifier)
        {
            return GetQuestionResultDbSet()
                .Include(q => q.TestResult)
                .Include(q => q.QuestionTemplate)
                .Include(q => q.QuestionTemplate.TestTemplate)
                .Include(q => q.TestResult.TestTemplate.Owner)
                .Include(q => q.QuestionTemplate.SubquestionTemplateList)
                .Include(q => q.TestResult.Student)
                .Include(q => q.SubquestionResultList)
                .Where(q => q.TestResultIdentifier == testResultIdentifier && q.TestResult.Student.Login == studentLogin
                    && q.OwnerLogin == ownerLogin);
        }

        public IQueryable<SubquestionResult> GetSubquestionResultsByOwnerLogin(string login, string testResultIdentifier, string questionNumberIdentifier)
        {
            return GetSubquestionResultDbSet()
                .Include(s => s.SubquestionTemplate)
                .Include(s => s.QuestionResult)
                .Include(s => s.QuestionResult.QuestionTemplate)
                .Include(s => s.QuestionResult.QuestionTemplate.TestTemplate)
                .Where(s => s.TestResultIdentifier == testResultIdentifier
                && s.QuestionNumberIdentifier == questionNumberIdentifier && s.OwnerLogin == login);
        }

        public IQueryable<SubquestionResult> GetSubquestionResultsByStudentLogin(string studentLogin, string ownerLogin, string testResultIdentifier, string questionNumberIdentifier)
        {
            return GetSubquestionResultDbSet()
                .Include(s => s.SubquestionTemplate)
                .Include(s => s.QuestionResult)
                .Include(s => s.QuestionResult.QuestionTemplate)
                .Include(s => s.QuestionResult.QuestionTemplate.TestTemplate)
                .Where(s => s.TestResultIdentifier == testResultIdentifier && s.QuestionNumberIdentifier == questionNumberIdentifier &&
                s.QuestionResult.TestResult.Student.Login == studentLogin && s.OwnerLogin == ownerLogin);
        }

        public SubquestionResult GetSubquestionResult(string login, string testResultIdentifier, string questionNumberIdentifier, string subquestionIdentifier)
        {
            return GetSubquestionResultDbSet()
                .Include(s => s.SubquestionTemplate)
                .Include(s => s.SubquestionTemplate.QuestionTemplate)
                .Include(s => s.SubquestionTemplate.QuestionTemplate.TestTemplate)
                .First(s => s.TestResultIdentifier == testResultIdentifier && s.QuestionNumberIdentifier == questionNumberIdentifier
                && s.SubquestionIdentifier == subquestionIdentifier && s.OwnerLogin == login);
        }

        public IQueryable<SubquestionResult> GetSubquestionResults(string login, string testResultIdentifier, string questionNumberIdentifier)
        {
            return GetSubquestionResultDbSet()
                .Include(s => s.QuestionResult)
                .Include(s => s.QuestionResult.TestResult)
                .Where(s => s.QuestionNumberIdentifier == questionNumberIdentifier
                    && s.OwnerLogin == login
                    && s.TestResultIdentifier == testResultIdentifier).AsQueryable();
        }

        public DbSet<SubquestionResultStatistics> GetSubquestionResultStatisticsDbSet()
        {
            return dataFunctions.GetSubquestionResultStatisticsDbSet();
        }

        public DbSet<TestDifficultyStatistics> GetTestDifficultyStatisticsDbSet()
        {
            return dataFunctions.GetTestDifficultyStatisticsDbSet();
        }

        public SubquestionResultStatistics? GetSubquestionResultStatistics(string login)
        {
            return dataFunctions.GetSubquestionResultStatisticsDbSet().FirstOrDefault(s => s.UserLogin == login);
        }

        public TestDifficultyStatistics? GetTestDifficultyStatistics(string login)
        {
            return dataFunctions.GetTestDifficultyStatisticsDbSet().FirstOrDefault(s => s.UserLogin == login);
        }

        public async Task<string> SetSubquestionResultPoints(string subquestionPoints, string studentsPoints, string negativePoints, SubquestionResult subquestionResult)
        {
            string message;
            if (subquestionPoints == null)
            {
                message = "Chyba: nelze upravit studentův počet bodů. Nejprve je nutné nastavit počet bodů u zadání podotázky.";
            }
            else if (studentsPoints == null)
            {
                message = "Chyba: nebyl zadán žádný počet bodů.";
            }
            else if (!double.TryParse(studentsPoints, out _))
            {
                message = "Chyba: \"" + studentsPoints + "\" není korektní formát počtu bodů. Je nutné zadat číslo.";
            }
            else if (Convert.ToDouble(studentsPoints) > Convert.ToDouble(subquestionPoints))
            {
                message = "Chyba: příliš vysoký počet bodů. Nejvyšší počet bodů, které může za tuto podotázku student obdržet, je " + subquestionPoints + ".";
            }
            else if (Convert.ToDouble(studentsPoints) < Convert.ToDouble(subquestionPoints) * (-1))
            {
                message = "Chyba: příliš nízký počet bodů. Nejnížší počet bodů, které může za tuto podotázku student obdržet, je -" + subquestionPoints + ".";
            }
            else if (negativePoints == "negativePoints_no" && (Convert.ToDouble(studentsPoints) < 0))
            {
                message = "Chyba: v tomto testu nemůže student za podotázku obdržet záporný počet bodů. Změnu je možné provést v nastavení šablony testu.";
            }
            else
            {
                message = "Studentův počet bodů byl úspěšně změněn.";
                subquestionResult.StudentsPoints = Math.Round(Convert.ToDouble(studentsPoints), 2);
                await dataFunctions.SaveChangesAsync();
            }
            return message;
        }

        public async Task UpdateStudentsPoints(string login, string questionNumberIdentifier, string subquestionIdentifier)
        {
            List<SubquestionResult> subquestionResults = dataFunctions.GetSubquestionResults(questionNumberIdentifier, subquestionIdentifier, login);
            for(int i = 0; i < subquestionResults.Count; i++)
            {
                SubquestionResult subquestionResult = subquestionResults[i];
                SubquestionTemplate subquestionTemplate = subquestionResult.SubquestionTemplate;
                (double? defaultStudentsPoints, _, _) = CommonFunctions.CalculateStudentsAnswerAttributes(subquestionTemplate.SubquestionType, subquestionTemplate.PossibleAnswerList,
                    subquestionTemplate.CorrectAnswerList, subquestionTemplate.SubquestionPoints, subquestionTemplate.WrongChoicePoints, subquestionResult.StudentsAnswerList);
                subquestionResult.DefaultStudentsPoints = defaultStudentsPoints;
                subquestionResult.StudentsPoints = defaultStudentsPoints;
            }
            await dataFunctions.SaveChangesAsync();
        }

        public List<TestTemplate> GetTestingDataTestTemplates()
        {
            var testTemplates = dataFunctions.GetTestTemplateDbSet()
                .Include(t => t.Subject)
                .Include(t => t.QuestionTemplateList)
                .ThenInclude(q => q.SubquestionTemplateList)
                .Where(t => t.IsTestingData).ToList();
            return testTemplates;
        }

        public List<TestResult> GetTestingDataTestResults()
        {
            var testResults = GetTestResultDbSet()
                .Include(t => t.QuestionResultList)
                .ThenInclude(q => q.SubquestionResultList)
                .Where(t => t.IsTestingData).ToList();
            return testResults;
        }

        public int GetTestingDataSubquestionResultsCount()
        {
            int testingDataSubquestionResults = 0;
            var testResults = GetTestingDataTestResults();
            for (int i = 0; i < testResults.Count; i++)
            {
                TestResult testResult = testResults[i];
                for (int j = 0; j < testResult.QuestionResultList.Count; j++)
                {
                    QuestionResult questionResult = testResult.QuestionResultList.ElementAt(j);
                    for (int k = 0; k < questionResult.SubquestionResultList.Count; k++)
                    {
                        testingDataSubquestionResults++;
                    }
                }
            }
            return testingDataSubquestionResults;
        }

        public async Task<string> CreateResultTestingData(string action, string amountOfSubquestionResults)
        {
            string message;
            await TestingUsersCheck();
            User? owner = dataFunctions.GetUserByLogin("login");
            var existingTestTemplates = GetTestingDataTestTemplates();
            if(existingTestTemplates.Count == 0)
            {
                return "Chyba: nejprve je nutné vytvořit zadání testů.";
            }
            int testingDataTestResultsCount = GetTestingDataTestResults().Count;

            List<TestResult> testResults = new List<TestResult>();
            if (action == "addSubquestionResultRandomData")
            {
                testResults = DataGenerator.GenerateRandomTestResults(existingTestTemplates, testingDataTestResultsCount, Convert.ToInt32(amountOfSubquestionResults));
            }
            else if (action == "addSubquestionResultCorrelationalData")
            {
                testResults = DataGenerator.GenerateCorrelationalTestResults(existingTestTemplates, testingDataTestResultsCount, Convert.ToInt32(amountOfSubquestionResults));
            }
            message = await dataFunctions.AddTestResults(testResults);
            string login = "login";
            owner = dataFunctions.GetUserByLoginAsNoTracking();

            //delete existing subquestion template records of this user
            dataFunctions.ExecuteSqlRaw("delete from SubquestionResultRecord where 'login' = '" + login + "'");
            await dataFunctions.SaveChangesAsync();

            //create subquestion template records
            var testResultsToRecord = GetTestResultList(login);

            var subquestionResultRecords = DataGenerator.CreateSubquestionResultRecords(testResultsToRecord);
            await dataFunctions.SaveSubquestionResultRecords(subquestionResultRecords, owner);

            dataFunctions.ClearChargeTracker();
            owner = dataFunctions.GetUserByLoginAsNoTracking();
            //managing subquestionResultStatistics
            var subquestionResultStatistics = GetSubquestionResultStatistics(owner.Login);
            if (subquestionResultStatistics == null)
            {
                subquestionResultStatistics = new SubquestionResultStatistics();
                subquestionResultStatistics.User = owner;
                subquestionResultStatistics.UserLogin = owner.Login;
                subquestionResultStatistics.NeuralNetworkAccuracy = PythonFunctions.GetNeuralNetworkAccuracy(true, login, "ResultNeuralNetwork.py");
                subquestionResultStatistics.MachineLearningAccuracy = PythonFunctions.GetNeuralNetworkAccuracy(true, login, "ResultMachineLearning.py");
                if (subquestionResultStatistics.NeuralNetworkAccuracy >= subquestionResultStatistics.MachineLearningAccuracy)
                {
                    subquestionResultStatistics.UsedModel = Model.NeuralNetwork;
                }
                else
                {
                    subquestionResultStatistics.UsedModel = Model.MachineLearning;
                }

                await dataFunctions.AddSubquestionResultStatistics(subquestionResultStatistics);
                dataFunctions.AttachUser(subquestionResultStatistics.User);
                await dataFunctions.SaveChangesAsync();
            }
            else
            {
                subquestionResultStatistics.NeuralNetworkAccuracy = PythonFunctions.GetNeuralNetworkAccuracy(true, login, "ResultNeuralNetwork.py");
                subquestionResultStatistics.MachineLearningAccuracy = PythonFunctions.GetNeuralNetworkAccuracy(true, login, "ResultMachineLearning.py");
                if (subquestionResultStatistics.NeuralNetworkAccuracy >= subquestionResultStatistics.MachineLearningAccuracy)
                {
                    subquestionResultStatistics.UsedModel = Model.NeuralNetwork;
                }
                else
                {
                    subquestionResultStatistics.UsedModel = Model.MachineLearning;
                }

                await dataFunctions.SaveChangesAsync();
            }

            //managing testDifficultyStatistics
            var testDifficultyStatistics = GetTestDifficultyStatistics(owner.Login);
            double[] subquestionTypeAveragePoints = DataGenerator.GetSubquestionTypeAverageStudentsPoints(testResults);
            List<(Subject, double)> subjectAveragePointsTuple = DataGenerator.GetSubjectAverageStudentsPoints(testResults);
            double[] subquestionTypeAverageAnswerCorrectness = DataGenerator.GetSubquestionTypeAverageAnswerCorrectness(testResults);
            double[] subjectAveragePointsToDelete = new double[subjectAveragePointsTuple.Count];
            for(int i = 0; i < subjectAveragePointsTuple.Count; i++)
            {
                subjectAveragePointsToDelete[i] = subjectAveragePointsTuple[i].Item2;
            }

            if (testDifficultyStatistics == null)
            {
                testDifficultyStatistics = new TestDifficultyStatistics();
                testDifficultyStatistics.User = owner;
                testDifficultyStatistics.UserLogin = owner.Login;
                testDifficultyStatistics.InternalSubquestionTypeAveragePoints = subquestionTypeAveragePoints;
                testDifficultyStatistics.InternalSubjectAveragePoints = subjectAveragePointsToDelete;//todo: pouzit tuple
                testDifficultyStatistics.InternalSubquestionTypeAverageAnswerCorrectness = subquestionTypeAverageAnswerCorrectness;
                await dataFunctions.AddTestDifficultyStatistics(testDifficultyStatistics);
                dataFunctions.AttachUser(testDifficultyStatistics.User);
                await dataFunctions.SaveChangesAsync();
            }
            else
            {
                //TODO: Update techto udaju pro skutecneho ucitele
                testDifficultyStatistics.InternalSubquestionTypeAveragePoints = subquestionTypeAveragePoints;
                testDifficultyStatistics.InternalSubjectAveragePoints = subjectAveragePointsToDelete;//todo: pouzit tuple
                testDifficultyStatistics.InternalSubquestionTypeAverageAnswerCorrectness = subquestionTypeAverageAnswerCorrectness;

                dataFunctions.AttachUser(testDifficultyStatistics.User);
                await dataFunctions.SaveChangesAsync();
            }

            return message;
        }

        public async Task TestingUsersCheck()
        {
            Student? student = dataFunctions.GetStudentByLogin("testingstudent");
            if (student == null)
            {
                student = new Student() { Login = "testingstudent", Email = "studentemail", StudentIdentifier = "testingstudent", FirstName = "name", LastName = "surname", IsTestingData = true };
                await dataFunctions.AddStudent(student);
            }
        }

        public async Task DeleteResultTestingData()
        {
            dataFunctions.ExecuteSqlRaw("delete from TestResult where IsTestingData = 1");
            dataFunctions.ExecuteSqlRaw("delete from SubquestionResultRecord where OwnerLogin = 'login'");
            dataFunctions.ExecuteSqlRaw("delete from SubquestionResultStatistics where UserLogin = 'login'");
            await dataFunctions.SaveChangesAsync();
        }

        public async Task<string> GetSubquestionResultPointsSuggestion(string login, string testResultIdentifier, string questionNumberIdentifier, string subquestionIdentifier)
        {
            User owner = dataFunctions.GetUserByLogin(login);

            //check if enough subquestion results have been added to warrant new model training
            bool retrainModel = false;
            int subquestionResultsAdded = GetSubquestionResultStatistics(login).SubquestionResultsAdded;
            if (subquestionResultsAdded >= 100)
            {
                retrainModel = true;
                await RetrainSubquestionResultModel(owner);
            }

            var subquestionResults = GetSubquestionResults(login, testResultIdentifier, questionNumberIdentifier);

            if (subquestionIdentifier == null)
            {
                subquestionIdentifier = subquestionResults.First().SubquestionIdentifier;
            }

            var subquestionResult = GetSubquestionResult(login, testResultIdentifier, questionNumberIdentifier, subquestionIdentifier);
            SubquestionTemplate subquestionTemplate = subquestionResult.SubquestionTemplate;

            var testResults = GetTestResultList(owner.Login);
            string[] subjectsArray = { "Chemie", "Zeměpis", "Matematika", "Dějepis", "Informatika" };
            double[] subquestionTypeAveragePoints = DataGenerator.GetSubquestionTypeAverageStudentsPoints(testResults);
            List<(Subject, double)> subjectAveragePointsTuple = DataGenerator.GetSubjectAverageStudentsPoints(testResults);
            TestTemplate testTemplate = subquestionTemplate.QuestionTemplate.TestTemplate;
            double? minimumPointsShare = DataGenerator.GetMinimumPointsShare(testTemplate);

            SubquestionResultRecord currentSubquestionResultRecord = DataGenerator.CreateSubquestionResultRecord(subquestionResult, owner,
                subjectAveragePointsTuple, subquestionTypeAveragePoints, minimumPointsShare);
            SubquestionResultStatistics? currectSubquestionResultStatistics = GetSubquestionResultStatistics(login);
            Model usedModel = currectSubquestionResultStatistics.UsedModel;
            string suggestedSubquestionPoints = PythonFunctions.GetSubquestionResultSuggestedPoints(login, retrainModel, currentSubquestionResultRecord, usedModel);
            if (subquestionResultsAdded >= 100)
            {
                SubquestionResultStatistics subquestionResultStatistics = GetSubquestionResultStatistics(login);
                subquestionResultStatistics.SubquestionResultsAdded = 0;
                subquestionResultStatistics.NeuralNetworkAccuracy = PythonFunctions.GetNeuralNetworkAccuracy(false, login, "ResultNeuralNetwork.py");
                subquestionResultStatistics.MachineLearningAccuracy = PythonFunctions.GetNeuralNetworkAccuracy(true, login, "ResultMachineLearning.py");
                if (subquestionResultStatistics.NeuralNetworkAccuracy >= subquestionResultStatistics.MachineLearningAccuracy)
                {
                    subquestionResultStatistics.UsedModel = Model.NeuralNetwork;
                }
                else
                {
                    subquestionResultStatistics.UsedModel = Model.MachineLearning;
                }
                await dataFunctions.SaveChangesAsync();
            }

            return suggestedSubquestionPoints;
        }

        public async Task RetrainSubquestionResultModel(User owner)
        {
            string login = owner.Login;
            //delete existing subquestion template records of this user
            dataFunctions.ExecuteSqlRaw("delete from SubquestionResultRecord where 'login' = '" + login + "'");
            await dataFunctions.SaveChangesAsync();

            //create subquestion template records
            var testResults = GetTestResultList(login);

            var subquestionResultRecords = DataGenerator.CreateSubquestionResultRecords(testResults);
            await dataFunctions.SaveSubquestionResultRecords(subquestionResultRecords, owner);
        }

        public async Task<string?> BeginStudentAttempt(TestTemplate testTemplate, Student student)
        {
            TestResult testResult = new TestResult();
            testResult.TestResultIdentifier = GenerateRandomString();
            testResult.TestNameIdentifier = testTemplate.TestNameIdentifier;
            testResult.TestNumberIdentifier = testTemplate.TestNumberIdentifier;
            testResult.TestTemplate = testTemplate;
            testResult.TimeStamp = DateTime.Now;
            testResult.Student = student;
            testResult.StudentLogin = student.Login;
            testResult.OwnerLogin = testTemplate.OwnerLogin;
            testResult.IsTestingData = false;
            testResult.QuestionResultList = new List<QuestionResult>();

            for(int i = 0; i < testTemplate.QuestionTemplateList.Count; i++)
            {
                QuestionTemplate questionTemplate = testTemplate.QuestionTemplateList.ElementAt(i);
                QuestionResult questionResult = new QuestionResult();
                questionResult.TestResultIdentifier = testResult.TestResultIdentifier;
                questionResult.QuestionNumberIdentifier = questionTemplate.QuestionNumberIdentifier;
                questionResult.OwnerLogin = testTemplate.OwnerLogin;
                questionResult.TestResult = testResult;
                questionResult.QuestionTemplate = questionTemplate;
                questionResult.SubquestionResultList = new List<SubquestionResult>();

                for(int j = 0; j < questionTemplate.SubquestionTemplateList.Count; j++)
                {
                    SubquestionTemplate subquestionTemplate = questionTemplate.SubquestionTemplateList.ElementAt(j);
                    SubquestionResult subquestionResult = new SubquestionResult();
                    subquestionResult.TestResultIdentifier = testResult.TestResultIdentifier;
                    subquestionResult.QuestionNumberIdentifier = questionResult.QuestionNumberIdentifier;
                    subquestionResult.SubquestionIdentifier = subquestionTemplate.SubquestionIdentifier;
                    subquestionResult.OwnerLogin = testResult.OwnerLogin;
                    subquestionResult.StudentsAnswerList = new string[0];
                    subquestionResult.StudentsPoints = 0;
                    subquestionResult.DefaultStudentsPoints = 0;
                    subquestionResult.AnswerCorrectness = 0;
                    subquestionResult.AnswerStatus = AnswerStatus.NotAnswered;
                    subquestionResult.SubquestionTemplate = subquestionTemplate;
                    subquestionResult.QuestionResult = questionResult;
                    questionResult.SubquestionResultList.Add(subquestionResult);
                }

                testResult.QuestionResultList.Add(questionResult);
            }

            return await dataFunctions.AddTestResult(testResult);
        }

        public async Task<TestResult> LoadLastStudentAttempt(Student student)
        {
            return await dataFunctions.LoadLastStudentAttempt(student);
        }

        public List<(int, AnswerCompleteness)> GetSubquestionResultsProperties(TestResult testResult)
        {
            List<(int, AnswerCompleteness)> subquestionResultsProperties = new List<(int, AnswerCompleteness)>();
            for(int i = 0; i < testResult.QuestionResultList.Count; i++)
            {
                QuestionResult questionResult = testResult.QuestionResultList.ElementAt(i);

                for(int j = 0; j < questionResult.SubquestionResultList.Count; j++)
                {
                    SubquestionResult subquestionResult = questionResult.SubquestionResultList.ElementAt(j);
                    SubquestionTemplate subquestionTemplate = subquestionResult.SubquestionTemplate;
                    AnswerCompleteness answerCompleteness = new AnswerCompleteness();
                    switch (subquestionTemplate.SubquestionType)
                    {
                        case SubquestionType.OrderingElements:
                            if(subquestionResult.StudentsAnswerList.Length == 0)
                            {
                                answerCompleteness = AnswerCompleteness.Unanswered;
                            }
                            else if(subquestionResult.StudentsAnswerList.Length == subquestionTemplate.PossibleAnswerList.Length)
                            {
                                answerCompleteness = AnswerCompleteness.Answered;
                            }
                            else
                            {
                                answerCompleteness = AnswerCompleteness.PartiallyAnswered;
                            }
                            break;
                        case SubquestionType n when (n == SubquestionType.MultiChoiceMultipleCorrectAnswers || n == SubquestionType.FreeAnswer ||
                                n == SubquestionType.MultiChoiceSingleCorrectAnswer || n == SubquestionType.MultiChoiceTextFill ||
                                n == SubquestionType.FreeAnswerWithDeterminedCorrectAnswer || n == SubquestionType.Slider):
                            if (subquestionResult.StudentsAnswerList.Length == 0)
                            {
                                answerCompleteness = AnswerCompleteness.Unanswered;
                            }
                            else
                            {
                                answerCompleteness = AnswerCompleteness.Answered;
                            }
                            break;
                        case SubquestionType.MatchingElements:
                            if (subquestionResult.StudentsAnswerList.Length == 0)
                            {
                                answerCompleteness = AnswerCompleteness.Unanswered;
                            }
                            else if (subquestionResult.StudentsAnswerList.Length < subquestionTemplate.PossibleAnswerList.Length / 2)
                            {
                                answerCompleteness = AnswerCompleteness.PartiallyAnswered;
                            }
                            else
                            {
                                answerCompleteness = AnswerCompleteness.Answered;
                            }
                            break;
                        case SubquestionType.MultipleQuestions:
                            int unansweredCount = 0;
                            for (int k = 0; k < subquestionResult.StudentsAnswerList.Length; k++)
                            {
                                if (subquestionResult.StudentsAnswerList[k] == "X")
                                {
                                    unansweredCount++;
                                }
                            }
                            if (subquestionResult.StudentsAnswerList.Length - unansweredCount == 0)
                            {
                                answerCompleteness = AnswerCompleteness.Unanswered;
                            }
                            else if(subquestionResult.StudentsAnswerList.Length - unansweredCount < subquestionTemplate.PossibleAnswerList.Length)
                            {
                                answerCompleteness = AnswerCompleteness.PartiallyAnswered;
                            }
                            else
                            {
                                answerCompleteness = AnswerCompleteness.Answered;
                            }
                            break;
                        case SubquestionType.GapMatch:
                            unansweredCount = 0;
                            for (int k = 0; k < subquestionResult.StudentsAnswerList.Length; k++)
                            {
                                if (subquestionResult.StudentsAnswerList[k] == "|")
                                {
                                    unansweredCount++;
                                }
                            }
                            if (subquestionResult.StudentsAnswerList.Length - unansweredCount == 0)
                            {
                                answerCompleteness = AnswerCompleteness.Unanswered;
                            }
                            else if (subquestionResult.StudentsAnswerList.Length - unansweredCount < subquestionTemplate.CorrectAnswerList.Length)
                            {
                                answerCompleteness = AnswerCompleteness.PartiallyAnswered;
                            }
                            else
                            {
                                answerCompleteness = AnswerCompleteness.Answered;
                            }
                            break;
                    }
                    subquestionResultsProperties.Add((subquestionResult.Id, answerCompleteness));
                }
            }
            return subquestionResultsProperties;
        }

        public (SubquestionResult, string?) ValidateSubquestionResult(SubquestionResult subquestionResult, string[] possibleAnswers)
        {
            SubquestionTemplate subquestionTemplate = subquestionResult.SubquestionTemplate;
            string? errorMessage = null;

            if (subquestionResult.StudentsAnswerList == null)
            {
                subquestionResult.StudentsAnswerList = new string[0];
            }

            for (int i = 0; i < subquestionResult.StudentsAnswerList.Length; i++)
            {
                if(subquestionResult.StudentsAnswerList[i] != null)
                {
                    subquestionResult.StudentsAnswerList[i].Replace("|", "");//replace gap separator
                    subquestionResult.StudentsAnswerList[i].Replace(";", "");//replace answer separator
                }
            }

            //placeholder text gets binded to the student's answer array as well - it's necessary to remove these elements
            string placeholderText = "-ZVOLTE MOŽNOST-";
            if(subquestionTemplate.SubquestionType != SubquestionType.GapMatch)
            {
                subquestionResult.StudentsAnswerList = subquestionResult.StudentsAnswerList.Where(s => s != placeholderText).ToArray();
            }
            else
            {
                for (int i = 0; i < subquestionResult.StudentsAnswerList.Length; i++)
                {
                    if (subquestionResult.StudentsAnswerList[i] == placeholderText)
                    {
                        subquestionResult.StudentsAnswerList[i] = "|";
                    }
                }
            }

            switch (subquestionTemplate.SubquestionType)
            {
                case SubquestionType.Error:
                    errorMessage = "Při ukládání řešení otázky došlo k chybě. Řešení otázky nebylo uloženo. Kód chyby 1001";
                    break;
                case SubquestionType.OrderingElements:
                    if(subquestionResult.StudentsAnswerList.Length > subquestionTemplate.PossibleAnswerList.Length)
                    {
                        errorMessage = "Při ukládání řešení otázky došlo k chybě. Řešení otázky nebylo uloženo. Kód chyby 1004";
                    }
                    for(int i = 0; i < subquestionResult.StudentsAnswerList.Length; i++)
                    {
                        if (!subquestionTemplate.PossibleAnswerList.Contains(subquestionResult.StudentsAnswerList[i]))
                        {
                            errorMessage = "Při ukládání řešení otázky došlo k chybě. Řešení otázky nebylo uloženo. Kód chyby 1003";
                        }
                    }
                    break;
                case SubquestionType.MultiChoiceMultipleCorrectAnswers:
                    if(subquestionResult.StudentsAnswerList.Length > subquestionTemplate.PossibleAnswerList.Length)
                    {
                        errorMessage = "Při ukládání řešení otázky došlo k chybě. Řešení otázky nebylo uloženo. Kód chyby 1004";
                    }
                    for (int i = 0; i < subquestionResult.StudentsAnswerList.Length; i++)
                    {
                        if (!subquestionTemplate.PossibleAnswerList.Contains(subquestionResult.StudentsAnswerList[i]))
                        {
                            errorMessage = "Při ukládání řešení otázky došlo k chybě. Řešení otázky nebylo uloženo. Kód chyby 1003";
                        }
                    }
                    break;
                case SubquestionType.MatchingElements:
                    if (subquestionResult.StudentsAnswerList.Length > subquestionTemplate.PossibleAnswerList.Length)
                    {
                        errorMessage = "Při ukládání řešení otázky došlo k chybě. Řešení otázky nebylo uloženo. Kód chyby 1004";
                    }
                    for (int i = 0; i < subquestionResult.StudentsAnswerList.Length; i++)
                    {
                        if (!subquestionTemplate.PossibleAnswerList.Contains(subquestionResult.StudentsAnswerList[i]))
                        {
                            errorMessage = "Při ukládání řešení otázky došlo k chybě. Řešení otázky nebylo uloženo. Kód chyby 1003";
                        }
                    }
                    if (subquestionResult.StudentsAnswerList.Length % 2 == 1)
                    {
                        errorMessage = "Při ukládání řešení otázky došlo k chybě. Řešení otázky nebylo uloženo. Kód chyby 1005";
                    }
                    else
                    {
                        string[] newstudentsAnswerList = new string[subquestionResult.StudentsAnswerList.Length / 2];
                        for (int i = 0; i < subquestionResult.StudentsAnswerList.Length; i++)
                        {
                            if (i % 2 == 1)
                            {
                                continue;
                            }
                            else
                            {
                                int index = i / 2;
                                newstudentsAnswerList[index] = subquestionResult.StudentsAnswerList[i] + "|" + subquestionResult.StudentsAnswerList[i + 1];
                            }
                        }
                        subquestionResult.StudentsAnswerList = newstudentsAnswerList;
                    }
                    break;
                case SubquestionType.MultipleQuestions:
                    //the answers are in wrong order because of shuffle - we have to rearrange them
                    string[] studentsAnswers = subquestionResult.StudentsAnswerList;
                    subquestionResult.StudentsAnswerList = new string[possibleAnswers.Length];
                    for (int i = 0; i < subquestionResult.SubquestionTemplate.PossibleAnswerList.Length; i++)   
                    {
                        for (int j = 0; j < possibleAnswers.Length; j++)
                        {
                            if (possibleAnswers[j] == subquestionResult.SubquestionTemplate.PossibleAnswerList[i])
                            {
                                subquestionResult.StudentsAnswerList[i] = studentsAnswers[j];
                                break;
                            }
                        }
                    }

                    for (int i = 0; i < subquestionResult.StudentsAnswerList.Length; i++)
                    {
                        if (subquestionResult.StudentsAnswerList[i] == null)
                        {
                            subquestionResult.StudentsAnswerList[i] = "X";//unanswered
                        }
                        if(subquestionResult.StudentsAnswerList[i] != "0" && subquestionResult.StudentsAnswerList[i] != "1" && subquestionResult.StudentsAnswerList[i] != "X")
                        {
                            errorMessage = "Při ukládání řešení otázky došlo k chybě. Řešení otázky nebylo uloženo. Kód chyby 1003";
                        }
                    }
                    if (subquestionResult.StudentsAnswerList.Length != subquestionTemplate.PossibleAnswerList.Length)
                    {
                        errorMessage = "Při ukládání řešení otázky došlo k chybě. Řešení otázky nebylo uloženo. Kód chyby 1004";
                    }
                    break;
                case SubquestionType.FreeAnswer:
                    break;
                case SubquestionType n when (n == SubquestionType.MultiChoiceSingleCorrectAnswer || n == SubquestionType.MultiChoiceTextFill):
                    if (subquestionResult.StudentsAnswerList.Length > 1)
                    {
                        errorMessage = "Při ukládání řešení otázky došlo k chybě. Řešení otázky nebylo uloženo. Kód chyby 1006";
                    }
                    else if (subquestionResult.StudentsAnswerList.Length == 1)
                    {
                        if (!subquestionTemplate.PossibleAnswerList.Contains(subquestionResult.StudentsAnswerList[0]))
                        {
                            errorMessage = "Při ukládání řešení otázky došlo k chybě. Řešení otázky nebylo uloženo. Kód chyby 1003";
                        }
                    }
                    break;
                case SubquestionType.FreeAnswerWithDeterminedCorrectAnswer:
                    if (subquestionResult.StudentsAnswerList.Length > 1)
                    {
                        errorMessage = "Při ukládání řešení otázky došlo k chybě. Řešení otázky nebylo uloženo. Kód chyby 1006";
                    }
                    break;
                case SubquestionType.GapMatch:
                    if (subquestionResult.StudentsAnswerList.Length > subquestionTemplate.CorrectAnswerList.Length)
                    {
                        errorMessage = "Při ukládání řešení otázky došlo k chybě. Řešení otázky nebylo uloženo. Kód chyby 1007";
                    }
                    for (int i = 0; i < subquestionResult.StudentsAnswerList.Length; i++)
                    {
                        if (!subquestionTemplate.CorrectAnswerList.Contains(subquestionResult.StudentsAnswerList[i]) && subquestionResult.StudentsAnswerList[i] != "|")
                        {
                            errorMessage = "Při ukládání řešení otázky došlo k chybě. Řešení otázky nebylo uloženo. Kód chyby 1003";
                        }
                    }
                    break;
                case SubquestionType.Slider:
                    if (subquestionResult.StudentsAnswerList.Length > 1)
                    {
                        errorMessage = "Při ukládání řešení otázky došlo k chybě. Řešení otázky nebylo uloženo. Kód chyby 1008";
                    }
                    if (subquestionResult.StudentsAnswerList[0] == "Nezodpovězeno")
                    {
                        subquestionResult.StudentsAnswerList = new string[0];
                    }
                    if (subquestionResult.StudentsAnswerList.Length == 1)
                    {
                        bool isNumber = int.TryParse(subquestionResult.StudentsAnswerList[0], out _);
                        if (!isNumber)
                        {
                            errorMessage = "Při ukládání řešení otázky došlo k chybě. Řešení otázky nebylo uloženo. Kód chyby 1009";
                        }
                    }
                    break;
            }
            return (subquestionResult, errorMessage);
        }

        public async Task UpdateSubquestionResultStudentsAnswers(SubquestionResult subquestionResult, int subquestionResultIndex, Student student)
        {
            TestResult testResult = await LoadLastStudentAttempt(student);
            for(int i = 0; i < testResult.QuestionResultList.Count; i++)
            {
                QuestionResult questionResult = testResult.QuestionResultList.ElementAt(i);

                for(int j = 0; j < questionResult.SubquestionResultList.Count; j++)
                {
                    if(j == subquestionResultIndex)
                    {
                        questionResult.SubquestionResultList.ElementAt(j).StudentsAnswerList = subquestionResult.StudentsAnswerList;
                        //todo: status, correctness
                        break;
                    }
                }
            }
            await dataFunctions.SaveChangesAsync();
        }

        public async Task<SubquestionTemplate> GetSubquestionTemplateBySubquestionResultIndex(int subquestionResultIndex, Student student)
        {
            TestResult testResult = await LoadLastStudentAttempt(student);
            for (int i = 0; i < testResult.QuestionResultList.Count; i++)
            {
                QuestionResult questionResult = testResult.QuestionResultList.ElementAt(i);

                for (int j = 0; j < questionResult.SubquestionResultList.Count; j++)
                {
                    if (j == subquestionResultIndex)
                    {
                        return questionResult.SubquestionResultList.ElementAt(j).SubquestionTemplate;
                    }
                }
            }
            throw Exceptions.SubquestionTemplateNotFoundException;
        }

        public async Task FinishStudentAttempt(Student student)
        {
            TestResult testResult = await dataFunctions.LoadLastStudentAttempt(student);
            testResult.TimeStamp = DateTime.Now;
            for(int i = 0; i < testResult.QuestionResultList.Count; i++)
            {
                QuestionResult questionResult = testResult.QuestionResultList.ElementAt(i);

                for(int j = 0; j < questionResult.SubquestionResultList.Count; j++)
                {
                    SubquestionResult subquestionResult = questionResult.SubquestionResultList.ElementAt(j);
                    SubquestionTemplate subquestionTemplate = subquestionResult.SubquestionTemplate;
                    (double? defaultStudentsPoints, double answerCorrectness, AnswerStatus answerStatus) = CommonFunctions.CalculateStudentsAnswerAttributes(
                        subquestionTemplate.SubquestionType, subquestionTemplate.PossibleAnswerList, subquestionTemplate.CorrectAnswerList,
                        subquestionTemplate.SubquestionPoints, subquestionTemplate.WrongChoicePoints, subquestionResult.StudentsAnswerList);
                    subquestionResult.DefaultStudentsPoints = defaultStudentsPoints;
                    subquestionResult.StudentsPoints = defaultStudentsPoints;
                    subquestionResult.AnswerCorrectness = answerCorrectness;
                    subquestionResult.AnswerStatus = answerStatus;
                }
            }

            await dataFunctions.SaveChangesAsync();
        }

        public SubquestionResult ProcessSubquestionResultForView(SubquestionResult subquestionResult)
        {
            switch (subquestionResult.SubquestionTemplate.SubquestionType)
            {
                case SubquestionType.MatchingElements:
                    for(int i = 0; i < subquestionResult.StudentsAnswerList.Length; i++)
                    {
                        subquestionResult.StudentsAnswerList[i] = subquestionResult.StudentsAnswerList[i].Replace("|", " -> ");
                    }
                    break;
                case SubquestionType.MultipleQuestions:
                    for (int i = 0; i < subquestionResult.StudentsAnswerList.Length; i++)
                    {
                        string studentsAnswer = "";
                        if (subquestionResult.StudentsAnswerList[i] == "1")
                        {
                            studentsAnswer = "Ano";
                        }
                        else if (subquestionResult.StudentsAnswerList[i] == "0")
                        {
                            studentsAnswer = "Ne";
                        }
                        else if (subquestionResult.StudentsAnswerList[i] == "X")
                        {
                            studentsAnswer = "Nezodpovězeno";
                        }
                        subquestionResult.StudentsAnswerList[i] = subquestionResult.SubquestionTemplate.PossibleAnswerList[i] + " -> " + studentsAnswer;
                    }
                    break;
                case SubquestionType.GapMatch:
                    for (int i = 0; i < subquestionResult.StudentsAnswerList.Length; i++)
                    {
                        subquestionResult.StudentsAnswerList[i] = "[" + (i + 1) + "] - " + subquestionResult.StudentsAnswerList[i];
                    }
                    break;
            }

            return subquestionResult;
        }

        public string GenerateRandomString()//random string - will be deleted after domain model change
        {
            // Creating object of random class
            Random rand = new Random();

            // Choosing the size of string
            // Using Next() string
            int stringlen = rand.Next(4, 10);
            int randValue;
            string str = "";
            char letter;
            for (int i = 0; i < stringlen; i++)
            {

                // Generating a random number.
                randValue = rand.Next(0, 26);

                // Generating random character by converting
                // the random number into character.
                letter = Convert.ToChar(randValue + 65);

                // Appending the letter to string.
                str = str + letter;
            }
            return str;
        }

        /// <summary>
        /// Returns the list of test results
        /// </summary>
        public List<TestResult> LoadTestResults(string login)
        {
            List<TestResult> testResults = new List<TestResult>();

            if (Directory.Exists(Config.GetResultsPath()))
            {
                foreach (var directory in Directory.GetDirectories(Config.GetResultsPath()))
                {
                    foreach (var file in Directory.GetFiles(directory))
                    {
                        if (Path.GetExtension(file) == ".xml")
                        {
                            string timeStampString = "";
                            string testStudentIdentifier = "";

                            XmlReader xmlReader = XmlReader.Create(file);
                            while (xmlReader.Read())
                            {
                                if (xmlReader.Name == "context")
                                {
                                    if (xmlReader.GetAttribute("sourcedId") != null)
                                    {
                                        testStudentIdentifier = xmlReader.GetAttribute("sourcedId")!;
                                    }
                                }

                                if (xmlReader.Name == "testResult")
                                {
                                    if (xmlReader.GetAttribute("datestamp") != null)
                                    {
                                        timeStampString = xmlReader.GetAttribute("datestamp")!;
                                    }
                                }
                            }
                            string[] attemptIdentifierSplitByUnderscore = Path.GetFileNameWithoutExtension(file).Split("_");
                            TestResult testResult = new TestResult();
                            testResult.TestResultIdentifier = attemptIdentifierSplitByUnderscore[2];

                            if (Path.GetFileName(Path.GetDirectoryName(file)) != null)
                            {
                                testResult.TestNameIdentifier = Path.GetFileName(Path.GetDirectoryName(file))!;
                            }
                            else
                            {
                                throw Exceptions.TestTemplateNotFoundException(testResult.TestResultIdentifier);
                            }

                            if (dataFunctions.GetTestTemplateDbSet().Count() == 0)
                            {
                                throw Exceptions.TestTemplatesNotImportedException;
                            }
                            if (dataFunctions.GetTestTemplate(login, testResult.TestNameIdentifier) != null)
                            {
                                testResult.TestTemplate = dataFunctions.GetTestTemplate(login, testResult.TestNameIdentifier);
                            }
                            else
                            {
                                throw Exceptions.TestTemplateNotFoundException(testResult.TestResultIdentifier);
                            }

                            testResult.TestNumberIdentifier = testResult.TestTemplate.TestNumberIdentifier;

                            if (dataFunctions.GetStudentDbSet().Count() == 0)
                            {
                                throw Exceptions.StudentsNotImportedException;
                            }
                            if (dataFunctions.GetStudentByIdentifier(testStudentIdentifier) != null)
                            {
                                testResult.Student = dataFunctions.GetStudentByIdentifier(testStudentIdentifier);
                                testResult.StudentLogin = testResult.Student.Login;
                            }
                            else
                            {
                                throw Exceptions.StudentNotFoundException(testStudentIdentifier);
                            }

                            testResult.OwnerLogin = login;

                            DateTime timeStamp = DateTime.ParseExact(timeStampString, "yyyy-MM-ddTHH:mm:ss.fff",
                                    System.Globalization.CultureInfo.InvariantCulture);
                            testResult.TimeStamp = timeStamp;

                            testResult.QuestionResultList = LoadQuestionResults(testResult, login);

                            testResults.Add(testResult);
                        }
                    }
                }
                return testResults;
            }
            else
            {
                throw Exceptions.TestResultsPathNotFoundException;
            }

        }

        public List<QuestionResult> LoadQuestionResults(TestResult testResult, string login)
        {
            List<QuestionResult> questionResults = new List<QuestionResult>();
            TestTemplate testTemplate = testResult.TestTemplate;
            ICollection<QuestionTemplate> questionTemplates = testTemplate.QuestionTemplateList;
            foreach (QuestionTemplate questionTemplate in questionTemplates)
            {
                QuestionResult questionResult = new QuestionResult();
                questionResult.TestResult = testResult;
                if (dataFunctions.GetQuestionTemplate(login, questionTemplate.QuestionNumberIdentifier) != null)
                {
                    questionResult.QuestionTemplate = dataFunctions.GetQuestionTemplate(login, questionTemplate.QuestionNumberIdentifier);
                }
                else
                {
                    throw Exceptions.QuestionTemplateNotFoundException(testResult.TestResultIdentifier, questionTemplate.QuestionNumberIdentifier);
                }
                questionResult.TestResultIdentifier = testResult.TestResultIdentifier;
                questionResult.QuestionNumberIdentifier = questionTemplate.QuestionNumberIdentifier;
                questionResult.SubquestionResultList = LoadSubquestionResults(questionResult, login);
                questionResult.OwnerLogin = login;
                questionResults.Add(questionResult);
            }
            return questionResults;
        }

        public List<SubquestionResult> LoadSubquestionResults(QuestionResult questionResult, string login)
        {
            List<SubquestionResult> subquestionResults = new List<SubquestionResult>();
            QuestionTemplate questionTemplate = questionResult.QuestionTemplate;
            TestTemplate testTemplate = questionTemplate.TestTemplate;
            XmlReader xmlReader;
            int gapCount = 0;

            ICollection<SubquestionTemplate> subquestionTemplateList = questionResult.QuestionTemplate.SubquestionTemplateList;
            foreach (SubquestionTemplate subquestionTemplate in subquestionTemplateList)
            {
                List<string> studentsAnswers = new List<string>();
                xmlReader = XmlReader.Create(Config.GetResultPath(testTemplate.TestNameIdentifier, questionResult.TestResultIdentifier));
                while (xmlReader.Read())
                {
                    //skip other question results
                    if (xmlReader.Name == "itemResult")
                    {
                        if (xmlReader.GetAttribute("identifier") != questionTemplate.QuestionNameIdentifier && xmlReader.GetAttribute("identifier") != null)
                        {
                            xmlReader.Skip();
                        }
                    }

                    if (xmlReader.Name == "responseVariable")
                    {
                        //skip these two tags, because they include a <value> child-tag that we don't want to read
                        if (xmlReader.GetAttribute("identifier") != subquestionTemplate.SubquestionIdentifier && xmlReader.GetAttribute("identifier") != null)
                        {
                            xmlReader.Skip();
                        }

                        if (xmlReader.NodeType == XmlNodeType.EndElement)
                        {
                            SubquestionResult subquestionResult = new SubquestionResult();
                            subquestionResult.TestResultIdentifier = questionResult.TestResultIdentifier;
                            subquestionResult.QuestionNumberIdentifier = questionResult.QuestionNumberIdentifier;
                            subquestionResult.QuestionResult = questionResult;
                            subquestionResult.SubquestionIdentifier = subquestionTemplate.SubquestionIdentifier;
                            subquestionResult.SubquestionTemplate = subquestionTemplate;
                            subquestionResult.StudentsAnswerList = studentsAnswers.ToArray();
                            EnumTypes.AnswerStatus answerStatus = AnswerStatus.NotDetermined;
                            (double? defaultStudentsPoints, double answerCorrectness, answerStatus) = CommonFunctions.CalculateStudentsAnswerAttributes(
                                subquestionTemplate.SubquestionType, subquestionTemplate.PossibleAnswerList, subquestionTemplate.CorrectAnswerList,
                                subquestionTemplate.SubquestionPoints, subquestionTemplate.WrongChoicePoints, subquestionResult.StudentsAnswerList);
                            if (subquestionTemplate.SubquestionPoints != null)
                            {
                                subquestionResult.DefaultStudentsPoints = defaultStudentsPoints;
                                subquestionResult.StudentsPoints = defaultStudentsPoints;
                            }
                            subquestionResult.AnswerCorrectness = answerCorrectness;
                            subquestionResult.AnswerStatus = answerStatus;
                            subquestionResult.OwnerLogin = login;
                            subquestionResults.Add(subquestionResult);
                        }
                    }

                    if (xmlReader.Name == "outcomeVariable")
                    {
                        xmlReader.Skip();
                    }

                    if (xmlReader.Name == "value")
                    {
                        string studentsAnswer = xmlReader.ReadString();//this may read only the answer's identifier instead of the answer itself

                        if (studentsAnswer.Length == 0)
                        {
                            studentsAnswer = "Nevyplněno";
                        }
                        else
                        {
                            //some of the strings may include invalid characters that must be removed
                            Regex regEx = new Regex("['<>]");
                            studentsAnswer = regEx.Replace(studentsAnswer, "");
                            if (studentsAnswer.Length > 0)
                            {
                                if (studentsAnswer[0] == ' ')
                                {
                                    studentsAnswer = studentsAnswer.Remove(0, 1);
                                }
                            }

                            if (subquestionTemplate.SubquestionType == EnumTypes.SubquestionType.OrderingElements || subquestionTemplate.SubquestionType == EnumTypes.SubquestionType.MultiChoiceMultipleCorrectAnswers ||
                                subquestionTemplate.SubquestionType == EnumTypes.SubquestionType.MultiChoiceSingleCorrectAnswer || subquestionTemplate.SubquestionType == EnumTypes.SubquestionType.MultiChoiceTextFill)
                            {
                                studentsAnswer = GetStudentsAnswerText(testTemplate.TestNameIdentifier, questionTemplate.QuestionNumberIdentifier, subquestionTemplate.SubquestionIdentifier, studentsAnswer);
                            }
                            else if (subquestionTemplate.SubquestionType == EnumTypes.SubquestionType.MatchingElements)
                            {
                                string[] studentsAnswerSplitBySpace = studentsAnswer.Split(" ");
                                studentsAnswer = GetStudentsAnswerText(testTemplate.TestNameIdentifier, questionTemplate.QuestionNumberIdentifier, subquestionTemplate.SubquestionIdentifier, studentsAnswerSplitBySpace[0])
                                    + " -> " + GetStudentsAnswerText(testTemplate.TestNameIdentifier, questionTemplate.QuestionNumberIdentifier, subquestionTemplate.SubquestionIdentifier, studentsAnswerSplitBySpace[1]);
                            }
                            else if (subquestionTemplate.SubquestionType == EnumTypes.SubquestionType.MultipleQuestions)
                            {
                                string[] studentsAnswerSplitBySpace = studentsAnswer.Split(" ");
                                studentsAnswer = GetStudentsAnswerText(testTemplate.TestNameIdentifier, questionTemplate.QuestionNumberIdentifier, subquestionTemplate.SubquestionIdentifier, studentsAnswerSplitBySpace[1])
                                    + " -> " + GetStudentsAnswerText(testTemplate.TestNameIdentifier, questionTemplate.QuestionNumberIdentifier, subquestionTemplate.SubquestionIdentifier, studentsAnswerSplitBySpace[0]);
                            }
                            else if (subquestionTemplate.SubquestionType == EnumTypes.SubquestionType.GapMatch)
                            {
                                gapCount++;
                                string[] studentsAnswerSplitBySpace = studentsAnswer.Split(" ");
                                studentsAnswer = "[" + gapCount + "] - " + GetStudentsAnswerText(testTemplate.TestNameIdentifier, questionTemplate.QuestionNumberIdentifier, subquestionTemplate.SubquestionIdentifier, studentsAnswerSplitBySpace[0]);
                            }
                        }

                        studentsAnswers.Add(studentsAnswer);
                    }
                }
            }

            return subquestionResults;
        }



        /// <summary>
        /// Returns the selected question result
        /// </summary>
        /// <returns>the selected question result</returns>
        /// <param name="testNameIdentifier">Name identifier of the test that the selected question belongs to</param>
        /// <param name="questionNumberIdentifier">Number identifier of the selected question</param>
        /// <param name="subquestionIdentifier">Subquestion identifier of the selected subquestion</param>
        /// <param name="studentsAnswer">String containing identifier of student's answer</param>
        string GetStudentsAnswerText(string testNameIdentifier, string questionNumberIdentifier, string subquestionIdentifier, string studentsAnswer)
        {
            XmlReader xmlReader = XmlReader.Create(Config.GetQuestionTemplateFilePath(testNameIdentifier, questionNumberIdentifier));
            while (xmlReader.Read())
            {
                var name = xmlReader.Name;
                if (name == "choiceInteraction" || name == "sliderInteraction" || name == "gapMatchInteraction" || name == "matchInteraction" ||
                    name == "extendedTextInteraction" || name == "orderInteraction" || name == "associateInteraction" || name == "inlineChoiceInteraction")
                {
                    //skip other subquestions
                    if (xmlReader.GetAttribute("responseIdentifier") != subquestionIdentifier)
                    {
                        xmlReader.Skip();
                    }
                }

                if ((name == "simpleChoice" || name == "simpleAssociableChoice" || name == "inlineChoice" || name == "gapText") && xmlReader.GetAttribute("identifier") == studentsAnswer)
                {
                    return xmlReader.ReadString();
                }
            }
            throw Exceptions.StudentsAnswerNotFoundException(testNameIdentifier, questionNumberIdentifier, subquestionIdentifier);
        }
    }
}