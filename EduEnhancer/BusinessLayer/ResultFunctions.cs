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

        public async Task<string> DeleteTestResults(string login)
        {
            return await dataFunctions.DeleteTestResults(login);
        }

        public async Task<string> DeleteTestResult(string login, int testResultId)
        {
            TestResult testResult = GetTestResultDbSet().First(t => t.OwnerLogin == login && t.TestResultId == testResultId);
            return await dataFunctions.DeleteTestResult(testResult);
        }

        public IQueryable<QuestionResult> GetQuestionResultsByOwnerLogin(string login, int testResultId)
        {
            return GetQuestionResultDbSet()
                .Include(t => t.TestResult)
                .Include(q => q.QuestionTemplate)
                .Include(q => q.QuestionTemplate.TestTemplate)
                .Include(q => q.QuestionTemplate.SubquestionTemplateList)
                .Include(s => s.TestResult.Student)
                .Include(q => q.SubquestionResultList)
                .Where(t => t.TestResultId == testResultId && t.OwnerLogin == login);
        }

        public IQueryable<QuestionResult> GetQuestionResultsByStudentLogin(string studentLogin, string ownerLogin, int testResultId)
        {
            return GetQuestionResultDbSet()
                .Include(q => q.TestResult)
                .Include(q => q.QuestionTemplate)
                .Include(q => q.QuestionTemplate.TestTemplate)
                .Include(q => q.TestResult.TestTemplate.Owner)
                .Include(q => q.QuestionTemplate.SubquestionTemplateList)
                .Include(q => q.TestResult.Student)
                .Include(q => q.SubquestionResultList)
                .Where(q => q.TestResultId == testResultId && q.TestResult.Student.Login == studentLogin
                    && q.OwnerLogin == ownerLogin);
        }

        public IQueryable<SubquestionResult> GetSubquestionResultsByOwnerLogin(string login, int questionResultId)
        {
            return GetSubquestionResultDbSet()
                .Include(s => s.SubquestionTemplate)
                .Include(s => s.QuestionResult)
                .Include(s => s.QuestionResult.QuestionTemplate)
                .Include(s => s.QuestionResult.QuestionTemplate.TestTemplate)
                .Where(s => s.QuestionResultId == questionResultId && s.OwnerLogin == login);
        }

        public IQueryable<SubquestionResult> GetSubquestionResultsByStudentLogin(string studentLogin, int questionResultId)
        {
            return GetSubquestionResultDbSet()
                .Include(s => s.SubquestionTemplate)
                .Include(s => s.QuestionResult)
                .Include(s => s.QuestionResult.QuestionTemplate)
                .Include(s => s.QuestionResult.QuestionTemplate.TestTemplate)
                .Where(s => s.QuestionResultId == questionResultId &&
                s.QuestionResult.TestResult.Student.Login == studentLogin);
        }

        public SubquestionResult GetSubquestionResult(string login, int subquestionResultId)
        {
            return GetSubquestionResultDbSet()
                .Include(s => s.SubquestionTemplate)
                .Include(s => s.SubquestionTemplate.QuestionTemplate)
                .Include(s => s.SubquestionTemplate.QuestionTemplate.TestTemplate)
                .First(s => s.SubquestionResultId == subquestionResultId && s.OwnerLogin == login);
        }

        public IQueryable<SubquestionResult> GetSubquestionResults(string login, int testResultId, int questionTemplateId)
        {
            return GetSubquestionResultDbSet()
                .Include(s => s.QuestionResult)
                .Include(s => s.QuestionResult.TestResult)
                .Where(s => s.QuestionTemplateId == questionTemplateId
                    && s.OwnerLogin == login
                    && s.TestResultId == testResultId).AsQueryable();
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

        public async Task UpdateStudentsPoints(string login, int questionTemplateId, int subquestionResultId)
        {
            List<SubquestionResult> subquestionResults = dataFunctions.GetSubquestionResults(questionTemplateId, subquestionResultId, login);
            for(int i = 0; i < subquestionResults.Count; i++)
            {
                SubquestionResult subquestionResult = subquestionResults[i];
                SubquestionTemplate subquestionTemplate = subquestionResult.SubquestionTemplate;
                (double defaultStudentsPoints, _, _) = CommonFunctions.CalculateStudentsAnswerAttributes(subquestionTemplate.SubquestionType, subquestionTemplate.PossibleAnswerList,
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
                student = new Student() { Login = "testingstudent", Email = "studentemail", FirstName = "name", LastName = "surname", IsTestingData = true };
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

        public async Task<string> GetSubquestionResultPointsSuggestion(string login, int subquestionResultId)
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

            //var subquestionResults = GetSubquestionResults(login, testResultId, questionTemplateId);

            /*if (subquestionIdentifier == null)
            {
                subquestionIdentifier = subquestionResults.First().SubquestionIdentifier;
            }*/

            var subquestionResult = GetSubquestionResult(login, subquestionResultId);
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
            testResult.TestTemplateId = testTemplate.TestTemplateId;
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
                questionResult.TestResultId = testResult.TestResultId;
                questionResult.QuestionTemplateId = questionTemplate.QuestionTemplateId;
                questionResult.OwnerLogin = testTemplate.OwnerLogin;
                questionResult.TestResult = testResult;
                questionResult.QuestionTemplate = questionTemplate;
                questionResult.SubquestionResultList = new List<SubquestionResult>();

                for(int j = 0; j < questionTemplate.SubquestionTemplateList.Count; j++)
                {
                    SubquestionTemplate subquestionTemplate = questionTemplate.SubquestionTemplateList.ElementAt(j);
                    SubquestionResult subquestionResult = new SubquestionResult();
                    subquestionResult.TestResultId = testResult.TestResultId;
                    subquestionResult.QuestionTemplateId = questionResult.QuestionTemplateId;
                    subquestionResult.SubquestionTemplateId = subquestionTemplate.SubquestionTemplateId;
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
                    subquestionResultsProperties.Add((subquestionResult.SubquestionResultId, answerCompleteness));
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
                    (double defaultStudentsPoints, double answerCorrectness, AnswerStatus answerStatus) = CommonFunctions.CalculateStudentsAnswerAttributes(
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
    }
}