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
                .ThenInclude(t => t.Subject)
                .Include(t => t.TestTemplate)
                .ThenInclude(t => t.QuestionTemplates)
                .ThenInclude(q => q.SubquestionTemplates)
                .Include(t => t.QuestionResults)
                .ThenInclude(q => q.SubquestionResults)
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
                .Include(q => q.QuestionTemplate.SubquestionTemplates)
                .Include(s => s.TestResult.Student)
                .Include(q => q.SubquestionResults)
                .Where(t => t.TestResultId == testResultId && t.OwnerLogin == login);
        }

        public IQueryable<QuestionResult> GetQuestionResultsByStudentLogin(string studentLogin, string ownerLogin, int testResultId)
        {
            return GetQuestionResultDbSet()
                .Include(q => q.TestResult)
                .Include(q => q.QuestionTemplate)
                .Include(q => q.QuestionTemplate.TestTemplate)
                .Include(q => q.TestResult.TestTemplate.Owner)
                .Include(q => q.QuestionTemplate.SubquestionTemplates)
                .Include(q => q.TestResult.Student)
                .Include(q => q.SubquestionResults)
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
                (double defaultStudentsPoints, _, _) = CommonFunctions.CalculateStudentsAnswerAttributes(subquestionTemplate.SubquestionType, subquestionTemplate.PossibleAnswers,
                    subquestionTemplate.CorrectAnswers, subquestionTemplate.SubquestionPoints, subquestionTemplate.WrongChoicePoints, subquestionResult.StudentsAnswers);
                subquestionResult.DefaultStudentsPoints = defaultStudentsPoints;
                subquestionResult.StudentsPoints = defaultStudentsPoints;
            }
            await dataFunctions.SaveChangesAsync();
        }

        public List<TestTemplate> GetTestingDataTestTemplates()
        {
            var testTemplates = dataFunctions.GetTestTemplateDbSet()
                .Include(t => t.Subject)
                .Include(t => t.QuestionTemplates)
                .ThenInclude(q => q.SubquestionTemplates)
                .Where(t => t.IsTestingData).ToList();
            return testTemplates;
        }

        public List<TestResult> GetTestingDataTestResults()
        {
            var testResults = GetTestResultDbSet()
                .Include(t => t.QuestionResults)
                .ThenInclude(q => q.SubquestionResults)
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
                for (int j = 0; j < testResult.QuestionResults.Count; j++)
                {
                    QuestionResult questionResult = testResult.QuestionResults.ElementAt(j);
                    for (int k = 0; k < questionResult.SubquestionResults.Count; k++)
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
            await ManageTestDifficultyStatistics(testResults, owner);

            return message;
        }

        public async Task ManageTestDifficultyStatistics(List<TestResult> testResults, User owner)
        {
            var testDifficultyStatistics = GetTestDifficultyStatistics(owner.Login);
            double[] subquestionTypeAveragePoints = DataGenerator.GetSubquestionTypeAverageStudentsPoints(testResults);
            List<(Subject, double)> subjectAveragePointsTuple = DataGenerator.GetSubjectAverageStudentsPoints(testResults);
            double[] subquestionTypeAverageAnswerCorrectness = DataGenerator.GetSubquestionTypeAverageAnswerCorrectness(testResults);
            double[] subjectIds = new double[subjectAveragePointsTuple.Count];
            for (int i = 0; i < subjectAveragePointsTuple.Count; i++)
            {
                subjectIds[i] = subjectAveragePointsTuple[i].Item1.SubjectId;
            }
            double[] subjectAveragePoints = new double[subjectAveragePointsTuple.Count];
            for (int i = 0; i < subjectAveragePointsTuple.Count; i++)
            {
                subjectAveragePoints[i] = subjectAveragePointsTuple[i].Item2;
            }

            if (testDifficultyStatistics == null)
            {
                testDifficultyStatistics = new TestDifficultyStatistics();
                testDifficultyStatistics.User = owner;
                testDifficultyStatistics.UserLogin = owner.Login;
                testDifficultyStatistics.InternalSubquestionTypeAveragePoints = subquestionTypeAveragePoints;
                testDifficultyStatistics.InternalSubjectIds = subjectIds;
                testDifficultyStatistics.InternalSubjectAveragePoints = subjectAveragePoints;
                testDifficultyStatistics.InternalSubquestionTypeAverageAnswerCorrectness = subquestionTypeAverageAnswerCorrectness;
                await dataFunctions.AddTestDifficultyStatistics(testDifficultyStatistics);
                dataFunctions.AttachUser(testDifficultyStatistics.User);
                await dataFunctions.SaveChangesAsync();
            }
            else
            {
                testDifficultyStatistics.InternalSubquestionTypeAveragePoints = subquestionTypeAveragePoints;
                testDifficultyStatistics.InternalSubjectIds = subjectIds;
                testDifficultyStatistics.InternalSubjectAveragePoints = subjectAveragePoints;
                testDifficultyStatistics.InternalSubquestionTypeAverageAnswerCorrectness = subquestionTypeAverageAnswerCorrectness;

                dataFunctions.AttachUser(testDifficultyStatistics.User);
                await dataFunctions.SaveChangesAsync();
            }
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
            int subquestionResultsAdded = GetSubquestionResultStatistics(login).SubquestionResultsAddedCount;
            if (subquestionResultsAdded >= 100)
            {
                retrainModel = true;
                await RetrainSubquestionResultModel(owner);
            }

            var subquestionResult = GetSubquestionResult(login, subquestionResultId);
            SubquestionTemplate subquestionTemplate = subquestionResult.SubquestionTemplate;

            var testResults = GetTestResultList(owner.Login);
            double[] subquestionTypeAveragePoints = DataGenerator.GetSubquestionTypeAverageStudentsPoints(testResults);
            List<(Subject, double)> subjectAveragePointsTuple = DataGenerator.GetSubjectAverageStudentsPoints(testResults);
            TestTemplate testTemplate = subquestionTemplate.QuestionTemplate.TestTemplate;
            double? minimumPointsShare = DataGenerator.GetMinimumPointsShare(testTemplate);

            SubquestionResultRecord currentSubquestionResultRecord = DataGenerator.CreateSubquestionResultRecord(subquestionResult, owner,
                subjectAveragePointsTuple, subquestionTypeAveragePoints, minimumPointsShare);
            SubquestionResultStatistics? currectSubquestionResultStatistics = GetSubquestionResultStatistics(login);
            if (!currectSubquestionResultStatistics.EnoughSubquestionResultsAdded)
            {
                return "Pro použití této funkce je nutné aby studenti vyplnili alespoň 100 podotázek.";
            }
            Model usedModel = currectSubquestionResultStatistics.UsedModel;
            string suggestedSubquestionPoints = PythonFunctions.GetSubquestionResultSuggestedPoints(login, retrainModel, currentSubquestionResultRecord, usedModel);
            if (subquestionResultsAdded >= 100)
            {
                SubquestionResultStatistics subquestionResultStatistics = GetSubquestionResultStatistics(login);
                subquestionResultStatistics.SubquestionResultsAddedCount = 0;
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

                await ManageTestDifficultyStatistics(testResults, owner);
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
            testResult.QuestionResults = new List<QuestionResult>();

            for(int i = 0; i < testTemplate.QuestionTemplates.Count; i++)
            {
                QuestionTemplate questionTemplate = testTemplate.QuestionTemplates.ElementAt(i);
                QuestionResult questionResult = new QuestionResult();
                questionResult.TestResultId = testResult.TestResultId;
                questionResult.QuestionTemplateId = questionTemplate.QuestionTemplateId;
                questionResult.OwnerLogin = testTemplate.OwnerLogin;
                questionResult.TestResult = testResult;
                questionResult.QuestionTemplate = questionTemplate;
                questionResult.SubquestionResults = new List<SubquestionResult>();

                for(int j = 0; j < questionTemplate.SubquestionTemplates.Count; j++)
                {
                    SubquestionTemplate subquestionTemplate = questionTemplate.SubquestionTemplates.ElementAt(j);
                    SubquestionResult subquestionResult = new SubquestionResult();
                    subquestionResult.TestResultId = questionResult.TestResultId;
                    subquestionResult.QuestionTemplateId = questionResult.QuestionTemplateId;
                    subquestionResult.SubquestionTemplateId = subquestionTemplate.SubquestionTemplateId;
                    subquestionResult.OwnerLogin = testResult.OwnerLogin;
                    subquestionResult.StudentsAnswers = new string[0];
                    subquestionResult.StudentsPoints = 0;
                    subquestionResult.DefaultStudentsPoints = 0;
                    subquestionResult.AnswerCorrectness = 0;
                    subquestionResult.AnswerStatus = AnswerStatus.NotAnswered;
                    subquestionResult.SubquestionTemplate = subquestionTemplate;
                    subquestionResult.QuestionResult = questionResult;
                    questionResult.SubquestionResults.Add(subquestionResult);
                }

                testResult.QuestionResults.Add(questionResult);
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
            for(int i = 0; i < testResult.QuestionResults.Count; i++)
            {
                QuestionResult questionResult = testResult.QuestionResults.ElementAt(i);

                for(int j = 0; j < questionResult.SubquestionResults.Count; j++)
                {
                    SubquestionResult subquestionResult = questionResult.SubquestionResults.ElementAt(j);
                    SubquestionTemplate subquestionTemplate = subquestionResult.SubquestionTemplate;
                    AnswerCompleteness answerCompleteness = new AnswerCompleteness();
                    switch (subquestionTemplate.SubquestionType)
                    {
                        case SubquestionType.OrderingElements:
                            if(subquestionResult.StudentsAnswers.Length == 0)
                            {
                                answerCompleteness = AnswerCompleteness.Unanswered;
                            }
                            else if(subquestionResult.StudentsAnswers.Length == subquestionTemplate.PossibleAnswers.Length)
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
                            if (subquestionResult.StudentsAnswers.Length == 0)
                            {
                                answerCompleteness = AnswerCompleteness.Unanswered;
                            }
                            else
                            {
                                answerCompleteness = AnswerCompleteness.Answered;
                            }
                            break;
                        case SubquestionType.MatchingElements:
                            if (subquestionResult.StudentsAnswers.Length == 0)
                            {
                                answerCompleteness = AnswerCompleteness.Unanswered;
                            }
                            else if (subquestionResult.StudentsAnswers.Length < subquestionTemplate.PossibleAnswers.Length / 2)
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
                            for (int k = 0; k < subquestionResult.StudentsAnswers.Length; k++)
                            {
                                if (subquestionResult.StudentsAnswers[k] == "X")
                                {
                                    unansweredCount++;
                                }
                            }
                            if (subquestionResult.StudentsAnswers.Length - unansweredCount == 0)
                            {
                                answerCompleteness = AnswerCompleteness.Unanswered;
                            }
                            else if(subquestionResult.StudentsAnswers.Length - unansweredCount < subquestionTemplate.PossibleAnswers.Length)
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
                            for (int k = 0; k < subquestionResult.StudentsAnswers.Length; k++)
                            {
                                if (subquestionResult.StudentsAnswers[k] == "|")
                                {
                                    unansweredCount++;
                                }
                            }
                            if (subquestionResult.StudentsAnswers.Length - unansweredCount == 0)
                            {
                                answerCompleteness = AnswerCompleteness.Unanswered;
                            }
                            else if (subquestionResult.StudentsAnswers.Length - unansweredCount < subquestionTemplate.CorrectAnswers.Length)
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

            if (subquestionResult.StudentsAnswers == null)
            {
                subquestionResult.StudentsAnswers = new string[0];
            }

            for (int i = 0; i < subquestionResult.StudentsAnswers.Length; i++)
            {
                if(subquestionResult.StudentsAnswers[i] != null)
                {
                    subquestionResult.StudentsAnswers[i].Replace("|", "");//replace gap separator
                    subquestionResult.StudentsAnswers[i].Replace(";", "");//replace answer separator
                }
            }

            //placeholder text gets binded to the student's answer array as well - it's necessary to remove these elements
            string placeholderText = "-ZVOLTE MOŽNOST-";
            if(subquestionTemplate.SubquestionType != SubquestionType.GapMatch)
            {
                subquestionResult.StudentsAnswers = subquestionResult.StudentsAnswers.Where(s => s != placeholderText).ToArray();
            }
            else
            {
                for (int i = 0; i < subquestionResult.StudentsAnswers.Length; i++)
                {
                    if (subquestionResult.StudentsAnswers[i] == placeholderText)
                    {
                        subquestionResult.StudentsAnswers[i] = "|";
                    }
                }
            }

            switch (subquestionTemplate.SubquestionType)
            {
                case SubquestionType.Error:
                    errorMessage = "Při ukládání řešení otázky došlo k chybě. Řešení otázky nebylo uloženo. Kód chyby 1001";
                    break;
                case SubquestionType.OrderingElements:
                    if(subquestionResult.StudentsAnswers.Length > subquestionTemplate.PossibleAnswers.Length)
                    {
                        errorMessage = "Při ukládání řešení otázky došlo k chybě. Řešení otázky nebylo uloženo. Kód chyby 1004";
                    }
                    for(int i = 0; i < subquestionResult.StudentsAnswers.Length; i++)
                    {
                        if (!subquestionTemplate.PossibleAnswers.Contains(subquestionResult.StudentsAnswers[i]))
                        {
                            errorMessage = "Při ukládání řešení otázky došlo k chybě. Řešení otázky nebylo uloženo. Kód chyby 1003";
                        }
                    }
                    break;
                case SubquestionType.MultiChoiceMultipleCorrectAnswers:
                    if(subquestionResult.StudentsAnswers.Length > subquestionTemplate.PossibleAnswers.Length)
                    {
                        errorMessage = "Při ukládání řešení otázky došlo k chybě. Řešení otázky nebylo uloženo. Kód chyby 1004";
                    }
                    for (int i = 0; i < subquestionResult.StudentsAnswers.Length; i++)
                    {
                        if (!subquestionTemplate.PossibleAnswers.Contains(subquestionResult.StudentsAnswers[i]))
                        {
                            errorMessage = "Při ukládání řešení otázky došlo k chybě. Řešení otázky nebylo uloženo. Kód chyby 1003";
                        }
                    }
                    break;
                case SubquestionType.MatchingElements:
                    if (subquestionResult.StudentsAnswers.Length > subquestionTemplate.PossibleAnswers.Length)
                    {
                        errorMessage = "Při ukládání řešení otázky došlo k chybě. Řešení otázky nebylo uloženo. Kód chyby 1004";
                    }
                    for (int i = 0; i < subquestionResult.StudentsAnswers.Length; i++)
                    {
                        if (!subquestionTemplate.PossibleAnswers.Contains(subquestionResult.StudentsAnswers[i]))
                        {
                            errorMessage = "Při ukládání řešení otázky došlo k chybě. Řešení otázky nebylo uloženo. Kód chyby 1003";
                        }
                    }
                    if (subquestionResult.StudentsAnswers.Length % 2 == 1)
                    {
                        errorMessage = "Při ukládání řešení otázky došlo k chybě. Řešení otázky nebylo uloženo. Kód chyby 1005";
                    }
                    else
                    {
                        string[] newstudentsAnswerList = new string[subquestionResult.StudentsAnswers.Length / 2];
                        for (int i = 0; i < subquestionResult.StudentsAnswers.Length; i++)
                        {
                            if (i % 2 == 1)
                            {
                                continue;
                            }
                            else
                            {
                                int index = i / 2;
                                newstudentsAnswerList[index] = subquestionResult.StudentsAnswers[i] + "|" + subquestionResult.StudentsAnswers[i + 1];
                            }
                        }
                        subquestionResult.StudentsAnswers = newstudentsAnswerList;
                    }
                    break;
                case SubquestionType.MultipleQuestions:
                    //the answers are in wrong order because of shuffle - we have to rearrange them
                    string[] studentsAnswers = subquestionResult.StudentsAnswers;
                    subquestionResult.StudentsAnswers = new string[possibleAnswers.Length];
                    for (int i = 0; i < subquestionResult.SubquestionTemplate.PossibleAnswers.Length; i++)   
                    {
                        for (int j = 0; j < possibleAnswers.Length; j++)
                        {
                            if (possibleAnswers[j] == subquestionResult.SubquestionTemplate.PossibleAnswers[i])
                            {
                                subquestionResult.StudentsAnswers[i] = studentsAnswers[j];
                                break;
                            }
                        }
                    }

                    for (int i = 0; i < subquestionResult.StudentsAnswers.Length; i++)
                    {
                        if (subquestionResult.StudentsAnswers[i] == null)
                        {
                            subquestionResult.StudentsAnswers[i] = "X";//unanswered
                        }
                        if(subquestionResult.StudentsAnswers[i] != "0" && subquestionResult.StudentsAnswers[i] != "1" && subquestionResult.StudentsAnswers[i] != "X")
                        {
                            errorMessage = "Při ukládání řešení otázky došlo k chybě. Řešení otázky nebylo uloženo. Kód chyby 1003";
                        }
                    }
                    if (subquestionResult.StudentsAnswers.Length != subquestionTemplate.PossibleAnswers.Length)
                    {
                        errorMessage = "Při ukládání řešení otázky došlo k chybě. Řešení otázky nebylo uloženo. Kód chyby 1004";
                    }
                    break;
                case SubquestionType.FreeAnswer:
                    break;
                case SubquestionType n when (n == SubquestionType.MultiChoiceSingleCorrectAnswer || n == SubquestionType.MultiChoiceTextFill):
                    if (subquestionResult.StudentsAnswers.Length > 1)
                    {
                        errorMessage = "Při ukládání řešení otázky došlo k chybě. Řešení otázky nebylo uloženo. Kód chyby 1006";
                    }
                    else if (subquestionResult.StudentsAnswers.Length == 1)
                    {
                        if (!subquestionTemplate.PossibleAnswers.Contains(subquestionResult.StudentsAnswers[0]))
                        {
                            errorMessage = "Při ukládání řešení otázky došlo k chybě. Řešení otázky nebylo uloženo. Kód chyby 1003";
                        }
                    }
                    break;
                case SubquestionType.FreeAnswerWithDeterminedCorrectAnswer:
                    if (subquestionResult.StudentsAnswers.Length > 1)
                    {
                        errorMessage = "Při ukládání řešení otázky došlo k chybě. Řešení otázky nebylo uloženo. Kód chyby 1006";
                    }
                    break;
                case SubquestionType.GapMatch:
                    if (subquestionResult.StudentsAnswers.Length > subquestionTemplate.CorrectAnswers.Length)
                    {
                        errorMessage = "Při ukládání řešení otázky došlo k chybě. Řešení otázky nebylo uloženo. Kód chyby 1007";
                    }
                    for (int i = 0; i < subquestionResult.StudentsAnswers.Length; i++)
                    {
                        if (!subquestionTemplate.CorrectAnswers.Contains(subquestionResult.StudentsAnswers[i]) && subquestionResult.StudentsAnswers[i] != "|")
                        {
                            errorMessage = "Při ukládání řešení otázky došlo k chybě. Řešení otázky nebylo uloženo. Kód chyby 1003";
                        }
                    }
                    break;
                case SubquestionType.Slider:
                    if (subquestionResult.StudentsAnswers.Length > 1)
                    {
                        errorMessage = "Při ukládání řešení otázky došlo k chybě. Řešení otázky nebylo uloženo. Kód chyby 1008";
                    }
                    if (subquestionResult.StudentsAnswers[0] == "Nezodpovězeno")
                    {
                        subquestionResult.StudentsAnswers = new string[0];
                    }
                    if (subquestionResult.StudentsAnswers.Length == 1)
                    {
                        bool isNumber = int.TryParse(subquestionResult.StudentsAnswers[0], out _);
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
            for(int i = 0; i < testResult.QuestionResults.Count; i++)
            {
                QuestionResult questionResult = testResult.QuestionResults.ElementAt(i);

                for(int j = 0; j < questionResult.SubquestionResults.Count; j++)
                {
                    if(j == subquestionResultIndex)
                    {
                        questionResult.SubquestionResults.ElementAt(j).StudentsAnswers = subquestionResult.StudentsAnswers;
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
            for (int i = 0; i < testResult.QuestionResults.Count; i++)
            {
                QuestionResult questionResult = testResult.QuestionResults.ElementAt(i);

                for (int j = 0; j < questionResult.SubquestionResults.Count; j++)
                {
                    if (j == subquestionResultIndex)
                    {
                        return questionResult.SubquestionResults.ElementAt(j).SubquestionTemplate;
                    }
                }
            }
            throw Exceptions.SubquestionTemplateNotFoundException;
        }

        public async Task FinishStudentAttempt(Student student)
        {
            TestResult testResult = await dataFunctions.LoadLastStudentAttempt(student);
            testResult.TimeStamp = DateTime.Now;
            for(int i = 0; i < testResult.QuestionResults.Count; i++)
            {
                QuestionResult questionResult = testResult.QuestionResults.ElementAt(i);

                for(int j = 0; j < questionResult.SubquestionResults.Count; j++)
                {
                    SubquestionResult subquestionResult = questionResult.SubquestionResults.ElementAt(j);
                    SubquestionTemplate subquestionTemplate = subquestionResult.SubquestionTemplate;
                    (double defaultStudentsPoints, double answerCorrectness, AnswerStatus answerStatus) = CommonFunctions.CalculateStudentsAnswerAttributes(
                        subquestionTemplate.SubquestionType, subquestionTemplate.PossibleAnswers, subquestionTemplate.CorrectAnswers,
                        subquestionTemplate.SubquestionPoints, subquestionTemplate.WrongChoicePoints, subquestionResult.StudentsAnswers);
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
                    for(int i = 0; i < subquestionResult.StudentsAnswers.Length; i++)
                    {
                        subquestionResult.StudentsAnswers[i] = subquestionResult.StudentsAnswers[i].Replace("|", " -> ");
                    }
                    break;
                case SubquestionType.MultipleQuestions:
                    for (int i = 0; i < subquestionResult.StudentsAnswers.Length; i++)
                    {
                        string studentsAnswer = "";
                        if (subquestionResult.StudentsAnswers[i] == "1")
                        {
                            studentsAnswer = "Ano";
                        }
                        else if (subquestionResult.StudentsAnswers[i] == "0")
                        {
                            studentsAnswer = "Ne";
                        }
                        else if (subquestionResult.StudentsAnswers[i] == "X")
                        {
                            studentsAnswer = "Nezodpovězeno";
                        }
                        subquestionResult.StudentsAnswers[i] = subquestionResult.SubquestionTemplate.PossibleAnswers[i] + " -> " + studentsAnswer;
                    }
                    break;
                case SubquestionType.GapMatch:
                    for (int i = 0; i < subquestionResult.StudentsAnswers.Length; i++)
                    {
                        subquestionResult.StudentsAnswers[i] = "[" + (i + 1) + "] - " + subquestionResult.StudentsAnswers[i];
                    }
                    break;
            }

            return subquestionResult;
        }

        /// <summary>
        /// Used to delete all question results of a certain question template
        /// </summary>
        public async Task DeleteQuestionResults(int questionTemplateId)
        {
            await dataFunctions.DeleteQuestionResults(questionTemplateId);
        }

        /// <summary>
        /// Used to delete all subquestion results of a certain subquestion template
        /// </summary>
        public async Task DeleteSubquestionResults(int subquestionTemplateId)
        {
            await dataFunctions.DeleteSubquestionResults(subquestionTemplateId);
        }
    }
}