using Common;
using DataLayer;
using DomainModel;
using Microsoft.EntityFrameworkCore;
using ArtificialIntelligenceTools;
using System.Diagnostics;
using System.Text.RegularExpressions;
using System.Xml;
using static Common.EnumTypes;
using System.Threading.Tasks;

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

        /// <summary>
        /// Returns the list of test results that have been turned in (students cannot make changes to them anymore)
        /// </summary>
        /// <param name="login">Teacher's login</param>
        public async Task<List<TestResult>> GetTurnedTestResults(string login)
        {
            return await GetTestResultDbSet().
                Include(s => s.Student).
                Include(t => t.TestTemplate).
                Where(t => t.OwnerLogin == login
                && t.IsTurnedIn == true).ToListAsync();
        }

        /// <summary>
        /// Returns the list of test results of the teacher
        /// </summary>
        /// <param name="login">Teacher's login</param>
        public List<TestResult> GetTestResultList(string login)
        {
            return GetTestResultDbSet()
                .Include(s => s.Student)
                .Include(t => t.TestTemplate)
                .ThenInclude(t => t.Subject)
                .Include(t => t.TestTemplate)
                .ThenInclude(t => t.Owner)
                .Include(t => t.TestTemplate)
                .ThenInclude(t => t.QuestionTemplates)
                .ThenInclude(q => q.SubquestionTemplates)
                .Include(t => t.QuestionResults)
                .ThenInclude(q => q.SubquestionResults)
                .Where(t => t.OwnerLogin == login).ToList();
        }

        /// <summary>
        /// Returns the list of test results that have been turned in by the student
        /// </summary>
        /// <param name="login">Student's login</param>
        public async Task<List<TestResult>> GetFinishedTestResultsByStudentLogin(string login)
        {
            return await GetTestResultDbSet()
                .Include(t => t.Student)
                .Include(t => t.TestTemplate)
                .Include(t => t.TestTemplate.Owner)
                .Where(t => t.Student.Login == login
                    && t.IsTurnedIn == true
                    && t.TestTemplate.EndDate < DateTime.Now).ToListAsync();
        }

        /// <summary>
        /// Returns the amount of test results (of the test template) that have been turned in by the student
        /// Function is used to make sure that the student cannot take the same test more than once
        /// </summary>
        /// <param name="login">Student's login</param>
        /// <param name="testTemplateId">Id of the test template</param>
        public async Task<int> GetAmountOfTurnedTestResultsByTestTemplate(string login, int testTemplateId)
        {
            List<TestResult> turnedTestResultsByTestTemplate = await GetTestResultDbSet()
                .Where(t => t.Student.Login == login
                    && t.IsTurnedIn == true
                    && t.TestTemplateId == testTemplateId).ToListAsync();
            return turnedTestResultsByTestTemplate.Count;
        }

        /// <summary>
        /// Returns the amount of test results (of the test template) that have been turned in by the student
        /// Function is used to make sure that the student cannot take the same test more than once
        /// </summary>
        /// <param name="login">Student's login</param>
        /// <param name="testTemplateId">Id of the test template</param>
        public async Task<int> GetAmountOfNotTurnedTestResultsByTestTemplate(string login, int testTemplateId)
        {
            List<TestResult> notTurnedTestResultsByTestTemplate = await GetTestResultDbSet()
                .Where(t => t.Student.Login == login
                    && t.IsTurnedIn == false
                    && t.TestTemplateId == testTemplateId).ToListAsync();
            return notTurnedTestResultsByTestTemplate.Count;
        }

        public async Task<TestResult> GetTestResult(int testResultId)
        {
            return await dataFunctions.GetTestResult(testResultId);
        }

        /// <summary>
        /// Deletes test results of the teacher
        /// </summary>
        /// <param name="login">Teacher's login</param>
        public async Task<string> DeleteTestResults(string login)
        {
            return await dataFunctions.DeleteTestResults(login);
        }

        /// <summary>
        /// Deletes a test result
        /// </summary>
        /// <param name="login">Teacher's login</param>
        public async Task<string> DeleteTestResult(string login, int testResultId)
        {
            TestResult testResult = GetTestResultDbSet()
                .Include(t => t.TestTemplate)
                .First(t => t.TestResultId == testResultId);
            if(!CanUserModifyResult(login, testResult.TestTemplate.OwnerLogin))
            {
                return "K této akci nemáte oprávnění.";
            }
            return await dataFunctions.DeleteTestResult(testResult);
        }

        public async Task<List<QuestionResult>> GetQuestionResults(int testResultId)
        {
            return await GetQuestionResultDbSet()
                .Include(t => t.TestResult)
                .Include(q => q.QuestionTemplate)
                .Include(q => q.QuestionTemplate.TestTemplate)
                .Include(q => q.QuestionTemplate.SubquestionTemplates)
                .Include(s => s.TestResult.Student)
                .Include(q => q.SubquestionResults)
                .Where(t => t.TestResultId == testResultId).ToListAsync();
        }

        public async Task<List<SubquestionResult>> GetSubquestionResults(int questionResultId)
        {
            return await GetSubquestionResultDbSet()
                .Include(s => s.SubquestionTemplate)
                .Include(s => s.QuestionResult)
                .Include(s => s.QuestionResult.QuestionTemplate)
                .Include(s => s.QuestionResult.QuestionTemplate.TestTemplate)
                .Where(s => s.QuestionResultId == questionResultId).ToListAsync();
        }

        public async Task<SubquestionResult> GetSubquestionResult(int subquestionResultId)
        {
            return await GetSubquestionResultDbSet()
                .Include(s => s.SubquestionTemplate)
                .Include(s => s.SubquestionTemplate.QuestionTemplate)
                .Include(s => s.SubquestionTemplate.QuestionTemplate.TestTemplate)
                .FirstAsync(s => s.SubquestionResultId == subquestionResultId);
        }

        public DbSet<SubquestionResultStatistics> GetSubquestionResultStatisticsDbSet()
        {
            return dataFunctions.GetSubquestionResultStatisticsDbSet();
        }

        /// <summary>
        /// Returns subquestion result statistics of the teacher - throws an exception if statistics are not found
        /// </summary>
        /// <param name="login">Teacher's login</param>
        public async Task<SubquestionResultStatistics> GetSubquestionResultStatistics(string login)
        {
            SubquestionResultStatistics? subquestionResultStatistics = await dataFunctions.GetSubquestionResultStatisticsDbSet().FirstOrDefaultAsync(s => s.UserLogin == login);
            if(subquestionResultStatistics == null)
            {
                throw Exceptions.SubquestionResultStatisticsNotFoundException(login);
            }
            return subquestionResultStatistics;
        }

        /// <summary>
        /// Returns subquestion result statistics of the teacher - doesn't throw an exception if statistics are not found
        /// </summary>
        /// <param name="login">Teacher's login</param>
        public async Task<SubquestionResultStatistics?> GetSubquestionResultStatisticsNullable(string login)
        {
            return await dataFunctions.GetSubquestionResultStatisticsDbSet().FirstOrDefaultAsync(s => s.UserLogin == login);
        }

        /// <summary>
        /// Returns test difficulty statistics of the teacher - doesn't throw an exception if statistics are not found
        /// </summary>
        /// <param name="login">Teacher's login</param>
        public async Task<TestDifficultyStatistics?> GetTestDifficultyStatisticsNullable(string login)
        {
            return await dataFunctions.GetTestDifficultyStatisticsDbSet().FirstOrDefaultAsync
                (s => s.UserLogin == login);
        }

        /// <summary>
        /// Checks whether (test/question/subquestion) result can be modified by the user (results can only be modified by creators of result's templates)
        /// </summary>
        /// <param name="currentUserlogin">Current user's login</param>
        /// <param name="resultOwnerLogin">Result owner's login</param>
        public bool CanUserModifyResult(string currentUserlogin, string resultOwnerLogin)
        {
            if (currentUserlogin == resultOwnerLogin)
            {
                return true;
            }
            return false;
        }

        /// <summary>
        /// Updates subquestion result's student's points
        /// </summary>
        /// <param name="subquestionPoints">Subquestion template's points</param>
        /// <param name="studentsPoints">Student's points</param>
        /// <param name="negativePoints">Negative points of the test template (enabled/enabled for question/disabled)</param>
        /// <param name="subquestionResult">Subquestion result whose points are being changed</param>
        /// <param name="login">Teacher's login</param>
        public async Task<string> SetSubquestionResultPoints(string subquestionPoints, string studentsPoints, string negativePoints,
                SubquestionResult subquestionResult, string login)
        {
            if (!CanUserModifyResult(login, subquestionResult.SubquestionTemplate.OwnerLogin))
            {
                return "K této akci nemáte oprávnění.";
            }
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

        /// <summary>
        /// Returns all test templates marked as testing data
        /// </summary>
        public async Task<List<TestTemplate>> GetTestingDataTestTemplates()
        {
            return await dataFunctions.GetTestTemplateDbSet()
                .Include(t => t.Subject)
                .Include(t => t.QuestionTemplates)
                .ThenInclude(q => q.SubquestionTemplates)
                .Where(t => t.IsTestingData).ToListAsync();
        }

        /// <summary>
        /// Returns all test results marked as testing data
        /// </summary>
        public async Task<List<TestResult>> GetTestingDataTestResults()
        {
            return await GetTestResultDbSet()
                .Include(t => t.QuestionResults)
                .ThenInclude(q => q.SubquestionResults)
                .Where(t => t.IsTestingData).ToListAsync();
        }

        /// <summary>
        /// Returns the amount of subquestion results marked as testing data
        /// </summary>
        public async Task<int> GetTestingDataSubquestionResultsCount()
        {
            int testingDataSubquestionResults = 0;
            var testResults = await GetTestingDataTestResults();
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

        /// <summary>
        /// Creates a number of subquestion results marked as testing data
        /// This function is accessible via the ManageArtificialIntelligence view
        /// </summary>
        /// <param name="action">addSubquestionResultRandomData or addSubquestionResultCorrelationalData</param>
        /// <param name="amountOfSubquestionResults">Amount of subquestion results to be created</param>
        public async Task<string> CreateResultTestingData(string action, string amountOfSubquestionResults)
        {
            string message;
            var existingTestTemplates = await GetTestingDataTestTemplates();
            if(existingTestTemplates.Count == 0)
            {
                return "Chyba: nejprve je nutné vytvořit zadání testů.";
            }
            var existingTestResults = await GetTestingDataTestResults();
            int testingDataTestResultsCount = existingTestResults.Count;

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
            User owner = await dataFunctions.GetUserByLoginAsNoTracking();

            //delete existing subquestion template records of this user
            dataFunctions.ExecuteSqlRaw("delete from SubquestionResultRecord where 'login' = '" + login + "'");
            await dataFunctions.SaveChangesAsync();

            //create subquestion template records
            var testResultsToRecord = GetTestResultList(login);

            var subquestionResultRecords = DataGenerator.CreateSubquestionResultRecords(testResultsToRecord);
            await dataFunctions.SaveSubquestionResultRecords(subquestionResultRecords, owner);

            dataFunctions.ClearChargeTracker();
            owner = await dataFunctions.GetUserByLoginAsNoTracking();
            //managing subquestionResultStatistics
            var subquestionResultStatistics = await GetSubquestionResultStatisticsNullable(owner.Login);
            if (subquestionResultStatistics == null)
            {
                subquestionResultStatistics = new SubquestionResultStatistics();
                subquestionResultStatistics.User = owner;
                subquestionResultStatistics.UserLogin = owner.Login;
                int amountOfExistingUserResults = dataFunctions.GetSubquestionResultDbSet().Where(s => s.OwnerLogin == login).Count();
                if ((amountOfExistingUserResults + int.Parse(amountOfSubquestionResults)) > 100)
                {
                    subquestionResultStatistics.EnoughSubquestionResultsAdded = true;
                }
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
                int amountOfExistingUserResults = dataFunctions.GetSubquestionResultDbSet().Where(s => s.OwnerLogin == login).Count();
                if ((amountOfExistingUserResults + int.Parse(amountOfSubquestionResults)) > 100)
                {
                    subquestionResultStatistics.EnoughSubquestionResultsAdded = true;
                }

                await dataFunctions.SaveChangesAsync();
            }

            //managing testDifficultyStatistics
            await ManageTestDifficultyStatistics(testResults, owner);

            //sign testing student up for every testing subject
            await dataFunctions.RefreshTestingStudentSubjects();
            return message;
        }

        /// <summary>
        /// Manages test difficulty statistics
        /// This function is called after testing data are created, or after every 100th subquestion result has been processed for the teacher
        /// <param name="testResults">List of teacher's test results</param>
        /// <param name="owner">Teacher</param>
        /// </summary>
        public async Task ManageTestDifficultyStatistics(List<TestResult> testResults, User owner)
        {
            var testDifficultyStatistics = await GetTestDifficultyStatisticsNullable(owner.Login);
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

        /// <summary>
        /// Checks whether student marked as testing data exists, and creates him if he doesn't
        /// </summary>
        public async Task TestingUsersCheck()
        {
            Student? student = await dataFunctions.GetStudentByLoginNullable("testingstudent");
            if (student == null)
            {
                student = new Student() { Login = "testingstudent", Email = "studentemail", FirstName = "name", LastName = "surname", IsTestingData = true };
                await dataFunctions.AddStudent(student);
            }
        }

        /// <summary>
        /// Deletes all testing data related to results
        /// </summary>
        public async Task DeleteResultTestingData()
        {
            dataFunctions.ExecuteSqlRaw("delete from TestResult where IsTestingData = 1");
            dataFunctions.ExecuteSqlRaw("delete from SubquestionResultRecord where OwnerLogin = 'login'");
            dataFunctions.ExecuteSqlRaw("delete from SubquestionResultStatistics where UserLogin = 'login'");
            dataFunctions.ExecuteSqlRaw("delete from TestDifficultyStatistics where UserLogin = 'login'");

            //delete all models (if they exist)
            string[] testingDataModels = new string[] {
                Path.GetDirectoryName(Environment.CurrentDirectory) + "\\ArtificialIntelligenceTools\\PythonScripts\\model\\results\\login_NN.pt",
                Path.GetDirectoryName(Environment.CurrentDirectory) + "\\ArtificialIntelligenceTools\\PythonScripts\\model\\results\\login_LR.sav"};
            for (int i = 0; i < testingDataModels.Length; i++)
            {
                if (File.Exists(testingDataModels[i]))
                {
                    File.Delete(testingDataModels[i]);
                }
            }

            await dataFunctions.SaveChangesAsync();
        }

        /// <summary>
        /// Suggests the amount of points that the teacher should award the student for the answer
        /// <param name="login">Teacher's login</param>
        /// <param name="subquestionResultId">Id of the subquestion result</param>
        /// </summary>
        public async Task<string> GetSubquestionResultPointsSuggestion(string login, int subquestionResultId)
        {
            User owner = await dataFunctions.GetUserByLogin(login);

            //check if enough subquestion results have been added to warrant new model training
            bool retrainModel = false;
            SubquestionResultStatistics subquestionResultStatistics = await GetSubquestionResultStatistics(login);
            int subquestionResultsAdded = subquestionResultStatistics.SubquestionResultsAddedCount;
            if (subquestionResultsAdded >= 100)
            {
                retrainModel = true;
                await RetrainSubquestionResultModel(owner);
            }

            var subquestionResult = await GetSubquestionResult(subquestionResultId);
            SubquestionTemplate subquestionTemplate = subquestionResult.SubquestionTemplate;

            var testResults = GetTestResultList(owner.Login);
            double[] subquestionTypeAveragePoints = DataGenerator.GetSubquestionTypeAverageStudentsPoints(testResults);
            List<(Subject, double)> subjectAveragePointsTuple = DataGenerator.GetSubjectAverageStudentsPoints(testResults);
            TestTemplate testTemplate = subquestionTemplate.QuestionTemplate.TestTemplate;
            double minimumPointsShare = DataGenerator.GetMinimumPointsShare(testTemplate);

            SubquestionResultRecord currentSubquestionResultRecord = DataGenerator.CreateSubquestionResultRecord(subquestionResult, owner,
                subjectAveragePointsTuple, subquestionTypeAveragePoints, minimumPointsShare);
            SubquestionResultStatistics currectSubquestionResultStatistics = await GetSubquestionResultStatistics(login);
            if (!currectSubquestionResultStatistics.EnoughSubquestionResultsAdded)
            {
                return "Pro použití této funkce je nutné aby studenti vyplnili alespoň 100 podotázek.";
            }
            Model usedModel = currectSubquestionResultStatistics.UsedModel;
            double suggestedSubquestionPoints = PythonFunctions.GetSubquestionResultSuggestedPoints(login, retrainModel, currentSubquestionResultRecord, usedModel);
            if (subquestionResultsAdded >= 100)
            {
                subquestionResultStatistics = await GetSubquestionResultStatistics(login);
                subquestionResultStatistics.EnoughSubquestionResultsAdded = true;
                subquestionResultStatistics.SubquestionResultsAddedCount = 0;
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

                await ManageTestDifficultyStatistics(testResults, owner);
                await dataFunctions.SaveChangesAsync();
            }

            return suggestedSubquestionPoints.ToString();
        }

        /// <summary>
        /// Deletes all existing subquestion result records of the teacher and replaces them with records of current subquestion results
        /// <param name="owner">Teacher</param>
        /// </summary>
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

        /// <summary>
        /// Creates a new test result, each student can only be assigned to a single test result of the test template
        /// <param name="testTemplate">Test template of the test result</param>
        /// <param name="student">Student</param>
        /// </summary>
        public async Task<string?> BeginStudentAttempt(TestTemplate testTemplate, Student student)
        {
            TestResult testResult = new TestResult();
            testResult.TestTemplateId = testTemplate.TestTemplateId;
            testResult.TestTemplate = testTemplate;
            testResult.TimeStamp = DateTime.Now;
            testResult.Student = student;
            testResult.StudentLogin = student.Login;
            testResult.OwnerLogin = testTemplate.OwnerLogin;
            testResult.IsTestingData = student.IsTestingData;
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

        /// <summary>
        /// Loads student's last test result (based on test result's timestamp)
        /// </summary>
        public async Task<TestResult> LoadLastStudentAttempt(Student student)
        {
            return await dataFunctions.LoadLastStudentAttempt(student);
        }

        /// <summary>
        /// Returns a list of tuples containing subquestion result's Id and completeness (Answered/PartiallyAnswered/Unanswered)
        /// This function is used mainly to obtain the list of subquestion results independent of question results
        /// </summary>
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

        /// <summary>
        /// Same as GetSubquestionResultsProperties, but tuples contain answer status
        /// (Correct/PartiallyCorrect/Incorrect/NotAnswered/CannotBeDetermined) instead of completeness
        /// </summary>
        public List<(int, AnswerStatus)> GetSubquestionResultsPropertiesFinished(TestResult testResult)
        {
            List<(int, AnswerStatus)> subquestionResultsProperties = new List<(int, AnswerStatus)>();
            for (int i = 0; i < testResult.QuestionResults.Count; i++)
            {
                QuestionResult questionResult = testResult.QuestionResults.ElementAt(i);

                for (int j = 0; j < questionResult.SubquestionResults.Count; j++)
                {
                    SubquestionResult subquestionResult = questionResult.SubquestionResults.ElementAt(j);
                    subquestionResultsProperties.Add((subquestionResult.SubquestionResultId, subquestionResult.AnswerStatus));
                }
            }
            return subquestionResultsProperties;
        }

        /// <summary>
        /// Validates whether student's answers are valid and can be added to the subquestion result
        /// This function is used to prevent the student from entering invalid data (which could result in unexpected behavior)
        /// Error codes (such as 1001) are used to prevent the "attacker" from learning what made his answers get rejected
        /// </summary>
        /// <param name="subquestionResult">Subquestion result that is yet to be validated</param>
        /// <param name="possibleAnswers">Array of student's answers (used only for subquestion type 4 - multiple questions)</param>
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
                    subquestionResult.StudentsAnswers[i] = subquestionResult.StudentsAnswers[i].Replace("|", "");//replace gap separator
                    subquestionResult.StudentsAnswers[i] = subquestionResult.StudentsAnswers[i].Replace(";", "");//replace answer separator
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
                        if (!subquestionTemplate.PossibleAnswers.Contains(subquestionResult.StudentsAnswers[0]) && subquestionResult.StudentsAnswers[0] != null)
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
                        errorMessage = "Při ukládání řešení otázky došlo k chybě. Řešení otázky nebylo uloženo. Kód chyby 1006";
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
                            errorMessage = "Při ukládání řešení otázky došlo k chybě. Řešení otázky nebylo uloženo. Kód chyby 1002";
                        }
                    }
                    break;
            }

            //check if test template's end date has already passed
            DateTime endDate = subquestionTemplate.QuestionTemplate.TestTemplate.EndDate;
            if(endDate < DateTime.Now && !subquestionTemplate.QuestionTemplate.TestTemplate.IsTestingData)
            {
                errorMessage = "Čas na odevdzání testu již vypršel. Test byl automaticky odevzdán.";
            }

            return (subquestionResult, errorMessage);
        }

        /// <summary>
        /// Updates subquestion result's student's answers
        /// </summary>
        /// <param name="subquestionResult">Subquestion result</param>
        /// <param name="subquestionResultIndex">Subquestion result index - used to locate the subquestion result within the test result</param>
        /// <param name="student">Student</param>
        public async Task UpdateSubquestionResultStudentsAnswers(SubquestionResult subquestionResult, int subquestionResultIndex, Student student)
        {
            TestResult testResult = await LoadLastStudentAttempt(student);
            int subquestionCounter = 0;
            bool subquestionFound = false;
            for(int i = 0; i < testResult.QuestionResults.Count; i++)
            {
                if (subquestionFound)
                {
                    break;
                }
                QuestionResult questionResult = testResult.QuestionResults.ElementAt(i);

                for(int j = 0; j < questionResult.SubquestionResults.Count; j++)
                {
                    if(subquestionCounter == subquestionResultIndex)
                    {
                        subquestionFound = true;
                        questionResult.SubquestionResults.ElementAt(j).StudentsAnswers = subquestionResult.StudentsAnswers;
                        break;
                    }
                    subquestionCounter++;
                }
            }
            if (!subquestionFound)
            {
                throw Exceptions.InvalidSubquestionResultIndexException;
            }
            await dataFunctions.SaveChangesAsync();
        }


        /// <summary>
        /// Returns subquestion template based on subquestion result index
        /// <summary>
        public async Task<SubquestionTemplate> GetSubquestionTemplateBySubquestionResultIndex(int subquestionResultIndex, Student student)
        {
            TestResult testResult = await LoadLastStudentAttempt(student);
            int subquestionCounter = 0;
            for (int i = 0; i < testResult.QuestionResults.Count; i++)
            {
                QuestionResult questionResult = testResult.QuestionResults.ElementAt(i);

                for (int j = 0; j < questionResult.SubquestionResults.Count; j++)
                {
                    if (subquestionCounter == subquestionResultIndex)
                    {
                        return questionResult.SubquestionResults.ElementAt(j).SubquestionTemplate;
                    }
                    subquestionCounter++;
                }
            }
            throw Exceptions.SubquestionTemplateNotFoundException;
        }

        /// <summary>
        /// Finishes student's attempt - test result is marked as IsTurnedIn
        /// After this function is executed, student cannot make changes to the test result while teacher can
        /// <summary>
        public async Task FinishStudentAttempt(Student student)
        {
            TestResult testResult = await dataFunctions.LoadLastStudentAttempt(student);
            testResult.TimeStamp = DateTime.Now;
            testResult.IsTurnedIn = true;
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

        /// <summary>
        /// Processes subquestion result for view - certain chars are replaced with more intuitive strings
        /// <summary>
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
        /// <param name="questionTemplateId">Question template Id</param>
        /// <param name="login">Teacher's login</param>
        public async Task DeleteQuestionResults(int questionTemplateId, string login)
        {
            QuestionTemplate questionTemplate = await dataFunctions.GetQuestionTemplate(questionTemplateId);
            if (CanUserModifyResult(login, questionTemplate.OwnerLogin))
            {
                await dataFunctions.DeleteQuestionResults(questionTemplateId);
            }
        }

        /// <summary>
        /// Returns the amount of points that the student has obtained for this test result
        /// </summary>
        public double GetTestResultPointsSum(TestResult testResult)
        {
            double testPoints = 0;
            for (int i = 0; i < testResult.QuestionResults.Count; i++)
            {
                QuestionResult questionResult = testResult.QuestionResults.ElementAt(i);

                for (int j = 0; j < questionResult.SubquestionResults.Count; j++)
                {
                    SubquestionResult subquestionResult = questionResult.SubquestionResults.ElementAt(j);
                    testPoints += subquestionResult.StudentsPoints;
                }
            }

            return Math.Round(testPoints, 2);
        }
    }
}