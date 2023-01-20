using Common;
using DataLayer;
using DomainModel;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.Configuration;
using static Common.EnumTypes;
using Microsoft.AspNetCore.Http;
using ArtificialIntelligenceTools;

namespace BusinessLayer
{
    /// <summary>
    /// A class handling the logic of application's functions and data
    /// </summary>
    public class BusinessLayerFunctions
    {
        private TemplateFunctions templateFunctions;
        private ResultFunctions resultFunctions;
        private UserFunctions userFunctions;
        private OtherFunctions otherFunctions;

        public BusinessLayerFunctions(CourseContext context, IConfiguration configuration)
        {
            templateFunctions = new TemplateFunctions(context);
            resultFunctions = new ResultFunctions(context);
            userFunctions = new UserFunctions(context);
            otherFunctions = new OtherFunctions(context, configuration);
        }

        //TemplateFunctions.cs

        public DbSet<TestTemplate> GetTestTemplateDbSet()
        {
            return templateFunctions.GetTestTemplateDbSet();
        }

        public DbSet<SubquestionTemplateStatistics> GetSubquestionTemplateStatisticsDbSet()
        {
            return templateFunctions.GetSubquestionTemplateStatisticsDbSet();
        }

        public async Task<List<TestTemplate>> GetTestTemplates(string login)
        {
            return await templateFunctions.GetTestTemplates(login);
        }

        public async Task<string> AddTestTemplate(TestTemplate testTemplate, string subjectId)
        {
            string login = GetCurrentUserLogin();
            User owner = await GetUserByLogin(login);
            testTemplate.OwnerLogin = login;
            testTemplate.Owner = owner;
            return await templateFunctions.AddTestTemplate(testTemplate, subjectId);
        }

        public async Task<string> EditTestTemplate(TestTemplate testTemplate, string subjectId)
        {
            string login = GetCurrentUserLogin();
            TestTemplate oldTestTemplate = await GetTestTemplate(testTemplate.TestTemplateId);
            testTemplate.Owner = oldTestTemplate.Owner;
            testTemplate.OwnerLogin = oldTestTemplate.OwnerLogin;
            return await templateFunctions.EditTestTemplate(login, testTemplate, subjectId);
        }

        public async Task<string> DeleteTestTemplates()
        {
            string login = GetCurrentUserLogin();
            return await templateFunctions.DeleteTestTemplates(login);
        }

        public async Task<string> DeleteTestTemplate(int testTemplateId, string webRootPath)
        {
            string login = GetCurrentUserLogin();
            return await templateFunctions.DeleteTestTemplate(login, testTemplateId, webRootPath);
        }

        public bool CanUserEditTestTemplate(TestTemplate testTemplate)
        {
            string login = GetCurrentUserLogin();
            return templateFunctions.CanUserEditTestTemplate(testTemplate, login);
        }

        public async Task<List<QuestionTemplate>> GetQuestionTemplates(int testTemplateId)
        {
            return await templateFunctions.GetQuestionTemplates(testTemplateId);
        }

        public async Task<QuestionTemplate> GetQuestionTemplate(int questionTemplateId)
        {
            return await templateFunctions.GetQuestionTemplate(questionTemplateId);
        }

        public async Task<string> AddQuestionTemplate(QuestionTemplate questionTemplate)
        {
            return await templateFunctions.AddQuestionTemplate(questionTemplate);
        }

        public async Task<string> EditQuestionTemplate(QuestionTemplate questionTemplate)
        {
            string login = GetCurrentUserLogin();
            return await templateFunctions.EditQuestionTemplate(questionTemplate, login);
        }

        public async Task<string> DeleteQuestionTemplate(int questionTemplateId, string webRootPath)
        {
            string login = GetCurrentUserLogin();
            await DeleteQuestionResults(questionTemplateId);
            return await templateFunctions.DeleteQuestionTemplate(login, questionTemplateId, webRootPath);
        }

        public async Task<List<SubquestionTemplate>> GetSubquestionTemplates(int questionTemplateId)
        {
            return await templateFunctions.GetSubquestionTemplates(questionTemplateId);
        }

        public async Task<string> AddSubquestionTemplate(SubquestionTemplate subquestionTemplate, IFormFile? image, string webRootPath)
        {
            return await templateFunctions.AddSubquestionTemplate(subquestionTemplate, image, webRootPath);
        }

        public (SubquestionTemplate, string?) ValidateSubquestionTemplate(SubquestionTemplate subquestionTemplate, string[] subquestionTextArray, string sliderValues, IFormFile? image)
        {
            return templateFunctions.ValidateSubquestionTemplate(subquestionTemplate, subquestionTextArray, sliderValues, image);
        }

        public async Task<string> EditSubquestionTemplate(SubquestionTemplate subquestionTemplate, IFormFile? image, string webRootPath)
        {
            string login = GetCurrentUserLogin();
            return await templateFunctions.EditSubquestionTemplate(subquestionTemplate, image, webRootPath, login);
        }

        public async Task<string> DeleteSubquestionTemplate(int subquestionTemplateId, string webRootPath)
        {
            string login = GetCurrentUserLogin();
            return await templateFunctions.DeleteSubquestionTemplate(login, subquestionTemplateId, webRootPath);
        }

        public List<SubquestionTemplate> ProcessSubquestionTemplatesForView(List<SubquestionTemplate> subquestionTemplates)
        {
            return templateFunctions.ProcessSubquestionTemplatesForView(subquestionTemplates);
        }

        public SubquestionTemplate ProcessSubquestionTemplateForView(SubquestionTemplate subquestionTemplate)
        {
            return templateFunctions.ProcessSubquestionTemplateForView(subquestionTemplate);
        }

        public async Task<TestTemplate> GetTestTemplate(int testTemplateId)
        {
            return await templateFunctions.GetTestTemplate(testTemplateId);
        }

        public async Task<SubquestionTemplate> GetSubquestionTemplate(int subquestionTemplateId)
        {
            return await templateFunctions.GetSubquestionTemplate(subquestionTemplateId);
        }

        public async Task<double> GetTestTemplatePointsSum(int testTemplateId)
        {
            TestTemplate testTemplate = await GetTestTemplate(testTemplateId);
            return templateFunctions.GetTestTemplatePointsSum(testTemplate);
        }

        public async Task<int> GetTestTemplateSubquestionsCount(int testTemplateId)
        {
            TestTemplate testTemplate = await GetTestTemplate(testTemplateId);
            return templateFunctions.GetTestTemplateSubquestionsCount(testTemplate);
        }

        public async Task<string> GetSubquestionTemplatePointsSuggestion(SubquestionTemplate subquestionTemplate, bool subquestionTemplateExists)
        {
            if (subquestionTemplateExists)
            {
                subquestionTemplate.OwnerLogin = GetCurrentUserLogin();
            }
            return await templateFunctions.GetSubquestionTemplatePointsSuggestion(subquestionTemplate, subquestionTemplateExists);
        }

        public async Task<int> GetTestingDataSubquestionTemplatesCount()
        {
            return await templateFunctions.GetTestingDataSubquestionTemplatesCount();
        }

        public async Task<string> CreateTemplateTestingData(string action, string amountOfSubquestionTemplates)
        {
            return await templateFunctions.CreateTemplateTestingData(action, amountOfSubquestionTemplates);
        }

        public async Task DeleteTemplateTestingData()
        {
            await templateFunctions.DeleteTemplateTestingData();
        }

        public string[] GetSubquestionTypeTextArray()
        {
            return templateFunctions.GetSubquestionTypeTextArray();
        }

        public async Task<string> GetTestDifficultyPrediction(string login, int testTemplateId)
        {
            return await templateFunctions.GetTestDifficultyPrediction(login, testTemplateId);
        }

        public DbSet<Subject> GetSubjectDbSet()
        {
            return templateFunctions.GetSubjectDbSet();
        }

        public async Task<List<Subject>> GetSubjects()
        {
            return await templateFunctions.GetSubjects();
        }

        public async Task<Subject> GetSubjectById(int subjectId)
        {
            return await templateFunctions.GetSubjectById(subjectId);
        }

        public async Task<string> AddSubject(Subject subject, string[] enrolledStudentLogin)
        {
            string login = GetCurrentUserLogin();
            User guarantor = await GetUserByLogin(login);
            subject.GuarantorLogin = login;
            subject.Guarantor = guarantor;

            List<Student> studentList = new List<Student>();
            for(int i = 0; i < enrolledStudentLogin.Length; i++)
            {
                Student student = await GetStudentByLogin(enrolledStudentLogin[i]);
                studentList.Add(student);
            }
            subject.Students = studentList;
            return await templateFunctions.AddSubject(subject);
        }

        public async Task<string> EditSubject(Subject subject, string[] enrolledStudentLogin)
        {
            string login = GetCurrentUserLogin();
            User guarantor = await GetUserByLogin(login);

            List<Student> studentList = new List<Student>();
            for (int i = 0; i < enrolledStudentLogin.Length; i++)
            {
                Student student = await GetStudentByLogin(enrolledStudentLogin[i]);
                studentList.Add(student);
            }
            subject.Students = studentList;
            return await templateFunctions.EditSubject(subject, guarantor);
        }

        public async Task<string> DeleteSubject(int subjectId)
        {
            string login = GetCurrentUserLogin();
            Subject subject = await GetSubjectById(subjectId);
            User guarantor = await GetUserByLogin(login);
            return await templateFunctions.DeleteSubject(subject, guarantor);
        }

        public async Task<List<TestTemplate>> GetStudentAvailableTestList(string login)
        {
            Student student = await GetStudentByLogin(login);
            if (student.IsTestingData)
            {
                return GetTestTemplateDbSet().Include(t => t.Owner).Where(t => student.Subjects.Contains(t.Subject)
                    && t.IsTestingData).ToList();
            }
            else
            {
                List<TestTemplate> testTemplates = GetTestTemplateDbSet().Include(t => t.Owner).Where(t => student.Subjects.Contains(t.Subject)
                    && t.StartDate < DateTime.Now && t.EndDate > DateTime.Now).ToList();
                foreach (TestTemplate testTemplate in testTemplates)
                {
                    if (await GetAmountOfTurnedTestResultsByTestTemplate(login, testTemplate.TestTemplateId) > 0)
                    {
                        testTemplates.Remove(testTemplate);
                    }
                }
                return testTemplates;
            }
        }

        public async Task GenerateTemplatesFile()
        {
            var testTemplates = await GetTestTemplates("login");
            List<Subject> testingDataSubjects = await GetSubjectDbSet().Where(t => t.IsTestingData == true).ToListAsync();
            bool dataColleration = true;
            DataGenerator.GenerateTemplatesFile(dataColleration, testingDataSubjects);
        }

        //ResultFunctions.cs

        public DbSet<TestResult> GetTestResultDbSet()
        {
            return resultFunctions.GetTestResultDbSet();
        }

        public async Task<List<TestResult>> GetTurnedTestResults(string login)
        {
            return await resultFunctions.GetTurnedTestResults(login);
        }

        public async Task<List<TestResult>> GetFinishedTestResultsByStudentLogin(string login)
        {
            return await resultFunctions.GetFinishedTestResultsByStudentLogin(login);
        }

        public async Task<int> GetAmountOfTurnedTestResultsByTestTemplate(string login, int testTemplateId)
        {
            return await resultFunctions.GetAmountOfTurnedTestResultsByTestTemplate(login, testTemplateId);
        }

        public async Task<int> GetAmountOfNotTurnedTestResultsByTestTemplate(string login, int testTemplateId)
        {
            return await resultFunctions.GetAmountOfNotTurnedTestResultsByTestTemplate(login, testTemplateId);
        }

        public async Task<TestResult> GetTestResult(int testResultId)
        {
            return await resultFunctions.GetTestResult(testResultId);
        }

        public async Task<string?> BeginStudentAttempt(int testTemplateId)
        {
            string login = GetCurrentUserLogin();
            TestTemplate testTemplate = await GetTestTemplate(testTemplateId);
            Student student = await GetStudentByLogin(login);
            return await resultFunctions.BeginStudentAttempt(testTemplate, student);
        }

        public async Task FinishStudentAttempt()
        {
            string login = GetCurrentUserLogin();
            Student student = await GetStudentByLogin(login);
            await resultFunctions.FinishStudentAttempt(student);
        }

        public async Task<TestResult> LoadLastStudentAttempt()
        {
            string login = GetCurrentUserLogin();
            Student student = await GetStudentByLogin(login);
            return await resultFunctions.LoadLastStudentAttempt(student);
        }

        public List<(int, AnswerCompleteness)> GetSubquestionResultsProperties(TestResult testResult)
        {
            return resultFunctions.GetSubquestionResultsProperties(testResult);
        }

        public List<(int, AnswerStatus)> GetSubquestionResultsPropertiesFinished(TestResult testResult)
        {
            return resultFunctions.GetSubquestionResultsPropertiesFinished(testResult);
        }

        public async Task UpdateSubquestionResultStudentsAnswers(SubquestionResult subquestionResult, int subquestionResultIndex)
        {
            string login = GetCurrentUserLogin();
            Student student = await GetStudentByLogin(login);
            await resultFunctions.UpdateSubquestionResultStudentsAnswers(subquestionResult, subquestionResultIndex, student);
        }

        public async Task<(SubquestionResult, string?)> ValidateSubquestionResult(SubquestionResult subquestionResult, int subquestionResultIndex, 
                string[] possibleAnswers)
        {
            string login = GetCurrentUserLogin();
            Student student = await GetStudentByLogin(login);
            SubquestionTemplate subquestionTemplate = await resultFunctions.GetSubquestionTemplateBySubquestionResultIndex(subquestionResultIndex, student);
            subquestionResult.SubquestionTemplate = subquestionTemplate;
            return resultFunctions.ValidateSubquestionResult(subquestionResult, possibleAnswers);
        }

        public async Task<string> DeleteTestResults()
        {
            string login = GetCurrentUserLogin();
            return await resultFunctions.DeleteTestResults(login);
        }

        public async Task<string> DeleteTestResult(int testResultId)
        {
            string login = GetCurrentUserLogin();
            return await resultFunctions.DeleteTestResult(login, testResultId);
        }

        public async Task<List<QuestionResult>> GetQuestionResults(int testResultId)
        {
            return await resultFunctions.GetQuestionResults(testResultId);
        }

        public async Task<List<SubquestionResult>> GetSubquestionResults(int questionResultId)
        {
            return await resultFunctions.GetSubquestionResults(questionResultId);
        }

        public async Task<SubquestionResult> GetSubquestionResult(int subquestionResultId)
        {
            return await resultFunctions.GetSubquestionResult(subquestionResultId);
        }

        public async Task<string> SetSubquestionResultPoints(string subquestionPoints, string studentsPoints, string negativePoints, 
            SubquestionResult subquestionResult)
        {
            string login = GetCurrentUserLogin();
            return await resultFunctions.SetSubquestionResultPoints(subquestionPoints, studentsPoints, negativePoints, subquestionResult, login);
        }

        public async Task<int> GetTestingDataSubquestionResultsCount()
        {
            return await resultFunctions.GetTestingDataSubquestionResultsCount();
        }

        public async Task<string> CreateResultTestingData(string action, string amountOfSubquestionResults)
        {
            return await resultFunctions.CreateResultTestingData(action, amountOfSubquestionResults);
        }

        public async Task DeleteResultTestingData()
        {
            await resultFunctions.DeleteResultTestingData();
        }

        public async Task<string> GetSubquestionResultPointsSuggestion(string login, int subquestionResultId)
        {
            return await resultFunctions.GetSubquestionResultPointsSuggestion(login, subquestionResultId);
        }

        public DbSet<SubquestionResultStatistics> GetSubquestionResultStatisticsDbSet()
        {
            return resultFunctions.GetSubquestionResultStatisticsDbSet();
        }

        public SubquestionResult ProcessSubquestionResultForView(SubquestionResult subquestionResult)
        {
            return resultFunctions.ProcessSubquestionResultForView(subquestionResult);
        }

        public async Task DeleteQuestionResults(int questionTemplateId)
        {
            string login = GetCurrentUserLogin();
            await resultFunctions.DeleteQuestionResults(questionTemplateId, login);
        }

        public async Task<double> GetTestResultPointsSum(int testResultId)
        {
            TestResult testResult = await GetTestResult(testResultId);
            return resultFunctions.GetTestResultPointsSum(testResult);
        }

        public async Task UpdateTestResultTimeStamp(int testTemplateId)
        {
            string login = GetCurrentUserLogin();
            await resultFunctions.UpdateTestResultTimeStamp(login, testTemplateId);
        }

        public async Task GenerateResultsFile()
        {
            var testTemplates = await GetTestTemplates("login");
            bool dataColleration = true;
            DataGenerator.GenerateResultsFile(testTemplates, dataColleration);
        }

        //UserFunctions.cs

        public DbSet<User> GetUserDbSet()
        {
            return userFunctions.GetUserDbSet();
        }

        public async Task<User> GetUserByLogin(string login)
        {
            return await userFunctions.GetUserByLogin(login);
        }

        public async Task<User?> GetUserByLoginNullable(string login)
        {
            return await userFunctions.GetUserByLoginNullable(login);
        }

        public async Task<User?> GetUserByEmailNullable(string email)
        {
            return await userFunctions.GetUserByEmailNullable(email);
        }

        public DbSet<Student> GetStudentDbSet()
        {
            return userFunctions.GetStudentDbSet();
        }

        public async Task<Student> GetStudentByLogin(string login)
        {
            return await userFunctions.GetStudentByLogin(login);
        }

        public async Task<Student?> GetStudentByLoginNullable(string login)
        {
            return await userFunctions.GetStudentByLoginNullable(login);
        }

        public async Task<Student?> GetStudentByEmailNullable(string email)
        {
            return await userFunctions.GetStudentByEmailNullable(email);
        }

        public async Task EditStudent(string firstName, string lastName, string login, string email, Student studentLoginCheck)
        {
            await userFunctions.EditStudent(firstName, lastName, login, email, studentLoginCheck);
        }

        public async Task DeleteStudent(Student student)
        {
            await userFunctions.DeleteStudent(student);
        }

        public async Task AddTeacher(string firstName, string lastName, string login, string email)
        {
            await userFunctions.AddTeacher(firstName, lastName, login, email);
        }

        public async Task EditUser(User user, string firstName, string lastName, string login, string email, string role)
        {
            await userFunctions.EditUser(user, firstName, lastName, login, email, role);
        }

        public async Task ChangeMainAdmin(User newMainAdmin, string firstName, string lastName, string login, string email)
        {
            await userFunctions.ChangeMainAdmin(newMainAdmin, firstName, lastName, login, email);
        }

        public async Task<string> ApproveUserRegistration(string firstName, string lastName, string login, string email, string role)
        {
            return await userFunctions.ApproveUserRegistration(firstName, lastName, login, email, role);
        }

        public async Task<string> RefuseRegistration(string email)
        {
            return await userFunctions.RefuseRegistration(email);
        }

        public async Task<string> DeleteRegistration(string email)
        {
            return await userFunctions.DeleteRegistration(email);
        }

        public async Task DeleteAllRegistrations()
        {
            await userFunctions.DeleteAllRegistrations();
        }

        public async Task DeleteUser(User user)
        {
            await userFunctions.DeleteUser(user);
        }

        public async Task DeleteAllTeachers()
        {
            await userFunctions.DeleteAllTeachers();
        }

        public async Task DeleteAllAdmins()
        {
            await userFunctions.DeleteAllAdmins();
        }

        public async Task AddAdmin(string firstName, string lastName, string login, string email)
        {
            await userFunctions.AddAdmin(firstName, lastName, login, email);
        }

        public DbSet<UserRegistration> GetUserRegistrationDbSet()
        {
            return userFunctions.GetUserRegistrationDbSet();
        }

        public async Task<List<UserRegistration>> GetUserRegistrations(string email)
        {
            return await userFunctions.GetUserRegistrations(email);
        }

        public async Task RegisterMainAdmin(string firstName, string lastName, string email, string login)
        {
            await userFunctions.RegisterMainAdmin(firstName, lastName,  email, login);
        }

        public async Task<string> CreateUserRegistration(string firstName, string lastName, string email, string login, string role)
        {
            return await userFunctions.CreateUserRegistration(firstName, lastName, email, login, role);
        }

        public async Task DeleteAllStudents()
        {
            await userFunctions.DeleteAllStudents();
        }

        public async Task<string> AddStudent(string firstName, string lastName, string login, string email)
        {
            return await userFunctions.AddStudent(firstName, lastName, login, email);
        }

        public async Task<bool> CanUserAccessPage(EnumTypes.Role requiredRole)
        {
            return await userFunctions.CanUserAccessPage(requiredRole);
        }

        public async Task<string?> CanStudentAccessTest(int testTemplateId)
        {
            string login = GetCurrentUserLogin();
            Student student = await GetStudentByLogin(login);
            TestTemplate testTemplate = await GetTestTemplate(testTemplateId);
            if(await GetAmountOfNotTurnedTestResultsByTestTemplate(login, testTemplateId) > 0)
            {
                return "Pokus probíhá.";
            }
            return userFunctions.CanStudentAccessTest(student, testTemplate);
        }

        //OtherFunctions.cs

        public DbSet<GlobalSettings> GetGlobalSettingsDbSet()
        {
            return otherFunctions.GetGlobalSettingsDbSet();
        }

        public async Task ChangeGlobalSettings(string testingMode)
        {
            await otherFunctions.ChangeGlobalSettings(testingMode);
        }

        /// <summary>
        /// In case testing mode has been previously enabled, it is automatically turned on after the application is started again
        /// </summary>
        public void InitialTestingModeSettings()
        {
            otherFunctions.InitialTestingModeSettings();
        }

        /// <summary>
        /// Sets the platform on which the application is running (Windows/Linux)
        /// </summary>
        public void SelectedPlatformSettings()
        {
            otherFunctions.SelectedPlatformSettings();
        }

        public string GetAIDeviceName()
        {
            return otherFunctions.GetAIDeviceName();
        }

        public string GetGoogleClientId()
        {
            return otherFunctions.GetGoogleClientId();
        }

        public string GetCurrentUserLogin()
        {
            return otherFunctions.GetCurrentUserLogin();
        }

        public void SetCurrentUserLogin(string login)
        {
            otherFunctions.SetCurrentUserLogin(login);
        }

        public string? GetStudentSubquestionResultId()
        {
            return otherFunctions.GetStudentSubquestionResultId();
        }

        public void SetStudentSubquestionResultId(int subquestionResultId)
        {
            otherFunctions.SetStudentSubquestionResultId(subquestionResultId.ToString());
        }
    }
}