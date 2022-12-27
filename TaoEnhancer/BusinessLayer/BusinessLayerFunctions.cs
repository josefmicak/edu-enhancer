using Common;
using DataLayer;
using DomainModel;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.Configuration;
using ArtificialIntelligenceTools;
using System;
using System.Diagnostics;
using static Common.EnumTypes;
using Microsoft.AspNetCore.Http;

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

        public List<TestTemplate> GetTestTemplatesByLogin(string login)//todo: duplikat
        {
            return templateFunctions.GetTestTemplatesByLogin(login);
        }

        public IQueryable<TestTemplate> GetTestTemplates(string login)
        {
            return templateFunctions.GetTestTemplates(login);
        }

        public async Task<string> AddTestTemplates(string login)
        {
            return await templateFunctions.AddTestTemplates(login);
        }

        public async Task<string> AddTestTemplate(TestTemplate testTemplate)
        {
            string login = GetCurrentUserLogin();
            testTemplate.OwnerLogin = login;
            testTemplate.Owner = GetUserByLogin(login);
            return await templateFunctions.AddTestTemplate(testTemplate);
        }

        public async Task<string> DeleteTestTemplates(string login)
        {
            return await templateFunctions.DeleteTestTemplates(login);
        }

        public async Task<string> DeleteTestTemplate(string login, string testNumberIdentifier, string webRootPath)
        {
            return await templateFunctions.DeleteTestTemplate(login, testNumberIdentifier, webRootPath);
        }

        public IQueryable<QuestionTemplate> GetQuestionTemplates(string login, string testNumberIdentifier)
        {
            return templateFunctions.GetQuestionTemplates(login, testNumberIdentifier);
        }

        public QuestionTemplate GetQuestionTemplate(string login, string questionNumberIdentifier)
        {
            return templateFunctions.GetQuestionTemplate(login, questionNumberIdentifier);
        }

        public async Task<string> AddQuestionTemplate(QuestionTemplate questionTemplate)
        {
            return await templateFunctions.AddQuestionTemplate(questionTemplate);
        }

        public async Task<string> DeleteQuestionTemplate(string questionNumberIdentifier, string webRootPath)
        {
            string login = GetCurrentUserLogin();
            return await templateFunctions.DeleteQuestionTemplate(login, questionNumberIdentifier, webRootPath);
        }

        public IQueryable<SubquestionTemplate> GetSubquestionTemplates(string login, string questionNumberIdentifier)
        {
            return templateFunctions.GetSubquestionTemplates(login, questionNumberIdentifier);
        }

        public async Task<string> AddSubquestionTemplate(SubquestionTemplate subquestionTemplate)
        {
            return await templateFunctions.AddSubquestionTemplate(subquestionTemplate);
        }

        public (SubquestionTemplate, string?) ValidateSubquestionTemplate(SubquestionTemplate subquestionTemplate, string[] subquestionTextArray, string sliderValues)
        {
            return templateFunctions.ValidateSubquestionTemplate(subquestionTemplate, subquestionTextArray, sliderValues);
        }

        public async Task<string> DeleteSubquestionTemplate(string questionNumberIdentifier, string subquestionIdentifier, string webRootPath)
        {
            string login = GetCurrentUserLogin();
            return await templateFunctions.DeleteSubquestionTemplate(login, questionNumberIdentifier, subquestionIdentifier, webRootPath);
        }

        public List<SubquestionTemplate> ProcessSubquestionTemplateForView(List<SubquestionTemplate> subquestionTemplates)
        {
            return templateFunctions.ProcessSubquestionTemplateForView(subquestionTemplates);
        }

        public TestTemplate GetTestTemplate(string login, string testNumberIdentifier)
        {
            return templateFunctions.GetTestTemplate(login, testNumberIdentifier);
        }

        public SubquestionTemplate GetSubquestionTemplate(string login, string questionNumberIdentifier, string subquestionIdentifier)
        {
            return templateFunctions.GetSubquestionTemplate(login, questionNumberIdentifier, subquestionIdentifier);
        }

        public async Task SetNegativePoints(TestTemplate testTemplate, EnumTypes.NegativePoints negativePoints)
        {
            await templateFunctions.SetNegativePoints(testTemplate, negativePoints);
        }

        public async Task<string> SetMinimumPoints(TestTemplate testTemplate, double minimumPoints, string testPointsDetermined)
        {
            return await templateFunctions.SetMinimumPoints(testTemplate, minimumPoints, testPointsDetermined);
        }

        public async Task<string> SetSubquestionTemplatePoints(string login, string questionNumberIdentifier, string subquestionIdentifier, string subquestionPoints, string wrongChoicePoints, bool defaultWrongChoicePoints)
        {
            return await templateFunctions.SetSubquestionTemplatePoints(login, questionNumberIdentifier, subquestionIdentifier, subquestionPoints, wrongChoicePoints, defaultWrongChoicePoints);
        }

        public async Task<string> GetSubquestionTemplatePointsSuggestion(string login, string questionNumberIdentifier, string subquestionIdentifier)
        {
            return await templateFunctions.GetSubquestionTemplatePointsSuggestion(login, questionNumberIdentifier, subquestionIdentifier);
        }

        public async Task<string> GetSubquestionTemplatePointsSuggestion(SubquestionTemplate subquestionTemplate)
        {
            return await templateFunctions.GetSubquestionTemplatePointsSuggestion(subquestionTemplate);
        }

        public int GetTestingDataSubquestionTemplatesCount()
        {
            return templateFunctions.GetTestingDataSubquestionTemplatesCount();
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

        public string GetTestDifficultyPrediction(string login, string testNumberIdentifier)
        {
            return templateFunctions.GetTestDifficultyPrediction(login, testNumberIdentifier);
        }

        public string? ValidateImage(IFormFile image)
        {
            return templateFunctions.ValidateImage(image);
        }

        public string SaveImage(IFormFile image, string webRootPath)
        {
            return templateFunctions.SaveImage(image, webRootPath);
        }

        //ResultFunctions.cs

        public DbSet<TestResult> GetTestResultDbSet()
        {
            return resultFunctions.GetTestResultDbSet();
        }

        public IQueryable<TestResult> GetTestResultsByOwnerLogin(string login)
        {
            return resultFunctions.GetTestResultsByOwnerLogin(login);
        }

        public IQueryable<TestResult> GetTestResultsByStudentLogin(string login)
        {
            return resultFunctions.GetTestResultsByStudentLogin(login);
        }

        public async Task<string> AddTestResults(string login)
        {
            return await resultFunctions.AddTestResults(login);
        }

        public async Task<string> DeleteTestResults(string login)
        {
            return await resultFunctions.DeleteTestResults(login);
        }

        public async Task<string> DeleteTestResult(string login, string testResultIdentifier)
        {
            return await resultFunctions.DeleteTestResult(login, testResultIdentifier);
        }

        public IQueryable<QuestionResult> GetQuestionResultsByOwnerLogin(string login, string testResultIdentifier)
        {
            return resultFunctions.GetQuestionResultsByOwnerLogin(login, testResultIdentifier);
        }

        public IQueryable<QuestionResult> GetQuestionResultsByStudentLogin(string studentLogin, string ownerLogin, string testResultIdentifier)
        {
            return resultFunctions.GetQuestionResultsByStudentLogin(studentLogin, ownerLogin, testResultIdentifier);
        }

        public IQueryable<SubquestionResult> GetSubquestionResultsByOwnerLogin(string login, string testResultIdentifier, string questionNumberIdentifier)
        {
            return resultFunctions.GetSubquestionResultsByOwnerLogin(login, testResultIdentifier, questionNumberIdentifier);
        }

        public IQueryable<SubquestionResult> GetSubquestionResultsByStudentLogin(string studentLogin, string ownerLogin, string testResultIdentifier, string questionNumberIdentifier)
        {
            return resultFunctions.GetSubquestionResultsByStudentLogin(studentLogin, ownerLogin, testResultIdentifier, questionNumberIdentifier);
        }

        public SubquestionResult GetSubquestionResult(string login, string testResultIdentifier, string questionNumberIdentifier, string subquestionIdentifier)
        {
            return resultFunctions.GetSubquestionResult(login, testResultIdentifier, questionNumberIdentifier, subquestionIdentifier);
        }

        public async Task<string> SetSubquestionResultPoints(string subquestionPoints, string studentsPoints, string negativePoints, SubquestionResult subquestionResult)
        {
            return await resultFunctions.SetSubquestionResultPoints(subquestionPoints, studentsPoints, negativePoints, subquestionResult);
        }

        public async Task UpdateStudentsPoints(string login, string questionNumberIdentifier, string subquestionIdentifier)
        {
            await resultFunctions.UpdateStudentsPoints(login, questionNumberIdentifier, subquestionIdentifier);
        }

        public int GetTestingDataSubquestionResultsCount()
        {
            return resultFunctions.GetTestingDataSubquestionResultsCount();
        }

        public async Task<string> CreateResultTestingData(string action, string amountOfSubquestionResults)
        {
            return await resultFunctions.CreateResultTestingData(action, amountOfSubquestionResults);
        }

        public async Task DeleteResultTestingData()
        {
            await resultFunctions.DeleteResultTestingData();
        }

        public async Task<string> GetSubquestionResultPointsSuggestion(string login, string testResultIdentifier, string questionNumberIdentifier, string subquestionIdentifier)
        {
            return await resultFunctions.GetSubquestionResultPointsSuggestion(login, testResultIdentifier, questionNumberIdentifier, subquestionIdentifier);
        }

        public SubquestionResultStatistics? GetSubquestionResultStatistics(string login)
        {
            return resultFunctions.GetSubquestionResultStatistics(login);
        }

        public DbSet<SubquestionResultStatistics> GetSubquestionResultStatisticsDbSet()
        {
            return resultFunctions.GetSubquestionResultStatisticsDbSet();
        }

        //UserFunctions.cs

        public DbSet<User> GetUserDbSet()
        {
            return userFunctions.GetUserDbSet();
        }

        public List<User> GetUserList()
        {
            return userFunctions.GetUserList();
        }

        public User? GetUserByLogin(string login)
        {
            return userFunctions.GetUserByLogin(login);
        }

        public User? GetUserByEmail(string email)
        {
            return userFunctions.GetUserByEmail(email);
        }

        public DbSet<Student> GetStudentDbSet()
        {
            return userFunctions.GetStudentDbSet();
        }

        public List<Student> GetStudentList()
        {
            return userFunctions.GetStudentList();
        }

        public Student? GetStudentByLogin(string login)
        {
            return userFunctions.GetStudentByLogin(login);
        }

        public Student? GetStudentByEmail(string email)
        {
            return userFunctions.GetStudentByEmail(email);
        }

        public Student? GetStudentByIdentifier(string studentIdentifier)
        {
            return userFunctions.GetStudentByIdentifier(studentIdentifier);
        }

        public async Task EditStudent(string studentIdentifier, string firstName, string lastName, string login, string email, Student studentLoginCheck)
        {
            await userFunctions.EditStudent(studentIdentifier, firstName, lastName, login, email, studentLoginCheck);
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

        public async Task<string> ApproveStudentRegistration(Student student, string firstName, string lastName, string login, string email)
        {
            return await userFunctions.ApproveStudentRegistration(student, firstName, lastName, login, email);
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

        public IQueryable<UserRegistration> GetUserRegistrations(string email)
        {
            return userFunctions.GetUserRegistrations(email);
        }

        public async Task RegisterMainAdmin(string firstName, string lastName, string email, string login)
        {
            await userFunctions.RegisterMainAdmin(firstName, lastName,  email, login);
        }

        public async Task<string> CreateUserRegistration(string firstName, string lastName, string email, string login, string role)
        {
            return await userFunctions.CreateUserRegistration(firstName, lastName, email, login, role);
        }

        public async Task<string> AddStudents(string login)
        {
            return await userFunctions.AddStudents(login);
        }

        public async Task DeleteAllStudents()
        {
            await userFunctions.DeleteAllStudents();
        }

        public async Task<string> AddStudent(string studentIdentifier, string firstName, string lastName, string login, string email, Student? studentLoginCheck)
        {
            return await userFunctions.AddStudent(studentIdentifier, firstName, lastName, login, email, studentLoginCheck);
        }


        public bool CanUserAccessPage(EnumTypes.Role requiredRole)
        {
            return userFunctions.CanUserAccessPage(requiredRole);
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
    }
}