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
using System.Xml.Linq;

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

        public async Task<string> AddTestTemplate(TestTemplate testTemplate, string subjectId)
        {
            string login = GetCurrentUserLogin();
            testTemplate.OwnerLogin = login;
            testTemplate.Owner = GetUserByLogin(login);
            return await templateFunctions.AddTestTemplate(testTemplate, subjectId);
        }

        public async Task<string> DeleteTestTemplates(string login)
        {
            return await templateFunctions.DeleteTestTemplates(login);
        }

        public async Task<string> DeleteTestTemplate(string login, int testTemplateId, string webRootPath)
        {
            return await templateFunctions.DeleteTestTemplate(login, testTemplateId, webRootPath);
        }

        public IQueryable<QuestionTemplate> GetQuestionTemplates(string login, int testTemplateId)
        {
            return templateFunctions.GetQuestionTemplates(login, testTemplateId);
        }

        public QuestionTemplate GetQuestionTemplate(string login, int questionTemplateId)
        {
            return templateFunctions.GetQuestionTemplate(login, questionTemplateId);
        }

        public async Task<string> AddQuestionTemplate(QuestionTemplate questionTemplate)
        {
            return await templateFunctions.AddQuestionTemplate(questionTemplate);
        }

        public async Task<string> DeleteQuestionTemplate(int questionTemplateId, string webRootPath)
        {
            string login = GetCurrentUserLogin();
            await DeleteQuestionResults(questionTemplateId);
            return await templateFunctions.DeleteQuestionTemplate(login, questionTemplateId, webRootPath);
        }

        public IQueryable<SubquestionTemplate> GetSubquestionTemplates(string login, int questionTemplateId)
        {
            return templateFunctions.GetSubquestionTemplates(login, questionTemplateId);
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
            return await templateFunctions.EditSubquestionTemplate(subquestionTemplate, image, webRootPath);
        }

        public async Task<string> DeleteSubquestionTemplate(int questionTemplateId, int subquestionTemplateId, string webRootPath)
        {
            string login = GetCurrentUserLogin();
            await DeleteSubquestionResults(subquestionTemplateId);
            return await templateFunctions.DeleteSubquestionTemplate(login, questionTemplateId, subquestionTemplateId, webRootPath);
        }

        public List<SubquestionTemplate> ProcessSubquestionTemplatesForView(List<SubquestionTemplate> subquestionTemplates)
        {
            return templateFunctions.ProcessSubquestionTemplatesForView(subquestionTemplates);
        }

        public SubquestionTemplate ProcessSubquestionTemplateForView(SubquestionTemplate subquestionTemplate)
        {
            return templateFunctions.ProcessSubquestionTemplateForView(subquestionTemplate);
        }

        public TestTemplate GetTestTemplate(int testTemplateId)
        {
            return templateFunctions.GetTestTemplate(testTemplateId);
        }

        public TestTemplate GetTestTemplate(string login, int testTemplateId)
        {
            return templateFunctions.GetTestTemplate(login, testTemplateId);
        }

        public SubquestionTemplate GetSubquestionTemplate(string login, int subquestionTemplateId)
        {
            return templateFunctions.GetSubquestionTemplate(login, subquestionTemplateId);
        }

        public async Task SetNegativePoints(TestTemplate testTemplate, EnumTypes.NegativePoints negativePoints)
        {
            await templateFunctions.SetNegativePoints(testTemplate, negativePoints);
        }

        public async Task<string> SetMinimumPoints(TestTemplate testTemplate, double minimumPoints)
        {
            return await templateFunctions.SetMinimumPoints(testTemplate, minimumPoints);
        }

        public async Task<string> SetSubquestionTemplatePoints(string login, string subquestionTemplateId, string subquestionPoints, string wrongChoicePoints, bool defaultWrongChoicePoints)
        {
            return await templateFunctions.SetSubquestionTemplatePoints(login, int.Parse(subquestionTemplateId), subquestionPoints, wrongChoicePoints, defaultWrongChoicePoints);
        }

        public double GetTestTemplatePointsSum(int testTemplateId)
        {
            TestTemplate testTemplate = GetTestTemplate(testTemplateId);
            return templateFunctions.GetTestTemplatePointsSum(testTemplate);
        }

        public int GetTestTemplateSubquestionsCount(int testTemplateId)
        {
            TestTemplate testTemplate = GetTestTemplate(testTemplateId);
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

        public string GetTestDifficultyPrediction(string login, int testTemplateId)
        {
            return templateFunctions.GetTestDifficultyPrediction(login, testTemplateId);
        }

        public DbSet<Subject> GetSubjectDbSet()
        {
            return templateFunctions.GetSubjectDbSet();
        }

        public IQueryable<Subject> GetSubjects()
        {
            return templateFunctions.GetSubjects();
        }

        public Subject? GetSubjectById(int subjectId)
        {
            return templateFunctions.GetSubjectById(subjectId);
        }

        public async Task<string> AddSubject(Subject subject, string[] enrolledStudentLogin)
        {
            string login = GetCurrentUserLogin();
            subject.GuarantorLogin = login;
            subject.Guarantor = GetUserByLogin(login);

            List<Student> studentList = new List<Student>();
            for(int i = 0; i < enrolledStudentLogin.Length; i++)
            {
                Student? student = GetStudentByLogin(enrolledStudentLogin[i]);
                if(student != null)
                {
                    studentList.Add(student);
                }
            }
            subject.Students = studentList;
            return await templateFunctions.AddSubject(subject);
        }

        public async Task<string> EditSubject(Subject subject, string[] enrolledStudentLogin)
        {
            string login = GetCurrentUserLogin();
            User user = GetUserByLogin(login);

            List<Student> studentList = new List<Student>();
            for (int i = 0; i < enrolledStudentLogin.Length; i++)
            {
                Student? student = GetStudentByLogin(enrolledStudentLogin[i]);
                if (student != null)
                {
                    studentList.Add(student);
                }
            }
            subject.Students = studentList;
            return await templateFunctions.EditSubject(subject, user);
        }

        public async Task<string> DeleteSubject(int subjectId)
        {
            string login = GetCurrentUserLogin();
            Subject? subject = GetSubjectById(subjectId);
            User user = GetUserByLogin(login);
            return await templateFunctions.DeleteSubject(subject, user);
        }

        public IQueryable<TestTemplate> GetStudentAvailableTestList(string login)
        {
            Student? student = GetStudentByLogin(login);
            if(student != null)
            {
                return GetTestTemplateDbSet().Include(t => t.Owner).Where(t => student.Subjects.Contains(t.Subject));
            }
            else
            {
                throw Exceptions.UserNotFoundException;
            }
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

        public async Task<string?> BeginStudentAttempt(int testTemplateId, string login)
        {
            TestTemplate testTemplate = GetTestTemplate(testTemplateId);
            Student? student = GetStudentByLogin(login);
            if (student == null)
            {
                throw Exceptions.UserNotFoundException;
            }
            else
            {
                return await resultFunctions.BeginStudentAttempt(testTemplate, student);
            }
        }

        public async Task FinishStudentAttempt(string login)
        {
            Student? student = GetStudentByLogin(login);
            if (student == null)
            {
                throw Exceptions.UserNotFoundException;
            }
            else
            {
                await resultFunctions.FinishStudentAttempt(student);
            }
        }

        public async Task<TestResult> LoadLastStudentAttempt(string login)
        {
            Student? student = GetStudentByLogin(login);
            if (student == null)
            {
                throw Exceptions.UserNotFoundException;
            }
            else
            {
                return await resultFunctions.LoadLastStudentAttempt(student);
            }
        }

        public List<(int, AnswerCompleteness)> GetSubquestionResultsProperties(TestResult testResult)
        {
            return resultFunctions.GetSubquestionResultsProperties(testResult);
        }

        public async Task UpdateSubquestionResultStudentsAnswers(SubquestionResult subquestionResult, int subquestionResultIndex, string login)
        {
            Student? student = GetStudentByLogin(login);
            if (student == null)
            {
                throw Exceptions.UserNotFoundException;
            }
            else
            {
                await resultFunctions.UpdateSubquestionResultStudentsAnswers(subquestionResult, subquestionResultIndex, student);
            }
        }

        public async Task<(SubquestionResult, string?)> ValidateSubquestionResult(SubquestionResult subquestionResult, int subquestionResultIndex, string login, 
            string[] possibleAnswers)
        {
            Student? student = GetStudentByLogin(login);
            if (student == null)
            {
                throw Exceptions.UserNotFoundException;
            }
            else
            {
                SubquestionTemplate subquestionTemplate = await resultFunctions.GetSubquestionTemplateBySubquestionResultIndex(subquestionResultIndex, student);
                subquestionResult.SubquestionTemplate = subquestionTemplate;
                return resultFunctions.ValidateSubquestionResult(subquestionResult, possibleAnswers);
            }
        }

        public async Task<string> DeleteTestResults(string login)
        {
            return await resultFunctions.DeleteTestResults(login);
        }

        public async Task<string> DeleteTestResult(string login, int testResultId)
        {
            return await resultFunctions.DeleteTestResult(login, testResultId);
        }

        public IQueryable<QuestionResult> GetQuestionResultsByOwnerLogin(string login, int testResultId)
        {
            return resultFunctions.GetQuestionResultsByOwnerLogin(login, testResultId);
        }

        public IQueryable<QuestionResult> GetQuestionResultsByStudentLogin(string studentLogin, string ownerLogin, int testResultId)
        {
            return resultFunctions.GetQuestionResultsByStudentLogin(studentLogin, ownerLogin, testResultId);
        }

        public IQueryable<SubquestionResult> GetSubquestionResultsByOwnerLogin(string login, int questionResultId)
        {
            return resultFunctions.GetSubquestionResultsByOwnerLogin(login, questionResultId);
        }

        public IQueryable<SubquestionResult> GetSubquestionResultsByStudentLogin(string studentLogin, int questionResultId)
        {
            return resultFunctions.GetSubquestionResultsByStudentLogin(studentLogin, questionResultId);
        }

        public SubquestionResult GetSubquestionResult(string login, int subquestionResultId)
        {
            return resultFunctions.GetSubquestionResult(login, subquestionResultId);
        }

        public async Task<string> SetSubquestionResultPoints(string subquestionPoints, string studentsPoints, string negativePoints, SubquestionResult subquestionResult)
        {
            return await resultFunctions.SetSubquestionResultPoints(subquestionPoints, studentsPoints, negativePoints, subquestionResult);
        }

        public async Task UpdateStudentsPoints(string login, int questionTemplateId, string subquestionTemplateId)
        {
            await resultFunctions.UpdateStudentsPoints(login, questionTemplateId, int.Parse(subquestionTemplateId));
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

        public async Task<string> GetSubquestionResultPointsSuggestion(string login, int subquestionResultId)
        {
            return await resultFunctions.GetSubquestionResultPointsSuggestion(login, subquestionResultId);
        }

        public SubquestionResultStatistics? GetSubquestionResultStatistics(string login)
        {
            return resultFunctions.GetSubquestionResultStatistics(login);
        }

        public DbSet<SubquestionResultStatistics> GetSubquestionResultStatisticsDbSet()
        {
            return resultFunctions.GetSubquestionResultStatisticsDbSet();
        }

        public SubquestionResult ProcessSubquestionResultForView(SubquestionResult subquestionResult)
        {
            return resultFunctions.ProcessSubquestionResultForView(subquestionResult);
        }

        public async Task DeleteSubquestionResults(int subquestionTemplateId)
        {
            await resultFunctions.DeleteSubquestionResults(subquestionTemplateId);
        }

        public async Task DeleteQuestionResults(int questionTemplateId)
        {
            await resultFunctions.DeleteQuestionResults(questionTemplateId);
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

        public IQueryable<Student> GetStudents()
        {
            return userFunctions.GetStudents();
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

        public async Task DeleteAllStudents()
        {
            await userFunctions.DeleteAllStudents();
        }

        public async Task<string> AddStudent(string firstName, string lastName, string login, string email, Student? studentLoginCheck)
        {
            return await userFunctions.AddStudent(firstName, lastName, login, email, studentLoginCheck);
        }


        public bool CanUserAccessPage(EnumTypes.Role requiredRole)
        {
            return userFunctions.CanUserAccessPage(requiredRole);
        }

        public bool CanStudentAccessTest(string login, int testTemplateId)
        {
            Student? student = GetStudentByLogin(login);
            TestTemplate testTemplate = GetTestTemplate(testTemplateId);
            if(student == null)
            {
                throw Exceptions.UserNotFoundException;
            }
            else
            {
                return userFunctions.CanStudentAccessTest(student, testTemplate);
            }
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