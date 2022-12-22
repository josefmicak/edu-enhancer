using Common;
using DomainModel;
using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Storage;
using System.Diagnostics;

namespace DataLayer
{
    /// <summary>
    /// The only place in the application that handles direct work with data stored in the database
    /// </summary>
    public class DataFunctions
    {
        private readonly CourseContext _context;

        public DataFunctions(CourseContext context)
        {
            _context = context;
        }

        //TemplateFunctions.cs

        public DbSet<TestTemplate> GetTestTemplateDbSet()
        {
            return _context.TestTemplates;
        }

        public DbSet<QuestionTemplate> GetQuestionTemplateDbSet()
        {
            return _context.QuestionTemplates;
        }

        public DbSet<SubquestionTemplate> GetSubquestionTemplateDbSet()
        {
            return _context.SubquestionTemplates;
        }

        public DbSet<SubquestionTemplateStatistics> GetSubquestionTemplateStatisticsDbSet()
        {
            return _context.SubquestionTemplateStatistics;
        }

        public List<TestTemplate> GetTestTemplateList(string login)
        {
            return GetTestTemplateDbSet()
                .Include(t => t.QuestionTemplateList)
                .ThenInclude(q => q.SubquestionTemplateList)
                .Where(t => t.OwnerLogin == login).ToList();
        }

        public async Task<string> AddTestTemplates(List<TestTemplate> testTemplates, User? owner)
        {
            int successCount = 0;
            int errorCount = 0;

            for (int i = 0; i < testTemplates.Count; i++)
            {
                _context.ChangeTracker.Clear();
                try
                {
                    TestTemplate testTemplate = testTemplates[i];
                    _context.ChangeTracker.Clear();
                    _context.Users.Attach(testTemplate.Owner);
                    _context.TestTemplates.Add(testTemplate);
                    await _context.SaveChangesAsync();
                    successCount++;
                }
                catch (Exception ex)
                {
                    Debug.WriteLine(ex.Message);
                    errorCount++;
                }
            }

            string message = "Přidáno " + successCount + " šablon testů (" + errorCount + " duplikátů nebo chyb).";
            return message;
        }

        public async Task<string> AddTestTemplate(TestTemplate testTemplate)
        {
            string message;
            try
            {
                _context.Users.Attach(testTemplate.Owner);
                _context.TestTemplates.Add(testTemplate);
                await _context.SaveChangesAsync();
                message = "Zadání testu bylo úspěšně přidáno.";
            }
            catch (Exception ex)
            {
                Debug.WriteLine(ex.Message);
                message = "Při přidání testu nastala neočekávaná chyba.";

            }
            return message;
        }

        public async Task<string> DeleteTestTemplates(string login)
        {
            var testTemplateList = GetTestTemplateDbSet().Where(t => t.OwnerLogin == login);
            foreach (TestTemplate testTemplate in testTemplateList)
            {
                _context.TestTemplates.Remove(testTemplate);
            }
            await _context.SaveChangesAsync();
            string message = "Byly smazány všechny vaše testy.";
            return message;
        }

        public async Task<string> DeleteTestTemplate(TestTemplate testTemplate, string webRootPath)
        {
            for(int i = 0; i < testTemplate.QuestionTemplateList.Count; i++)
            {
                QuestionTemplate questionTemplate = testTemplate.QuestionTemplateList.ElementAt(i);
                for(int j = 0; j < questionTemplate.SubquestionTemplateList.Count; j++)
                {
                    SubquestionTemplate subquestionTemplate = questionTemplate.SubquestionTemplateList.ElementAt(j);
                    if(subquestionTemplate.ImageSource != null)
                    {
                        DeleteSubquestionTemplateImage(webRootPath, subquestionTemplate.ImageSource);
                    }
                }
            }
            _context.TestTemplates.Remove(testTemplate);
            await _context.SaveChangesAsync();
            string message = "Test byl úspěšně smazán.";
            return message;
        }

        public async Task SaveSubquestionTemplateRecords(List<SubquestionTemplateRecord> subquestionTemplateRecords, User owner)
        {
            for (int i = 0; i < subquestionTemplateRecords.Count; i++)
            {
                try
                {
                    var subquestionTemplateRecord = subquestionTemplateRecords[i];
                    _context.Users.Attach(owner);
                    _context.SubquestionTemplateRecords.Add(subquestionTemplateRecord);
                    await _context.SaveChangesAsync();
                }

                catch (Exception ex)
                {
                    Debug.WriteLine(ex.Message);
                }
            }
        }

        public List<TestTemplate> GetTestTemplatesByLogin(string login)
        {
            return _context.TestTemplates
                .Include(t => t.Owner)
                .Include(t => t.QuestionTemplateList)
                .ThenInclude(q => q.SubquestionTemplateList)
                .Where(t => t.OwnerLogin == login).ToList();
        }

        public TestTemplate GetTestTemplate(string login, string testNameIdentifier)
        {
            return GetTestTemplateDbSet()
                .Include(t => t.QuestionTemplateList)
                .First(t => t.TestNameIdentifier == testNameIdentifier && t.OwnerLogin == login);
        }

        public TestTemplate GetTestTemplate(string login, string testNumberIdentifier, string _)
        {//todo: sjednotit
            return GetTestTemplateDbSet()
                .Include(t => t.QuestionTemplateList)
                .First(t => t.TestNumberIdentifier == testNumberIdentifier && t.OwnerLogin == login);
        }

        public QuestionTemplate GetQuestionTemplate(string login, string questionNumberIdentifier)
        {
            return GetQuestionTemplateDbSet()
                .Include(q => q.TestTemplate)
                .Include(q => q.SubquestionTemplateList)
                .First(q => q.QuestionNumberIdentifier == questionNumberIdentifier && q.OwnerLogin == login);
        }

        public async Task<string> AddQuestionTemplate(QuestionTemplate questionTemplate, string testNumberIdentifier)
        {
            string message;
            try
            {
                TestTemplate testTemplate = GetTestTemplate(questionTemplate.OwnerLogin, testNumberIdentifier, "");
                ICollection<QuestionTemplate> questionTemplates = testTemplate.QuestionTemplateList;
                questionTemplates.Add(questionTemplate);
                await _context.SaveChangesAsync();
                message = "Zadání otázky bylo úspěšně přidáno.";
            }
            catch (Exception ex)
            {
                Debug.WriteLine(ex.Message);
                message = "Při přidání otázky nastala neočekávaná chyba.";
            }
            return message;
        }

        public async Task<string> DeleteQuestionTemplate(string login, string questionNumberIdentifier, string webRootPath)
        {
            QuestionTemplate questionTemplate = GetQuestionTemplate(login, questionNumberIdentifier);
            for(int i = 0; i < questionTemplate.SubquestionTemplateList.Count; i++)
            {
                SubquestionTemplate subquestionTemplate = questionTemplate.SubquestionTemplateList.ElementAt(i);
                if(subquestionTemplate.ImageSource != null)
                {
                    DeleteSubquestionTemplateImage(webRootPath, subquestionTemplate.ImageSource);
                }
            }
            _context.QuestionTemplates.Remove(questionTemplate);
            await _context.SaveChangesAsync();
            string message = "Otázka byla úspěšně smazána.";
            return message;
        }

        public SubquestionTemplate GetSubquestionTemplate(string questionNumberIdentifier, string subquestionIdentifier, string login)
        {
            return GetSubquestionTemplateDbSet()
                .First(s => s.QuestionNumberIdentifier == questionNumberIdentifier && s.SubquestionIdentifier == subquestionIdentifier && s.OwnerLogin == login);
        }

        public async Task<string> AddSubquestionTemplate(SubquestionTemplate subquestionTemplate)
        {
            string message;
            try
            {
                QuestionTemplate questionTemplate = GetQuestionTemplate(subquestionTemplate.OwnerLogin, subquestionTemplate.QuestionNumberIdentifier);
                ICollection<SubquestionTemplate> subquestionTemplates = questionTemplate.SubquestionTemplateList;
                subquestionTemplates.Add(subquestionTemplate);
                await _context.SaveChangesAsync();
                message = "Zadání podotázky bylo úspěšně přidáno.";
            }
            catch (Exception ex)
            {
                Debug.WriteLine(ex.Message);
                message = "Při přidání podotázky nastala neočekávaná chyba.";
            }
            return message;
        }

        public async Task<string> DeleteSubquestionTemplate(string login, string questionNumberIdentifier, string subquestionIdentifier, string webRootPath)
        {
            SubquestionTemplate subquestionTemplate = GetSubquestionTemplate(questionNumberIdentifier, subquestionIdentifier, login);
            _context.SubquestionTemplates.Remove(subquestionTemplate);
            await _context.SaveChangesAsync();
            if(subquestionTemplate.ImageSource != null)
            {
                DeleteSubquestionTemplateImage(webRootPath, subquestionTemplate.ImageSource);
            }
            
            string message = "Podotázka byla úspěšně smazána.";
            return message;
        }

        public void DeleteSubquestionTemplateImage(string webRootPath, string imageSource)
        {
            string uploadsFolder = Path.Combine(webRootPath, "Uploads/");
            File.Delete(uploadsFolder + imageSource);
        }

        public async Task AddSubquestionTemplateStatistics(SubquestionTemplateStatistics subquestionTemplateStatistics)
        {
            _context.SubquestionTemplateStatistics.Add(subquestionTemplateStatistics);
            AttachUser(subquestionTemplateStatistics.User);
            await _context.SaveChangesAsync();
        }

        //ResultFunctions.cs

        public DbSet<TestResult> GetTestResultDbSet()
        {
            return _context.TestResults;
        }

        public DbSet<QuestionResult> GetQuestionResultDbSet()
        {
            return _context.QuestionResults;
        }

        public DbSet<SubquestionResult> GetSubquestionResultDbSet()
        {
            return _context.SubquestionResults;
        }

        public DbSet<SubquestionResultStatistics> GetSubquestionResultStatisticsDbSet()
        {
            return _context.SubquestionResultStatistics;
        }

        public SubquestionResultStatistics? GetSubquestionResultStatistics(string login)
        {
            return GetSubquestionResultStatisticsDbSet().FirstOrDefault(s => s.UserLogin == login);
        }

        public DbSet<TestDifficultyStatistics> GetTestDifficultyStatisticsDbSet()
        {
            return _context.TestDifficultyStatistics;
        }

        public async Task<string> AddTestResults(List<TestResult> testResults)
        {
            int successCount = 0;
            int errorCount = 0;
            for (int i = 0; i < testResults.Count; i++)
            {
                _context.ChangeTracker.Clear();
                try
                {
                    TestResult testResult = testResults[i];
                    _context.ChangeTracker.Clear();
                    _context.Students.Attach(testResult.Student);
                    _context.TestTemplates.Attach(testResult.TestTemplate);
                    _context.TestResults.Add(testResult);
                    await _context.SaveChangesAsync();
                    successCount++;
                }
                catch (Exception ex)
                {
                    Debug.WriteLine(ex.Message);
                    errorCount++;
                }
            }

            string message = "Přidáno " + successCount + " řešení testů (" + errorCount + " duplikátů nebo chyb).";
            return message;
        }

        public async Task<string> DeleteTestResults(string login)
        {
            var testResultList = GetTestResultDbSet().Where(t => t.OwnerLogin == login);
            foreach (TestResult testResult in testResultList)
            {
                _context.TestResults.Remove(testResult);
            }
            await _context.SaveChangesAsync();
            string message = "Byly smazány všechna vaše řešení testů.";
            return message;
        }

        public async Task<string> DeleteTestResult(TestResult testResult)
        {
            _context.TestResults.Remove(testResult);
            await _context.SaveChangesAsync();
            string message = "Test byl úspěšně smazán.";
            return message;
        }

        public SubquestionResult GetSubquestionResult(string testResultIdentifier, string questionNumberIdentifier, string subquestionIdentifier, string login)
        {
            SubquestionResult subquestionResult = GetSubquestionResultDbSet()
                /*     .Include(s => s.QuestionResult)
                     .Include(s => s.QuestionResult.TestResult)
                     .Include(s => s.QuestionResult.TestResult.TestTemplate)*/
                .First(s => s.TestResultIdentifier == testResultIdentifier && s.QuestionNumberIdentifier == questionNumberIdentifier
                && s.SubquestionIdentifier == subquestionIdentifier && s.OwnerLogin == login);
            return subquestionResult;
        }

        public List<SubquestionResult> GetSubquestionResults(string questionNumberIdentifier, string subquestionIdentifier, string login)
        {
            List<SubquestionResult> subquestionResults = GetSubquestionResultDbSet()
                    .Include(s => s.SubquestionTemplate)
                    .Where(s => s.QuestionNumberIdentifier == questionNumberIdentifier
                    && s.SubquestionIdentifier == subquestionIdentifier && s.OwnerLogin == login).ToList();
            return subquestionResults;
        }

        public List<TestResult> GetTestResultsByLogin(string login)
        {
            List<TestResult> testResults = GetTestResultDbSet()
                .Include(s => s.QuestionResultList)
                .ThenInclude(s => s.SubquestionResultList)
                .Include(s => s.TestTemplate)
                .ThenInclude(s => s.QuestionTemplateList)
                .ThenInclude(s => s.SubquestionTemplateList)
                .Where(s => s.OwnerLogin == login).ToList();
            return testResults;
        }

        public async Task SaveSubquestionResultRecords(List<SubquestionResultRecord> subquestionResultRecords, User owner)
        {
            for (int i = 0; i < subquestionResultRecords.Count; i++)
            {
                try
                {
                    var subquestionResultRecord = subquestionResultRecords[i];
                    _context.Users.Attach(owner);
                    _context.SubquestionResultRecords.Add(subquestionResultRecord);
                    await _context.SaveChangesAsync();
                }

                catch (Exception ex)
                {
                    Debug.WriteLine(ex.Message);
                }
            }
        }

        public async Task AddSubquestionResultStatistics(SubquestionResultStatistics subquestionResultStatistics)
        {
            _context.SubquestionResultStatistics.Add(subquestionResultStatistics);
            AttachUser(subquestionResultStatistics.User);
            await _context.SaveChangesAsync();
        }

        public async Task AddTestDifficultyStatistics(TestDifficultyStatistics testDifficultyStatistics)
        {
            _context.TestDifficultyStatistics.Add(testDifficultyStatistics);
            AttachUser(testDifficultyStatistics.User);
            await _context.SaveChangesAsync();
        }

        //UserFunctions.cs

        public DbSet<User> GetUserDbSet()
        {
            return _context.Users;
        }

        public List<User> GetUserList()
        {
            return GetUserDbSet().ToList();
        }

        public User? GetUserByLogin(string login)
        {
            return GetUserDbSet().FirstOrDefault(u => u.Login == login);
        }

        public User? GetUserByLoginAsNoTracking()
        {
            return GetUserDbSet().AsNoTracking().FirstOrDefault(u => u.Login == "login");
        }

        public DbSet<Student> GetStudentDbSet()
        {
            return _context.Students;
        }

        public Student? GetStudentByLogin(string login)
        {
            return GetStudentDbSet().FirstOrDefault(s => s.Login == login);
        }

        public Student? GetStudentByIdentifier(string studentIdentifier)
        {
            return GetStudentDbSet().FirstOrDefault(s => s.StudentIdentifier == studentIdentifier);
        }

        public DbSet<UserRegistration> GetUserRegistrationDbSet()
        {
            return _context.UserRegistrations;
        }

        public async Task AddUser(User user)
        {
            _context.Users.Add(user);
            await _context.SaveChangesAsync();
        }

        public async Task AddUserRegistration(UserRegistration userRegistration)
        {
            _context.UserRegistrations.Add(userRegistration);
            await _context.SaveChangesAsync();
        }

        public async Task<string> AddStudents(string login, List<Student> students)
        {
            string message = string.Empty;
            int successCount = 0;
            int errorCount = 0;
            for (int i = 0; i < students.Count; i++)
            {
                _context.ChangeTracker.Clear();
                try
                {
                    Student student = students[i];
                    var userLoginCheck = GetUserByLogin(login);
                    if (userLoginCheck != null)
                    {
                        throw Exceptions.UserAlreadyExistsException(login);
                    }
                    _context.Students.Add(student);
                    //in case the student has registered before the student's file has been imported, we add the Student to the UserRegistration
                    var userRegistrationList = _context.UserRegistrations.Where(u => u.Login == student.Login && u.State == EnumTypes.RegistrationState.Waiting);
                    foreach (UserRegistration userRegistration in userRegistrationList)
                    {
                        userRegistration.Student = student;
                    }
                    await _context.SaveChangesAsync();
                    successCount++;
                }
                catch (Exception ex)
                {
                    Debug.WriteLine(ex.Message);
                    errorCount++;
                }
            }
            message = "Přidáno " + successCount + " studentů (" + errorCount + " duplikátů nebo chyb).";
            return message;
        }

        public async Task<string> AddStudent(Student student)
        {
            _context.Students.Add(student);
            await _context.SaveChangesAsync();
            return "Student s loginem \"" + student.Login + "\" úspěšně přidán.";
        }

        public async Task<string> AddEmailStudent(Student student)
        {
            _context.Students.Add(student);
            await _context.SaveChangesAsync();
            return "Student s loginem \"" + student.Login + "\" úspěšně přidán.";
        }

        public async Task DeleteStudent(Student student)
        {
            _context.Students.Remove(student);
            await _context.SaveChangesAsync();
        }

        public async Task DeleteUser(User user)
        {
            _context.Users.Remove(user);
            await _context.SaveChangesAsync();
        }

        public async Task DeleteRegistration(UserRegistration userRegistration)
        {
            _context.UserRegistrations.Remove(userRegistration);
            await _context.SaveChangesAsync();
        }

        //OtherFunctions.cs

        public DbSet<GlobalSettings> GetGlobalSettingsDbSet()
        {
            return _context.GlobalSettings;
        }

        public GlobalSettings GetGlobalSettings()
        {
            if(_context.GlobalSettings.First() != null)
            {
                return _context.GlobalSettings.First();
            }
            else
            {
                //todo: throw exception globalni nastaveni nenalezena
                return null;
            }
        }

        //General functions

        public async Task SaveChangesAsync()
        {
            await _context.SaveChangesAsync();
        }

        public void ExecuteSqlRaw(string command)
        {
            _context.Database.ExecuteSqlRaw(command);
        }

        public void ClearChargeTracker()
        {
            _context.ChangeTracker.Clear();
        }

        public void AttachUser(User user)
        {
            _context.Users.Attach(user);
        }
    }
}
