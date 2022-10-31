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

        public async Task<string> AddTestTemplates(List<TestTemplate> testTemplates, User? owner, bool attachOwner)
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
                    if(attachOwner)
                    {
                        if(owner != null)
                        {
                            _context.Users.Attach(testTemplate.Owner);
                        }
                    }
                    else
                    {
                        _context.Users.Attach(testTemplate.Owner);
                    }
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

        public async Task<string> DeleteTestTemplate(TestTemplate testTemplate)
        {
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

        public TestTemplate GetTestTemplate(string login, string testNameIdentifier)
        {
            return GetTestTemplateDbSet()
                .Include(t => t.QuestionTemplateList)
                .First(t => t.TestNameIdentifier == testNameIdentifier && t.OwnerLogin == login);
        }

        public QuestionTemplate GetQuestionTemplate(string login, string questionNumberIdentifier)
        {
            return GetQuestionTemplateDbSet()
                .Include(q => q.SubquestionTemplateList)
                .First(q => q.QuestionNumberIdentifier == questionNumberIdentifier && q.OwnerLogin == login);
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
