using Common;
using DomainModel;
using Microsoft.AspNetCore.Http;
using Microsoft.EntityFrameworkCore;
using System.Diagnostics;

namespace DataLayer
{
    /// <summary>
    /// The primary place in the application that handles direct work with data stored in the database
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

        public DbSet<Subject> GetSubjectDbSet()
        {
            return _context.Subjects;
        }

        public async Task<string> AddSubject(Subject subject)
        {
            string message;
            try
            {
                _context.Users.Attach(subject.Guarantor);
                _context.Subjects.Add(subject);
                await _context.SaveChangesAsync();
                message = "Předmět byl úspěšně přidán.";
            }
            catch (Exception ex)
            {
                Debug.WriteLine(ex.Message);
                message = "Při přidání předmětu nastala neočekávaná chyba.";

            }
            _context.ChangeTracker.Clear();
            return message;
        }

        public async Task<string> DeleteSubject(Subject subject)
        {
            string message;
            try
            {
                _context.Subjects.Remove(subject);
                await _context.SaveChangesAsync();
                message = "Předmět byl úspěšně smazán.";
            }
            catch (Exception ex)
            {
                Debug.WriteLine(ex.Message);
                message = "Při mazání předmětu nastala neočekávaná chyba.";

            }
            return message;
        }

        /*public async Task DeleteSubjects(List<string> adminLogins)
        {
            string message;
            try
            {
                _context.Subjects.Remove(subject);
                await _context.SaveChangesAsync();
                message = "Předmět byl úspěšně smazán.";
            }
            catch (Exception ex)
            {
                Debug.WriteLine(ex.Message);
                message = "Při mazání předmětu nastala neočekávaná chyba.";

            }
            return message;
        }*/

        public async Task<List<Subject>> GetTestingDataSubjects()
        {
            return await GetSubjectDbSet().AsNoTracking().Where(s => s.IsTestingData == true).ToListAsync();
        }

        public async Task<List<TestTemplate>> GetTestTemplateList(string login)
        {
            return await GetTestTemplateDbSet()
                .Include(t => t.Subject)
                .Include(t => t.QuestionTemplates)
                .ThenInclude(q => q.SubquestionTemplates)
                .Where(t => t.OwnerLogin == login).ToListAsync();
        }

        public async Task<string> AddTestTemplates(List<TestTemplate> testTemplates)
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
                    _context.Subjects.Attach(testTemplate.Subject);
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

        public async Task<string> EditTestTemplate(TestTemplate testTemplate)
        {
            string message;
            try
            {
                TestTemplate oldTestTemplate = await GetTestTemplate(testTemplate.TestTemplateId);
                oldTestTemplate.Title = testTemplate.Title;
                oldTestTemplate.Subject = testTemplate.Subject;
                oldTestTemplate.MinimumPoints = testTemplate.MinimumPoints;
                oldTestTemplate.NegativePoints = testTemplate.NegativePoints;
                await _context.SaveChangesAsync();
                message = "Zadání testu bylo úspěšně upraveno.";
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
            if(testTemplate.QuestionTemplates != null)
            {
                for (int i = 0; i < testTemplate.QuestionTemplates.Count; i++)
                {
                    QuestionTemplate questionTemplate = testTemplate.QuestionTemplates.ElementAt(i);
                    for (int j = 0; j < questionTemplate.SubquestionTemplates.Count; j++)
                    {
                        SubquestionTemplate subquestionTemplate = questionTemplate.SubquestionTemplates.ElementAt(j);
                        if (subquestionTemplate.ImageSource != null)
                        {
                            DeleteSubquestionTemplateImage(webRootPath, subquestionTemplate.ImageSource);
                        }
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

        public async Task<TestTemplate> GetTestTemplate(int testTemplateId)
        {
            return await GetTestTemplateDbSet()
                .Include(t => t.QuestionTemplates)
                .FirstAsync(t => t.TestTemplateId == testTemplateId);
        }

        public async Task<QuestionTemplate> GetQuestionTemplate(int questionTemplateId)
        {
            return await GetQuestionTemplateDbSet()
                .Include(q => q.TestTemplate)
                .Include(q => q.SubquestionTemplates)
                .FirstAsync(q => q.QuestionTemplateId == questionTemplateId);
        }

        public async Task<string> AddQuestionTemplate(QuestionTemplate questionTemplate)
        {
            string message;
            try
            {
                TestTemplate testTemplate = questionTemplate.TestTemplate;
                ICollection<QuestionTemplate> questionTemplates = testTemplate.QuestionTemplates;
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

        public async Task<string> EditQuestionTemplate(QuestionTemplate questionTemplate)
        {
            string message;
            try
            {
                QuestionTemplate oldQuestionTemplate = await GetQuestionTemplate(questionTemplate.QuestionTemplateId);
                oldQuestionTemplate.Title = questionTemplate.Title;
                await _context.SaveChangesAsync();
                message = "Zadání otázky bylo úspěšně upraveno.";
            }
            catch (Exception ex)
            {
                Debug.WriteLine(ex.Message);
                message = "Při úpravě otázky nastala neočekávaná chyba.";
            }
            return message;
        }

        public async Task<string> DeleteQuestionTemplate(int questionTemplateId, string webRootPath)
        {
            QuestionTemplate questionTemplate = await GetQuestionTemplate(questionTemplateId);
            if(questionTemplate.SubquestionTemplates != null)
            {
                for (int i = 0; i < questionTemplate.SubquestionTemplates.Count; i++)
                {
                    SubquestionTemplate subquestionTemplate = questionTemplate.SubquestionTemplates.ElementAt(i);
                    if (subquestionTemplate.ImageSource != null)
                    {
                        DeleteSubquestionTemplateImage(webRootPath, subquestionTemplate.ImageSource);
                    }
                }
            }
            _context.QuestionTemplates.Remove(questionTemplate);
            await _context.SaveChangesAsync();
            string message = "Otázka byla úspěšně smazána.";
            return message;
        }

        public async Task<SubquestionTemplate> GetSubquestionTemplate(int subquestionTemplateId)
        {
            return await GetSubquestionTemplateDbSet()
                .FirstAsync(s => s.SubquestionTemplateId == subquestionTemplateId);
        }

        public async Task<string> AddSubquestionTemplate(SubquestionTemplate subquestionTemplate, IFormFile? image, string webRootPath)
        {
            string message;
            try
            {
                await IncrementSubquestionTemplateStatistics(subquestionTemplate.OwnerLogin);
                QuestionTemplate questionTemplate = await GetQuestionTemplate(subquestionTemplate.QuestionTemplateId);
                ICollection<SubquestionTemplate> subquestionTemplates = questionTemplate.SubquestionTemplates;
                subquestionTemplates.Add(subquestionTemplate);
                await _context.SaveChangesAsync();
                message = "Zadání podotázky bylo úspěšně přidáno.";

                //only save image in case one has been uploaded, and all validity checks have already been passed
                if(image != null)
                {
                    subquestionTemplate.ImageSource = SaveImage(image, webRootPath, null);
                    await _context.SaveChangesAsync();
                }
            }
            catch (Exception ex)
            {
                Debug.WriteLine(ex.Message);
                message = "Při přidání podotázky nastala neočekávaná chyba.";
            }
            return message;
        }

        /// <summary>
        /// Saves image to the wwwroot/Uploads folder
        /// </summary>
        /// <param name="image">Image to be saved</param>
        /// <param name="webRootPath">Path to the wwwroot folder where all images are to be saved</param>
        /// <param name="newFileName">Indicates whether file name should be changed (in case the subquestion is being edited, it's done in advance)</param>
        public string SaveImage(IFormFile image, string webRootPath, string? newFileName)
        {
            string uploadsFolder = Path.Combine(webRootPath, "Uploads");
            if (!Directory.Exists(uploadsFolder))
            {
                Directory.CreateDirectory(uploadsFolder);
            }
           
            string newFilePath;
            if (newFileName == null)
            {
                newFileName = Guid.NewGuid().ToString() + "_" + image.FileName;
            }
            newFilePath = Path.Combine(uploadsFolder, newFileName);
            using (var fileStream = new FileStream(newFilePath, FileMode.Create))
            {
                image.CopyTo(fileStream);
            }

            return newFileName;
        }

        public async Task<string> DeleteSubquestionTemplate(int subquestionTemplateId, string webRootPath)
        {
            string message;
            try
            {
                await DeleteSubquestionResults(subquestionTemplateId);
                SubquestionTemplate subquestionTemplate = await GetSubquestionTemplate(subquestionTemplateId);
                _context.SubquestionTemplates.Remove(subquestionTemplate);
                await _context.SaveChangesAsync();
                if (subquestionTemplate.ImageSource != null)
                {
                    DeleteSubquestionTemplateImage(webRootPath, subquestionTemplate.ImageSource);
                }
                message = "Podotázka byla úspěšně smazána.";
            }
            catch(Exception ex)
            {
                Debug.WriteLine(ex.Message);
                message = "Při mazání podotázky nastala neočekávaná chyba.";
            }
            return message;
        }

        public void DeleteSubquestionTemplateImage(string webRootPath, string imageSource)
        {
            //TestingImage doesn't get deleted (all testing data subquestion templates share this image)
            if(imageSource != "TestingImage.png")
            {
                string uploadsFolder = Path.Combine(webRootPath, "Uploads/");
                File.Delete(uploadsFolder + imageSource);
            }
        }

        public async Task AddSubquestionTemplateStatistics(SubquestionTemplateStatistics subquestionTemplateStatistics)
        {
            _context.SubquestionTemplateStatistics.Add(subquestionTemplateStatistics);
            AttachUser(subquestionTemplateStatistics.User);
            await _context.SaveChangesAsync();
        }

        /// <summary>
        /// After a subquestion template is added, teacher's subquestion template statistics' field is incremented by one
        /// </summary>
        /// <param name="login">Teacher's login</param>
        public async Task IncrementSubquestionTemplateStatistics(string login)
        {
            SubquestionTemplateStatistics subquestionTemplateStatistics = await GetSubquestionTemplateStatisticsDbSet().FirstAsync(s => s.UserLogin == login);
            subquestionTemplateStatistics.SubquestionTemplatesAddedCount += 1;
            if(subquestionTemplateStatistics.SubquestionTemplatesAddedCount == 100)
            {
                subquestionTemplateStatistics.EnoughSubquestionTemplatesAdded = true;
            }
            await SaveChangesAsync();
        }

        public int GetTestTemplateSubquestionsCount(TestTemplate testTemplate)
        {
            int subquestionsCount = 0;
            for (int i = 0; i < testTemplate.QuestionTemplates.Count; i++)
            {
                QuestionTemplate questionTemplate = testTemplate.QuestionTemplates.ElementAt(i);

                for(int j = 0; j < questionTemplate.SubquestionTemplates.Count; j++)
                {
                    subquestionsCount++;
                }
            }
            return subquestionsCount;
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

        public async Task<SubquestionResultStatistics?> GetSubquestionResultStatisticsNullable(string login)
        {
            return await GetSubquestionResultStatisticsDbSet().FirstOrDefaultAsync(s => s.UserLogin == login);
        }

        public DbSet<TestDifficultyStatistics> GetTestDifficultyStatisticsDbSet()
        {
            return _context.TestDifficultyStatistics;
        }

        public async Task<TestDifficultyStatistics?> GetTestDifficultyStatisticsNullable(string login)
        {
            return await GetTestDifficultyStatisticsDbSet().FirstOrDefaultAsync(s => s.UserLogin == login);
        }

        public async Task<TestResult> GetTestResult(int testResultId)
        {
            return await GetTestResultDbSet()
                .Include(t => t.QuestionResults)
                .ThenInclude(q => q.SubquestionResults)
                .Include(t => t.TestTemplate)
                .ThenInclude(t => t.QuestionTemplates)
                .ThenInclude(t => t.SubquestionTemplates)
                .Include(t => t.TestTemplate)
                .ThenInclude(t => t.Subject)
                .Include(t => t.TestTemplate)
                .ThenInclude(t => t.Owner)
                .FirstAsync(t => t.TestResultId == testResultId);
        }

        public async Task<string?> AddTestResult(TestResult testResult)
        {
            try
            {
                _context.ChangeTracker.Clear();
                _context.Students.Attach(testResult.Student);
                _context.TestTemplates.Attach(testResult.TestTemplate);
                _context.TestResults.Add(testResult);
                await IncrementSubquestionResultStatistics(testResult.OwnerLogin, GetTestTemplateSubquestionsCount(testResult.TestTemplate));
                await _context.SaveChangesAsync();
                return null;
            }
            catch (Exception ex)
            {
                Debug.WriteLine(ex.Message);
                return "Při načítání testu došlo k nečekané chybě";
            }
        }

        public async Task<TestResult> LoadLastStudentAttempt(Student student)
        {
            return await GetTestResultDbSet()
                .Include(t => t.QuestionResults)
                .ThenInclude(q => q.SubquestionResults)
                .Include(t => t.TestTemplate)
                .ThenInclude(q => q.QuestionTemplates)
                .ThenInclude(q => q.SubquestionTemplates)
                .OrderByDescending(t => t.TimeStamp)
                .FirstAsync(t => t.StudentLogin == student.Login);
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

        public async Task<List<TestResult>> GetTestResultsByLogin(string login)
        {
            return await GetTestResultDbSet()
                .Include(s => s.QuestionResults)
                .ThenInclude(s => s.SubquestionResults)
                .Include(s => s.TestTemplate)
                .ThenInclude(s => s.QuestionTemplates)
                .ThenInclude(s => s.SubquestionTemplates)
                .Where(s => s.OwnerLogin == login).ToListAsync();
        }

        public async Task DeleteQuestionResults(int questionTemplateId)
        {
            IQueryable<QuestionResult> questionResults = GetQuestionResultDbSet().Where(q => q.QuestionTemplateId == questionTemplateId);
            _context.QuestionResults.RemoveRange(questionResults);
            await _context.SaveChangesAsync();
        }

        public async Task DeleteSubquestionResults(int subquestionTemplateId)
        {
            IQueryable<SubquestionResult> subquestionResults = GetSubquestionResultDbSet().Where(s => s.SubquestionTemplateId == subquestionTemplateId);
            _context.SubquestionResults.RemoveRange(subquestionResults);
            await _context.SaveChangesAsync();
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

        /// <summary>
        /// After a number of subquestion result is added, teacher's subquestion result statistics' field is incremented by this number
        /// </summary>
        /// <param name="login">Teacher's login</param>
        /// <param name="subquestionsCount">Amount of added subquestion results</param>
        public async Task IncrementSubquestionResultStatistics(string login, int subquestionsCount)
        {
            SubquestionResultStatistics subquestionResultStatistics = await GetSubquestionResultStatisticsDbSet().FirstAsync(s => s.UserLogin == login);
            subquestionResultStatistics.SubquestionResultsAddedCount += subquestionsCount;
            if (subquestionResultStatistics.SubquestionResultsAddedCount >= 100)
            {
                subquestionResultStatistics.EnoughSubquestionResultsAdded = true;
            }
            await SaveChangesAsync();
        }

        //UserFunctions.cs

        public DbSet<User> GetUserDbSet()
        {
            return _context.Users;
        }

        public async Task<User> GetUserByLogin(string login)
        {
            User? user = await GetUserDbSet().FirstOrDefaultAsync(u => u.Login == login);
            if(user == null)
            {
                throw Exceptions.UserNotFoundException(login);
            }
            return user;
        }

        public async Task<User?> GetUserByLoginNullable(string login)
        {
            return await GetUserDbSet().FirstOrDefaultAsync(u => u.Login == login);
        }

        public async Task<User> GetUserByLoginAsNoTracking()
        {
            User? user = await GetUserDbSet().AsNoTracking().FirstOrDefaultAsync(u => u.Login == "login");
            if (user == null)
            {
                throw Exceptions.UserNotFoundException("login");
            }
            return user;
        }

        public DbSet<Student> GetStudentDbSet()
        {
            return _context.Students;
        }

        public async Task<Student> GetStudentByLogin(string login)
        {
            Student? student = await GetStudentDbSet().FirstOrDefaultAsync(u => u.Login == login);
            if(student == null)
            {
                throw Exceptions.StudentNotFoundException(login);
            }
            return student;
        }

        public async Task<Student?> GetStudentByLoginNullable(string login)
        {
            return await GetStudentDbSet().FirstOrDefaultAsync(u => u.Login == login);
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

        public async Task<string> AddStudent(Student student)
        {
            _context.Students.Add(student);
            await _context.SaveChangesAsync();
            return "Student s loginem \"" + student.Login + "\" úspěšně přidán.";
        }

        /// <summary>
        /// After subquestion results marked as testing data are added, the student marked as testing data is re-assigned to all subjects marked as testing data
        /// </summary>
        public async Task RefreshTestingStudentSubjects()
        {
            ExecuteSqlRaw("delete from StudentSubject");
            Student student = await GetStudentByLogin("testingstudent");
            List<Subject> subjects = await GetSubjectDbSet().Where(s => s.IsTestingData == true).ToListAsync();
            student.Subjects = subjects;
            await _context.SaveChangesAsync();
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
                throw Exceptions.GlobalSettingsNotFound;
            }
        }

        //General functions

        public async Task SaveChangesAsync()
        {
            await _context.SaveChangesAsync();
        }

        /// <summary>
        /// Directly executes the entered SQl command
        /// </summary>
        public void ExecuteSqlRaw(string command)
        {
            _context.Database.ExecuteSqlRaw(command);
        }

        /// <summary>
        /// Clears any tracked changes in EF Core
        /// </summary>
        public void ClearChargeTracker()
        {
            _context.ChangeTracker.Clear();
        }

        /// <summary>
        /// Attaches an existing entity (in case one exists)
        /// </summary>
        public void AttachUser(User user)
        {
            if(user != null)
            {
                _context.Users.Attach(user);
            }
        }
    }
}
