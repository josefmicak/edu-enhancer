using Common;
using DataLayer;
using DomainModel;
using Microsoft.EntityFrameworkCore;
using System.Diagnostics;

namespace BusinessLayer
{
    /// <summary>
    /// Functions related to users (entities: User, Student, UserRegistration)
    /// </summary>
    public class UserFunctions
    {
        private DataFunctions dataFunctions;

        public UserFunctions(CourseContext context)
        {
            dataFunctions = new DataFunctions(context);
        }

        public DbSet<User> GetUserDbSet()
        {
            return dataFunctions.GetUserDbSet();
        }

        /// <summary>
        /// Returns user by login - throws exception if user doesn't exist
        /// </summary>
        public async Task<User> GetUserByLogin(string login)
        {
            return await dataFunctions.GetUserByLogin(login);
        }

        /// <summary>
        /// Returns user by login - doesn't throw an exception if user doesn't exist
        /// </summary>
        public async Task<User?> GetUserByLoginNullable(string login)
        {
            return await dataFunctions.GetUserByLoginNullable(login);
        }

        /// <summary>
        /// Returns user by email - doesn't throw an exception if user doesn't exist
        /// </summary>
        public async Task<User?> GetUserByEmailNullable(string email)
        {
            return await GetUserDbSet().FirstOrDefaultAsync(u => u.Email == email);
        }

        public DbSet<Student> GetStudentDbSet()
        {
            return dataFunctions.GetStudentDbSet();
        }

        /// <summary>
        /// Returns student by login - throws exception if student doesn't exist
        /// </summary>
        public async Task<Student> GetStudentByLogin(string login)
        {
            Student? student = await GetStudentDbSet().Include(s => s.Subjects).FirstOrDefaultAsync(s => s.Login == login);
            if(student == null)
            {
                throw Exceptions.StudentNotFoundException(login);
            }
            return student;
        }

        /// <summary>
        /// Returns student by login - doesn't throw an exception if student doesn't exist
        /// </summary>
        public async Task<Student?> GetStudentByLoginNullable(string login)
        {
            return await GetStudentDbSet().Include(s => s.Subjects).FirstOrDefaultAsync(s => s.Login == login);;
        }

        /// <summary>
        /// Returns student by email - doesn't throw an exception if user doesn't exist
        /// </summary>
        public async Task<Student?> GetStudentByEmailNullable(string email)
        {
            return await GetStudentDbSet().FirstOrDefaultAsync(s => s.Email == email);
        }

        /// <summary>
        /// Returns main admin - doesn't throw an exception if main admin doesn't exist
        /// </summary>
        public async Task<User?> GetMainAdmin()
        {
            return await GetUserDbSet().FirstOrDefaultAsync(u => u.Role == EnumTypes.Role.MainAdmin);
        }

        public DbSet<UserRegistration> GetUserRegistrationDbSet()
        {
            return dataFunctions.GetUserRegistrationDbSet();
        }

        /// <summary>
        /// Returns all registrations made by the email address
        /// </summary>
        public async Task<List<UserRegistration>> GetUserRegistrations(string email)
        {
            return await GetUserRegistrationDbSet()
                .Where(u => u.Email == email).ToListAsync();
        }

        /// <summary>
        /// Returns the registration made by the email address (in case one exists)
        /// </summary>
        public async Task<UserRegistration?> GetUserRegistration(string email)
        {
            return await GetUserRegistrationDbSet()
                .FirstOrDefaultAsync(u => u.Email == email);
        }

        /// <summary>
        /// Creates a main admin account
        /// Function is called after the very first user logs in after the application is launched
        /// </summary>
        public async Task RegisterMainAdmin(string firstName, string lastName, string email, string login)
        {
            User mainAdmin = new User();
            mainAdmin.FirstName = firstName;
            mainAdmin.LastName = lastName;
            mainAdmin.Email = email;
            mainAdmin.Login = login;
            mainAdmin.Role = EnumTypes.Role.MainAdmin;
            Config.Application["login"] = login;
            await dataFunctions.AddUser(mainAdmin);

            SubquestionTemplateStatistics subquestionTemplateStatistics = new SubquestionTemplateStatistics();
            subquestionTemplateStatistics.User = mainAdmin;
            subquestionTemplateStatistics.UserLogin = mainAdmin.Login;
            await dataFunctions.AddSubquestionTemplateStatistics(subquestionTemplateStatistics);

            SubquestionResultStatistics subquestionResultStatistics = new SubquestionResultStatistics();
            subquestionResultStatistics.User = mainAdmin;
            subquestionResultStatistics.UserLogin = mainAdmin.Login;
            await dataFunctions.AddSubquestionResultStatistics(subquestionResultStatistics);
        }

        /// <summary>
        /// Creates an user registration, only one registration per user is permitted
        /// </summary>
        /// <param name="role">Role of the user (student/teacher/admin/main admin)</param>
        public async Task<string> CreateUserRegistration(string firstName, string lastName, string email, string login, string role)
        {
            string message;
            var user = await GetUserRegistration(email);
            if (user != null)
            {
                message = "Chyba: již jste zaregistrován. Nyní je nutné vyčkat na potvrzení registrace správcem.";
            }
            else if (firstName == null || lastName == null || login == null || email == null)
            {
                message = "Chyba: všechny položky musí být vyplněny.";
            }
            else
            {
                try
                {
                    UserRegistration userRegistration = new UserRegistration();
                    userRegistration.FirstName = firstName;
                    userRegistration.LastName = lastName;
                    userRegistration.Login = login;
                    userRegistration.Email = email;
                    userRegistration.State = EnumTypes.RegistrationState.Waiting;
                    userRegistration.CreationDate = DateTime.Now;
                    userRegistration.Role = (EnumTypes.Role)Convert.ToInt32(role);

                    await dataFunctions.AddUserRegistration(userRegistration);
                    message = "Registrace úspěšně vytvořena. Nyní je nutné vyčkat na potvrzení registrace správcem.";
                }
                catch (Exception ex)
                {
                    Debug.WriteLine(ex.Message);
                    message = "Při registraci nastala chyba.";
                }
            }
            return message;
        }

        public async Task DeleteAllStudents()
        {
            dataFunctions.ExecuteSqlRaw("delete from Student");
            await dataFunctions.SaveChangesAsync();
        }

        public async Task<string> AddStudent(string firstName, string lastName, string login, string email)
        {
            Student student = new Student();
            student.Login = login;
            student.Email = email;
            student.FirstName = firstName;
            student.LastName = lastName;
            return await dataFunctions.AddStudent(student);
        }

        public async Task EditStudent(string firstName, string lastName, string login, string email, Student studentLoginCheck)
        {
            studentLoginCheck.Login = login;
            studentLoginCheck.Email = email;
            studentLoginCheck.FirstName = firstName;
            studentLoginCheck.LastName = lastName;
            await dataFunctions.SaveChangesAsync();
        }

        public async Task DeleteStudent(Student student)
        {
            await dataFunctions.DeleteStudent(student);
        }

        public async Task AddTeacher(string firstName, string lastName, string login, string email)
        {
            User teacher = new User();
            teacher.FirstName = firstName;
            teacher.LastName = lastName;
            teacher.Login = login;
            teacher.Email = email;
            teacher.Role = EnumTypes.Role.Teacher;
            await dataFunctions.AddUser(teacher);

            SubquestionTemplateStatistics subquestionTemplateStatistics = new SubquestionTemplateStatistics();
            subquestionTemplateStatistics.User = teacher;
            subquestionTemplateStatistics.UserLogin = teacher.Login;
            await dataFunctions.AddSubquestionTemplateStatistics(subquestionTemplateStatistics);

            SubquestionResultStatistics subquestionResultStatistics = new SubquestionResultStatistics();
            subquestionResultStatistics.User = teacher;
            subquestionResultStatistics.UserLogin = teacher.Login;
            await dataFunctions.AddSubquestionResultStatistics(subquestionResultStatistics);
        }

        public async Task EditUser(User user, string firstName, string lastName, string login, string email, string role)
        {
            user.Login = login;
            user.Email = email;
            user.FirstName = firstName;
            user.LastName = lastName;
            if (role != null)
            {
                user.Role = (EnumTypes.Role)Convert.ToInt32(role);
            }
            await dataFunctions.SaveChangesAsync();
        }

        public async Task DeleteUser(User user)
        {
            await dataFunctions.DeleteUser(user);
        }

        public async Task DeleteAllTeachers()
        {
            List<Subject> teacherSubjects = await dataFunctions.GetSubjectDbSet()
                .Include(s => s.Guarantor)
                .Where(s => s.Guarantor.Role == EnumTypes.Role.Teacher).ToListAsync();
            dataFunctions.GetSubjectDbSet().RemoveRange(teacherSubjects);
            await dataFunctions.SaveChangesAsync();
            dataFunctions.ExecuteSqlRaw("delete from [User] where role = 2");
            await dataFunctions.SaveChangesAsync();
        }

        public async Task DeleteAllAdmins()
        {
            List<Subject> adminSubjects = await dataFunctions.GetSubjectDbSet()
                .Include(s => s.Guarantor)
                .Where(s => s.Guarantor.Role == EnumTypes.Role.Admin).ToListAsync();
            dataFunctions.GetSubjectDbSet().RemoveRange(adminSubjects);
            await dataFunctions.SaveChangesAsync();
            dataFunctions.ExecuteSqlRaw("delete from [User] where role = 3");
            await dataFunctions.SaveChangesAsync();
        }

        public async Task AddAdmin(string firstName, string lastName, string login, string email)
        {
            User admin = new User();
            admin.FirstName = firstName;
            admin.LastName = lastName;
            admin.Login = login;
            admin.Email = email;
            admin.Role = EnumTypes.Role.Admin;
            await dataFunctions.AddUser(admin);

            SubquestionTemplateStatistics subquestionTemplateStatistics = new SubquestionTemplateStatistics();
            subquestionTemplateStatistics.User = admin;
            subquestionTemplateStatistics.UserLogin = admin.Login;
            await dataFunctions.AddSubquestionTemplateStatistics(subquestionTemplateStatistics);

            SubquestionResultStatistics subquestionResultStatistics = new SubquestionResultStatistics();
            subquestionResultStatistics.User = admin;
            subquestionResultStatistics.UserLogin = admin.Login;
            await dataFunctions.AddSubquestionResultStatistics(subquestionResultStatistics);
        }

        /// <summary>
        /// Changes the role of the selected account to main admin
        /// Only one main admin can exist in the application, so the old account's role is changed to regular admin
        /// </summary>
        public async Task ChangeMainAdmin(User newMainAdmin, string firstName, string lastName, string login, string email)
        {
            var oldMainAdmin = await GetMainAdmin();
            if (oldMainAdmin != null)
            {
                oldMainAdmin.Role = EnumTypes.Role.Admin;
            }

            newMainAdmin.Login = login;
            newMainAdmin.Email = email;
            newMainAdmin.FirstName = firstName;
            newMainAdmin.LastName = lastName;
            newMainAdmin.Role = EnumTypes.Role.MainAdmin;

            await dataFunctions.SaveChangesAsync();
        }

        public async Task<string> ApproveUserRegistration(string firstName, string lastName, string login, string email, string role)
        {
            string message;
            if(role == "Student")
            {
                Student student = new Student();
                student.Email = email;
                student.FirstName = firstName;
                student.LastName = lastName;
                student.Login = login;
                await dataFunctions.AddStudent(student);
                var userRegistration = await GetUserRegistration(email);
                if (userRegistration != null)
                {
                    userRegistration.State = EnumTypes.RegistrationState.Accepted;
                    await dataFunctions.SaveChangesAsync();
                    message = "Registrace úspěšně schválena.";
                }
                else
                {
                    message = "Chyba: registrace nebyla nalezena";
                }
            }
            else
            {
                User user = new User();
                user.Email = email;
                user.FirstName = firstName;
                user.LastName = lastName;
                user.Login = login;
                user.Role = Enum.Parse<EnumTypes.Role>(role);
                await dataFunctions.AddUser(user);
                var userRegistration = await GetUserRegistration(email);
                if (userRegistration != null)
                {
                    userRegistration.State = EnumTypes.RegistrationState.Accepted;
                    await dataFunctions.SaveChangesAsync();
                    message = "Registrace úspěšně schválena.";

                    SubquestionTemplateStatistics subquestionTemplateStatistics = new SubquestionTemplateStatistics();
                    subquestionTemplateStatistics.User = user;
                    subquestionTemplateStatistics.UserLogin = user.Login;
                    await dataFunctions.AddSubquestionTemplateStatistics(subquestionTemplateStatistics);

                    SubquestionResultStatistics subquestionResultStatistics = new SubquestionResultStatistics();
                    subquestionResultStatistics.User = user;
                    subquestionResultStatistics.UserLogin = user.Login;
                    await dataFunctions.AddSubquestionResultStatistics(subquestionResultStatistics);
                }
                else
                {
                    message = "Chyba: registrace nebyla nalezena";
                }
            }

            return message;
        }

        public async Task<string> RefuseRegistration(string email)
        {
            string message;
            var userRegistration = await GetUserRegistration(email);
            if (userRegistration != null)
            {
                userRegistration.State = EnumTypes.RegistrationState.Rejected;
                await dataFunctions.SaveChangesAsync();
                message = "Registrace úspěšně zamítnuta.";
            }
            else
            {
                message = "Chyba: registrace nebyla nalezena";
            }
            return message;
        }

        public async Task<string> DeleteRegistration(string email)
        {
            string message;
            var userRegistration = await GetUserRegistration(email);
            if (userRegistration != null)
            {
                await dataFunctions.DeleteRegistration(userRegistration);
                message = "Registrace úspěšně zamítnuta.";
            }
            else
            {
                message = "Chyba: registrace nebyla nalezena";
            }
            return message;
        }

        public async Task DeleteAllRegistrations()
        {
            dataFunctions.ExecuteSqlRaw("delete from UserRegistration");
            await dataFunctions.SaveChangesAsync();
        }

        /// <summary>
        /// Checks whether the user can access the page (each page has a minimum required role to be accessed)
        /// </summary>
        /// <param name="requiredRole">Minimum required role, users with a lower role cannot access the page</param>
        public async Task<bool> CanUserAccessPage(EnumTypes.Role requiredRole)
        {
            if (Config.TestingMode)//in case the testing mode is on, no authentication is required at all
            {
                return true;
            }

            string login = Config.Application["login"];
            var user = await GetUserByLoginNullable(login);
            var student = await GetStudentByLoginNullable(login);

            if (requiredRole > EnumTypes.Role.Student)//staff member
            {
                if (user == null)
                {
                    return false;
                }
                if (user.Role < requiredRole)
                {
                    return false;
                }
                return true;
            }
            else//student
            {
                if (student == null)
                {
                    return false;
                }
                return true;
            }
        }

        /// <summary>
        /// Checks whether the student can access the test (in regards to test subject and start/end dates of the test)
        /// </summary>
        /// <param name="student">Student trying to access the test</param>
        /// <param name="testTemplate">Test that the student is trying to access</param>
        public string? CanStudentAccessTest(Student student, TestTemplate testTemplate)
        {
            string? errorMessage = null;
            if (!student.Subjects.Contains(testTemplate.Subject))
            {
                errorMessage = "Chyba: nejste přihlášen k tomuto předmětu.";
            }
            if (!testTemplate.IsTestingData)
            {
                if (testTemplate.StartDate > DateTime.Now)
                {
                    errorMessage = "Chyba: tento test můžete vyplnit až po " + testTemplate.StartDate;
                }
                if (testTemplate.EndDate < DateTime.Now)
                {
                    errorMessage = "Chyba: tento test bylo možné vyplnit pouze do " + testTemplate.EndDate;
                }
            }
            return errorMessage;
        }
    }
}