using Common;
using DataLayer;
using DomainModel;
using Microsoft.EntityFrameworkCore;
using static Common.EnumTypes;
using System.Diagnostics;
using System.Xml;

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

        public List<User> GetUserList()
        {
            return dataFunctions.GetUserList();
        }

        public User? GetUserByLogin(string login)
        {
            return dataFunctions.GetUserByLogin(login);
        }

        public User? GetUserByEmail(string email)
        {
            return GetUserDbSet().FirstOrDefault(u => u.Email == email);
        }

        public DbSet<Student> GetStudentDbSet()
        {
            return dataFunctions.GetStudentDbSet();
        }

        public IQueryable<Student> GetStudents()
        {
            return GetStudentDbSet();
        }

        public List<Student> GetStudentList()
        {
            return GetStudentDbSet().ToList();
        }
        
        public Student? GetStudentByLogin(string login)
        {
            return GetStudentDbSet().Include(s => s.SubjectList).FirstOrDefault(s => s.Login == login);
        }

        public Student? GetStudentByEmail(string email)
        {
            return GetStudentDbSet().FirstOrDefault(s => s.Email == email);
        }

        public Student? GetStudentByIdentifier(string studentIdentifier)
        {
            return GetStudentDbSet().FirstOrDefault(s => s.StudentIdentifier == studentIdentifier);
        }

        public User? GetMainAdmin()
        {
            return GetUserDbSet().FirstOrDefault(u => u.Role == EnumTypes.Role.MainAdmin);
        }

        public DbSet<UserRegistration> GetUserRegistrationDbSet()
        {
            return dataFunctions.GetUserRegistrationDbSet();
        }

        public IQueryable<UserRegistration> GetUserRegistrations(string email)
        {
            return GetUserRegistrationDbSet()
                .Where(u => u.Email == email);
        }

        public UserRegistration? GetUserRegistration(string email)
        {
            return GetUserRegistrationDbSet()
                .FirstOrDefault(u => u.Email == email);
        }

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

        public async Task<string> CreateUserRegistration(string firstName, string lastName, string email, string login, string role)
        {
            string message;
            var user = GetUserRegistration(email);
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
                    var importedStudent = dataFunctions.GetStudentByLogin(login);
                    if (importedStudent != null)
                    {
                        userRegistration.Student = importedStudent;
                    }

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

        public async Task<string> AddStudents(string login)
        {
            List<Student> students = LoadStudents();
            return await dataFunctions.AddStudents(login, students);
        }

        public async Task DeleteAllStudents()
        {
            dataFunctions.ExecuteSqlRaw("delete from Student");
            await dataFunctions.SaveChangesAsync();
        }

        public async Task<string> AddStudent(string studentIdentifier, string firstName, string lastName, string login, string email, Student? studentLoginCheck)
        {
            var studentByIdentifier = GetStudentByIdentifier(studentIdentifier);
            if (studentLoginCheck == null && studentByIdentifier == null)
            {
                Student student = new Student();
                student.StudentIdentifier = studentIdentifier;
                student.Login = login;
                student.Email = email;
                student.FirstName = firstName;
                student.LastName = lastName;
                return await dataFunctions.AddStudent(student);
            }
            else
            {
                if (studentLoginCheck != null)
                {
                    studentLoginCheck.Email = email;
                    await dataFunctions.SaveChangesAsync();
                    return "Studentovi s loginem " + login + " byla úspěšně přiřazena emailová adresa.";
                }
            }
            return "Při přidávání studenta nastala neočekávaná chyba.";
        }

        public async Task EditStudent(string studentIdentifier, string firstName, string lastName, string login, string email, Student studentLoginCheck)
        {
            studentLoginCheck.Login = login;
            studentLoginCheck.Email = email;
            studentLoginCheck.FirstName = firstName;
            studentLoginCheck.LastName = lastName;
            studentLoginCheck.StudentIdentifier = studentIdentifier;
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
            dataFunctions.ExecuteSqlRaw("delete from [User] where role = 2");
            await dataFunctions.SaveChangesAsync();
        }

        public async Task DeleteAllAdmins()
        {
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

        public async Task ChangeMainAdmin(User newMainAdmin, string firstName, string lastName, string login, string email)
        {
            var oldMainAdmin = GetMainAdmin();
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

        public async Task<string> ApproveStudentRegistration(Student student, string firstName, string lastName, string login, string email)
        {
            string message;
            student.Email = email;
            student.FirstName = firstName;
            student.LastName = lastName;

            var userRegistration = GetUserRegistration(email);
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
            return message;
        }

        public async Task<string> ApproveUserRegistration(string firstName, string lastName, string login, string email, string role)
        {
            string message;
            User user = new User();
            user.Email = email;
            user.FirstName = firstName;
            user.LastName = lastName;
            user.Login = login;
            user.Role = Enum.Parse<EnumTypes.Role>(role);
            await dataFunctions.AddUser(user);
            var userRegistration = GetUserRegistration(email);
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
            return message;
        }

        public async Task<string> RefuseRegistration(string email)
        {
            string message;
            var userRegistration = GetUserRegistration(email);
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
            var userRegistration = GetUserRegistration(email);
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

        public bool CanUserAccessPage(EnumTypes.Role requiredRole)
        {
            if (Config.TestingMode)//in case the testing mode is on, no authentication is required at all
            {
                return true;
            }

            string login = Config.Application["login"];
            var user = GetUserByLogin(login);
            var student = GetStudentByLogin(login);

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

        public bool CanStudentAccessTest(Student student, TestTemplate testTemplate)
        {
            if (student.SubjectList.Contains(testTemplate.Subject))
            {
                return true;
            }
            return false;
        }

        public List<Student> LoadStudents()
        {
            List<Student> students = new List<Student>();
            if (Directory.Exists(Config.GetStudentsPath()))
            {
                foreach (var studentFile in Directory.GetFiles(Config.GetStudentsPath()))
                {
                    if (new FileInfo(studentFile).Extension == ".rdf")
                    {
                        Student student = new Student();
                        XmlReader xmlReader = XmlReader.Create(studentFile);
                        while (xmlReader.Read())
                        {
                            if (xmlReader.Name == "rdf:Description" && xmlReader.NodeType != XmlNodeType.EndElement)
                            {
                                if (xmlReader.GetAttribute("rdf:about") != null)
                                {
                                    string[] studentDescription = xmlReader.GetAttribute("rdf:about")!.Split("#");
                                    if (studentDescription[1] != null)
                                    {
                                        student.StudentIdentifier = studentDescription[1];
                                    }
                                }
                            }

                            if (xmlReader.Name == "ns0:login" && xmlReader.NodeType != XmlNodeType.EndElement)
                            {
                                student.Login = xmlReader.ReadInnerXml();
                            }

                            if (xmlReader.Name == "ns0:userFirstName" && xmlReader.NodeType != XmlNodeType.EndElement)
                            {
                                student.FirstName = xmlReader.ReadInnerXml();
                            }

                            if (xmlReader.Name == "ns0:userLastName" && xmlReader.NodeType != XmlNodeType.EndElement)
                            {
                                student.LastName = xmlReader.ReadInnerXml();
                            }

                        }
                        students.Add(student);
                    }
                }
            }
            else { throw Exceptions.StudentsPathNotFoundException; }
            return students;
        }

    }
}