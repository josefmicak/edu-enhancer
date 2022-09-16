using Microsoft.AspNetCore.Mvc;
using System.Diagnostics;
using ViewLayer.Models;
using DomainModel;
using Microsoft.EntityFrameworkCore;
using DataLayer;
using Microsoft.AspNetCore.Authorization;
using System.Net.Mail;
using System.Dynamic;
using Common;

namespace ViewLayer.Controllers
{
    [Authorize]
    public class HomeController : Controller
    {
        private readonly CourseContext _context;
        private QuestionController questionController;
        private StudentController studentController = new StudentController();
        private TestController testController;

        private readonly ILogger<HomeController> _logger;
        private readonly IConfiguration _configuration;

        public HomeController(ILogger<HomeController> logger, IConfiguration configuration, CourseContext context)
        {
            _logger = logger;
            _configuration = configuration;
            _context = context;
            questionController = new QuestionController(context);
            testController = new TestController(context);

            if(_context.GlobalSettings.First() != null)
            {
                Config.TestingMode = _context.GlobalSettings.First().TestingMode;
                Config.SelectedPlatform = _context.GlobalSettings.First().SelectedPlatform;
            }

            Config.GoogleClientId = configuration["Authentication:Google:ClientId"];
        }

        [AllowAnonymous]
        public async Task<IActionResult> Index()
        {
            if(_context.Users.Count() == 0)
            {
                ViewBag.Message = "Prozatím není zaregistrován žádný uživatel. Pro vytvoření účtu s právy hlavního administrátora se přihlašte.";
            }
            else
            {
                ViewBag.Message = "Pro přístup do aplikace se přihlašte.";
            }

            //due to security reasons, the list of users is passed to the view only in case the application is in testing mode
            if(!Config.TestingMode)
            {
                return View();
            }
            else
            {
                dynamic model = new ExpandoObject();
                model.Users = await _context.Users.ToListAsync();
                model.Students = await _context.Students.ToListAsync();
                return (_context.Users != null && _context.Students != null) ?
                    View(model) :
                Problem("Entity set 'CourseContext.Users' or 'CourseContext.Students'  is null.");
            }
        }

        [HttpPost]
        [AllowAnonymous]
        public IActionResult Index(string selectedUserLogin)
        {
            User? user = _context.Users.FirstOrDefault(u => u.Login == selectedUserLogin);
            Student? student = _context.Students.FirstOrDefault(s => s.Login == selectedUserLogin);

            Config.Application["login"] = selectedUserLogin;
            if (user != null)
            {
                return RedirectToAction("TestingModeLogin", "Account", new {name = user.FullName(), email = user.Email });
            }
            else if (student != null)
            {
                return RedirectToAction("TestingModeLogin", "Account", new { name = student.FullName(), email = student.Email });
            }
            //throw an exception in case no user or student with this login exists
            throw Exceptions.UserNotFoundException;
        }

        public async Task<IActionResult> TestTemplateList()
        {
            if (!CanUserAccessPage(Config.Role.Teacher))
            {
                return AccessDeniedAction();
            }

            string login = Config.Application["login"];
            var user = _context.Users.First(u => u.Login == login);
            if(user != null)
            {
                if (user.Role == Config.Role.Teacher)
                {
                    ViewBag.Return = "TeacherMenu";
                }
                else if (user.Role == Config.Role.Admin)
                {
                    ViewBag.Return = "AdminMenu";
                }
                else if (user.Role == Config.Role.MainAdmin)
                {
                    ViewBag.Return = "MainAdminMenu";
                }
            }
            if (TempData["Message"] != null)
            {
                ViewBag.Message = TempData["Message"]!.ToString();
            }
            return _context.TestTemplates != null ?
            View(await _context.TestTemplates
                .Where(t => t.OwnerLogin == login).ToListAsync()) :
            Problem("Entity set 'CourseContext.TestTemplates'  is null.");
        }

        [HttpPost]
        [ValidateAntiForgeryToken]
        public async Task<IActionResult> TestTemplateList(string action, string testNumberIdentifier)
        {
            string login = Config.Application["login"];
            if (action == "add")
            {
                List<TestTemplate> testTemplates = testController.LoadTestTemplates(login);
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

                TempData["Message"] = "Přidáno " + successCount + " šablon testů (" + errorCount + " duplikátů nebo chyb).";
                successCount = 0;
                errorCount = 0;
            }
            else if(action == "deleteAllTemplates")
            {
                var testTemplateList = _context.TestTemplates.Where(t => t.OwnerLogin == login);
                foreach(TestTemplate testTemplate in testTemplateList)
                {
                    _context.TestTemplates.Remove(testTemplate);
                }
                await _context.SaveChangesAsync();
                TempData["Message"] = "Byly smazány všechny vaše testy.";
            }
            else if(action == "deleteTemplate")
            {
                var testTemplateList = _context.TestTemplates.Where(t => t.OwnerLogin == login && t.TestNumberIdentifier == testNumberIdentifier);
                foreach (TestTemplate testTemplate in testTemplateList)
                {
                    _context.TestTemplates.Remove(testTemplate);
                }
                await _context.SaveChangesAsync();
                TempData["Message"] = "Test byl úspěšně smazán.";
            }
            return RedirectToAction(nameof(TestTemplateList));
        }

        public async Task<IActionResult> TestTemplate(string testNumberIdentifier)
        {
            if (!CanUserAccessPage(Config.Role.Teacher))
            {
                return AccessDeniedAction();
            }

            if (TempData["Message"] != null)
            {
                ViewBag.Message = TempData["Message"]!.ToString();
            }
            string login = Config.Application["login"];

            var questionTemplates = _context.QuestionTemplates
                .Include(q => q.TestTemplate)
                .Include(q => q.SubquestionTemplateList)
                .Where(q => q.TestTemplate.TestNumberIdentifier == testNumberIdentifier && q.OwnerLogin == login);

            if(questionTemplates.Count() > 0)
            {
                return View(await questionTemplates.ToListAsync());
            }
            else
            {
                return NoElementsFoundAction();
            }
        }

        [HttpPost]
        public async Task<IActionResult> TestTemplate(string testNumberIdentifier, string negativePoints)
        {
            string? message = null;
            //the teacher is changing negative points setting for the test template
            var testTemplate = _context.TestTemplates.FirstOrDefault(t => t.TestNumberIdentifier == testNumberIdentifier);
            if(testTemplate != null)
            {
                testTemplate.NegativePoints = negativePoints;
                message = "Změny úspěšně uloženy.";
                await _context.SaveChangesAsync();
            }

            TempData["Message"] = message;
            return RedirectToAction("TestTemplate", "Home", new { testNumberIdentifier = testNumberIdentifier });
        }

        public async Task<IActionResult> QuestionTemplate(string questionNumberIdentifier)
        {
            if (!CanUserAccessPage(Config.Role.Teacher))
            {
                return AccessDeniedAction();
            }

            string login = Config.Application["login"];
            if (TempData["Message"] != null)
            {
                ViewBag.Message = TempData["Message"]!.ToString();
            }
            if (TempData["subquestionIdentifier"] != null)
            {
                ViewBag.subquestionIdentifier = TempData["subquestionIdentifier"]!.ToString();
            }
            ViewBag.SubquestionTypeTextArray = questionController.SubquestionTypeTextArray;

            var subquestionTemplates = _context.SubquestionTemplates
                .Include(q => q.QuestionTemplate)
                .Include(q => q.QuestionTemplate.TestTemplate)
                .Where(q => q.QuestionNumberIdentifier == questionNumberIdentifier && q.OwnerLogin == login);

            if (subquestionTemplates.Count() > 0)
            {
                return View(await subquestionTemplates.ToListAsync());
            }
            else
            {
                return NoElementsFoundAction();
            }
        }

        [HttpPost]
        public async Task<IActionResult> QuestionTemplate(string questionNumberIdentifier, string subquestionIdentifier, string subquestionPoints)
        {
            string login = Config.Application["login"];
            if (subquestionPoints != null)
            {
                subquestionPoints = subquestionPoints.Replace(".", ",");
            }
            string? message = null;
            //the teacher is changing points of the subquestion
            if (subquestionPoints != null)
            {
                var subquestionTemplate = _context.SubquestionTemplates.FirstOrDefault(s => s.QuestionNumberIdentifier == questionNumberIdentifier
                && s.SubquestionIdentifier == subquestionIdentifier && s.OwnerLogin == login);
                if(subquestionTemplate != null)
                {
                    if (subquestionPoints == null)
                    {
                        message = "Chyba: nebyl zadán žádný počet bodů.";
                    }
                    else if (!double.TryParse(subquestionPoints, out _))
                    {
                        message = "Chyba: \"" + subquestionPoints + "\" není korektní formát počtu bodů. Je nutné zadat číslo.";
                    }
                    else if(Math.Round(Convert.ToDouble(subquestionPoints), 2) <= 0)
                    {
                        message = "Chyba: otázce je nutné přidělit kladný počet bodů.";
                    }
                    else//todo: overit jestli nema za otazku nektery student pridelen vyssi pocet bodu nez soucasny pocet bodu
                    {
                        message = "Počet bodů byl úspěšně změněn.";
                        subquestionTemplate.SubquestionPoints = Math.Round(Convert.ToDouble(subquestionPoints), 2);
                        await _context.SaveChangesAsync();
                    }
                }
            }
            TempData["Message"] = message;
            TempData["subquestionIdentifier"] = subquestionIdentifier;
            return RedirectToAction("QuestionTemplate", "Home", new { questionNumberIdentifier = questionNumberIdentifier });
        }

        public async Task<IActionResult> ManageSolvedTestList()
        {
            if (!CanUserAccessPage(Config.Role.Teacher))
            {
                return AccessDeniedAction();
            }

            string login = Config.Application["login"];
            var user = _context.Users.First(u => u.Login == login);
            if(user != null)
            {
                if (user.Role == Config.Role.Teacher)
                {
                    ViewBag.Return = "TeacherMenu";
                }
                else if (user.Role == Config.Role.Admin)
                {
                    ViewBag.Return = "AdminMenu";
                }
                else if (user.Role == Config.Role.MainAdmin)
                {
                    ViewBag.Return = "MainAdminMenu";
                }
            }
            if (TempData["Message"] != null)
            {
                ViewBag.Message = TempData["Message"]!.ToString();
            }
            return _context.TestResults != null ?
            View(await _context.TestResults.
                Include(s => s.Student).
                Include(t => t.TestTemplate)
                .Where(t => t.TestTemplate.OwnerLogin == login).ToListAsync()) :
            Problem("Entity set 'CourseContext.TestResults'  is null.");
        }

        [HttpPost]
        [ValidateAntiForgeryToken]
        public async Task<IActionResult> ManageSolvedTestList(string action, string testResultIdentifier)
        {
            string login = Config.Application["login"];
            if (action == "add")
            {
                List<TestResult> testResults = testController.LoadTestResults(login);
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

                TempData["Message"] = "Přidáno " + successCount + " řešení testů (" + errorCount + " duplikátů nebo chyb).";
                successCount = 0;
                errorCount = 0;
            }
            else if(action == "deleteAllResults")
            {
                var testResultList = _context.TestResults.Where(t => t.TestTemplate.OwnerLogin == login);
                foreach (TestResult testResult in testResultList)
                {
                    _context.TestResults.Remove(testResult);
                }
                await _context.SaveChangesAsync();
                TempData["Message"] = "Byly smazány všechna vaše řešení testů.";
            }
            else if(action == "deleteResult")
            {
                var testResultList = _context.TestResults.Where(t => t.TestTemplate.OwnerLogin == login && t.TestResultIdentifier == testResultIdentifier);
                foreach (TestResult testResult in testResultList)
                {
                    _context.TestResults.Remove(testResult);
                }
                await _context.SaveChangesAsync();
                TempData["Message"] = "Test byl úspěšně smazán.";
            }
            return RedirectToAction(nameof(ManageSolvedTestList));
        }

        public async Task<IActionResult> ManageSolvedTest(string testResultIdentifier)
        {
            if (!CanUserAccessPage(Config.Role.Teacher))
            {
                return AccessDeniedAction();
            }

            string login = Config.Application["login"];

            var questionResults = _context.QuestionResults
                .Include(t => t.TestResult)
                .Include(q => q.QuestionTemplate)
                .Include(q => q.QuestionTemplate.TestTemplate)
                .Include(q => q.QuestionTemplate.SubquestionTemplateList)
                .Include(s => s.TestResult.Student)
                .Include(q => q.SubquestionResultList)
                .Where(t => t.TestResultIdentifier == testResultIdentifier && t.OwnerLogin == login);

            if (questionResults.Count() > 0)
            {
                return View(await questionResults.ToListAsync());
            }
            else
            {
                return NoElementsFoundAction();
            }
        }

        public async Task<IActionResult> ManageSolvedQuestion(string questionNumberIdentifier, string testResultIdentifier)
        {
            if (!CanUserAccessPage(Config.Role.Teacher))
            {
                return AccessDeniedAction();
            }

            if (TempData["Message"] != null)
            {
                ViewBag.Message = TempData["Message"]!.ToString();
            }
            if (TempData["subquestionIdentifier"] != null)
            {
                ViewBag.subquestionIdentifier = TempData["subquestionIdentifier"]!.ToString();
            }
            ViewBag.SubquestionTypeTextArray = questionController.SubquestionTypeTextArray;
            string login = Config.Application["login"];

            var subquestionResults = _context.SubquestionResults
                .Include(s => s.SubquestionTemplate)
                .Include(s => s.QuestionResult)
                .Include(s => s.QuestionResult.QuestionTemplate)
                .Include(s => s.QuestionResult.QuestionTemplate.TestTemplate)
                .Where(s => s.TestResultIdentifier == testResultIdentifier
                && s.QuestionNumberIdentifier == questionNumberIdentifier && s.OwnerLogin == login);

            if (subquestionResults.Count() > 0)
            {
                return View(await subquestionResults.ToListAsync());
            }
            else
            {
                return NoElementsFoundAction();
            }
        }

        [HttpPost]
        public async Task<IActionResult> ManageSolvedQuestion(string questionNumberIdentifier, string testResultIdentifier, string subquestionIdentifier, string studentsPoints, string subquestionPoints, string negativePoints)
        {
            string login = Config.Application["login"];
            if(studentsPoints != null)
            {
                studentsPoints = studentsPoints.Replace(".", ",");
            }
            string? message = null;
            //the teacher is changing student's points
            if(studentsPoints != null)
            {
                var subquestionResult = _context.SubquestionResults
                    .FirstOrDefault(s => s.TestResultIdentifier == testResultIdentifier && s.QuestionNumberIdentifier == questionNumberIdentifier 
                    && s.SubquestionIdentifier == subquestionIdentifier && s.OwnerLogin == login);
                if(subquestionResult != null)
                {
                    if(subquestionPoints == null)
                    {
                        message = "Chyba: nelze upravit studentův počet bodů. Nejprve je nutné nastavit počet bodů u zadání podotázky.";
                    }
                    else if(studentsPoints == null)
                    {
                        message = "Chyba: nebyl zadán žádný počet bodů.";
                    }
                    else if(!double.TryParse(studentsPoints, out _))
                    {
                        message = "Chyba: \"" + studentsPoints + "\" není korektní formát počtu bodů. Je nutné zadat číslo.";
                    }
                    else if(Convert.ToDouble(studentsPoints) > Convert.ToDouble(subquestionPoints))
                    {
                        message = "Chyba: příliš vysoký počet bodů. Nejvyšší počet bodů, které může za tuto podotázku student obdržet, je " + subquestionPoints + ".";
                    }
                    else if (Convert.ToDouble(studentsPoints) < Convert.ToDouble(subquestionPoints) * (-1))
                    {
                        message = "Chyba: příliš nízký počet bodů. Nejnížší počet bodů, které může za tuto podotázku student obdržet, je -" + subquestionPoints + ".";
                    }
                    else if(negativePoints == "negativePoints_no" && (Convert.ToDouble(studentsPoints) < 0))
                    {
                        message = "Chyba: v tomto testu nemůže student za podotázku obdržet záporný počet bodů. Změnu je možné provést v nastavení šablony testu.";
                    }
                    else
                    {
                        message = "Studentův počet bodů byl úspěšně změněn.";
                        subquestionResult.StudentsPoints = Math.Round(Convert.ToDouble(studentsPoints), 2);
                        await _context.SaveChangesAsync();
                    }
                }
            }
            TempData["Message"] = message;
            TempData["subquestionIdentifier"] = subquestionIdentifier;
            return RedirectToAction("ManageSolvedQuestion", "Home", new { questionNumberIdentifier = questionNumberIdentifier, testResultIdentifier = testResultIdentifier });
        }

        public async Task<IActionResult> BrowseSolvedTestList()
        {
            if (!CanUserAccessPage(Config.Role.Student))
            {
                return AccessDeniedAction();
            }

            string login = Config.Application["login"];
            dynamic model = new ExpandoObject();
            model.TestResults = await _context.TestResults
                .Include(t => t.Student)
                .Include(t => t.TestTemplate)
                .Include(t => t.TestTemplate.Owner)
                .Where(t => t.Student.Login == login).ToListAsync();
            model.Student = _context.Students.FirstOrDefault(s => s.Login == login);
            return _context.TestResults != null ?
                View(model) :
            Problem("Entity set 'CourseContext.TestResults'  is null.");
        }

        public async Task<IActionResult> BrowseSolvedTest(string testResultIdentifier, string ownerLogin)
        {
            if (!CanUserAccessPage(Config.Role.Student))
            {
                return AccessDeniedAction();
            }

            string login = Config.Application["login"];

            var questionResults = _context.QuestionResults
                .Include(q => q.TestResult)
                .Include(q => q.QuestionTemplate)
                .Include(q => q.QuestionTemplate.TestTemplate)
                .Include(q => q.TestResult.TestTemplate.Owner)
                .Include(q => q.QuestionTemplate.SubquestionTemplateList)
                .Include(q => q.TestResult.Student)
                .Include(q => q.SubquestionResultList)
                .Where(q => q.TestResultIdentifier == testResultIdentifier && q.TestResult.Student.Login == login
                    && q.OwnerLogin == ownerLogin);

            if (questionResults.Count() > 0)
            {
                return View(await questionResults.ToListAsync());
            }
            else
            {
                return NoElementsFoundAction();
            }
        }

        public async Task<IActionResult> BrowseSolvedQuestion(string questionNumberIdentifier, string testResultIdentifier, string ownerLogin)
        {
            if (!CanUserAccessPage(Config.Role.Student))
            {
                return AccessDeniedAction();
            }

            if (TempData["Message"] != null)
            {
                ViewBag.Message = TempData["Message"]!.ToString();
            }
            if (TempData["subquestionIdentifier"] != null)
            {
                ViewBag.subquestionIdentifier = TempData["subquestionIdentifier"]!.ToString();
            }
            ViewBag.SubquestionTypeTextArray = questionController.SubquestionTypeTextArray;
            string login = Config.Application["login"];

            var subquestionResults = _context.SubquestionResults
                .Include(s => s.SubquestionTemplate)
                .Include(s => s.QuestionResult)
                .Include(s => s.QuestionResult.QuestionTemplate)
                .Include(s => s.QuestionResult.QuestionTemplate.TestTemplate)
                .Where(s => s.TestResultIdentifier == testResultIdentifier && s.QuestionNumberIdentifier == questionNumberIdentifier &&
                s.QuestionResult.TestResult.Student.Login == login && s.OwnerLogin == ownerLogin);

            if (subquestionResults.Count() > 0)
            {
                return View(await subquestionResults.ToListAsync());
            }
            else
            {
                return NoElementsFoundAction();
            }
        }

        [HttpPost]
        public IActionResult BrowseSolvedQuestion(string questionNumberIdentifier, string testResultIdentifier, string subquestionIdentifier, string ownerLogin)
        {
            TempData["subquestionIdentifier"] = subquestionIdentifier;
            return RedirectToAction("BrowseSolvedQuestion", "Home", new { questionNumberIdentifier = questionNumberIdentifier, testResultIdentifier = testResultIdentifier, ownerLogin = ownerLogin });
        }

        [AllowAnonymous]
        public async Task<IActionResult> UserRegistration()
        {
            ViewBag.firstName = Config.Application["firstName"];
            ViewBag.lastName = Config.Application["lastName"];
            ViewBag.email = Config.Application["email"];
            ViewBag.message = TempData["message"];
            if(_context.Users.Count() == 0)
            {
                ViewBag.firstRegistrationMessage = "Po zaregistrování vám bude automaticky vytvořen účet hlavního administrátora.";
            }
            return _context.UserRegistrations != null ?
            View(await _context.UserRegistrations
                .Where(u => u.Email == Config.Application["email"]).ToListAsync()) :
            Problem("Entity set 'CourseContext.UserRegistrations'  is null.");
        }

        [HttpPost]
        public async Task<IActionResult> UserRegistration(string? firstName, string? lastName, string? login, string role)
        {
            string email = Config.Application["email"];
            if(_context.Users.Count() == 0)
            {
                if (firstName == null || lastName == null || login == null || email == null)
                {
                    TempData["Message"] = "Chyba: všechny položky musí být vyplněny.";
                    return RedirectToAction(nameof(UserRegistration));
                }
                else
                {
                    User mainAdmin = new User();
                    mainAdmin.FirstName = firstName;
                    mainAdmin.LastName = lastName;
                    mainAdmin.Email = email;
                    mainAdmin.Login = login;
                    mainAdmin.Role = Config.Role.MainAdmin;
                    Config.Application["login"] = login;
                    _context.Users.Add(mainAdmin);
                    await _context.SaveChangesAsync();
                    return RedirectToAction(nameof(MainAdminMenu));
                }
            }
            else
            {
                var user = _context.UserRegistrations.FirstOrDefault(u => u.Email == email);
                if (user != null)
                {
                    TempData["Message"] = "Chyba: již jste zaregistrován. Nyní je nutné vyčkat na potvrzení registrace správcem.";
                }
                else if (firstName == null || lastName == null || login == null || email == null)
                {
                    TempData["Message"] = "Chyba: všechny položky musí být vyplněny.";
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
                        userRegistration.State = Config.RegistrationState.Waiting;
                        userRegistration.CreationDate = DateTime.Now;
                        userRegistration.Role = (Config.Role)Convert.ToInt32(role);
                        var importedStudent = _context.Students.FirstOrDefault(s => s.Login == login);
                        if (importedStudent != null)
                        {
                            userRegistration.Student = importedStudent;
                        }

                        _context.UserRegistrations.Add(userRegistration);
                        await _context.SaveChangesAsync();
                        TempData["Message"] = "Registrace úspěšně vytvořena. Nyní je nutné vyčkat na potvrzení registrace správcem.";
                    }
                    catch (Exception ex)
                    {
                        Debug.WriteLine(ex.Message);
                        TempData["Message"] = "Při registraci nastala chyba.";
                    }
                }
                return RedirectToAction(nameof(UserRegistration));
            }
        }

        public async Task<IActionResult> ManageUserList()
        {
            if (!CanUserAccessPage(Config.Role.MainAdmin))
            {
                return AccessDeniedAction();
            }

            if (TempData["StudentMessage"] != null)
            {
                ViewBag.StudentMessage = TempData["StudentMessage"]!.ToString();
            }
            if (TempData["TeacherMessage"] != null)
            {
                ViewBag.TeacherMessage = TempData["TeacherMessage"]!.ToString();
            }
            if (TempData["AdminMessage"] != null)
            {
                ViewBag.AdminMessage = TempData["AdminMessage"]!.ToString();
            }
            dynamic model = new ExpandoObject();
            model.Users = await _context.Users.ToListAsync();
            model.Students = await _context.Students.ToListAsync();
            return (_context.Users != null && _context.Students != null) ?
                View(model) :
            Problem("Entity set 'CourseContext.TestResults' or 'CourseContext.Students' is null.");
        }

        [HttpPost]
        public async Task<IActionResult> ManageUserList(string action, string studentIdentifier, string firstName, string lastName, string login, string email, string role)
        {
            bool isEmailValid;
            try
            {
                MailAddress m = new MailAddress(email);
                isEmailValid = true;
            }
            catch
            {
                isEmailValid = false;
            }

            var studentLoginCheck = _context.Students.FirstOrDefault(u => u.Login == login);
            var studentEmailCheck = _context.Students.FirstOrDefault(u => u.Email == email);

            var userLoginCheck = _context.Users.FirstOrDefault(u => u.Login == login);
            var userEmailCheck = _context.Users.FirstOrDefault(u => u.Email == email);

            if (action == "addStudents")
            {
                List<Student> students = studentController.LoadStudents();

                int successCount = 0;
                int errorCount = 0;
                for (int i = 0; i < students.Count; i++)
                {
                    _context.ChangeTracker.Clear();
                    try
                    {
                        Student student = students[i];
                        userLoginCheck = _context.Users.FirstOrDefault(u => u.Login == student.Login);
                        if(userLoginCheck != null)
                        {
                            throw Exceptions.UserAlreadyExistsException(login);
                        }
                        _context.Students.Add(student);
                        //in case the student has registered before the student's file has been imported, we add the Student to the UserRegistration
                        var userRegistrationList = _context.UserRegistrations.Where(u => u.Login == student.Login && u.State == Config.RegistrationState.Waiting);
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
                TempData["StudentMessage"] = "Přidáno " + successCount + " studentů (" + errorCount + " duplikátů nebo chyb).";
            }
            else if(action == "deleteAllStudents")
            {
                _context.Database.ExecuteSqlRaw("delete from Student");
                TempData["StudentMessage"] = "Byly smazáni všichni existující studenti.";
            }
            else if(action == "addStudent")
            {
                if(!isEmailValid)
                {
                    TempData["StudentMessage"] = "Chyba: \"" + email + "\" není správně formátovaná emailová adresa.";
                }
                else if (studentEmailCheck != null || userEmailCheck != null)
                {
                    TempData["StudentMessage"] = "Chyba: uživatel s emailem \"" + email + "\" již existuje.";
                }
                else if (userLoginCheck != null)
                {
                    TempData["StudentMessage"] = "Chyba: uživatel s loginem \"" + login + "\" již existuje.";
                }
                else
                {
                    var studentByIdentifier = _context.Students.FirstOrDefault(s => s.StudentIdentifier == studentIdentifier);
                    if (studentLoginCheck == null && studentByIdentifier == null)
                    {
                        Student student = new Student();
                        student.StudentIdentifier = studentIdentifier;
                        student.Login = login;
                        student.Email = email;
                        student.FirstName = firstName;
                        student.LastName = lastName;
                        _context.Students.Add(student);
                        await _context.SaveChangesAsync();
                        TempData["StudentMessage"] = "Student s loginem \"" + login + "\" úspěšně přidán.";
                    }
                    else
                    {
                        if(studentLoginCheck != null)
                        {
                            studentLoginCheck.Email = email;
                            await _context.SaveChangesAsync();
                            TempData["StudentMessage"] = "Studentovi s loginem " + login + " byla úspěšně přiřazena emailová adresa.";
                        }
                    }
                }
            }
            else if(action == "editStudent")
            {
                //it's necessary to ensure that there won't be two or more users with the same email/identifier
                if(studentLoginCheck != null)
                {
                    var studentByNewEmail = _context.Students.FirstOrDefault(s => s.Email == email);
                    if (!isEmailValid)
                    {
                        TempData["StudentMessage"] = "Chyba: \"" + email + "\" není správně formátovaná emailová adresa.";
                    }
                    else if (firstName == null || lastName == null || login == null)
                    {
                        TempData["StudentMessage"] = "Chyba: je nutné vyplnit všechna pole.";
                    }
                    else if (studentByNewEmail != null && studentLoginCheck.Email != studentByNewEmail.Email)
                    {
                        TempData["StudentMessage"] = "Chyba: uživatel s emailem \"" + studentByNewEmail.Email + "\" již existuje.";
                    }
                    else if(userEmailCheck != null)
                    {
                        TempData["StudentMessage"] = "Chyba: uživatel s emailem \"" + userEmailCheck.Email + "\" již existuje.";
                    }
                    else if (userLoginCheck != null)
                    {
                        TempData["StudentMessage"] = "Chyba: uživatel s loginem \"" + userLoginCheck.Login + "\" již existuje.";
                    }
                    else
                    {
                        var studentByNewIdentifier = _context.Students.FirstOrDefault(s => s.StudentIdentifier == studentIdentifier);
                        if (studentByNewIdentifier != null && studentLoginCheck.StudentIdentifier != studentByNewIdentifier.StudentIdentifier)
                        {
                            TempData["StudentMessage"] = "Chyba: student s identifikátorem \"" + studentByNewIdentifier.Email + "\" již existuje.";
                        }
                        else
                        {
                            studentLoginCheck.Login = login;
                            studentLoginCheck.Email = email;
                            studentLoginCheck.FirstName = firstName;
                            studentLoginCheck.LastName = lastName;
                            studentLoginCheck.StudentIdentifier = studentIdentifier;
                            await _context.SaveChangesAsync();
                            TempData["StudentMessage"] = "Student úspěšně upraven.";
                        }
                    }
                }
                else
                {
                    TempData["StudentMessage"] = "Chyba: student s loginem \"" + studentLoginCheck + "\" nebyl nalezen.";
                }
            }
            else if(action == "deleteStudent")
            {
                var user = _context.Students.FirstOrDefault(s => s.Login == login);
                if(user != null)
                {
                    _context.Students.Remove(user);
                    await _context.SaveChangesAsync();
                    TempData["StudentMessage"] = "Student úspěšně smazán.";
                }
                else
                {
                    TempData["StudentMessage"] = "Chyba: student s loginem \"" + login + "\" nebyl nalezen.";
                }
            }
            else if(action == "addTeacher")
            {
                if (!isEmailValid)
                {
                    TempData["TeacherMessage"] = "Chyba: \"" + email + "\" není správně formátovaná emailová adresa.";
                }
                else if(firstName == null || lastName == null || login == null)
                {
                    TempData["TeacherMessage"] = "Chyba: je nutné vyplnit všechna pole.";
                }
                else if(userLoginCheck != null || studentLoginCheck != null)
                {
                    TempData["TeacherMessage"] = "Chyba: uživatel s loginem \"" + login + "\" již existuje.";
                }
                else if (userEmailCheck != null || studentEmailCheck != null)
                {
                    TempData["TeacherMessage"] = "Chyba: uživatel s emailem \"" + email + "\" již existuje.";
                }
                else
                {
                    User teacher = new User();
                    teacher.FirstName = firstName;
                    teacher.LastName = lastName;
                    teacher.Login = login;
                    teacher.Email = email;
                    teacher.Role = Config.Role.Teacher;
                    _context.Users.Add(teacher);
                    await _context.SaveChangesAsync();
                    TempData["TeacherMessage"] = "Učitel byl úspěšně přidán.";
                }
            }
            else if (action == "editTeacher")
            {
                //it's necessary to ensure that there won't be two or more users with the same email
                if (userLoginCheck != null)
                {
                    if (!isEmailValid)
                    {
                        TempData["TeacherMessage"] = "Chyba: \"" + email + "\" není správně formátovaná emailová adresa.";
                    }
                    else if (firstName == null || lastName == null || login == null)
                    {
                        TempData["TeacherMessage"] = "Chyba: je nutné vyplnit všechna pole.";
                    }
                    else if (userEmailCheck != null && userLoginCheck.Email != userEmailCheck.Email)
                    {
                        TempData["TeacherMessage"] = "Chyba: uživatel s emailem \"" + userEmailCheck.Email + "\" již existuje.";
                    }
                    else if (studentEmailCheck != null)
                    {
                        TempData["TeacherMessage"] = "Chyba: uživatel s emailem \"" + studentEmailCheck.Email + "\" již existuje.";
                    }
                    else if (studentLoginCheck != null)
                    {
                        TempData["TeacherMessage"] = "Chyba: uživatel s loginem \"" + studentLoginCheck.Login + "\" již existuje.";
                    }
                    else
                    {
                        userLoginCheck.Login = login;
                        userLoginCheck.Email = email;
                        userLoginCheck.FirstName = firstName;
                        userLoginCheck.LastName = lastName;
                        if (role != null)
                        {
                            userLoginCheck.Role = (Config.Role)Convert.ToInt32(role);
                        }
                        await _context.SaveChangesAsync();
                        TempData["TeacherMessage"] = "Učitel úspěšně upraven.";
                    }
                }
                else
                {
                    TempData["TeacherMessage"] = "Chyba: učitel s loginem \"" + login + "\" nebyl nalezen.";
                }
            }
            else if (action == "deleteTeacher")
            {
                if (userLoginCheck != null)
                {
                    _context.Users.Remove(userLoginCheck);
                    await _context.SaveChangesAsync();
                    TempData["TeacherMessage"] = "Učitel úspěšně smazán.";
                }
                else
                {
                    TempData["TeacherMessage"] = "Chyba: učitel s loginem \"" + login + "\" nebyl nalezen.";
                }
            }
            else if(action == "deleteAllTeachers")
            {
                _context.Database.ExecuteSqlRaw("delete from [User] where role = 2");
                TempData["TeacherMessage"] = "Byly smazáni všichni existující učitelé.";
            }
            else if (action == "addAdmin")
            {
                if (!isEmailValid)
                {
                    TempData["AdminMessage"] = "Chyba: \"" + email + "\" není správně formátovaná emailová adresa.";
                }
                else if (firstName == null || lastName == null || login == null)
                {
                    TempData["AdminMessage"] = "Chyba: je nutné vyplnit všechna pole.";
                }
                else if (userLoginCheck != null || studentLoginCheck != null)
                {
                    TempData["AdminMessage"] = "Chyba: uživatel s loginem \"" + login + "\" již existuje.";
                }
                else if (userEmailCheck != null || studentEmailCheck != null)
                {
                    TempData["AdminMessage"] = "Chyba: uživatel s emailem \"" + email + "\" již existuje.";
                }
                else
                {
                    User admin = new User();
                    admin.FirstName = firstName;
                    admin.LastName = lastName;
                    admin.Login = login;
                    admin.Email = email;
                    admin.Role = Config.Role.Admin;
                    _context.Users.Add(admin);
                    await _context.SaveChangesAsync();
                    TempData["AdminMessage"] = "Správce byl úspěšně přidán.";
                }
            }
            else if (action == "editAdmin" || action == "changeMainAdmin")
            {
                //it's necessary to ensure that there won't be two or more users with the same email
                if (userLoginCheck != null)
                {
                    if (!isEmailValid)
                    {
                        TempData["AdminMessage"] = "Chyba: \"" + email + "\" není správně formátovaná emailová adresa.";
                    }
                    else if (firstName == null || lastName == null || login == null)
                    {
                        TempData["AdminMessage"] = "Chyba: je nutné vyplnit všechna pole.";
                    }
                    else if (userEmailCheck != null && userLoginCheck.Email != userEmailCheck.Email)
                    {
                        TempData["AdminMessage"] = "Chyba: správce s emailem \"" + userEmailCheck.Email + "\" již existuje.";
                    }
                    else if (studentEmailCheck != null)
                    {
                        TempData["AdminMessage"] = "Chyba: uživatel s emailem \"" + studentEmailCheck.Email + "\" již existuje.";
                    }
                    else if (studentLoginCheck != null)
                    {
                        TempData["AdminMessage"] = "Chyba: uživatel s loginem \"" + studentLoginCheck.Login + "\" již existuje.";
                    }
                    else
                    {
                        if(action == "editAdmin")
                        {
                            if(Convert.ToInt32(role) == 2 || Convert.ToInt32(role) == 3)
                            {
                                TempData["AdminMessage"] = "Chyba: hlavnímu administrátorovi nelze změnit roli.";
                            }
                            else
                            {
                                userLoginCheck.Login = login;
                                userLoginCheck.Email = email;
                                userLoginCheck.FirstName = firstName;
                                userLoginCheck.LastName = lastName;
                                if (role != null)
                                {
                                    userLoginCheck.Role = (Config.Role)Convert.ToInt32(role);
                                }

                                await _context.SaveChangesAsync();
                                TempData["AdminMessage"] = "Správce úspěšně upraven.";
                            }
                        }
                        else if(action == "changeMainAdmin")
                        {
                            var oldMainAdmin = _context.Users.FirstOrDefault(u => u.Role == Config.Role.MainAdmin);
                            if(oldMainAdmin != null)
                            {
                                oldMainAdmin.Role = Config.Role.Admin;
                            }

                            userLoginCheck.Login = login;
                            userLoginCheck.Email = email;
                            userLoginCheck.FirstName = firstName;
                            userLoginCheck.LastName = lastName;
                            userLoginCheck.Role = Config.Role.MainAdmin;

                            await _context.SaveChangesAsync();
                            TempData["AdminMessage"] = "Nový hlavní administrátor úspěšně nastaven.";
                            return RedirectToAction(nameof(ManageUserListForAdmin));
                        }
                    }
                }
                else
                {
                    TempData["AdminMessage"] = "Chyba: správce s loginem \"" + login + "\" nebyl nalezen.";
                }
            }
            else if (action == "deleteAdmin")
            {
                if (userLoginCheck != null)
                {
                    if(userLoginCheck.Role == Config.Role.MainAdmin)
                    {
                        TempData["AdminMessage"] = "Chyba: účet hlavního administrátora nelze smazat.";
                    }
                    else
                    {
                        _context.Users.Remove(userLoginCheck);
                        await _context.SaveChangesAsync();
                        TempData["AdminMessage"] = "Správce úspěšně smazán.";
                    }
                }
                else
                {
                    TempData["AdminMessage"] = "Chyba: správce s loginem \"" + login + "\" nebyl nalezen.";
                }
            }
            else if (action == "deleteAllAdmins")
            {
                _context.Database.ExecuteSqlRaw("delete from [User] where role = 3");
                TempData["AdminMessage"] = "Byly smazáni všichni existující správci.";
            }
            return RedirectToAction(nameof(ManageUserList));
        }

        public async Task<IActionResult> ManageUserListForAdmin()
        {
            if (!CanUserAccessPage(Config.Role.Admin))
            {
                return AccessDeniedAction();
            }

            if (TempData["StudentMessage"] != null)
            {
                ViewBag.StudentMessage = TempData["StudentMessage"]!.ToString();
            }
            if (TempData["TeacherMessage"] != null)
            {
                ViewBag.TeacherMessage = TempData["TeacherMessage"]!.ToString();
            }
            if (TempData["AdminMessage"] != null)
            {
                ViewBag.TeacherMessage = TempData["AdminMessage"]!.ToString();
            }
            dynamic model = new ExpandoObject();
            model.Users = await _context.Users.ToListAsync();
            model.Students = await _context.Students.ToListAsync();
            return (_context.Users != null && _context.Students != null) ?
                View(model) :
            Problem("Entity set 'CourseContext.TestResults' or 'CourseContext.Students' is null.");
        }

        [HttpPost]
        public async Task<IActionResult> ManageUserListForAdmin(string action, string firstName, string lastName, string login, string email, string studentIdentifier)
        {
            bool isEmailValid;
            try
            {
                MailAddress m = new MailAddress(email);
                isEmailValid = true;
            }
            catch
            {
                isEmailValid = false;
            }
            
            var studentLoginCheck = _context.Students.FirstOrDefault(s => s.Login == login);
            var studentEmailCheck = _context.Students.FirstOrDefault(s => s.Email == email);

            var userLoginCheck = _context.Users.FirstOrDefault(u => u.Login == login);
            var userEmailCheck = _context.Users.FirstOrDefault(u => u.Email == email);

            if (action == "addStudents")
            {
                List<Student> students = studentController.LoadStudents();

                int successCount = 0;
                int errorCount = 0;
                for (int i = 0; i < students.Count; i++)
                {
                    try
                    {
                        Student student = students[i];
                        userLoginCheck = _context.Users.FirstOrDefault(u => u.Login == student.Login);
                        if (userLoginCheck != null)
                        {
                            throw Exceptions.UserAlreadyExistsException(login);
                        }
                        _context.Students.Add(student);
                        //in case the user has registered before the student has been imported, we add the User to the UserRegistration
                        var userRegistrationList = _context.UserRegistrations.Where(u => u.Login == student.Login && u.State == Config.RegistrationState.Waiting);
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
                TempData["StudentMessage"] = "Přidáno " + successCount + " studentů (" + errorCount + " duplikátů nebo chyb).";
            }
            else if (action == "addStudent")
            {
                if (!isEmailValid)
                {
                    TempData["StudentMessage"] = "Chyba: \"" + email + "\" není správně formátovaná emailová adresa.";
                }
                else if (studentEmailCheck != null || userEmailCheck != null)
                {
                    TempData["StudentMessage"] = "Chyba: uživatel s emailem \"" + email + "\" již existuje.";
                }
                else if(userLoginCheck != null)
                {
                    TempData["StudentMessage"] = "Chyba: uživatel s loginem \"" + login + "\" již existuje.";
                }
                else
                {
                    var studentByIdentifier = _context.Students.FirstOrDefault(s => s.StudentIdentifier == studentIdentifier);
                    if (studentLoginCheck == null && studentByIdentifier == null)
                    {
                        Student student = new Student();
                        student.StudentIdentifier = studentIdentifier;
                        student.Login = login;
                        student.Email = email;
                        student.FirstName = firstName;
                        student.LastName = lastName;
                        _context.Students.Add(student);
                        await _context.SaveChangesAsync();
                        TempData["StudentMessage"] = "Student s loginem \"" + login + "\" úspěšně přidán.";
                    }
                    else
                    {
                        if(studentLoginCheck != null)
                        {
                            studentLoginCheck.Email = email;
                            await _context.SaveChangesAsync();
                            TempData["StudentMessage"] = "Studentovi s loginem " + login + " byla úspěšně přiřazena emailová adresa.";
                        }
                    }
                }
            }
            else if (action == "editStudent")
            {
                //it's necessary to ensure that there won't be two or more users with the same email/identifier
                if (studentLoginCheck != null)
                {
                    if (!isEmailValid)
                    {
                        TempData["StudentMessage"] = "Chyba: \"" + email + "\" není správně formátovaná emailová adresa.";
                    }
                    else if (firstName == null || lastName == null || login == null)
                    {
                        TempData["StudentMessage"] = "Chyba: je nutné vyplnit všechna pole.";
                    }
                    else if (studentEmailCheck != null && studentLoginCheck.Email != studentEmailCheck.Email)
                    {
                        TempData["StudentMessage"] = "Chyba: uživatel s emailem \"" + studentEmailCheck.Email + "\" již existuje.";
                    }
                    else if (userEmailCheck != null)
                    {
                        TempData["StudentMessage"] = "Chyba: uživatel s emailem \"" + userEmailCheck.Email + "\" již existuje.";
                    }
                    else if (userLoginCheck != null)
                    {
                        TempData["StudentMessage"] = "Chyba: uživatel s loginem \"" + userLoginCheck.Login + "\" již existuje.";
                    }
                    else
                    {
                        var studentByNewIdentifier = _context.Students.FirstOrDefault(s => s.StudentIdentifier == studentIdentifier);
                        if (studentByNewIdentifier != null && studentLoginCheck.StudentIdentifier != studentByNewIdentifier.StudentIdentifier)
                        {
                            TempData["StudentMessage"] = "Chyba: student s identifikátorem \"" + studentByNewIdentifier.Email + "\" již existuje.";
                        }
                        else
                        {
                            studentLoginCheck.Login = login;
                            studentLoginCheck.Email = email;
                            studentLoginCheck.FirstName = firstName;
                            studentLoginCheck.LastName = lastName;
                            studentLoginCheck.StudentIdentifier = studentIdentifier;
                            await _context.SaveChangesAsync();
                            TempData["StudentMessage"] = "Student úspěšně upraven.";
                        }
                    }
                }
                else
                {
                    TempData["StudentMessage"] = "Chyba: student s loginem \"" + studentLoginCheck + "\" nebyl nalezen.";
                }
            }
            else if (action == "deleteStudent")
            {
                var user = _context.Students.FirstOrDefault(s => s.Login == login);
                if (user != null)
                {
                    _context.Students.Remove(user);
                    await _context.SaveChangesAsync();
                    TempData["StudentMessage"] = "Student úspěšně smazán.";
                }
                else
                {
                    TempData["StudentMessage"] = "Chyba: student s loginem \"" + login + "\" nebyl nalezen.";
                }
            }
            else if (action == "addTeacher")
            {
                if (!isEmailValid)
                {
                    TempData["TeacherMessage"] = "Chyba: \"" + email + "\" není správně formátovaná emailová adresa.";
                }
                else if (firstName == null || lastName == null || login == null)
                {
                    TempData["TeacherMessage"] = "Chyba: je nutné vyplnit všechna pole.";
                }
                else if (userLoginCheck != null || studentLoginCheck != null)
                {
                    TempData["TeacherMessage"] = "Chyba: uživatel s loginem \"" + login + "\" již existuje.";
                }
                else if (userEmailCheck != null || studentEmailCheck != null)
                {
                    TempData["TeacherMessage"] = "Chyba: uživatel s emailem \"" + email + "\" již existuje.";
                }
                else
                {
                    User teacher = new User();
                    teacher.FirstName = firstName;
                    teacher.LastName = lastName;
                    teacher.Login = login;
                    teacher.Email = email;
                    teacher.Role = Config.Role.Teacher;
                    _context.Users.Add(teacher);
                    await _context.SaveChangesAsync();
                    TempData["TeacherMessage"] = "Učitel byl úspěšně přidán.";
                }
            }
            else if (action == "editTeacher")
            {
                //it's necessary to ensure that there won't be two or more users with the same email
                if (userLoginCheck != null)
                {
                    if (!isEmailValid)
                    {
                        TempData["TeacherMessage"] = "Chyba: \"" + email + "\" není správně formátovaná emailová adresa.";
                    }
                    else if (firstName == null || lastName == null || login == null)
                    {
                        TempData["TeacherMessage"] = "Chyba: je nutné vyplnit všechna pole.";
                    }
                    else if (userEmailCheck != null && userLoginCheck.Email != userEmailCheck.Email)
                    {
                        TempData["TeacherMessage"] = "Chyba: uživatel s emailem \"" + userEmailCheck.Email + "\" již existuje.";
                    }
                    else if (studentEmailCheck != null)
                    {
                        TempData["TeacherMessage"] = "Chyba: uživatel s emailem \"" + studentEmailCheck.Email + "\" již existuje.";
                    }
                    else if (studentLoginCheck != null)
                    {
                        TempData["TeacherMessage"] = "Chyba: uživatel s loginem \"" + studentLoginCheck.Login + "\" již existuje.";
                    }
                    else
                    {
                        userLoginCheck.Login = login;
                        userLoginCheck.Email = email;
                        userLoginCheck.FirstName = firstName;
                        userLoginCheck.LastName = lastName;
                        await _context.SaveChangesAsync();
                        TempData["TeacherMessage"] = "Učitel úspěšně upraven.";
                    }
                }
                else
                {
                    TempData["TeacherMessage"] = "Chyba: učitel s loginem \"" + login + "\" nebyl nalezen.";
                }
            }
            else if (action == "deleteTeacher")
            {
                var user = _context.Users.FirstOrDefault(u => u.Login == login);
                if (user != null)
                {
                    _context.Users.Remove(user);
                    await _context.SaveChangesAsync();
                    TempData["TeacherMessage"] = "Učitel úspěšně smazán.";
                }
                else
                {
                    TempData["TeacherMessage"] = "Chyba: učitel s loginem \"" + login + "\" nebyl nalezen.";
                }
            }
            return RedirectToAction(nameof(ManageUserListForAdmin));
        }

        public async Task<IActionResult> ManageUserRegistrationList()
        {
            if (!CanUserAccessPage(Config.Role.MainAdmin))
            {
                return AccessDeniedAction();
            }

            if (TempData["Message"] != null)
            {
                ViewBag.Message = TempData["Message"]!.ToString();
            }
            return _context.UserRegistrations != null ?
            View(await _context.UserRegistrations
                .Include(u => u.Student).ToListAsync()) :
            Problem("Entity set 'CourseContext.UserRegistrations'  is null.");
        }

        [HttpPost]
        [ValidateAntiForgeryToken]
        public async Task<IActionResult> ManageUserRegistrationList(string login, string email, string firstName, string lastName, string role, string action)
        {
            string? message = null;
            if(action == "acceptRegistration")
            {
                //for students, the entry has to already exist in the database (just without the email)
                if(role == "Student")
                {
                    var userByLogin = _context.Users.FirstOrDefault(u => u.Login == login);
                    var studentByLogin = _context.Students.FirstOrDefault(s => s.Login == login);

                    if (studentByLogin != null)
                    {
                        if (userByLogin != null)
                        {
                            message = "Chyba: uživatel s loginem \"" + login + "\" již existuje.";
                        }
                        else
                        {
                            var userByEmail = _context.Users.FirstOrDefault(u => u.Email == email);
                            var studentByEmail = _context.Students.FirstOrDefault(s => s.Email == email);
                            if (userByEmail != null || studentByEmail != null)
                            {
                                message = "Chyba: uživatel s emailem \"" + email + "\" již existuje.";
                            }
                            else
                            {
                                studentByLogin.Email = email;
                                studentByLogin.FirstName = firstName;
                                studentByLogin.LastName = lastName;

                                var userRegistration = _context.UserRegistrations.FirstOrDefault(u => u.Email == email);
                                if (userRegistration != null)
                                {
                                    userRegistration.State = Config.RegistrationState.Accepted;
                                    await _context.SaveChangesAsync();
                                    message = "Registrace úspěšně schválena.";
                                }
                                else
                                {
                                    message = "Chyba: registrace nebyla nalezena";
                                }
                            }
                        }
                    }
                    else
                    {
                        message = "Chyba: uživatel nebyl nalezen.";
                    }
                }
                else
                {
                    var userByEmail = _context.Users.FirstOrDefault(u => u.Email == email);
                    var studentByEmail = _context.Students.FirstOrDefault(s => s.Email == email);

                    var userByLogin = _context.Users.FirstOrDefault(u => u.Login == login);
                    var studentByLogin = _context.Students.FirstOrDefault(s => s.Login == login);

                    if (userByEmail != null || studentByEmail != null)
                    {
                        message = "Chyba: uživatel s emailem \"" + email + "\" již existuje.";
                    }
                    else if (userByLogin != null || studentByLogin != null)
                    {
                        message = "Chyba: uživatel s loginem \"" + login + "\" již existuje.";
                    }
                    else
                    {
                        User user = new User();
                        user.Email = email;
                        user.FirstName = firstName;
                        user.LastName = lastName;
                        user.Login = login;
                        user.Role = Enum.Parse<Config.Role>(role);
                        _context.Users.Add(user);
                        var userRegistration = _context.UserRegistrations.FirstOrDefault(u => u.Email == email);
                        if (userRegistration != null)
                        {
                            userRegistration.State = Config.RegistrationState.Accepted;
                            await _context.SaveChangesAsync();
                            message = "Registrace úspěšně schválena.";
                        }
                        else
                        {
                            message = "Chyba: registrace nebyla nalezena";
                        }
                    }
                }
            }
            else if(action == "refuseRegistration")
            {
                var userRegistration = _context.UserRegistrations.FirstOrDefault(u => u.Email == email);
                if (userRegistration != null)
                {
                    userRegistration.State = Config.RegistrationState.Rejected;
                    await _context.SaveChangesAsync();
                    message = "Registrace úspěšně zamítnuta.";
                }
                else
                {
                    message = "Chyba: registrace nebyla nalezena";
                }
            }
            else if(action == "deleteRegistration")
            {
                var userRegistration = _context.UserRegistrations.FirstOrDefault(u => u.Email == email);
                if (userRegistration != null)
                {
                    _context.UserRegistrations.Remove(userRegistration);
                    await _context.SaveChangesAsync();
                    message = "Registrace úspěšně smazána.";
                }
                else
                {
                    message = "Chyba: registrace nebyla nalezena";
                }
            }
            else if(action == "deleteAllRegistrations")
            {
                _context.Database.ExecuteSqlRaw("delete from UserRegistration");
                await _context.SaveChangesAsync();
                message = "Všechny registrace byly úspěšně smazány.";
            }
            TempData["Message"] = message;
            return RedirectToAction(nameof(ManageUserRegistrationList));
        }

        public async Task<IActionResult> ManageUserRegistrationListForAdmin()
        {
            if (!CanUserAccessPage(Config.Role.Admin))
            {
                return AccessDeniedAction();
            }

            if (TempData["Message"] != null)
            {
                ViewBag.Message = TempData["Message"]!.ToString();
            }
            return _context.UserRegistrations != null ?
            View(await _context.UserRegistrations
                .Include(u => u.Student).ToListAsync()) :
            Problem("Entity set 'CourseContext.UserRegistrations'  is null.");
        }

        [HttpPost]
        [ValidateAntiForgeryToken]
        public async Task<IActionResult> ManageUserRegistrationListForAdmin(string login, string email, string firstName, string lastName, string role, string action)
        {
            string? message = null;
            if (action == "acceptRegistration")
            {
                //for students, the entry has to already exist in the database (just without the email), for staff members it does not
                if (role == "Student")
                {
                    var userByLogin = _context.Students.FirstOrDefault(u => u.Login == login);
                    if (userByLogin != null)
                    {
                        if (userByLogin.Email != null)
                        {
                            message = "Chyba: uživatel s loginem \"" + login + "\" již existuje.";
                        }
                        else
                        {
                            var userByEmail = _context.Users.FirstOrDefault(u => u.Email == email);
                            if (userByEmail != null)
                            {
                                message = "Chyba: uživatel s emailem \"" + email + "\" již existuje.";
                            }
                            else
                            {
                                userByLogin.Email = email;
                                userByLogin.FirstName = firstName;
                                userByLogin.LastName = lastName;

                                var userRegistration = _context.UserRegistrations.FirstOrDefault(u => u.Email == email);
                                if (userRegistration != null)
                                {
                                    userRegistration.State = Config.RegistrationState.Accepted;
                                    await _context.SaveChangesAsync();
                                    message = "Registrace úspěšně schválena.";
                                }
                                else
                                {
                                    message = "Chyba: registrace nebyla nalezena";
                                }
                            }
                        }
                    }
                    else
                    {
                        message = "Chyba: uživatel nebyl nalezen.";
                    }
                }
                else
                {
                    var userByEmail = _context.Users.FirstOrDefault(u => u.Email == email);
                    var userByLogin = _context.Users.FirstOrDefault(u => u.Login == login);
                    if (userByEmail != null)
                    {
                        message = "Chyba: uživatel s emailem \"" + email + "\" již existuje.";
                    }
                    else if (userByLogin != null)
                    {
                        message = "Chyba: uživatel s loginem \"" + login + "\" již existuje.";
                    }
                    else
                    {
                        User user = new User();
                        user.Email = email;
                        user.FirstName = firstName;
                        user.LastName = lastName;
                        user.Login = login;
                        user.Role = Enum.Parse<Config.Role>(role);
                        _context.Users.Add(user);
                        var userRegistration = _context.UserRegistrations.FirstOrDefault(u => u.Email == email);
                        if (userRegistration != null)
                        {
                            userRegistration.State = Config.RegistrationState.Accepted;
                            await _context.SaveChangesAsync();
                            message = "Registrace úspěšně schválena.";
                        }
                        else
                        {
                            message = "Chyba: registrace nebyla nalezena";
                        }
                    }
                }
            }
            else if(action == "refuseRegistration")
            {
                var userRegistration = _context.UserRegistrations.FirstOrDefault(u => u.Email == email);
                if (userRegistration != null)
                {
                    userRegistration.State = Config.RegistrationState.Rejected;
                    await _context.SaveChangesAsync();
                    message = "Registrace úspěšně zamítnuta.";
                }
                else
                {
                    message = "Chyba: registrace nebyla nalezena";
                }
            }
            else if(action == "deleteRegistration")
            {
                var userRegistration = _context.UserRegistrations.FirstOrDefault(u => u.Email == email);
                if (userRegistration != null)
                {
                    _context.UserRegistrations.Remove(userRegistration);
                    await _context.SaveChangesAsync();
                    message = "Registrace úspěšně smazána.";
                }
                else
                {
                    message = "Chyba: registrace nebyla nalezena";
                }
            }
            TempData["Message"] = message;
            return RedirectToAction(nameof(ManageUserRegistrationListForAdmin));
        }

        public async Task<IActionResult> TeacherMenu()
        {
            if (!CanUserAccessPage(Config.Role.Teacher))
            {
                return AccessDeniedAction();
            }

            string login = Config.Application["login"];
            return _context.Users != null ?
                View(await _context.Users.FirstOrDefaultAsync(u => u.Login == login)) :
                Problem("Entity set 'CourseContext.Users'  is null.");
        }

        public async Task<IActionResult> AdminMenu()
        {
            if (!CanUserAccessPage(Config.Role.Admin))
            {
                return AccessDeniedAction();
            }

            string login = Config.Application["login"];
            return _context.Users != null ?
                View(await _context.Users.FirstOrDefaultAsync(u => u.Login == login)) :
                Problem("Entity set 'CourseContext.Users'  is null.");
        }

        public async Task<IActionResult> MainAdminMenu()
        {
            if (!CanUserAccessPage(Config.Role.MainAdmin))
            {
                return AccessDeniedAction();
            }

            return _context.Users != null ?
                View(await _context.Users.FirstOrDefaultAsync(u => u.Role == Config.Role.MainAdmin)) :
                Problem("Entity set 'CourseContext.Users'  is null.");
        }

        public async Task<IActionResult> GlobalSettings()
        {
            if (!CanUserAccessPage(Config.Role.MainAdmin))
            {
                return AccessDeniedAction();
            }

            if (TempData["Message"] != null)
            {
                ViewBag.Message = TempData["Message"]!.ToString();
            }
            return _context.GlobalSettings != null ?
                View(await _context.GlobalSettings.FirstOrDefaultAsync()) :
                Problem("Entity set 'CourseContext.GlobalSettings'  is null.");
        }

        [HttpPost]
        public async Task<IActionResult> GlobalSettings(string testingMode, string selectedPlatform)
        {
            var globalSettings = _context.GlobalSettings.FirstOrDefault();
            if(globalSettings != null)
            {
                if (testingMode == "testingModeOff")
                {
                    globalSettings.TestingMode = false;
                    Config.TestingMode = false;
                }
                else if (testingMode == "testingModeOn")
                {
                    globalSettings.TestingMode = true;
                    Config.TestingMode = true;
                }

                if(selectedPlatform == "windows")
                {
                    globalSettings.SelectedPlatform = Config.Platform.Windows;
                    Config.SelectedPlatform = Config.Platform.Windows;
                }
                else if(selectedPlatform == "ubuntu")
                {
                    globalSettings.SelectedPlatform = Config.Platform.Ubuntu;
                    Config.SelectedPlatform = Config.Platform.Ubuntu;
                }
                TempData["Message"] = "Změny úspěšně uloženy.";
                await _context.SaveChangesAsync();
            }
            return RedirectToAction(nameof(GlobalSettings));
        }

        public bool CanUserAccessPage(Config.Role requiredRole)
        {
            if(Config.TestingMode)//in case the testing mode is on, no authentication is required at all
            {
                return true;
            }

            string login = Config.Application["login"];
            var user = _context.Users.FirstOrDefault(u => u.Login == login);
            var student = _context.Students.FirstOrDefault(s => s.Login == login);

            if(requiredRole > Config.Role.Student)//staff member
            {
                if(user == null)
                {
                    return false;
                }
                if(user.Role < requiredRole)
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

        public IActionResult AccessDeniedAction()
        {
            throw Exceptions.AccessDeniedException;
        }

        public IActionResult NoElementsFoundAction()
        {
            throw Exceptions.NoElementsFoundException;
        }

        [ResponseCache(Duration = 0, Location = ResponseCacheLocation.None, NoStore = true)]
        public IActionResult Error()
        {
            return View(new ErrorViewModel { RequestId = Activity.Current?.Id ?? HttpContext.TraceIdentifier });
        }
    }
}