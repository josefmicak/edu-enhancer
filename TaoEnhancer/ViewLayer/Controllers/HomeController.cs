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
using BusinessLayer;

namespace ViewLayer.Controllers
{
    [Authorize]
    public class HomeController : Controller
    {
        private readonly CourseContext _context;
        private BusinessLayerFunctions businessLayerFunctions;

        private readonly ILogger<HomeController> _logger;
        private readonly IConfiguration _configuration;

        public HomeController(ILogger<HomeController> logger, IConfiguration configuration, CourseContext context)
        {
            _logger = logger;
            _configuration = configuration;
            _context = context;
            businessLayerFunctions = new BusinessLayerFunctions(context, configuration);

            businessLayerFunctions.InitialTestingModeSettings();
            businessLayerFunctions.SelectedPlatformSettings();

           // DataGenerator.GenerateTemplatesFile("none");

            List<TestTemplate> testTemplates = businessLayerFunctions.GetTestTemplatesByLogin("login");
          //  DataGenerator.GenerateResultsFile(testTemplates, "on");

            Config.GoogleClientId = businessLayerFunctions.GetGoogleClientId();
        }

        [AllowAnonymous]
        public async Task<IActionResult> Index()
        {
            if(businessLayerFunctions.GetUserList().Count() == 0)
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
                model.Users = await businessLayerFunctions.GetUserDbSet().ToListAsync();
                model.Students = await businessLayerFunctions.GetStudentDbSet().ToListAsync();
                return (businessLayerFunctions.GetUserDbSet() != null || businessLayerFunctions.GetStudentDbSet() != null) ?
                    View(model) :
                Problem("Entity set 'CourseContext.Users' or 'CourseContext.Students'  is null.");
            }
        }

        [HttpPost]
        [AllowAnonymous]
        public IActionResult Index(string selectedUserLogin)
        {
            User? user = businessLayerFunctions.GetUserByLogin(selectedUserLogin);
            Student? student = businessLayerFunctions.GetStudentByLogin(selectedUserLogin);

            businessLayerFunctions.SetCurrentUserLogin(selectedUserLogin);
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
            if (!businessLayerFunctions.CanUserAccessPage(EnumTypes.Role.Teacher))
            {
                return AccessDeniedAction();
            }

            string login = businessLayerFunctions.GetCurrentUserLogin();
            var user = businessLayerFunctions.GetUserByLogin(login);
            if(user != null)
            {
                if (user.Role == EnumTypes.Role.Teacher)
                {
                    ViewBag.Return = "TeacherMenu";
                }
                else if (user.Role == EnumTypes.Role.Admin)
                {
                    ViewBag.Return = "AdminMenu";
                }
                else if (user.Role == EnumTypes.Role.MainAdmin)
                {
                    ViewBag.Return = "MainAdminMenu";
                }
            }
            if (TempData["Message"] != null)
            {
                ViewBag.Message = TempData["Message"]!.ToString();
            }

            var testTemplates = businessLayerFunctions.GetTestTemplates(login);
            return businessLayerFunctions.GetTestTemplateDbSet() != null ?
            View(await testTemplates.ToListAsync()) :
            Problem("Entity set 'CourseContext.TestTemplates'  is null.");
        }

        [HttpPost]
        [ValidateAntiForgeryToken]
        public async Task<IActionResult> TestTemplateList(string action, string testNumberIdentifier)
        {
            string login = businessLayerFunctions.GetCurrentUserLogin();
            if (action == "add")
            {
                TempData["Message"] = await businessLayerFunctions.AddTestTemplates(login);
            }
            else if(action == "deleteAllTemplates")
            {
                TempData["Message"] = await businessLayerFunctions.DeleteTestTemplates(login);
            }
            else if(action == "deleteTemplate")
            {
                TempData["Message"] = await businessLayerFunctions.DeleteTestTemplate(login, testNumberIdentifier);
            }
            return RedirectToAction(nameof(TestTemplateList));
        }

        public async Task<IActionResult> TestTemplate(string testNumberIdentifier)
        {
            if (!businessLayerFunctions.CanUserAccessPage(EnumTypes.Role.Teacher))
            {
                return AccessDeniedAction();
            }

            if (TempData["NegativePointsMessage"] != null)
            {
                ViewBag.NegativePointsMessage = TempData["NegativePointsMessage"]!.ToString();
            }
            if (TempData["MinimumPointsMessage"] != null)
            {
                ViewBag.MinimumPointsMessage = TempData["MinimumPointsMessage"]!.ToString();
            }
            if (TempData["TestDifficultyMessage"] != null)
            {
                ViewBag.TestDifficultyMessage = TempData["TestDifficultyMessage"]!.ToString();
            }
            string login = businessLayerFunctions.GetCurrentUserLogin();

            var questionTemplates = businessLayerFunctions.GetQuestionTemplates(login, testNumberIdentifier);

            if (questionTemplates.Count() > 0)
            {
                return View(await questionTemplates.ToListAsync());
            }
            else
            {
                //no question templates exist for this test template - we add a dummy question template to hold the test template data
                List<QuestionTemplate> questionTemplatesPlaceholder = new List<QuestionTemplate>();
                QuestionTemplate questionTemplate = new QuestionTemplate();
                questionTemplate.TestTemplate = businessLayerFunctions.GetTestTemplate(login, testNumberIdentifier);
                questionTemplate.QuestionNumberIdentifier = "placeholder";
                questionTemplatesPlaceholder.Add(questionTemplate);
                return View(questionTemplatesPlaceholder);
            }
        }

        [HttpPost]
        public async Task<IActionResult> TestTemplate(string action, string testNumberIdentifier, string negativePoints,
            string minimumPointsAmount, string testPointsDetermined)
        {
            string login = businessLayerFunctions.GetCurrentUserLogin();
            if (minimumPointsAmount != null)
            {
                if (Config.SelectedPlatform == EnumTypes.Platform.Windows)
                {
                    minimumPointsAmount = minimumPointsAmount.Replace(".", ",");
                }
                else if (Config.SelectedPlatform == EnumTypes.Platform.Linux)
                {
                    minimumPointsAmount = minimumPointsAmount.Replace(",", ".");
                }
            }
            string? negativePointsMessage = null;
            string? minimumPointsMessage = null;
            string? testDifficultyMessage = null;

            var testTemplate = businessLayerFunctions.GetTestTemplate(login, testNumberIdentifier);
            if (action == "setNegativePoints")
            {
                negativePointsMessage = "Změny úspěšně uloženy.";
                await businessLayerFunctions.SetNegativePoints(testTemplate, (EnumTypes.NegativePoints)Convert.ToInt32(negativePoints));
            }
            else if(action == "setMinimumPoints")
            {
                if (testTemplate != null)
                {
                    minimumPointsMessage = await businessLayerFunctions.SetMinimumPoints(testTemplate, Math.Round(Convert.ToDouble(minimumPointsAmount), 2), testPointsDetermined);
                }
            }
            else if (action == "getDifficultyPrediction")
            {
                testDifficultyMessage = businessLayerFunctions.GetTestDifficultyPrediction(login, testNumberIdentifier);

            }

            TempData["NegativePointsMessage"] = negativePointsMessage;
            TempData["MinimumPointsMessage"] = minimumPointsMessage;
            TempData["TestDifficultyMessage"] = testDifficultyMessage;
            return RedirectToAction("TestTemplate", "Home", new { testNumberIdentifier = testNumberIdentifier });
        }

        public async Task<IActionResult> QuestionTemplate(string questionNumberIdentifier)
        {
            if (!businessLayerFunctions.CanUserAccessPage(EnumTypes.Role.Teacher))
            {
                return AccessDeniedAction();
            }

            string login = businessLayerFunctions.GetCurrentUserLogin();
            if (TempData["Message"] != null)
            {
                ViewBag.Message = TempData["Message"]!.ToString();
            }

            if (TempData["subquestionIdentifier"] != null)//the user selected a subquestion from the dropdown menu
            {
                ViewBag.subquestionIdentifier = TempData["subquestionIdentifier"]!.ToString();
            }
            ViewBag.SubquestionTypeTextArray = businessLayerFunctions.GetSubquestionTypeTextArray();

            if (TempData["SuggestedSubquestionPoints"] != null)
            {
                ViewBag.SuggestedSubquestionPoints = TempData["SuggestedSubquestionPoints"]!.ToString();
            }

            var subquestionTemplates = businessLayerFunctions.GetSubquestionTemplates(login, questionNumberIdentifier);

            if (subquestionTemplates.Count() > 0)
            {
                return View(await subquestionTemplates.ToListAsync());
            }
            else
            {
                //no subquestion templates exist for this test template - we add a dummy subquestion template to hold the question template data
                List<SubquestionTemplate> subquestionTemplatesPlaceholder = new List<SubquestionTemplate>();
                SubquestionTemplate subquestionTemplate = new SubquestionTemplate();
                subquestionTemplate.QuestionTemplate = businessLayerFunctions.GetQuestionTemplate(login, questionNumberIdentifier);
                subquestionTemplate.SubquestionIdentifier = "placeholder";
                subquestionTemplatesPlaceholder.Add(subquestionTemplate);
                return View(subquestionTemplatesPlaceholder);
            }
        }

        [HttpPost]
        public async Task<IActionResult> QuestionTemplate(string action, string questionNumberIdentifier, string subquestionIdentifier, string subquestionPoints, string wrongChoicePoints, string wrongChoicePointsRadio)
        {
            string login = businessLayerFunctions.GetCurrentUserLogin();
            if (subquestionPoints != null)
            {
                if (Config.SelectedPlatform == EnumTypes.Platform.Windows)
                {
                    subquestionPoints = subquestionPoints.Replace(".", ",");
                }
                else if (Config.SelectedPlatform == EnumTypes.Platform.Linux)
                {
                    subquestionPoints = subquestionPoints.Replace(",", ".");
                }
            }
            string? message = null;

            if (action == "savePoints")
            {
                //the teacher is changing points of the subquestion
                if (subquestionPoints != null)
                {
                    bool defaultWrongChoicePoints = false;
                    if(wrongChoicePointsRadio == "wrongChoicePoints_automatic_radio")
                    {
                        defaultWrongChoicePoints = true;
                    }
                    message = await businessLayerFunctions.SetSubquestionTemplatePoints(login, questionNumberIdentifier, subquestionIdentifier, subquestionPoints, wrongChoicePoints, defaultWrongChoicePoints);
                    
                    //in case the subquestion points have been changed, we change the amount of student's points accordingly
                    if(message == "Počet bodů byl úspěšně změněn.")
                    {
                        await businessLayerFunctions.UpdateStudentsPoints(login, questionNumberIdentifier, subquestionIdentifier);
                    }
                }
            }
            else if(action == "getPointsSuggestion")
            {
                TempData["SuggestedSubquestionPoints"] = await businessLayerFunctions.GetSubquestionTemplatePointsSuggestion(login, questionNumberIdentifier, subquestionIdentifier);
            }

            TempData["Message"] = message;
            TempData["subquestionIdentifier"] = subquestionIdentifier;
            return RedirectToAction("QuestionTemplate", "Home", new { questionNumberIdentifier = questionNumberIdentifier });
        }
        
        public async Task<IActionResult> ManageSolvedTestList()
        {
            if (!businessLayerFunctions.CanUserAccessPage(EnumTypes.Role.Teacher))
            {
                return AccessDeniedAction();
            }

            string login = businessLayerFunctions.GetCurrentUserLogin();
            var user = businessLayerFunctions.GetUserByLogin(login);
            if(user != null)
            {
                if (user.Role == EnumTypes.Role.Teacher)
                {
                    ViewBag.Return = "TeacherMenu";
                }
                else if (user.Role == EnumTypes.Role.Admin)
                {
                    ViewBag.Return = "AdminMenu";
                }
                else if (user.Role == EnumTypes.Role.MainAdmin)
                {
                    ViewBag.Return = "MainAdminMenu";
                }
            }
            if (TempData["Message"] != null)
            {
                ViewBag.Message = TempData["Message"]!.ToString();
            }

            var testResults = businessLayerFunctions.GetTestResultsByOwnerLogin(login);
            return businessLayerFunctions.GetTestResultDbSet() != null ?
            View(await testResults.ToListAsync()) :
            Problem("Entity set 'CourseContext.TestResults'  is null.");
        }

        [HttpPost]
        [ValidateAntiForgeryToken]
        public async Task<IActionResult> ManageSolvedTestList(string action, string testResultIdentifier)
        {
            string login = businessLayerFunctions.GetCurrentUserLogin();
            if (action == "add")
            {
                TempData["Message"] = await businessLayerFunctions.AddTestResults(login);
            }
            else if(action == "deleteAllResults")
            {
                TempData["Message"] = await businessLayerFunctions.DeleteTestResults(login);
            }
            else if(action == "deleteResult")
            {
                TempData["Message"] = await businessLayerFunctions.DeleteTestResult(login, testResultIdentifier);
            }
            return RedirectToAction(nameof(ManageSolvedTestList));
        }

        public async Task<IActionResult> ManageSolvedTest(string testResultIdentifier)
        {
            if (!businessLayerFunctions.CanUserAccessPage(EnumTypes.Role.Teacher))
            {
                return AccessDeniedAction();
            }

            string login = businessLayerFunctions.GetCurrentUserLogin();

            var questionResults = businessLayerFunctions.GetQuestionResultsByOwnerLogin(login, testResultIdentifier);

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
            if (!businessLayerFunctions.CanUserAccessPage(EnumTypes.Role.Teacher))
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

            ViewBag.SubquestionTypeTextArray = businessLayerFunctions.GetSubquestionTypeTextArray();

            if (TempData["SuggestedSubquestionPoints"] != null)
            {
                ViewBag.SuggestedSubquestionPoints = TempData["SuggestedSubquestionPoints"]!.ToString();
            }

            string login = businessLayerFunctions.GetCurrentUserLogin();

            var subquestionResults = businessLayerFunctions.GetSubquestionResultsByOwnerLogin(login, testResultIdentifier, questionNumberIdentifier);

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
        public async Task<IActionResult> ManageSolvedQuestion(string action, string questionNumberIdentifier, string testResultIdentifier, string subquestionIdentifier,
            string studentsPoints, string subquestionPoints, string negativePoints, string subquestionType)
        {
            string login = businessLayerFunctions.GetCurrentUserLogin();
            string? message = null;
            if (action == "setPoints") 
            {
                if (studentsPoints != null)
                {
                    studentsPoints = studentsPoints.Replace(".", ",");//todo: formatting
                }
                
                //the teacher is changing student's points
                if (studentsPoints != null)
                {
                    var subquestionResult = businessLayerFunctions.GetSubquestionResult(login, testResultIdentifier, questionNumberIdentifier, subquestionIdentifier);

                    if (subquestionResult != null)
                    {
                        message = await businessLayerFunctions.SetSubquestionResultPoints(subquestionPoints, studentsPoints, negativePoints, subquestionResult);
                    }
                }
            }
            else if(action == "getPointsSuggestion")
            {
                int subquestionTypeInt = Convert.ToInt32(subquestionType);
                //for these types of subquestion AI can't predict the amount of points
                if (subquestionTypeInt == 0 || subquestionTypeInt == 5)
                {
                    TempData["SuggestedSubquestionPoints"] = "0";
                }
                else
                {
                    var subquestionResult = businessLayerFunctions.GetSubquestionResult(login, testResultIdentifier, questionNumberIdentifier, subquestionIdentifier);
                    double answerCorrectness = subquestionResult.AnswerCorrectness;
                    if(answerCorrectness == 1)//in case student's answer is entirely correct, full amount of subquestion points gets recommended
                    {
                        TempData["SuggestedSubquestionPoints"] = subquestionPoints;
                    }
                    else
                    {
                        TempData["SuggestedSubquestionPoints"] = await businessLayerFunctions.GetSubquestionResultPointsSuggestion(login, testResultIdentifier, questionNumberIdentifier, subquestionIdentifier);
                    }
                }
            }
            TempData["Message"] = message;
            TempData["subquestionIdentifier"] = subquestionIdentifier;
            return RedirectToAction("ManageSolvedQuestion", "Home", new { questionNumberIdentifier = questionNumberIdentifier, testResultIdentifier = testResultIdentifier });
        }

        public async Task<IActionResult> BrowseSolvedTestList()
        {
            if (!businessLayerFunctions.CanUserAccessPage(EnumTypes.Role.Student))
            {
                return AccessDeniedAction();
            }

            string login = businessLayerFunctions.GetCurrentUserLogin();
            dynamic model = new ExpandoObject();
            model.TestResults = await businessLayerFunctions.GetTestResultsByStudentLogin(login).ToListAsync();
            model.Student = businessLayerFunctions.GetStudentByLogin(login);
            return businessLayerFunctions.GetTestResultDbSet() != null ?
                View(model) :
            Problem("Entity set 'CourseContext.TestResults'  is null.");
        }

        public async Task<IActionResult> BrowseSolvedTest(string testResultIdentifier, string ownerLogin)
        {
            if (!businessLayerFunctions.CanUserAccessPage(EnumTypes.Role.Student))
            {
                return AccessDeniedAction();
            }

            string studentLogin = businessLayerFunctions.GetCurrentUserLogin();

            var questionResults = businessLayerFunctions.GetQuestionResultsByStudentLogin(studentLogin, ownerLogin, testResultIdentifier);

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
            if (!businessLayerFunctions.CanUserAccessPage(EnumTypes.Role.Student))
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
            ViewBag.SubquestionTypeTextArray = businessLayerFunctions.GetSubquestionTypeTextArray();
            string studentLogin = businessLayerFunctions.GetCurrentUserLogin();

            var subquestionResults = businessLayerFunctions.GetSubquestionResultsByStudentLogin(studentLogin, ownerLogin, testResultIdentifier, questionNumberIdentifier);
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

            if(businessLayerFunctions.GetUserDbSet().Count() == 0)
            {
                ViewBag.firstRegistrationMessage = "Po zaregistrování vám bude automaticky vytvořen účet hlavního administrátora.";
            }

            var userRegistrations = businessLayerFunctions.GetUserRegistrations(Config.Application["email"]);
            return businessLayerFunctions.GetUserRegistrationDbSet() != null ?
            View(await userRegistrations.ToListAsync()) :
            Problem("Entity set 'CourseContext.UserRegistrations'  is null.");
        }

        [HttpPost]
        public async Task<IActionResult> UserRegistration(string? firstName, string? lastName, string? login, string role)
        {
            string email = Config.Application["email"];
            if(businessLayerFunctions.GetUserDbSet().Count() == 0)
            {
                if (firstName == null || lastName == null || login == null || email == null)
                {
                    TempData["Message"] = "Chyba: všechny položky musí být vyplněny.";
                    return RedirectToAction(nameof(UserRegistration));
                }
                else
                {
                    await businessLayerFunctions.RegisterMainAdmin(firstName, lastName, email, login);
                    return RedirectToAction(nameof(MainAdminMenu));
                }
            }
            else
            {
                TempData["Message"] = await businessLayerFunctions.CreateUserRegistration(firstName, lastName, email, login, role);
                return RedirectToAction(nameof(UserRegistration));
            }
        }

        public async Task<IActionResult> ManageUserList()
        {
            if (!businessLayerFunctions.CanUserAccessPage(EnumTypes.Role.MainAdmin))
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
            model.Users = await businessLayerFunctions.GetUserDbSet().ToListAsync();
            model.Students = await businessLayerFunctions.GetStudentDbSet().ToListAsync();
            return (businessLayerFunctions.GetUserDbSet() != null || businessLayerFunctions.GetStudentDbSet() != null) ?
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

            var studentLoginCheck = businessLayerFunctions.GetStudentByLogin(login);
            var studentEmailCheck = businessLayerFunctions.GetStudentByEmail(email);

            var userLoginCheck = businessLayerFunctions.GetUserByLogin(login);
            var userEmailCheck = businessLayerFunctions.GetUserByEmail(email);

            if (action == "addStudents")
            {
                TempData["StudentMessage"] = await businessLayerFunctions.AddStudents(login);
            }
            else if(action == "deleteAllStudents")
            {
                await businessLayerFunctions.DeleteAllStudents();
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
                    TempData["StudentMessage"] = await businessLayerFunctions.AddStudent(studentIdentifier, firstName, lastName, login, email, studentLoginCheck);
                }
            }
            else if(action == "editStudent")
            {
                //it's necessary to ensure that there won't be two or more users with the same email/identifier
                if(studentLoginCheck != null)
                {
                    var studentByNewEmail = businessLayerFunctions.GetStudentByEmail(email);
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
                        var studentByNewIdentifier = businessLayerFunctions.GetStudentByIdentifier(studentIdentifier);
                        if (studentByNewIdentifier != null && studentLoginCheck.StudentIdentifier != studentByNewIdentifier.StudentIdentifier)
                        {
                            TempData["StudentMessage"] = "Chyba: student s identifikátorem \"" + studentByNewIdentifier.Email + "\" již existuje.";
                        }
                        else
                        {
                            await businessLayerFunctions.EditStudent(studentIdentifier, firstName, lastName, login, email, studentLoginCheck);
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
                var student = businessLayerFunctions.GetStudentByLogin(login);
                if(student != null)
                {
                    await businessLayerFunctions.DeleteStudent(student);
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
                    await businessLayerFunctions.AddTeacher(firstName, lastName, login, email);
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
                        await businessLayerFunctions.EditUser(userLoginCheck, firstName, lastName, login, email, role);
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
                    await businessLayerFunctions.DeleteUser(userLoginCheck);
                    TempData["TeacherMessage"] = "Učitel úspěšně smazán.";
                }
                else
                {
                    TempData["TeacherMessage"] = "Chyba: učitel s loginem \"" + login + "\" nebyl nalezen.";
                }
            }
            else if(action == "deleteAllTeachers")
            {
                await businessLayerFunctions.DeleteAllTeachers();
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
                    await businessLayerFunctions.AddAdmin(firstName, lastName, login, email);
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
                                await businessLayerFunctions.EditUser(userLoginCheck, firstName, lastName, login, email, role);
                                TempData["AdminMessage"] = "Správce úspěšně upraven.";
                            }
                        }
                        else if(action == "changeMainAdmin")
                        {
                            await businessLayerFunctions.ChangeMainAdmin(userLoginCheck, firstName, lastName, login, email);
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
                    if(userLoginCheck.Role == EnumTypes.Role.MainAdmin)
                    {
                        TempData["AdminMessage"] = "Chyba: účet hlavního administrátora nelze smazat.";
                    }
                    else
                    {
                        await businessLayerFunctions.DeleteUser(userLoginCheck);
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
                await businessLayerFunctions.DeleteAllAdmins();
                TempData["AdminMessage"] = "Byly smazáni všichni existující správci.";
            }
            return RedirectToAction(nameof(ManageUserList));
        }

        public async Task<IActionResult> ManageUserListForAdmin()
        {
            if (!businessLayerFunctions.CanUserAccessPage(EnumTypes.Role.Admin))
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
            model.Users = await businessLayerFunctions.GetUserDbSet().ToListAsync();
            model.Students = await businessLayerFunctions.GetStudentDbSet().ToListAsync();
            return (businessLayerFunctions.GetUserDbSet() != null || businessLayerFunctions.GetStudentDbSet() != null) ?
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

            var studentLoginCheck = businessLayerFunctions.GetStudentByLogin(login);
            var studentEmailCheck = businessLayerFunctions.GetStudentByEmail(email);

            var userLoginCheck = businessLayerFunctions.GetUserByLogin(login);
            var userEmailCheck = businessLayerFunctions.GetUserByEmail(email);

            if (action == "addStudents")
            {
                TempData["StudentMessage"] = await businessLayerFunctions.AddStudents(login);
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
                    TempData["StudentMessage"] = await businessLayerFunctions.AddStudent(studentIdentifier, firstName, lastName, login, email, studentLoginCheck);
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
                        var studentByNewIdentifier = businessLayerFunctions.GetStudentByIdentifier(studentIdentifier);
                        if (studentByNewIdentifier != null && studentLoginCheck.StudentIdentifier != studentByNewIdentifier.StudentIdentifier)
                        {
                            TempData["StudentMessage"] = "Chyba: student s identifikátorem \"" + studentByNewIdentifier.Email + "\" již existuje.";
                        }
                        else
                        {
                            await businessLayerFunctions.EditStudent(studentIdentifier, firstName, lastName, login, email, studentLoginCheck);
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
                var student = businessLayerFunctions.GetStudentByLogin(login);
                if (student != null)
                {
                    await businessLayerFunctions.DeleteStudent(student);
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
                    await businessLayerFunctions.AddTeacher(firstName, lastName, login, email);
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
                        await businessLayerFunctions.EditUser(userLoginCheck, firstName, lastName, login, email, "2");
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
                var user = businessLayerFunctions.GetUserByLogin(login);
                if (user != null)
                {
                    await businessLayerFunctions.DeleteUser(user);
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
            if (!businessLayerFunctions.CanUserAccessPage(EnumTypes.Role.MainAdmin))
            {
                return AccessDeniedAction();
            }

            if (TempData["Message"] != null)
            {
                ViewBag.Message = TempData["Message"]!.ToString();
            }
            return businessLayerFunctions.GetUserRegistrationDbSet() != null ?
            View(await businessLayerFunctions.GetUserRegistrationDbSet()
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
                    var userByLogin = businessLayerFunctions.GetUserByLogin(login);
                    var studentByLogin = businessLayerFunctions.GetStudentByLogin(login);

                    if (studentByLogin != null)
                    {
                        if (userByLogin != null)
                        {
                            message = "Chyba: uživatel s loginem \"" + login + "\" již existuje.";
                        }
                        else
                        {
                            var userByEmail = businessLayerFunctions.GetUserByEmail(email);
                            var studentByEmail = businessLayerFunctions.GetStudentByEmail(email);
                            if (userByEmail != null || studentByEmail != null)
                            {
                                message = "Chyba: uživatel s emailem \"" + email + "\" již existuje.";
                            }
                            else
                            {
                                message = await businessLayerFunctions.ApproveStudentRegistration(studentByLogin, firstName, lastName, login, email);
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
                    var userByEmail = businessLayerFunctions.GetUserByEmail(email);
                    var studentByEmail = businessLayerFunctions.GetStudentByEmail(email);

                    var userByLogin = businessLayerFunctions.GetUserByLogin(login);
                    var studentByLogin = businessLayerFunctions.GetStudentByLogin(login);

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
                        message = await businessLayerFunctions.ApproveUserRegistration(firstName, lastName, login, email, role);
                    }
                }
            }
            else if(action == "refuseRegistration")
            {
                message = await businessLayerFunctions.RefuseRegistration(email);
            }
            else if(action == "deleteRegistration")
            {
                message = await businessLayerFunctions.DeleteRegistration(email);
            }
            else if(action == "deleteAllRegistrations")
            {
                await businessLayerFunctions.DeleteAllRegistrations();
                message = "Všechny registrace byly úspěšně smazány.";
            }
            TempData["Message"] = message;
            return RedirectToAction(nameof(ManageUserRegistrationList));
        }

        public async Task<IActionResult> ManageUserRegistrationListForAdmin()
        {
            if (!businessLayerFunctions.CanUserAccessPage(EnumTypes.Role.Admin))
            {
                return AccessDeniedAction();
            }

            if (TempData["Message"] != null)
            {
                ViewBag.Message = TempData["Message"]!.ToString();
            }
            return businessLayerFunctions.GetUserRegistrationDbSet() != null ?
            View(await businessLayerFunctions.GetUserRegistrationDbSet()
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
                //for students, the entry has to already exist in the database (just without the email)
                if (role == "Student")
                {
                    var userByLogin = businessLayerFunctions.GetUserByLogin(login);
                    var studentByLogin = businessLayerFunctions.GetStudentByLogin(login);

                    if (studentByLogin != null)
                    {
                        if (userByLogin != null)
                        {
                            message = "Chyba: uživatel s loginem \"" + login + "\" již existuje.";
                        }
                        else
                        {
                            var userByEmail = businessLayerFunctions.GetUserByEmail(email);
                            var studentByEmail = businessLayerFunctions.GetStudentByEmail(email);
                            if (userByEmail != null || studentByEmail != null)
                            {
                                message = "Chyba: uživatel s emailem \"" + email + "\" již existuje.";
                            }
                            else
                            {
                                message = await businessLayerFunctions.ApproveStudentRegistration(studentByLogin, firstName, lastName, login, email);
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
                    var userByEmail = businessLayerFunctions.GetUserByEmail(email);
                    var studentByEmail = businessLayerFunctions.GetStudentByEmail(email);

                    var userByLogin = businessLayerFunctions.GetUserByLogin(login);
                    var studentByLogin = businessLayerFunctions.GetStudentByLogin(login);

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
                        message = await businessLayerFunctions.ApproveUserRegistration(firstName, lastName, login, email, role);
                    }
                }
            }
            else if (action == "refuseRegistration")
            {
                message = await businessLayerFunctions.RefuseRegistration(email);
            }
            else if (action == "deleteRegistration")
            {
                message = await businessLayerFunctions.DeleteRegistration(email);
            }
            TempData["Message"] = message;
            return RedirectToAction(nameof(ManageUserRegistrationListForAdmin));
        }

        public async Task<IActionResult> TeacherMenu()
        {
            if (!businessLayerFunctions.CanUserAccessPage(EnumTypes.Role.Teacher))
            {
                return AccessDeniedAction();
            }

            string login = businessLayerFunctions.GetCurrentUserLogin();
            return businessLayerFunctions.GetUserDbSet() != null ?
                View(await businessLayerFunctions.GetUserDbSet().FirstOrDefaultAsync(u => u.Login == login)) :
                Problem("Entity set 'CourseContext.Users'  is null.");
        }

        public async Task<IActionResult> AdminMenu()
        {
            if (!businessLayerFunctions.CanUserAccessPage(EnumTypes.Role.Admin))
            {
                return AccessDeniedAction();
            }

            string login = businessLayerFunctions.GetCurrentUserLogin();
            return businessLayerFunctions.GetUserDbSet() != null ?
                View(await businessLayerFunctions.GetUserDbSet().FirstOrDefaultAsync(u => u.Login == login)) :
                Problem("Entity set 'CourseContext.Users'  is null.");
        }

        public async Task<IActionResult> MainAdminMenu()
        {
            if (!businessLayerFunctions.CanUserAccessPage(EnumTypes.Role.MainAdmin))
            {
                return AccessDeniedAction();
            }

            return businessLayerFunctions.GetUserDbSet() != null ?
                View(await businessLayerFunctions.GetUserDbSet().FirstOrDefaultAsync(u => u.Role == EnumTypes.Role.MainAdmin)) :
                Problem("Entity set 'CourseContext.Users'  is null.");
        }

        public async Task<IActionResult> GlobalSettings()
        {
            if (!businessLayerFunctions.CanUserAccessPage(EnumTypes.Role.MainAdmin))
            {
                return AccessDeniedAction();
            }

            if (TempData["Message"] != null)
            {
                ViewBag.Message = TempData["Message"]!.ToString();
            }
            return businessLayerFunctions.GetGlobalSettingsDbSet() != null ?
                View(await businessLayerFunctions.GetGlobalSettingsDbSet().FirstOrDefaultAsync()) :
                Problem("Entity set 'CourseContext.GlobalSettings'  is null.");
        }

        [HttpPost]
        public async Task<IActionResult> GlobalSettings(string testingMode)
        {
            await businessLayerFunctions.ChangeGlobalSettings(testingMode);
            TempData["Message"] = "Změny úspěšně uloženy.";
            return RedirectToAction(nameof(GlobalSettings));
        }

        public async Task<IActionResult> ManageArtificialIntelligence()
        {
            string login = businessLayerFunctions.GetCurrentUserLogin();
            if (!businessLayerFunctions.CanUserAccessPage(EnumTypes.Role.MainAdmin))
            {
                return AccessDeniedAction();
            }

            if (TempData["Message"] != null)
            {
                ViewBag.Message = TempData["Message"]!.ToString();
            }

            if (TempData["DeviceName"] != null)
            {
                ViewBag.DeviceName = TempData["DeviceName"]!.ToString();
            }
            ViewBag.TestingDataSubquestionTemplates = businessLayerFunctions.GetTestingDataSubquestionTemplatesCount();
            ViewBag.TestingDataSubquestionResults = businessLayerFunctions.GetTestingDataSubquestionResultsCount();

            dynamic model = new ExpandoObject();
            model.SubquestionTemplateStatistics = await businessLayerFunctions.GetSubquestionTemplateStatisticsDbSet().ToListAsync();
            model.SubquestionResultStatistics = await businessLayerFunctions.GetSubquestionResultStatisticsDbSet().ToListAsync();

            return (businessLayerFunctions.GetSubquestionTemplateStatisticsDbSet() != null || businessLayerFunctions.GetSubquestionResultStatisticsDbSet() != null) ?
                View(model) :
            Problem("Entity set 'CourseContext.SubquestionTemplateStatistics' or 'CourseContext.SubquestionResultStatistics' is null.");
        }

        [HttpPost]
        public async Task<IActionResult> ManageArtificialIntelligence(string action, string amountOfSubquestionTemplates, string amountOfSubquestionResults)
        {
            if(action == "addSubquestionTemplateRandomData" || action == "addSubquestionTemplateCorrelationalData")
            {
                TempData["Message"] = await businessLayerFunctions.CreateTemplateTestingData(action, amountOfSubquestionTemplates);
            }
            else if(action == "deleteSubquestionTemplateTestingData")
            {
                await businessLayerFunctions.DeleteTemplateTestingData();
                TempData["Message"] = "Testovací data úspěšně vymazány.";
            }
            else if (action == "addSubquestionResultRandomData" || action == "addSubquestionResultCorrelationalData")
            {
                TempData["Message"] = await businessLayerFunctions.CreateResultTestingData(action, amountOfSubquestionResults);
            }
            else if (action == "deleteSubquestionResultTestingData")
            {
                await businessLayerFunctions.DeleteResultTestingData();
                TempData["Message"] = "Testovací data úspěšně vymazány.";
            }
            else if(action == "getDeviceName")
            {
                TempData["DeviceName"] = businessLayerFunctions.GetAIDeviceName();
            }

            return RedirectToAction(nameof(ManageArtificialIntelligence));
        }

        public IActionResult AddTestTemplate()
        {
            return View();
        }

        [HttpPost]
        public async Task<IActionResult> AddTestTemplate(string title)
        {
            string message = await businessLayerFunctions.AddTestTemplate(title);
            TempData["Message"] = message;
            return RedirectToAction(nameof(TestTemplateList));
        }

        public IActionResult AddQuestionTemplate(string testNumberIdentifier)
        {
            ViewBag.TestNumberIdentifier = testNumberIdentifier;
            return View();
        }

        [HttpPost]
        public async Task<IActionResult> AddQuestionTemplate(string title, string testNumberIdentifier)
        {
            string message = await businessLayerFunctions.AddQuestionTemplate(testNumberIdentifier, title);
            TempData["Message"] = message;
            return RedirectToAction("TestTemplate", "Home", new { testNumberIdentifier = testNumberIdentifier });
        }

        public IActionResult AddSubquestionTemplate(string questionNumberIdentifier)
        {
            ViewBag.QuestionNumberIdentifier = questionNumberIdentifier;
            ViewBag.SubquestionTypeTextArray = businessLayerFunctions.GetSubquestionTypeTextArray();
            if (TempData["SelectedSubquestionType"] != null)
            {
                ViewBag.SelectedSubquestionType = TempData["SelectedSubquestionType"]!.ToString();
            }
            return View();
        }

        [HttpPost]
        public async Task<IActionResult> AddSubquestionTemplate(string action, SubquestionTemplate subquestionTemplate, string questionNumberIdentifier,
            string subquestionPoints, string correctChoicePoints, string wrongChoicePointsRadio, string wrongChoicePoints_manual)
        {
            string login = businessLayerFunctions.GetCurrentUserLogin();
            string message = string.Empty;

            if (action == "selectType")
            {
                TempData["SubquestionType"] = subquestionTemplate.SubquestionType;
            }
            else if (action == "addSubquestion")
            {
                if (Config.SelectedPlatform == EnumTypes.Platform.Windows)
                {
                    subquestionPoints = subquestionPoints.Replace(".", ",");
                    correctChoicePoints = correctChoicePoints.Replace(".", ",");
                    wrongChoicePoints_manual = wrongChoicePoints_manual.Replace(".", ",");
                }
                else if (Config.SelectedPlatform == EnumTypes.Platform.Linux)
                {
                    subquestionPoints = subquestionPoints.Replace(",", ".");
                    correctChoicePoints = correctChoicePoints.Replace(".", ",");
                    wrongChoicePoints_manual = wrongChoicePoints_manual.Replace(".", ",");
                }
                /*SubquestionTemplate subquestionTemplate = new SubquestionTemplate();
                subquestionTemplate.SubquestionIdentifier = "sub";
                subquestionTemplate.QuestionTemplate = businessLayerFunctions.GetQuestionTemplate(login, questionNumberIdentifier);
                subquestionTemplate.QuestionNumberIdentifier = questionNumberIdentifier;
                subquestionTemplate.OwnerLogin = login;
                subquestionTemplate.SubquestionType = EnumTypes.SubquestionType.MultipleQuestions;
                subquestionTemplate.SubquestionText = subquestionTemplatexxx.SubquestionText;
                subquestionTemplate.ImageSource = "";
                subquestionTemplate.PossibleAnswerList = new string[] { "test1", "test2" };
                subquestionTemplate.CorrectAnswerList = new string[] { "test3", "test4" };
                subquestionTemplate.SubquestionPoints = 10;
                subquestionTemplate.CorrectChoicePoints = 10;
                subquestionTemplate.DefaultWrongChoicePoints = 10;
                subquestionTemplate.WrongChoicePoints = 10;*/
                subquestionTemplate.OwnerLogin = login;
                subquestionTemplate.QuestionTemplate = businessLayerFunctions.GetQuestionTemplate(login, questionNumberIdentifier);

                subquestionTemplate.SubquestionPoints = double.Parse(subquestionPoints);
                subquestionTemplate.CorrectChoicePoints = double.Parse(correctChoicePoints);
                subquestionTemplate.DefaultWrongChoicePoints = double.Parse(correctChoicePoints) * (-1);
                if (wrongChoicePointsRadio == "wrongChoicePoints_automatic_radio")
                {
                    subquestionTemplate.WrongChoicePoints = double.Parse(correctChoicePoints) * (-1);
                }
                else if (wrongChoicePointsRadio == "wrongChoicePoints_manual_radio")
                {
                    subquestionTemplate.WrongChoicePoints = double.Parse(wrongChoicePoints_manual);
                }
                message = await businessLayerFunctions.AddSubquestionTemplate(subquestionTemplate);
            }

            TempData["Message"] = message;
            if (action == "selectType")
            {
                TempData["SelectedSubquestionType"] = subquestionTemplate.SubquestionType;
                return RedirectToAction("AddSubquestionTemplate", "Home", new { questionNumberIdentifier = questionNumberIdentifier });
                //return RedirectToAction(nameof(AddSubquestionTemplate));
            }
            else
            {
                return RedirectToAction("QuestionTemplate", "Home", new { questionNumberIdentifier = questionNumberIdentifier });
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