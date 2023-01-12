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
using static Common.EnumTypes;
using NuGet.Protocol.Plugins;

namespace ViewLayer.Controllers
{
    [Authorize]
    public class HomeController : Controller
    {
        private readonly CourseContext _context;
        private BusinessLayerFunctions businessLayerFunctions;

        private readonly ILogger<HomeController> _logger;
        private readonly IConfiguration _configuration;
        private readonly IWebHostEnvironment _environment;

        public HomeController(ILogger<HomeController> logger, IConfiguration configuration, IWebHostEnvironment environment, CourseContext context)
        {
            _logger = logger;
            _configuration = configuration;
            _environment = environment;
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
            View(await testTemplates.Include(t => t.Subject).ToListAsync()) :
            Problem("Entity set 'CourseContext.TestTemplates'  is null.");
        }

        [HttpPost]
        [ValidateAntiForgeryToken]
        public async Task<IActionResult> TestTemplateList(string action, string testTemplateId)
        {
            string login = businessLayerFunctions.GetCurrentUserLogin();
            if(action == "deleteAllTemplates")
            {
                TempData["Message"] = await businessLayerFunctions.DeleteTestTemplates(login);
            }
            else if(action == "deleteTemplate")
            {
                TempData["Message"] = await businessLayerFunctions.DeleteTestTemplate(login, int.Parse(testTemplateId), _environment.WebRootPath);
            }
            return RedirectToAction(nameof(TestTemplateList));
        }

        public async Task<IActionResult> TestTemplate(string testTemplateId)
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
            if (TempData["Message"] != null)
            {
                ViewBag.Message = TempData["Message"]!.ToString();
            }
            string login = businessLayerFunctions.GetCurrentUserLogin();

            var questionTemplates = businessLayerFunctions.GetQuestionTemplates(login, int.Parse(testTemplateId));

            if (questionTemplates.Count() > 0)
            {
                return View(await questionTemplates.ToListAsync());
            }
            else
            {
                //no question templates exist for this test template - we add a dummy question template to hold the test template data
                List<QuestionTemplate> questionTemplatesPlaceholder = new List<QuestionTemplate>();
                QuestionTemplate questionTemplate = new QuestionTemplate();
                questionTemplate.TestTemplate = businessLayerFunctions.GetTestTemplate(login, int.Parse(testTemplateId));
                questionTemplate.QuestionTemplateId = -1;
                questionTemplatesPlaceholder.Add(questionTemplate);
                return View(questionTemplatesPlaceholder);
            }
        }

        [HttpPost]
        public async Task<IActionResult> TestTemplate(string action, string testTemplateId, string negativePoints,
            string minimumPointsAmount, string questionTemplateId)
        {
            string login = businessLayerFunctions.GetCurrentUserLogin();
            string? negativePointsMessage = null;
            string? minimumPointsMessage = null;
            string? testDifficultyMessage = null;
            string? message = null;

            var testTemplate = businessLayerFunctions.GetTestTemplate(login, int.Parse(testTemplateId));
            if (action == "setNegativePoints")
            {
                negativePointsMessage = "Změny úspěšně uloženy.";
                await businessLayerFunctions.SetNegativePoints(testTemplate, (EnumTypes.NegativePoints)Convert.ToInt32(negativePoints));
            }
            else if(action == "setMinimumPoints")
            {
                if (testTemplate != null)
                {
                    minimumPointsMessage = await businessLayerFunctions.SetMinimumPoints(testTemplate, Math.Round(Convert.ToDouble(minimumPointsAmount), 2));
                }
            }
            else if (action == "getDifficultyPrediction")
            {
                testDifficultyMessage = businessLayerFunctions.GetTestDifficultyPrediction(login, int.Parse(testTemplateId));
            }
            else if (action == "deleteQuestionTemplate")
            {
                message = await businessLayerFunctions.DeleteQuestionTemplate(int.Parse(questionTemplateId), _environment.WebRootPath);
            }

            TempData["NegativePointsMessage"] = negativePointsMessage;
            TempData["MinimumPointsMessage"] = minimumPointsMessage;
            TempData["TestDifficultyMessage"] = testDifficultyMessage;
            TempData["Message"] = message;
            return RedirectToAction("TestTemplate", "Home", new { testTemplateId = testTemplateId });
        }

        public async Task<IActionResult> QuestionTemplate(string questionTemplateId)
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

            if (ViewBag.Message != "Podotázka byla úspěšně smazána." && TempData["subquestionTemplateId"] != null)//the user selected a subquestion from the dropdown menu
            {
                ViewBag.subquestionTemplateId = TempData["subquestionTemplateId"]!.ToString();
            }
            ViewBag.SubquestionTypeTextArray = businessLayerFunctions.GetSubquestionTypeTextArray();

            if (TempData["SuggestedSubquestionPoints"] != null)
            {
                ViewBag.SuggestedSubquestionPoints = TempData["SuggestedSubquestionPoints"]!.ToString();
            }

            var subquestionTemplates = businessLayerFunctions.GetSubquestionTemplates(login, int.Parse(questionTemplateId));
            List<SubquestionTemplate> subquestionTemplateList = await subquestionTemplates.ToListAsync();

            //users wants to browse either the previous or the following subquestion
            if(ViewBag.Message == "previousSubquestion" || ViewBag.Message == "nextSubquestion")
            {
                for(int i = 0; i < subquestionTemplateList.Count; i++)
                {
                    if (subquestionTemplateList[i].SubquestionTemplateId == ViewBag.subquestionTemplateId)
                    {
                        if (ViewBag.Message == "previousSubquestion" && i != 0)
                        {
                            ViewBag.subquestionTemplateId = subquestionTemplateList[i - 1].SubquestionTemplateId;
                            break;
                        }
                        if (ViewBag.Message == "nextSubquestion" && i != subquestionTemplateList.Count - 1)
                        {
                            ViewBag.subquestionTemplateId = subquestionTemplateList[i + 1].SubquestionTemplateId;
                            break;
                        }
                    }
                }
                ViewBag.Message = null;
            }

            subquestionTemplateList = businessLayerFunctions.ProcessSubquestionTemplatesForView(subquestionTemplateList);

            if (subquestionTemplates.Count() > 0)
            {
                return View(subquestionTemplateList);
            }
            else
            {
                //no subquestion templates exist for this test template - we add a dummy subquestion template to hold the question template data
                List<SubquestionTemplate> subquestionTemplatesPlaceholder = new List<SubquestionTemplate>();
                SubquestionTemplate subquestionTemplate = new SubquestionTemplate();
                subquestionTemplate.QuestionTemplate = businessLayerFunctions.GetQuestionTemplate(login, int.Parse(questionTemplateId));
                subquestionTemplate.SubquestionTemplateId = -1;
                subquestionTemplatesPlaceholder.Add(subquestionTemplate);
                return View(subquestionTemplatesPlaceholder);
            }
        }

        [HttpPost]
        public async Task<IActionResult> QuestionTemplate(string action, string questionTemplateId, string subquestionTemplateId, string subquestionPoints, 
            string wrongChoicePoints, string wrongChoicePointsRadio, string currentSubquestionTemplateId)
        {
            string login = businessLayerFunctions.GetCurrentUserLogin();
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
                    message = await businessLayerFunctions.SetSubquestionTemplatePoints(login, subquestionTemplateId, subquestionPoints, wrongChoicePoints, defaultWrongChoicePoints);
                    
                    //in case the subquestion points have been changed, we change the amount of student's points accordingly
                    if(message == "Počet bodů byl úspěšně změněn.")
                    {
                        await businessLayerFunctions.UpdateStudentsPoints(login, int.Parse(questionTemplateId), subquestionTemplateId);
                    }
                }
            }
            else if(action == "getPointsSuggestion")
            {
                TempData["SuggestedSubquestionPoints"] = await businessLayerFunctions.GetSubquestionTemplatePointsSuggestion(login, int.Parse(questionTemplateId), int.Parse(subquestionTemplateId));
            }
            else if (action == "editSubquestionTemplate")
            {
                return RedirectToAction("EditSubquestionTemplate", "Home", new { subquestionTemplateId = subquestionTemplateId });
            }
            else if (action == "deleteSubquestionTemplate")
            {
                message = await businessLayerFunctions.DeleteSubquestionTemplate(int.Parse(questionTemplateId), int.Parse(subquestionTemplateId), _environment.WebRootPath);
            }
            else if(action == "previousSubquestion" || action == "nextSubquestion")
            {
                subquestionTemplateId = currentSubquestionTemplateId;
                message = action;
            }

            TempData["Message"] = message;
            TempData["subquestionTemplateId"] = subquestionTemplateId;
            return RedirectToAction("QuestionTemplate", "Home", new { questionTemplateId = questionTemplateId });
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
        public async Task<IActionResult> ManageSolvedTestList(string action, string testResultId)
        {
            string login = businessLayerFunctions.GetCurrentUserLogin();
            if(action == "deleteAllResults")
            {
                TempData["Message"] = await businessLayerFunctions.DeleteTestResults(login);
            }
            else if(action == "deleteResult")
            {
                TempData["Message"] = await businessLayerFunctions.DeleteTestResult(login, int.Parse(testResultId));
            }
            return RedirectToAction(nameof(ManageSolvedTestList));
        }

        public async Task<IActionResult> ManageSolvedTest(string testResultId)
        {
            if (!businessLayerFunctions.CanUserAccessPage(EnumTypes.Role.Teacher))
            {
                return AccessDeniedAction();
            }

            string login = businessLayerFunctions.GetCurrentUserLogin();

            var questionResults = businessLayerFunctions.GetQuestionResultsByOwnerLogin(login, int.Parse(testResultId));

            if (questionResults.Count() > 0)
            {
                return View(await questionResults.ToListAsync());
            }
            else
            {
                return NoElementsFoundAction();
            }
        }

        public async Task<IActionResult> ManageSolvedQuestion(string questionResultId)
        {
            if (!businessLayerFunctions.CanUserAccessPage(EnumTypes.Role.Teacher))
            {
                return AccessDeniedAction();
            }

            if (TempData["Message"] != null)
            {
                ViewBag.Message = TempData["Message"]!.ToString();
            }

            if (TempData["subquestionResultId"] != null)
            {
                ViewBag.subquestionResultId = TempData["subquestionResultId"]!.ToString();
            }

            ViewBag.SubquestionTypeTextArray = businessLayerFunctions.GetSubquestionTypeTextArray();

            if (TempData["SuggestedSubquestionPoints"] != null)
            {
                ViewBag.SuggestedSubquestionPoints = TempData["SuggestedSubquestionPoints"]!.ToString();
            }

            string login = businessLayerFunctions.GetCurrentUserLogin();

            var subquestionResults = businessLayerFunctions.GetSubquestionResultsByOwnerLogin(login, int.Parse(questionResultId));
            List<SubquestionResult> subquestionResultList = await subquestionResults.ToListAsync();
            for(int i = 0; i < subquestionResultList.Count; i++)
            {
                subquestionResultList[i] = businessLayerFunctions.ProcessSubquestionResultForView(subquestionResultList[i]);
                subquestionResultList[i].SubquestionTemplate = businessLayerFunctions.ProcessSubquestionTemplateForView(subquestionResultList[i].SubquestionTemplate);
            }
            if (subquestionResults.Count() > 0)
            {
                return View(subquestionResultList);
            }
            else
            {
                return NoElementsFoundAction();
            }
        }

        [HttpPost]
        public async Task<IActionResult> ManageSolvedQuestion(string action, string questionResultId, string subquestionResultId,
            string studentsPoints, string subquestionPoints, string negativePoints, string subquestionType)
        {
            string login = businessLayerFunctions.GetCurrentUserLogin();
            string? message = null;
            if (action == "setPoints") 
            { 
                //the teacher is changing student's points
                if (studentsPoints != null)
                {
                    var subquestionResult = businessLayerFunctions.GetSubquestionResult(login, int.Parse(subquestionResultId));

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
                    var subquestionResult = businessLayerFunctions.GetSubquestionResult(login, int.Parse(subquestionResultId));
                    double answerCorrectness = subquestionResult.AnswerCorrectness;
                    if(answerCorrectness == 1)//in case student's answer is entirely correct, full amount of subquestion points gets recommended
                    {
                        TempData["SuggestedSubquestionPoints"] = subquestionPoints;
                    }
                    else
                    {
                        TempData["SuggestedSubquestionPoints"] = await businessLayerFunctions.GetSubquestionResultPointsSuggestion(login, int.Parse(subquestionResultId));
                    }
                }
            }
            TempData["Message"] = message;
            TempData["subquestionResultId"] = subquestionResultId;
            return RedirectToAction("ManageSolvedQuestion", "Home", new { questionResultId = questionResultId });
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

        public async Task<IActionResult> BrowseSolvedTest(string testResultId, string ownerLogin)
        {
            if (!businessLayerFunctions.CanUserAccessPage(EnumTypes.Role.Student))
            {
                return AccessDeniedAction();
            }

            string studentLogin = businessLayerFunctions.GetCurrentUserLogin();

            var questionResults = businessLayerFunctions.GetQuestionResultsByStudentLogin(studentLogin, ownerLogin, int.Parse(testResultId));

            if (questionResults.Count() > 0)
            {
                return View(await questionResults.ToListAsync());
            }
            else
            {
                return NoElementsFoundAction();
            }
        }

        public async Task<IActionResult> BrowseSolvedQuestion(string questionResultId, string ownerLogin)
        {
            if (!businessLayerFunctions.CanUserAccessPage(EnumTypes.Role.Student))
            {
                return AccessDeniedAction();
            }

            if (TempData["Message"] != null)
            {
                ViewBag.Message = TempData["Message"]!.ToString();
            }
            if (TempData["subquestionTemplateId"] != null)
            {
                ViewBag.subquestionTemplateId = TempData["subquestionTemplateId"]!.ToString();
            }
            ViewBag.SubquestionTypeTextArray = businessLayerFunctions.GetSubquestionTypeTextArray();
            string studentLogin = businessLayerFunctions.GetCurrentUserLogin();

            var subquestionResults = businessLayerFunctions.GetSubquestionResultsByStudentLogin(studentLogin, int.Parse(questionResultId));
            List<SubquestionResult> subquestionResultList = await subquestionResults.ToListAsync();
            for (int i = 0; i < subquestionResultList.Count; i++)
            {
                subquestionResultList[i] = businessLayerFunctions.ProcessSubquestionResultForView(subquestionResultList[i]);
                subquestionResultList[i].SubquestionTemplate = businessLayerFunctions.ProcessSubquestionTemplateForView(subquestionResultList[i].SubquestionTemplate);
            }
            if (subquestionResults.Count() > 0)
            {
                return View(subquestionResultList);
            }
            else
            {
                return NoElementsFoundAction();
            }
        }

        [HttpPost]
        public IActionResult BrowseSolvedQuestion(string questionTemplateId, string testResultId, string subquestionTemplateId, string ownerLogin)
        {
            TempData["subquestionTemplateId"] = subquestionTemplateId;
            return RedirectToAction("BrowseSolvedQuestion", "Home", new { questionTemplateId = questionTemplateId, testResultId = testResultId, ownerLogin = ownerLogin });
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
        public async Task<IActionResult> ManageUserList(string action, string firstName, string lastName, string login, string email, string role)
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

            if(action == "deleteAllStudents")
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
                    TempData["StudentMessage"] = await businessLayerFunctions.AddStudent(firstName, lastName, login, email, studentLoginCheck);
                }
            }
            else if(action == "editStudent")
            {
                //it's necessary to ensure that there won't be two or more users with the same email/login
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
                        await businessLayerFunctions.EditStudent(firstName, lastName, login, email, studentLoginCheck);
                        TempData["StudentMessage"] = "Student úspěšně upraven.";
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
        public async Task<IActionResult> ManageUserListForAdmin(string action, string firstName, string lastName, string login, string email)
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

            if (action == "addStudent")
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
                    TempData["StudentMessage"] = await businessLayerFunctions.AddStudent(firstName, lastName, login, email, studentLoginCheck);
                }
            }
            else if (action == "editStudent")
            {
                //it's necessary to ensure that there won't be two or more users with the same email/login
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
                        await businessLayerFunctions.EditStudent(firstName, lastName, login, email, studentLoginCheck);
                        TempData["StudentMessage"] = "Student úspěšně upraven.";
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

        public async Task<IActionResult> AddTestTemplate()
        {
            dynamic model = new ExpandoObject();
            model.TestTemplate = new TestTemplate();
            model.Subjects = await businessLayerFunctions.GetSubjectDbSet().Include(s => s.Guarantor).ToListAsync();
            return View(model);
        }

        [HttpPost]
        public async Task<IActionResult> AddTestTemplate(TestTemplate testTemplate, string subjectId)
        {
            string message = await businessLayerFunctions.AddTestTemplate(testTemplate, subjectId);
            TempData["Message"] = message;
            return RedirectToAction(nameof(TestTemplateList));
        }

        public IActionResult AddQuestionTemplate(string testTemplateId)
        {
            ViewBag.testTemplateId = testTemplateId;
            return View();
        }

        [HttpPost]
        public async Task<IActionResult> AddQuestionTemplate(QuestionTemplate questionTemplate, string testTemplateId)
        {
            string login = businessLayerFunctions.GetCurrentUserLogin();
            questionTemplate.OwnerLogin = login;
            questionTemplate.TestTemplate = businessLayerFunctions.GetTestTemplate(login, int.Parse(testTemplateId));
            string message = await businessLayerFunctions.AddQuestionTemplate(questionTemplate);
            TempData["Message"] = message;
            return RedirectToAction("TestTemplate", "Home", new { testTemplateId = testTemplateId });
        }

        public IActionResult AddSubquestionTemplate(string? questionTemplateId, SubquestionTemplate? subquestionTemplate)
        {
            if(questionTemplateId != null)
            {
                ViewBag.QuestionTemplateId = questionTemplateId;
            }
            else
            {
                ViewBag.QuestionTemplateId = subquestionTemplate.QuestionTemplateId;
            }
            ViewBag.SubquestionTypeTextArray = businessLayerFunctions.GetSubquestionTypeTextArray();
            if (TempData["SelectedSubquestionType"] != null)
            {
                ViewBag.SelectedSubquestionType = TempData["SelectedSubquestionType"]!.ToString();
            }
            if (TempData["Message"] != null)
            {
                ViewBag.Message = TempData["Message"]!.ToString();
            }
            if (TempData["SuggestedSubquestionPoints"] != null)
            {
                ViewBag.SuggestedSubquestionPoints = TempData["SuggestedSubquestionPoints"]!.ToString();
            }
            return View(subquestionTemplate);
        }

        [HttpPost]
        public async Task<IActionResult> AddSubquestionTemplate(string action, SubquestionTemplate subquestionTemplate,
            IFormFile image, string[] subquestionTextArray, string sliderValues)
        {
            string login = businessLayerFunctions.GetCurrentUserLogin();
            string message = string.Empty;

            if (action == "selectType")
            {
                TempData["SubquestionType"] = subquestionTemplate.SubquestionType;
            }
            else if (action == "addSubquestion")
            {
                subquestionTemplate.OwnerLogin = login;
                subquestionTemplate.QuestionTemplate = businessLayerFunctions.GetQuestionTemplate(login, subquestionTemplate.QuestionTemplateId);

                (subquestionTemplate, string? errorMessage) = businessLayerFunctions.ValidateSubquestionTemplate(subquestionTemplate, subquestionTextArray, sliderValues, image);
                if (errorMessage != null)
                {
                    message = errorMessage;
                }
                else
                {
                    message = await businessLayerFunctions.AddSubquestionTemplate(subquestionTemplate, image, _environment.WebRootPath);
                }
            }
            else if (action == "getPointsSuggestion")
            {
                subquestionTemplate.OwnerLogin = login;
                subquestionTemplate.QuestionTemplate = businessLayerFunctions.GetQuestionTemplate(login, subquestionTemplate.QuestionTemplateId);
                //TempData["SuggestedSubquestionPoints"] = await businessLayerFunctions.GetSubquestionTemplatePointsSuggestion(subquestionTemplate);
                TempData["SuggestedSubquestionPoints"] = "todo - subject";
            }

            TempData["Message"] = message;
            if (action == "selectType")
            {
                TempData["SelectedSubquestionType"] = subquestionTemplate.SubquestionType;
                return RedirectToAction("AddSubquestionTemplate", "Home", new { questionTemplateId = subquestionTemplate.QuestionTemplateId });
            }
            else if(action == "addSubquestion")
            {
                if(message != "Zadání podotázky bylo úspěšně přidáno.")
                {
                    return RedirectToAction("AddSubquestionTemplate", "Home", new { questionTemplateId = subquestionTemplate.QuestionTemplateId });
                }
                else
                {
                    TempData["subquestionTemplateId"] = subquestionTemplate.SubquestionTemplateId;
                    return RedirectToAction("QuestionTemplate", "Home", new { questionTemplateId = subquestionTemplate.QuestionTemplateId });
                }
            }
            else //getPointsSuggestion redirection
            {
                (subquestionTemplate, string? _) = businessLayerFunctions.ValidateSubquestionTemplate(subquestionTemplate, subquestionTextArray, sliderValues, null);
                TempData["SelectedSubquestionType"] = subquestionTemplate.SubquestionType;
                return RedirectToAction("AddSubquestionTemplate", "Home", new RouteValueDictionary(subquestionTemplate));
            }
        }

        public IActionResult EditSubquestionTemplate(string subquestionTemplateId)
        {
            string login = businessLayerFunctions.GetCurrentUserLogin();
            //todo: zmenit "-1"
            SubquestionTemplate subquestionTemplate = businessLayerFunctions.GetSubquestionTemplate(login, int.Parse(subquestionTemplateId));
            ViewBag.SubquestionTypeTextArray = businessLayerFunctions.GetSubquestionTypeTextArray();
            if (TempData["Message"] != null)
            {
                ViewBag.Message = TempData["Message"]!.ToString();
            }
            if (TempData["SuggestedSubquestionPoints"] != null)
            {
                ViewBag.SuggestedSubquestionPoints = TempData["SuggestedSubquestionPoints"]!.ToString();
            }
            return View(subquestionTemplate);
        }

        [HttpPost]
        public async Task<IActionResult> EditSubquestionTemplate(string action, SubquestionTemplate subquestionTemplate,
            IFormFile image, string[] subquestionTextArray, string sliderValues)
        {
            string login = businessLayerFunctions.GetCurrentUserLogin();
            string message = string.Empty;

            if (action == "editSubquestion")
            {
                subquestionTemplate.OwnerLogin = login;
                subquestionTemplate.QuestionTemplate = businessLayerFunctions.GetQuestionTemplate(login, subquestionTemplate.QuestionTemplateId);

                (subquestionTemplate, string? errorMessage) = businessLayerFunctions.ValidateSubquestionTemplate(subquestionTemplate, subquestionTextArray, sliderValues, image);
                if (errorMessage != null)
                {
                    message = errorMessage;
                }
                else
                {
                    message = await businessLayerFunctions.EditSubquestionTemplate(subquestionTemplate, image, _environment.WebRootPath);
                }
            }
            else if (action == "getPointsSuggestion")
            {
                subquestionTemplate.OwnerLogin = login;
                subquestionTemplate.QuestionTemplate = businessLayerFunctions.GetQuestionTemplate(login, subquestionTemplate.QuestionTemplateId);
                //TempData["SuggestedSubquestionPoints"] = await businessLayerFunctions.GetSubquestionTemplatePointsSuggestion(subquestionTemplate);
                TempData["SuggestedSubquestionPoints"] = "todo - subject";
            }

            TempData["Message"] = message;
            if (action == "editSubquestion")
            {
                if (message != "Zadání podotázky bylo úspěšně upraveno.")
                {
                    return RedirectToAction("EditSubquestionTemplate", "Home", new { subquestionTemplateId = subquestionTemplate.SubquestionTemplateId });
                }
                else
                {
                    TempData["subquestionTemplateId"] = subquestionTemplate.SubquestionTemplateId;
                    return RedirectToAction("QuestionTemplate", "Home", new { questionTemplateId = subquestionTemplate.QuestionTemplateId });
                }
            }
            else //getPointsSuggestion redirection
            {
                (subquestionTemplate, string? _) = businessLayerFunctions.ValidateSubquestionTemplate(subquestionTemplate, subquestionTextArray, sliderValues, null);
                return RedirectToAction("EditSubquestionTemplate", "Home", new RouteValueDictionary(subquestionTemplate));
            }
        }

        public async Task<IActionResult> ManageSubjects()
        {
            string login = businessLayerFunctions.GetCurrentUserLogin();
            ViewBag.login = login;
            var user = businessLayerFunctions.GetUserByLogin(login);
            if (user != null)
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

            var subjects = businessLayerFunctions.GetSubjects();
            return businessLayerFunctions.GetSubjectDbSet() != null ?
                View(await subjects.Include(s => s.Guarantor).ToListAsync()) :
                Problem("Entity set 'CourseContext.Subjects'  is null.");
        }

        [HttpPost]
        public async Task<IActionResult> ManageSubjects(string action, string subjectId)
        {
            if(action == "deleteSubject")
            {
                string message = await businessLayerFunctions.DeleteSubject(int.Parse(subjectId));
                TempData["Message"] = message;
                return RedirectToAction(nameof(ManageSubjects));
            }
            else
            {
                return RedirectToAction(nameof(ManageSubjects));
            }
        }

        public async Task<IActionResult> AddSubject()
        {
            dynamic model = new ExpandoObject();
            model.Subject = new Subject();
            model.Students = await businessLayerFunctions.GetStudentDbSet().ToListAsync();
            return (businessLayerFunctions.GetStudentDbSet() != null) ?
                View(model) :
            Problem("Entity set 'CourseContext.Students'  is null.");
        }

        [HttpPost]
        public async Task<IActionResult> AddSubject(Subject subject, string[] enrolledStudentLogin)
        {
            string message = await businessLayerFunctions.AddSubject(subject, enrolledStudentLogin);
            TempData["Message"] = message;
            return RedirectToAction(nameof(ManageSubjects));
        }

        public async Task<IActionResult> EditSubject(string subjectId)
        {
            Subject? subject = businessLayerFunctions.GetSubjectById(int.Parse(subjectId));
            dynamic model = new ExpandoObject();
            model.Subject = subject;
            model.Students = await businessLayerFunctions.GetStudentDbSet().ToListAsync();

            if (TempData["Message"] != null)
            {
                ViewBag.Message = TempData["Message"]!.ToString();
            }

            return (businessLayerFunctions.GetStudentDbSet() != null) ?
                View(model) :
            Problem("Entity set 'CourseContext.Students'  is null.");
        }

        [HttpPost]
        public async Task<IActionResult> EditSubject(Subject subject, string[] enrolledStudentLogin)
        {
            string message = await businessLayerFunctions.EditSubject(subject, enrolledStudentLogin);
            TempData["Message"] = message;
            if(message == "Předmět byl úspěšně upraven.")
            {
                return RedirectToAction(nameof(ManageSubjects));
            }
            else
            {
                return RedirectToAction("EditSubject", "Home", new { subjectId = subject.SubjectId });
            }
        }

        public async Task<IActionResult> StudentMenu()
        {
            string login = businessLayerFunctions.GetCurrentUserLogin();

            if (TempData["Message"] != null)
            {
                ViewBag.Message = TempData["Message"]!.ToString();
            }

            return businessLayerFunctions.GetStudentDbSet() != null ?
                View(await businessLayerFunctions.GetStudentDbSet().FirstOrDefaultAsync(s => s.Login == login)) :
                Problem("Entity set 'CourseContext.Students'  is null.");
        }

        public async Task<IActionResult> StudentAvailableTestList()
        {
            string login = businessLayerFunctions.GetCurrentUserLogin();
            return businessLayerFunctions.GetTestTemplateDbSet() != null ?
                View(await businessLayerFunctions.GetStudentAvailableTestList(login).ToListAsync()) :
                Problem("Entity set 'CourseContext.TestTemplates'  is null.");
        }
        
        public IActionResult StudentAvailableTest(string testTemplateId)
        {
            string login = businessLayerFunctions.GetCurrentUserLogin();
            TestTemplate testTemplate = new TestTemplate();

            if (TempData["Message"] != null)
            {
                ViewBag.Message = TempData["Message"]!.ToString();
            }

            if (businessLayerFunctions.CanStudentAccessTest(login, int.Parse(testTemplateId)))
            {
                testTemplate = businessLayerFunctions.GetTestTemplate(int.Parse(testTemplateId));
            }
            else
            {
                ViewBag.Message = "K tomuto testu nemáte přístup.";
            }
            return businessLayerFunctions.GetTestTemplateDbSet() != null ?
                View(testTemplate) :
                Problem("Entity set 'CourseContext.TestTemplates'  is null.");
        }

        [HttpPost]
        public async Task<IActionResult> StudentAvailableTest(string action, string testTemplateId)
        {
            string login = businessLayerFunctions.GetCurrentUserLogin();
            string? errorMessage = null;
            if (action == "beginAttempt")
            {
                if (businessLayerFunctions.CanStudentAccessTest(login, int.Parse(testTemplateId)))
                {
                    errorMessage = await businessLayerFunctions.BeginStudentAttempt(int.Parse(testTemplateId), login);
                }
            }
            else
            {
                return RedirectToAction("StudentAvailableTest", "Home", new { testTemplateId = testTemplateId });
            }
            if(errorMessage == null)
            {
                return RedirectToAction("SolveQuestion", "Home");
            }
            else
            {
                TempData["Message"] = errorMessage;
                return RedirectToAction("StudentAvailableTest", "Home", new { testTemplateId = testTemplateId });
            }
        }

        public async Task<IActionResult> SolveQuestion(int questionNr)
        {
            string login = businessLayerFunctions.GetCurrentUserLogin();
            TestResult testResult = await businessLayerFunctions.LoadLastStudentAttempt(login);
            List<(int, AnswerCompleteness)> subquestionResultsProperties = businessLayerFunctions.GetSubquestionResultsProperties(testResult);
            SubquestionResult subquestionResult = new SubquestionResult();

            if (TempData["Message"] != null)
            {
                ViewBag.Message = TempData["Message"]!.ToString();
            }

            //the user has just started the attempt - the very first subquestion is shown to him then
            if (businessLayerFunctions.GetStudentSubquestionResultId() == null || !subquestionResultsProperties.Any(s => s.Item1 == int.Parse(businessLayerFunctions.GetStudentSubquestionResultId())))
            {
                for(int i = 0; i < testResult.QuestionResultList.Count; i++)
                {
                    QuestionResult questionResult = testResult.QuestionResultList.ElementAt(i);

                    //the question must include at least one subquestion
                    if(questionResult.SubquestionResultList.Count > 0)
                    {
                        subquestionResult = questionResult.SubquestionResultList.ElementAt(0);
                        break;
                    }
                }
                businessLayerFunctions.SetStudentSubquestionResultId(subquestionResult.SubquestionResultId);

                ViewBag.SubquestionsCount = subquestionResultsProperties.Count;
                ViewBag.SubquestionResultIdIndex = 0;
            }
            else
            {
                int newSubquestionResultId;
                if(questionNr < 0 || questionNr > subquestionResultsProperties.Count())
                {
                    newSubquestionResultId = subquestionResultsProperties[0].Item1;
                    ViewBag.Message = "Chyba: tato otázka neexistuje.";
                }
                else
                {
                    newSubquestionResultId = subquestionResultsProperties[questionNr].Item1;
                }
                businessLayerFunctions.SetStudentSubquestionResultId(newSubquestionResultId);

                for (int i = 0; i < testResult.QuestionResultList.Count; i++)
                {
                    QuestionResult questionResult = testResult.QuestionResultList.ElementAt(i);

                    for(int j = 0; j < questionResult.SubquestionResultList.Count; j++)
                    {
                        SubquestionResult subquestionResultTemp = questionResult.SubquestionResultList.ElementAt(j);
                        if(subquestionResultTemp.SubquestionResultId == newSubquestionResultId)
                        {
                            subquestionResult = subquestionResultTemp;
                            break;
                        }
                    }
                }

                ViewBag.SubquestionsCount = subquestionResultsProperties.Count;
                ViewBag.SubquestionResultIdIndex = questionNr;
            }
            int[] answerCompleteness = new int[subquestionResultsProperties.Count];
            for(int i = 0; i < subquestionResultsProperties.Count; i++)
            {
                answerCompleteness[i] = (int)subquestionResultsProperties[i].Item2;
            }
            ViewBag.AnswerCmpleteness = answerCompleteness;
            return View(subquestionResult);
        }

        [HttpPost]
        public async Task<IActionResult> SolveQuestion(SubquestionResult subquestionResult, string newSubquestionResultIndex, int subquestionResultIndex, 
            string[] possibleAnswers, string action)
        {
            string login = businessLayerFunctions.GetCurrentUserLogin();
            string? errorMessage;
            (subquestionResult, errorMessage) = await businessLayerFunctions.ValidateSubquestionResult(subquestionResult, subquestionResultIndex, login, possibleAnswers);
            if(errorMessage != null)
            {
                TempData["Message"] = errorMessage;
                return RedirectToAction("SolveQuestion", "Home", new { questionNr = newSubquestionResultIndex });
            }
            else
            {
                await businessLayerFunctions.UpdateSubquestionResultStudentsAnswers(subquestionResult, subquestionResultIndex, login);
                if(action == "turnTestIn")
                {
                    await businessLayerFunctions.FinishStudentAttempt(login);
                    TempData["Message"] = "Test byl odevdzán.";
                    return RedirectToAction("StudentMenu", "Home", new { questionNr = newSubquestionResultIndex });
                }
                else
                {
                    return RedirectToAction("SolveQuestion", "Home", new { questionNr = newSubquestionResultIndex });
                }
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