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
using static Microsoft.EntityFrameworkCore.DbLoggerCategory;
using Microsoft.EntityFrameworkCore.Storage.ValueConversion.Internal;

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
            Config.GoogleClientId = businessLayerFunctions.GetGoogleClientId();
        }

        [AllowAnonymous]
        public async Task<IActionResult> Index()
        {
            List<User> userList = await businessLayerFunctions.GetUserDbSet().ToListAsync();
            if (userList.Count == 0)
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
        public async Task<IActionResult> Index(string selectedUserLogin)
        {
            User? user = await businessLayerFunctions.GetUserByLoginNullable(selectedUserLogin);
            Student? student = await businessLayerFunctions.GetStudentByLoginNullable(selectedUserLogin);

            SetCurrentUserLogin(selectedUserLogin);
            if (user != null)
            {
                return RedirectToAction("TestingModeLogin", "Account", new {name = user.FullName(), email = user.Email });
            }
            else if (student != null)
            {
                return RedirectToAction("TestingModeLogin", "Account", new { name = student.FullName(), email = student.Email });
            }
            //throw an exception in case no user or student with this login exists
            throw Exceptions.UserNotFoundException(selectedUserLogin);
        }

        /// <summary>
        /// Returns login of currently logged in user
        /// </summary>
        public string GetCurrentUserLogin()
        {
            if (HttpContext.Session.GetString("login") == null)
            {
                throw Exceptions.UserLoggedOutException();
            }
            else
            {
                return HttpContext.Session.GetString("login")!;
            }
        }

        /// <summary>
        /// Sets user's login after he connects using OAuth 2.0 (in case he's previously made an account)
        /// </summary>
        public void SetCurrentUserLogin(string login)
        {
            HttpContext.Session.SetString("login", login);
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

            string login = GetCurrentUserLogin();
            var user = await businessLayerFunctions.GetUserByLoginNullable(login);
            var student = await businessLayerFunctions.GetStudentByLoginNullable(login);

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

        public async Task<IActionResult> TestTemplateList()
        {
            if (!await CanUserAccessPage(EnumTypes.Role.Teacher))
            {
                return AccessDeniedAction();
            }

            string login = GetCurrentUserLogin();
            User user = await businessLayerFunctions.GetUserByLogin(login);
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
            if (TempData["Message"] != null)
            {
                ViewBag.Message = TempData["Message"]!.ToString();
            }

            List<TestTemplate> testTemplates = await businessLayerFunctions.GetTestTemplates(login);
            return businessLayerFunctions.GetTestTemplateDbSet() != null ?
            View(testTemplates) :
            Problem("Entity set 'CourseContext.TestTemplates'  is null.");
        }

        [HttpPost]
        [ValidateAntiForgeryToken]
        public async Task<IActionResult> TestTemplateList(string action, string testTemplateId)
        {
            if(action == "deleteAllTemplates")
            {
                TempData["Message"] = await businessLayerFunctions.DeleteTestTemplates(GetCurrentUserLogin());
            }
            else if(action == "deleteTemplate")
            {
                TempData["Message"] = await businessLayerFunctions.DeleteTestTemplate(int.Parse(testTemplateId), _environment.WebRootPath, GetCurrentUserLogin());
            }
            return RedirectToAction(nameof(TestTemplateList));
        }

        public async Task<IActionResult> TestTemplate(string testTemplateId)
        {
            if (!await CanUserAccessPage(EnumTypes.Role.Teacher))
            {
                return AccessDeniedAction();
            }
            if (TempData["TestDifficultyMessage"] != null)
            {
                ViewBag.TestDifficultyMessage = TempData["TestDifficultyMessage"]!.ToString();
            }
            if (TempData["Message"] != null)
            {
                ViewBag.Message = TempData["Message"]!.ToString();
            }

            List<QuestionTemplate> questionTemplates = await businessLayerFunctions.GetQuestionTemplates(int.Parse(testTemplateId));

            if (questionTemplates.Count > 0)
            {
                return View(questionTemplates);
            }
            else
            {
                //no question templates exist for this test template - we add a dummy question template to hold the test template data
                List<QuestionTemplate> questionTemplatesPlaceholder = new List<QuestionTemplate>();
                QuestionTemplate questionTemplate = new QuestionTemplate();
                questionTemplate.TestTemplate = await businessLayerFunctions.GetTestTemplate(int.Parse(testTemplateId));
                questionTemplate.QuestionTemplateId = -1;
                questionTemplatesPlaceholder.Add(questionTemplate);
                return View(questionTemplatesPlaceholder);
            }
        }

        [HttpPost]
        public async Task<IActionResult> TestTemplate(string action, string testTemplateId, string questionTemplateId)
        {
            string? testDifficultyMessage = null;
            string? message = null;

            TestTemplate testTemplate = await businessLayerFunctions.GetTestTemplate(int.Parse(testTemplateId));
            if (action == "getDifficultyPrediction")
            {
                testDifficultyMessage = await businessLayerFunctions.GetTestDifficultyPrediction(GetCurrentUserLogin(), int.Parse(testTemplateId));
            }
            else if (action == "deleteQuestionTemplate")
            {
                if (!businessLayerFunctions.CanUserEditTestTemplate(testTemplate, GetCurrentUserLogin()))
                {
                    message = "Chyba: tento test nemůžete upravovat (nemáte oprávnění nebo již test začal).";
                }
                else
                {
                    message = await businessLayerFunctions.DeleteQuestionTemplate(int.Parse(questionTemplateId), _environment.WebRootPath, GetCurrentUserLogin());
                }
            }

            TempData["TestDifficultyMessage"] = testDifficultyMessage;
            TempData["Message"] = message;
            return RedirectToAction("TestTemplate", "Home", new { testTemplateId = testTemplateId });
        }

        public async Task<IActionResult> QuestionTemplate(string questionTemplateId, string subquestionTemplateId)
        {
            if (!await CanUserAccessPage(EnumTypes.Role.Teacher))
            {
                return AccessDeniedAction();
            }

            if (TempData["Message"] != null)
            {
                ViewBag.Message = TempData["Message"]!.ToString();
            }

            ViewBag.SubquestionTypeTextArray = businessLayerFunctions.GetSubquestionTypeTextArray();

            if (TempData["SuggestedSubquestionPoints"] != null)
            {
                ViewBag.SuggestedSubquestionPoints = TempData["SuggestedSubquestionPoints"]!.ToString();
            }

            List<SubquestionTemplate> subquestionTemplateList = await businessLayerFunctions.GetSubquestionTemplates(int.Parse(questionTemplateId));

            //users wants to browse either the previous or the following subquestion
            if (ViewBag.Message == "previousSubquestion" || ViewBag.Message == "nextSubquestion")
            {
                for(int i = 0; i < subquestionTemplateList.Count; i++)
                {
                    if (subquestionTemplateList[i].SubquestionTemplateId == int.Parse(subquestionTemplateId))
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
            //user has deleted a subquestion
            else if(ViewBag.Message == "Podotázka byla úspěšně smazána.")
            {
                ViewBag.subquestionTemplateId = subquestionTemplateList[0].SubquestionTemplateId;
            }
            //user has selected a subquestion from the dropdown menu
            else
            {
                int id;
                if(int.TryParse(subquestionTemplateId, out id))
                {
                    ViewBag.subquestionTemplateId = id;
                }
            }

            subquestionTemplateList = businessLayerFunctions.ProcessSubquestionTemplatesForView(subquestionTemplateList);

            if (subquestionTemplateList.Count > 0)
            {
                return View(subquestionTemplateList);
            }
            else
            {
                //no subquestion templates exist for this test template - we add a dummy subquestion template to hold the question template data
                List<SubquestionTemplate> subquestionTemplatesPlaceholder = new List<SubquestionTemplate>();
                SubquestionTemplate subquestionTemplate = new SubquestionTemplate();
                subquestionTemplate.QuestionTemplate = await businessLayerFunctions.GetQuestionTemplate(int.Parse(questionTemplateId));
                subquestionTemplate.SubquestionTemplateId = -1;
                subquestionTemplatesPlaceholder.Add(subquestionTemplate);
                return View(subquestionTemplatesPlaceholder);
            }
        }

        [HttpPost]
        public async Task<IActionResult> QuestionTemplate(string action, string questionTemplateId, string subquestionTemplateId, 
            string currentSubquestionTemplateId)
        {
            string? message = null;

            if (action == "editSubquestionTemplate")
            {
                QuestionTemplate questionTemplate = await businessLayerFunctions.GetQuestionTemplate(int.Parse(questionTemplateId));
                if (!businessLayerFunctions.CanUserEditTestTemplate(questionTemplate.TestTemplate, GetCurrentUserLogin()))
                {
                    message = "Chyba: tento test nemůžete upravovat (nemáte oprávnění nebo již test začal).";
                }
                else
                {
                    return RedirectToAction("EditSubquestionTemplate", "Home", new { subquestionTemplateId = subquestionTemplateId });
                }
            }
            else if (action == "deleteSubquestionTemplate")
            {
                QuestionTemplate questionTemplate = await businessLayerFunctions.GetQuestionTemplate(int.Parse(questionTemplateId));
                if (!businessLayerFunctions.CanUserEditTestTemplate(questionTemplate.TestTemplate, GetCurrentUserLogin()))
                {
                    message = "Chyba: tento test nemůžete upravovat (nemáte oprávnění nebo již test začal).";
                }
                else
                {
                    message = await businessLayerFunctions.DeleteSubquestionTemplate(int.Parse(subquestionTemplateId), _environment.WebRootPath, GetCurrentUserLogin());
                }
            }
            else if(action == "previousSubquestion" || action == "nextSubquestion")
            {
                subquestionTemplateId = currentSubquestionTemplateId;
                message = action;
            }

            TempData["Message"] = message;
            return RedirectToAction("QuestionTemplate", "Home", new { questionTemplateId = questionTemplateId, subquestionTemplateId = subquestionTemplateId });
        }
        
        public async Task<IActionResult> ManageSolvedTestList()
        {
            if (!await CanUserAccessPage(EnumTypes.Role.Teacher))
            {
                return AccessDeniedAction();
            }

            string login = GetCurrentUserLogin();
            User user = await businessLayerFunctions.GetUserByLogin(login);
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
            if (TempData["Message"] != null)
            {
                ViewBag.Message = TempData["Message"]!.ToString();
            }

            List<TestResult> testResults = await businessLayerFunctions.GetTurnedTestResults(login);
            return businessLayerFunctions.GetTestResultDbSet() != null ?
            View(testResults) :
            Problem("Entity set 'CourseContext.TestResults'  is null.");
        }

        [HttpPost]
        [ValidateAntiForgeryToken]
        public async Task<IActionResult> ManageSolvedTestList(string action, string testResultId)
        {
            if(action == "deleteAllResults")
            {
                TempData["Message"] = await businessLayerFunctions.DeleteTestResults(GetCurrentUserLogin());
            }
            else if(action == "deleteResult")
            {
                TempData["Message"] = await businessLayerFunctions.DeleteTestResult(int.Parse(testResultId), GetCurrentUserLogin());
            }
            return RedirectToAction(nameof(ManageSolvedTestList));
        }

        public async Task<IActionResult> ManageSolvedTest(string testResultId)
        {
            if (!await CanUserAccessPage(EnumTypes.Role.Teacher))
            {
                return AccessDeniedAction();
            }
            TestResult testResult = await businessLayerFunctions.GetTestResult(int.Parse(testResultId));
            if (!testResult.IsTurnedIn)
            {
                TempData["Message"] = "Chyba: tento test ještě nebyl odevdzán a nemůže být upravován.";
                return RedirectToAction(nameof(ManageSolvedTestList));
            }

            string login = GetCurrentUserLogin();

            List<QuestionResult> questionResults = await businessLayerFunctions.GetQuestionResults(int.Parse(testResultId));

            if (questionResults.Count > 0)
            {
                return View(questionResults);
            }
            else
            {
                return NoElementsFoundAction();
            }
        }

        public async Task<IActionResult> ManageSolvedQuestion(string questionResultId, string subquestionResultId)
        {
            if (!await CanUserAccessPage(EnumTypes.Role.Teacher))
            {
                return AccessDeniedAction();
            }

            if (TempData["Message"] != null)
            {
                ViewBag.Message = TempData["Message"]!.ToString();
            }

            ViewBag.SubquestionTypeTextArray = businessLayerFunctions.GetSubquestionTypeTextArray();

            if (TempData["SuggestedSubquestionPoints"] != null)
            {
                ViewBag.SuggestedSubquestionPoints = TempData["SuggestedSubquestionPoints"]!.ToString();
            }

            List<SubquestionResult> subquestionResultList = await businessLayerFunctions.GetSubquestionResults(int.Parse(questionResultId));

            //users wants to browse either the previous or the following subquestion
            if (ViewBag.Message == "previousSubquestion" || ViewBag.Message == "nextSubquestion")
            {
                for (int i = 0; i < subquestionResultList.Count; i++)
                {
                    if (subquestionResultList[i].SubquestionResultId == int.Parse(subquestionResultId))
                    {
                        if (ViewBag.Message == "previousSubquestion" && i != 0)
                        {
                            ViewBag.subquestionResultId = subquestionResultList[i - 1].SubquestionResultId;
                            break;
                        }
                        if (ViewBag.Message == "nextSubquestion" && i != subquestionResultList.Count - 1)
                        {
                            ViewBag.subquestionResultId = subquestionResultList[i + 1].SubquestionResultId;
                            break;
                        }
                    }
                }
                ViewBag.Message = null;
            }
            //user has selected a subquestion from the dropdown menu
            else
            {
                int id;
                if (int.TryParse(subquestionResultId, out id))
                {
                    ViewBag.subquestionResultId = id;
                }
            }

            for (int i = 0; i < subquestionResultList.Count; i++)
            {
                subquestionResultList[i] = businessLayerFunctions.ProcessSubquestionResultForView(subquestionResultList[i]);
                subquestionResultList[i].SubquestionTemplate = businessLayerFunctions.ProcessSubquestionTemplateForView(subquestionResultList[i].SubquestionTemplate);
            }
            if (subquestionResultList.Count > 0)
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
            string studentsPoints, string subquestionPoints, string negativePoints, string subquestionType, string currentSubquestionResultId)
        {
            string? message = null;
            if (action == "setPoints") 
            { 
                //the teacher is changing student's points
                if (studentsPoints != null)
                {
                    SubquestionResult subquestionResult = await businessLayerFunctions.GetSubquestionResult(int.Parse(subquestionResultId));

                    if (subquestionResult != null)
                    {
                        message = await businessLayerFunctions.SetSubquestionResultPoints(subquestionPoints, studentsPoints, negativePoints, subquestionResult, GetCurrentUserLogin());
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
                    SubquestionResult subquestionResult = await businessLayerFunctions.GetSubquestionResult(int.Parse(subquestionResultId));
                    double answerCorrectness = subquestionResult.AnswerCorrectness;
                    if(answerCorrectness == 1)//in case student's answer is entirely correct, full amount of subquestion points gets recommended
                    {
                        TempData["SuggestedSubquestionPoints"] = subquestionPoints;
                    }
                    else if(answerCorrectness == -1 || subquestionResult.AnswerStatus == AnswerStatus.NotAnswered)//in case student's answer is entirely incorrect, lowest amount of points possible gets recommended
                    {
                        TempData["SuggestedSubquestionPoints"] = "-" + subquestionPoints;
                    }
                    else
                    {
                        TempData["SuggestedSubquestionPoints"] = await businessLayerFunctions.GetSubquestionResultPointsSuggestion(GetCurrentUserLogin(), int.Parse(subquestionResultId));
                    }
                }
            }
            else if (action == "previousSubquestion" || action == "nextSubquestion")
            {
                subquestionResultId = currentSubquestionResultId;
                message = action;
            }
            TempData["Message"] = message;
            //TempData["subquestionResultId"] = subquestionResultId;
            return RedirectToAction("ManageSolvedQuestion", "Home", new { questionResultId = questionResultId, subquestionResultId = subquestionResultId });
        }

        public async Task<IActionResult> BrowseSolvedTestList()
        {
            if (!await CanUserAccessPage(EnumTypes.Role.Student))
            {
                return AccessDeniedAction();
            }

            if (TempData["Message"] != null)
            {
                ViewBag.Message = TempData["Message"]!.ToString();
            }

            string login = GetCurrentUserLogin();
            dynamic model = new ExpandoObject();
            model.TestResults = await businessLayerFunctions.GetFinishedTestResultsByStudentLogin(login);
            model.Student = await businessLayerFunctions.GetStudentByLogin(login);
            return businessLayerFunctions.GetTestResultDbSet() != null ?
                View(model) :
            Problem("Entity set 'CourseContext.TestResults'  is null.");
        }

        public async Task<IActionResult> SolvedQuestion(string testResultId, int questionNr)
        {
            string login = GetCurrentUserLogin();
            TestResult testResult = await businessLayerFunctions.GetTestResult(int.Parse(testResultId));
            List<(int, AnswerStatus)> subquestionResultsPropertiesFinished = businessLayerFunctions.GetSubquestionResultsPropertiesFinished(testResult);
            SubquestionResult subquestionResult = new SubquestionResult();

            if (testResult.StudentLogin != login)
            {
                ViewBag.Message = "Chyba: na prohlížení tohoto testu nemáte právo.";
                return RedirectToAction(nameof(BrowseSolvedTestList));
            }
            else if(testResult.TestTemplate.EndDate > DateTime.Now && !testResult.TestTemplate.IsTestingData)
            {
                ViewBag.Message = "Chyba: tento test budete moct prohlížet až po " + testResult.TestTemplate.EndDate.ToString();
                return RedirectToAction(nameof(BrowseSolvedTestList));
            }
            else
            {
                int subquestionResultId;
                if (questionNr < 0 || questionNr > subquestionResultsPropertiesFinished.Count)
                {
                    subquestionResultId = subquestionResultsPropertiesFinished[0].Item1;
                    ViewBag.Message = "Chyba: tato otázka neexistuje.";
                }
                else
                {
                    subquestionResultId = subquestionResultsPropertiesFinished[questionNr].Item1;
                }

                for (int i = 0; i < testResult.QuestionResults.Count; i++)
                {
                    QuestionResult questionResult = testResult.QuestionResults.ElementAt(i);

                    for (int j = 0; j < questionResult.SubquestionResults.Count; j++)
                    {
                        SubquestionResult subquestionResultTemp = questionResult.SubquestionResults.ElementAt(j);
                        if (subquestionResultTemp.SubquestionResultId == subquestionResultId)
                        {
                            subquestionResult = subquestionResultTemp;
                            break;
                        }
                    }
                }

                int[] answerStatus = new int[subquestionResultsPropertiesFinished.Count];
                for (int i = 0; i < subquestionResultsPropertiesFinished.Count; i++)
                {
                    answerStatus[i] = (int)subquestionResultsPropertiesFinished[i].Item2;
                }
                ViewBag.AnswerStatus = answerStatus;

                ViewBag.SubquestionTypeTextArray = businessLayerFunctions.GetSubquestionTypeTextArray();
                ViewBag.SubquestionsCount = subquestionResultsPropertiesFinished.Count;
                ViewBag.SubquestionResultIndex = questionNr;
                ViewBag.TestResultPointsSum = await businessLayerFunctions.GetTestResultPointsSum(int.Parse(testResultId));
                ViewBag.TestTemplatePointsSum = await businessLayerFunctions.GetTestTemplatePointsSum(testResult.TestTemplateId);
                subquestionResult = businessLayerFunctions.ProcessSubquestionResultForView(subquestionResult);
                subquestionResult.SubquestionTemplate = businessLayerFunctions.ProcessSubquestionTemplateForView(subquestionResult.SubquestionTemplate);
                return View(subquestionResult);
            }
        }

        [HttpPost]
        public IActionResult SolvedQuestion(string action, string subquestionResultIndex, string testResultId)
        {
            int questionNr = int.Parse(subquestionResultIndex);
            if (action == "previousSubquestion")
            {
                questionNr = int.Parse(subquestionResultIndex) - 1;
            }
            else if (action == "nextSubquestion")
            {
                questionNr = int.Parse(subquestionResultIndex) + 1;
            }
            return RedirectToAction("SolvedQuestion", "Home", new { testResultId = testResultId, questionNr = questionNr });
        }

        [AllowAnonymous]
        public async Task<IActionResult> UserRegistration()
        {
            ViewBag.firstName = HttpContext.Session.GetString("firstName");
            ViewBag.lastName = HttpContext.Session.GetString("lastName");
            string email = HttpContext.Session.GetString("email");
            ViewBag.email = email;
            HttpContext.Session.SetString("emailNew", email);
            ViewBag.message = TempData["message"];

            if(businessLayerFunctions.GetUserDbSet().Count() == 0)
            {
                ViewBag.firstRegistrationMessage = "Po zaregistrování vám bude automaticky vytvořen účet hlavního administrátora.";
            }

            List<UserRegistration> userRegistrations = await businessLayerFunctions.GetUserRegistrations(email);
            return businessLayerFunctions.GetUserRegistrationDbSet() != null ?
            View(userRegistrations) :
            Problem("Entity set 'CourseContext.UserRegistrations'  is null.");
        }

        [HttpPost]
        public async Task<IActionResult> UserRegistration(string firstName, string lastName, string login, string role)
        {
            string email = HttpContext.Session.GetString("emailNew");
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
                    SetCurrentUserLogin(login);
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
            if (!await CanUserAccessPage(EnumTypes.Role.MainAdmin))
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

            Student? studentLoginCheck = await businessLayerFunctions.GetStudentByLoginNullable(login);
            Student? studentEmailCheck = await businessLayerFunctions.GetStudentByEmailNullable(email);

            User? userLoginCheck = await businessLayerFunctions.GetUserByLoginNullable(login);
            User? userEmailCheck = await businessLayerFunctions.GetUserByEmailNullable(email);

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
                else if (studentLoginCheck != null || userLoginCheck != null)
                {
                    TempData["StudentMessage"] = "Chyba: uživatel s loginem \"" + login + "\" již existuje.";
                }
                else
                {
                    TempData["StudentMessage"] = await businessLayerFunctions.AddStudent(firstName, lastName, login, email);
                }
            }
            else if(action == "editStudent")
            {
                //it's necessary to ensure that there won't be two or more users with the same email/login
                if(studentLoginCheck != null)
                {
                    Student? studentByNewEmail = await businessLayerFunctions.GetStudentByEmailNullable(email);
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
                Student student = await businessLayerFunctions.GetStudentByLogin(login);
                await businessLayerFunctions.DeleteStudent(student);
                TempData["StudentMessage"] = "Student úspěšně smazán.";
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
            if (!await CanUserAccessPage(EnumTypes.Role.Admin))
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

            Student? studentLoginCheck = await businessLayerFunctions.GetStudentByLoginNullable(login);
            Student? studentEmailCheck = await businessLayerFunctions.GetStudentByEmailNullable(email);

            User? userLoginCheck = await businessLayerFunctions.GetUserByLoginNullable(login);
            User? userEmailCheck = await businessLayerFunctions.GetUserByEmailNullable(email);

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
                else if (studentLoginCheck != null || userLoginCheck != null)
                {
                    TempData["StudentMessage"] = "Chyba: uživatel s loginem \"" + login + "\" již existuje.";
                }
                else
                {
                    TempData["StudentMessage"] = await businessLayerFunctions.AddStudent(firstName, lastName, login, email);
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
                Student student = await businessLayerFunctions.GetStudentByLogin(login);
                await businessLayerFunctions.DeleteStudent(student);
                TempData["StudentMessage"] = "Student úspěšně smazán.";
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
                User user = await businessLayerFunctions.GetUserByLogin(login);
                await businessLayerFunctions.DeleteUser(user);
                TempData["TeacherMessage"] = "Učitel úspěšně smazán.";
            }
            return RedirectToAction(nameof(ManageUserListForAdmin));
        }

        public async Task<IActionResult> ManageUserRegistrationList()
        {
            if (!await CanUserAccessPage(EnumTypes.Role.MainAdmin))
            {
                return AccessDeniedAction();
            }

            if (TempData["Message"] != null)
            {
                ViewBag.Message = TempData["Message"]!.ToString();
            }
            return businessLayerFunctions.GetUserRegistrationDbSet() != null ?
            View(await businessLayerFunctions.GetUserRegistrationDbSet().ToListAsync()) :
            Problem("Entity set 'CourseContext.UserRegistrations'  is null.");
        }

        [HttpPost]
        [ValidateAntiForgeryToken]
        public async Task<IActionResult> ManageUserRegistrationList(string login, string email, string firstName, string lastName, string role, string action)
        {
            string? message = null;
            if(action == "acceptRegistration")
            {
                User? userByEmail = await businessLayerFunctions.GetUserByEmailNullable(email);
                Student? studentByEmail = await businessLayerFunctions.GetStudentByEmailNullable(email);

                User? userByLogin = await businessLayerFunctions.GetUserByLoginNullable(login);
                Student? studentByLogin = await businessLayerFunctions.GetStudentByLoginNullable(login);

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
            if (!await CanUserAccessPage(EnumTypes.Role.Admin))
            {
                return AccessDeniedAction();
            }

            if (TempData["Message"] != null)
            {
                ViewBag.Message = TempData["Message"]!.ToString();
            }
            return businessLayerFunctions.GetUserRegistrationDbSet() != null ?
            View(await businessLayerFunctions.GetUserRegistrationDbSet().ToListAsync()) :
            Problem("Entity set 'CourseContext.UserRegistrations'  is null.");
        }

        [HttpPost]
        [ValidateAntiForgeryToken]
        public async Task<IActionResult> ManageUserRegistrationListForAdmin(string login, string email, string firstName, string lastName, string role, string action)
        {
            string? message = null;
            if (action == "acceptRegistration")
            {
                User? userByEmail = await businessLayerFunctions.GetUserByEmailNullable(email);
                Student? studentByEmail = await businessLayerFunctions.GetStudentByEmailNullable(email);

                User? userByLogin = await businessLayerFunctions.GetUserByLoginNullable(login);
                Student? studentByLogin = await businessLayerFunctions.GetStudentByLoginNullable(login);

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
            if (!await CanUserAccessPage(EnumTypes.Role.Teacher))
            {
                return AccessDeniedAction();
            }

            string login = GetCurrentUserLogin();
            return businessLayerFunctions.GetUserDbSet() != null ?
                View(await businessLayerFunctions.GetUserDbSet().FirstOrDefaultAsync(u => u.Login == login)) :
                Problem("Entity set 'CourseContext.Users'  is null.");
        }

        public async Task<IActionResult> AdminMenu()
        {
            if (!await CanUserAccessPage(EnumTypes.Role.Admin))
            {
                return AccessDeniedAction();
            }

            string login = GetCurrentUserLogin();
            return businessLayerFunctions.GetUserDbSet() != null ?
                View(await businessLayerFunctions.GetUserDbSet().FirstOrDefaultAsync(u => u.Login == login)) :
                Problem("Entity set 'CourseContext.Users'  is null.");
        }

        public async Task<IActionResult> MainAdminMenu()
        {
            if (!await CanUserAccessPage(EnumTypes.Role.MainAdmin))
            {
                return AccessDeniedAction();
            }

            return businessLayerFunctions.GetUserDbSet() != null ?
                View(await businessLayerFunctions.GetUserDbSet().FirstOrDefaultAsync(u => u.Role == EnumTypes.Role.MainAdmin)) :
                Problem("Entity set 'CourseContext.Users'  is null.");
        }

        public async Task<IActionResult> GlobalSettings()
        {
            if (!await CanUserAccessPage(EnumTypes.Role.MainAdmin))
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
        public async Task<IActionResult> GlobalSettings(string action, string testingMode)
        {
            if(action == "setTestingMode")
            {
                await businessLayerFunctions.ChangeGlobalSettings(testingMode);
                TempData["Message"] = "Změny úspěšně uloženy.";
            }
            else if(action == "addNUnitData")
            {
                string message = await businessLayerFunctions.CreateNUnitData();
                TempData["Message"] = message;
            }
            else if (action == "deleteNUnitData")
            {
                string message = await businessLayerFunctions.DeleteNUnitData();
                TempData["Message"] = message;
            }

            return RedirectToAction(nameof(GlobalSettings));
        }

        public async Task<IActionResult> ManageArtificialIntelligence()
        {
            if (!await CanUserAccessPage(EnumTypes.Role.MainAdmin))
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
            ViewBag.TestingDataSubquestionTemplates = await businessLayerFunctions.GetTestingDataSubquestionTemplatesCount();
            ViewBag.TestingDataSubquestionResults = await businessLayerFunctions.GetTestingDataSubquestionResultsCount();

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
            if (action == "addSubquestionTemplateRandomData" || action == "addSubquestionTemplateCorrelationalData")
            {
                TempData["Message"] = await businessLayerFunctions.CreateTemplateTestingData(action, amountOfSubquestionTemplates);
            }
            else if (action == "deleteSubquestionTemplateTestingData")
            {
                await businessLayerFunctions.DeleteTemplateTestingData();
                TempData["Message"] = "Testovací data úspěšně vymazána.";
            }
            else if (action == "addSubquestionResultRandomData" || action == "addSubquestionResultCorrelationalData")
            {
                TempData["Message"] = await businessLayerFunctions.CreateResultTestingData(action, amountOfSubquestionResults);
            }
            else if (action == "deleteSubquestionResultTestingData")
            {
                await businessLayerFunctions.DeleteResultTestingData();
                TempData["Message"] = "Testovací data úspěšně vymazána.";
            }
            else if (action == "getDeviceName")
            {
                TempData["DeviceName"] = businessLayerFunctions.GetAIDeviceName();
            }
            else if (action == "deleteTestingData")
            {
                await businessLayerFunctions.DeleteTestingData();
                TempData["Message"] = "Všechna testovací data úspěšně vymazána.";
            }

            return RedirectToAction(nameof(ManageArtificialIntelligence));
        }

        public async Task<IActionResult> AddTestTemplate(TestTemplate testTemplate)
        {
            if (TempData["Message"] != null)
            {
                ViewBag.Message = TempData["Message"]!.ToString();
            }

            dynamic model = new ExpandoObject();
            model.TestTemplate = testTemplate;
            model.Subjects = await businessLayerFunctions.GetSubjectDbSet().Include(s => s.Guarantor).ToListAsync();
            return View(model);
        }

        [HttpPost]
        public async Task<IActionResult> AddTestTemplate(TestTemplate testTemplate, string subjectId, string _)
        {
            string message = await businessLayerFunctions.AddTestTemplate(testTemplate, subjectId, GetCurrentUserLogin());
            TempData["Message"] = message;
            if(message == "Zadání testu bylo úspěšně přidáno.")
            {
                return RedirectToAction(nameof(TestTemplateList));
            }
            else
            {
                return RedirectToAction("AddTestTemplate", "Home", new RouteValueDictionary(testTemplate));
            }
        }

        public async Task<IActionResult> EditTestTemplate(string testTemplateId)
        {
            if (TempData["Message"] != null)
            {
                ViewBag.Message = TempData["Message"]!.ToString();
            }

            TestTemplate testTemplate = await businessLayerFunctions.GetTestTemplate(int.Parse(testTemplateId));
            if (businessLayerFunctions.CanUserEditTestTemplate(testTemplate, GetCurrentUserLogin()))
            {
                dynamic model = new ExpandoObject();
                model.TestTemplate = await businessLayerFunctions.GetTestTemplate(int.Parse(testTemplateId));
                model.Subjects = await businessLayerFunctions.GetSubjectDbSet().Include(s => s.Guarantor).ToListAsync();
                return View(model);
            }
            else
            {
                TempData["Message"] = "Chyba: tento test nemůžete upravovat (nemáte oprávnění nebo již test začal).";
                return RedirectToAction(nameof(TestTemplateList));
            }
        }

        [HttpPost]
        public async Task<IActionResult> EditTestTemplate(TestTemplate testTemplate, string subjectId)
        {
            string message = await businessLayerFunctions.EditTestTemplate(testTemplate, subjectId, GetCurrentUserLogin());
            TempData["Message"] = message;
            if (message == "Zadání testu bylo úspěšně upraveno.")
            {
                return RedirectToAction(nameof(TestTemplateList));
            }
            else
            {
                return RedirectToAction("EditTestTemplate", "Home", new { testTemplateId = testTemplate.TestTemplateId });
            }
        }

        public async Task<IActionResult> AddQuestionTemplate(string testTemplateId)
        {
            TestTemplate testTemplate = await businessLayerFunctions.GetTestTemplate(int.Parse(testTemplateId));
            if (!businessLayerFunctions.CanUserEditTestTemplate(testTemplate, GetCurrentUserLogin()))
            {
                TempData["Message"] = "Chyba: tento test nemůžete upravovat (nemáte oprávnění nebo již test začal).";
                return RedirectToAction("TestTemplate", "Home", new { testTemplateId = testTemplateId });
            }
            else
            {
                if (TempData["Message"] != null)
                {
                    ViewBag.Message = TempData["Message"]!.ToString();
                }
                ViewBag.testTemplateId = testTemplateId;
                return View();
            }
        }

        [HttpPost]
        public async Task<IActionResult> AddQuestionTemplate(QuestionTemplate questionTemplate, string testTemplateId)
        {
            string login = GetCurrentUserLogin();
            questionTemplate.OwnerLogin = login;
            questionTemplate.TestTemplate = await businessLayerFunctions.GetTestTemplate(int.Parse(testTemplateId));
            string message = await businessLayerFunctions.AddQuestionTemplate(questionTemplate);
            TempData["Message"] = message;
            if(message == "Zadání otázky bylo úspěšně přidáno.")
            {
                return RedirectToAction("TestTemplate", "Home", new { testTemplateId = testTemplateId });
            }
            else
            {
                return RedirectToAction("AddQuestionTemplate", "Home", new { testTemplateId = testTemplateId });
            }
        }

        public async Task<IActionResult> EditQuestionTemplate(string questionTemplateId)
        {
            if (TempData["Message"] != null)
            {
                ViewBag.Message = TempData["Message"]!.ToString();
            }
            QuestionTemplate questionTemplate = await businessLayerFunctions.GetQuestionTemplate(int.Parse(questionTemplateId));
            if (businessLayerFunctions.CanUserEditTestTemplate(questionTemplate.TestTemplate, GetCurrentUserLogin()))
            {
                return View(questionTemplate);
            }
            else
            {
                TempData["Message"] = "Chyba: tento test nemůžete upravovat (nemáte oprávnění nebo již test začal).";
                return RedirectToAction("TestTemplate", "Home", new { testTemplateId = questionTemplate.TestTemplate.TestTemplateId });
            }
        }

        [HttpPost]
        public async Task<IActionResult> EditQuestionTemplate(QuestionTemplate questionTemplate, string testTemplateId)
        {
            string login = GetCurrentUserLogin();
            questionTemplate.OwnerLogin = login;
            questionTemplate.TestTemplate = await businessLayerFunctions.GetTestTemplate(int.Parse(testTemplateId));
            string message = await businessLayerFunctions.EditQuestionTemplate(questionTemplate, GetCurrentUserLogin());
            TempData["Message"] = message;
            if(message == "Zadání otázky bylo úspěšně upraveno.")
            {
                return RedirectToAction("TestTemplate", "Home", new { testTemplateId = testTemplateId });
            }
            else
            {
                return RedirectToAction("EditQuestionTemplate", "Home", new { questionTemplateId = questionTemplate.QuestionTemplateId });
            }
        }

        public async Task<IActionResult> AddSubquestionTemplate(string questionTemplateId, SubquestionTemplate subquestionTemplate)
        {
            if(questionTemplateId != null)
            {
                QuestionTemplate questionTemplate = await businessLayerFunctions.GetQuestionTemplate(int.Parse(questionTemplateId));
                if (!businessLayerFunctions.CanUserEditTestTemplate(questionTemplate.TestTemplate, GetCurrentUserLogin()))
                {
                    TempData["Message"] = "Chyba: tento test nemůžete upravovat (nemáte oprávnění nebo již test začal).";
                    return RedirectToAction("QuestionTemplate", "Home", new { questionTemplateId = questionTemplateId });
                }
            }
            
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
            string login = GetCurrentUserLogin();
            string message = string.Empty;

            if (action == "selectType")
            {
                TempData["SubquestionType"] = subquestionTemplate.SubquestionType;
            }
            else if (action == "addSubquestion")
            {
                subquestionTemplate.OwnerLogin = login;
                subquestionTemplate.QuestionTemplate = await businessLayerFunctions.GetQuestionTemplate(subquestionTemplate.QuestionTemplateId);

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
                subquestionTemplate.QuestionTemplate = await businessLayerFunctions.GetQuestionTemplate(subquestionTemplate.QuestionTemplateId);
                TempData["SuggestedSubquestionPoints"] = await businessLayerFunctions.GetSubquestionTemplatePointsSuggestion(subquestionTemplate, false, GetCurrentUserLogin());
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
                    return RedirectToAction("QuestionTemplate", "Home", new { questionTemplateId = subquestionTemplate.QuestionTemplateId, subquestionTemplateId = subquestionTemplate.SubquestionTemplateId });
                }
            }
            else //getPointsSuggestion redirection
            {
                (subquestionTemplate, string? _) = businessLayerFunctions.ValidateSubquestionTemplate(subquestionTemplate, subquestionTextArray, sliderValues, null);
                TempData["SelectedSubquestionType"] = subquestionTemplate.SubquestionType;
                return RedirectToAction("AddSubquestionTemplate", "Home", new RouteValueDictionary(subquestionTemplate));
            }
        }

        public async Task<IActionResult> EditSubquestionTemplate(string subquestionTemplateId)
        {
            SubquestionTemplate subquestionTemplate = await businessLayerFunctions.GetSubquestionTemplate(int.Parse(subquestionTemplateId));
            ViewBag.SubquestionTypeTextArray = businessLayerFunctions.GetSubquestionTypeTextArray();
            if (TempData["Message"] != null)
            {
                ViewBag.Message = TempData["Message"]!.ToString();
            }
            if (TempData["SuggestedSubquestionPoints"] != null)
            {
                ViewBag.SuggestedSubquestionPoints = TempData["SuggestedSubquestionPoints"]!.ToString();
            }
            if (businessLayerFunctions.CanUserEditTestTemplate(subquestionTemplate.QuestionTemplate.TestTemplate, GetCurrentUserLogin()))
            {
                return View(subquestionTemplate);
            }
            else
            {
                TempData["Message"] = "Chyba: tento test nemůžete upravovat (nemáte oprávnění nebo již test začal).";
                return RedirectToAction("QuestionTemplate", "Home", new { questionTemplateId = subquestionTemplate.QuestionTemplateId, subquestionTemplateId });
            }
        }

        [HttpPost]
        public async Task<IActionResult> EditSubquestionTemplate(string action, SubquestionTemplate subquestionTemplate,
            IFormFile image, string[] subquestionTextArray, string sliderValues)
        {
            string message = string.Empty;

            if (action == "editSubquestion")
            {
                subquestionTemplate.OwnerLogin = GetCurrentUserLogin();
                subquestionTemplate.QuestionTemplate = await businessLayerFunctions.GetQuestionTemplate(subquestionTemplate.QuestionTemplateId);

                (subquestionTemplate, string? errorMessage) = businessLayerFunctions.ValidateSubquestionTemplate(subquestionTemplate, subquestionTextArray, sliderValues, image);
                if (errorMessage != null)
                {
                    message = errorMessage;
                }
                else
                {
                    message = await businessLayerFunctions.EditSubquestionTemplate(subquestionTemplate, image, _environment.WebRootPath, GetCurrentUserLogin());
                }
            }
            else if (action == "getPointsSuggestion")
            {
                subquestionTemplate.OwnerLogin = GetCurrentUserLogin();
                subquestionTemplate.QuestionTemplate = await businessLayerFunctions.GetQuestionTemplate(subquestionTemplate.QuestionTemplateId);
                TempData["SuggestedSubquestionPoints"] = await businessLayerFunctions.GetSubquestionTemplatePointsSuggestion(subquestionTemplate, true, GetCurrentUserLogin());
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
                    return RedirectToAction("QuestionTemplate", "Home", new { questionTemplateId = subquestionTemplate.QuestionTemplateId, subquestionTemplateId = subquestionTemplate.SubquestionTemplateId });
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
            string login = GetCurrentUserLogin();
            ViewBag.login = login;
            User user = await businessLayerFunctions.GetUserByLogin(login);
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

            List<Subject> subjects = await businessLayerFunctions.GetSubjects();
            return businessLayerFunctions.GetSubjectDbSet() != null ?
                View(subjects) :
                Problem("Entity set 'CourseContext.Subjects'  is null.");
        }

        [HttpPost]
        public async Task<IActionResult> ManageSubjects(string action, string subjectId)
        {
            if(action == "deleteSubject")
            {
                string message = await businessLayerFunctions.DeleteSubject(int.Parse(subjectId), GetCurrentUserLogin());
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
            string message = await businessLayerFunctions.AddSubject(subject, enrolledStudentLogin, GetCurrentUserLogin());
            TempData["Message"] = message;
            return RedirectToAction(nameof(ManageSubjects));
        }

        public async Task<IActionResult> EditSubject(string subjectId)
        {
            Subject subject = await businessLayerFunctions.GetSubjectById(int.Parse(subjectId));
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
            string message = await businessLayerFunctions.EditSubject(subject, enrolledStudentLogin, GetCurrentUserLogin());
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
            string login = GetCurrentUserLogin();

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
            string login = GetCurrentUserLogin();
            return businessLayerFunctions.GetTestTemplateDbSet() != null ?
                View(await businessLayerFunctions.GetStudentAvailableTestList(login)) :
                Problem("Entity set 'CourseContext.TestTemplates'  is null.");
        }
        
        public async Task<IActionResult> StudentAvailableTest(string testTemplateId)
        {
            string login = GetCurrentUserLogin();
            TestTemplate testTemplate = new TestTemplate();
            string? errorMessage;

            if (TempData["Message"] != null)
            {
                ViewBag.Message = TempData["Message"]!.ToString();
            }

            errorMessage = await businessLayerFunctions.CanStudentAccessTest(int.Parse(testTemplateId), GetCurrentUserLogin());
            if (errorMessage == null || errorMessage == "Pokus probíhá.")
            {
                testTemplate = await businessLayerFunctions.GetTestTemplate(int.Parse(testTemplateId));
                ViewBag.TestTemplatePointsSum = await businessLayerFunctions.GetTestTemplatePointsSum(int.Parse(testTemplateId));
                bool notTurnedInExists = false;
                int amountOfNotTurnedIn = await businessLayerFunctions.GetAmountOfNotTurnedTestResultsByTestTemplate(login, int.Parse(testTemplateId));
                if(amountOfNotTurnedIn > 0)
                {
                    notTurnedInExists = true;
                }
                ViewBag.NotTurnedInExists = notTurnedInExists;
            }
            else
            {
                ViewBag.TestTemplatePointsSum = 0;
                ViewBag.NotTurnedInExists = false;
                ViewBag.Message = errorMessage;
            }
            return businessLayerFunctions.GetTestTemplateDbSet() != null ?
                View(testTemplate) :
                Problem("Entity set 'CourseContext.TestTemplates'  is null.");
        }

        [HttpPost]
        public async Task<IActionResult> StudentAvailableTest(string action, string testTemplateId)
        {
            string? errorMessage;
            if (action == "beginAttempt")
            {
                errorMessage = await businessLayerFunctions.CanStudentAccessTest(int.Parse(testTemplateId), GetCurrentUserLogin());
                if(errorMessage == "Pokus probíhá.")
                {
                    return RedirectToAction("SolveQuestion", "Home");
                }
                if(errorMessage == null)
                {
                    errorMessage = await businessLayerFunctions.BeginStudentAttempt(int.Parse(testTemplateId), GetCurrentUserLogin());
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
            TestResult testResult = await businessLayerFunctions.LoadLastStudentAttempt(GetCurrentUserLogin());
            List<(int, AnswerCompleteness)> subquestionResultsProperties = businessLayerFunctions.GetSubquestionResultsProperties(testResult);
            SubquestionResult subquestionResult = new SubquestionResult();

            if (TempData["Message"] != null)
            {
                ViewBag.Message = TempData["Message"]!.ToString();
            }

            //the user has just started the attempt - the very first subquestion is shown to him then
            if (GetStudentSubquestionResultId() == null 
                || !subquestionResultsProperties.Any(s => s.Item1 == GetStudentSubquestionResultId()))
            {
                for (int i = 0; i < testResult.QuestionResults.Count; i++)
                {
                    QuestionResult questionResult = testResult.QuestionResults.ElementAt(i);

                    //the question must include at least one subquestion
                    if (questionResult.SubquestionResults.Count > 0)
                    {
                        subquestionResult = questionResult.SubquestionResults.ElementAt(0);
                        break;
                    }
                }
                SetStudentSubquestionResultId(subquestionResult.SubquestionResultId);

                ViewBag.SubquestionsCount = subquestionResultsProperties.Count;
                ViewBag.SubquestionResultIdIndex = 0;
            }
            else
            {
                int newSubquestionResultId;
                if (questionNr < 0 || questionNr > subquestionResultsProperties.Count())
                {
                    newSubquestionResultId = subquestionResultsProperties[0].Item1;
                    ViewBag.Message = "Chyba: tato otázka neexistuje.";
                }
                else
                {
                    newSubquestionResultId = subquestionResultsProperties[questionNr].Item1;
                }
                SetStudentSubquestionResultId(newSubquestionResultId);

                for (int i = 0; i < testResult.QuestionResults.Count; i++)
                {
                    QuestionResult questionResult = testResult.QuestionResults.ElementAt(i);

                    for (int j = 0; j < questionResult.SubquestionResults.Count; j++)
                    {
                        SubquestionResult subquestionResultTemp = questionResult.SubquestionResults.ElementAt(j);
                        if (subquestionResultTemp.SubquestionResultId == newSubquestionResultId)
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
            ViewBag.AnswerCompleteness = answerCompleteness;
            return View(subquestionResult);
        }

        [HttpPost]
        public async Task<IActionResult> SolveQuestion(SubquestionResult subquestionResult, string newSubquestionResultIndex, int subquestionResultIndex, 
            string[] possibleAnswers, string action)
        {
            string? errorMessage;
            (subquestionResult, errorMessage) = await businessLayerFunctions.ValidateSubquestionResult(subquestionResult, subquestionResultIndex, possibleAnswers, GetCurrentUserLogin());
            if(errorMessage != null)
            {
                if(errorMessage == "Čas na odevdzání testu již vypršel. Test byl automaticky odevzdán.")
                {
                    TempData["Message"] = errorMessage;
                    await businessLayerFunctions.FinishStudentAttempt(GetCurrentUserLogin());
                    return RedirectToAction(nameof(BrowseSolvedTestList));
                }
                else
                {
                    TempData["Message"] = errorMessage;
                    return RedirectToAction("SolveQuestion", "Home", new { questionNr = newSubquestionResultIndex });
                }
            }
            else
            {
                await businessLayerFunctions.UpdateSubquestionResultStudentsAnswers(subquestionResult, subquestionResultIndex, GetCurrentUserLogin());
                if(action == "turnTestIn")
                {
                    await businessLayerFunctions.FinishStudentAttempt(GetCurrentUserLogin());
                    TempData["Message"] = "Test byl odevdzán.";
                    return RedirectToAction("StudentMenu", "Home", new { questionNr = newSubquestionResultIndex });
                }
                else
                {
                    return RedirectToAction("SolveQuestion", "Home", new { questionNr = newSubquestionResultIndex });
                }
            }
        }

        /// <summary>
        /// Returns ID of subquestion result that the student is solving
        /// </summary>
        public int? GetStudentSubquestionResultId()
        {
            return HttpContext.Session.GetInt32("subquestionResultId");
        }

        /// <summary>
        /// Sets ID of subquestion result that the student is solving
        /// </summary>
        public void SetStudentSubquestionResultId(int subquestionResultId)
        {
            HttpContext.Session.SetInt32("subquestionResultId", subquestionResultId);
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