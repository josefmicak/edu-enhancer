using Microsoft.AspNetCore.Mvc;
using System.Diagnostics;
using ViewLayer.Models;
using DomainModel;

namespace ViewLayer.Controllers
{
    public class HomeController : Controller
    {
        private QuestionController questionController = new QuestionController();
        private StudentController studentController = new StudentController();
        private TestController testController = new TestController();

        private readonly ILogger<HomeController> _logger;

        public HomeController(ILogger<HomeController> logger)
        {
            _logger = logger;
        }

        public IActionResult Index()
        {
            return View(new PageModel
            {
                Title = "Hlavní menu"
            });
        }

        public IActionResult TeacherMenu()
        {
            return View(new PageModel
            {
                Title = "Učitel"
            });
        }

        public IActionResult TestTemplateList()
        {
            return View(new TestTemplateListModel
            {
                Title = "Správa zadání testů",
                TestTemplates = testController.LoadTestTemplates()
            });
        }

        public IActionResult TestTemplate(string testNameIdentifier, string testNumberIdentifier)
        {
            return View(new TestTemplateModel
            {
                Title = "Správa zadání testu " + testNameIdentifier,
                TestTemplate = testController.LoadTestTemplate(testNameIdentifier, testNumberIdentifier)
            });
        }

        [HttpGet]
        public IActionResult QuestionTemplate(string testNameIdentifier, string testNumberIdentifier, string questionNameIdentifier, string questionNumberIdentifier)
        {
            return View(new QuestionTemplateModel
            {
                Title = "Správa zadání otázky " + questionNameIdentifier,
                QuestionTemplate = questionController.LoadQuestionTemplate(testNameIdentifier, testNumberIdentifier, questionNameIdentifier, questionNumberIdentifier),
                TestNameIdentifier = testNameIdentifier,
                TestNumberIdentifier = testNumberIdentifier,
                SubquestionTypeTextArray = questionController.SubquestionTypeTextArray
            });
        }

        [HttpPost]
        public IActionResult QuestionTemplate(string testNameIdentifier, string testNumberIdentifier, string questionNameIdentifier, string questionNumberIdentifier, string subquestionIdentifier)
        {
            return View(new QuestionTemplateModel
            {
                Title = "Správa zadání otázky " + questionNameIdentifier,
                QuestionTemplate = questionController.LoadQuestionTemplate(testNameIdentifier, testNumberIdentifier, questionNameIdentifier, questionNumberIdentifier),
                TestNameIdentifier = testNameIdentifier,
                TestNumberIdentifier = testNumberIdentifier,
                SubquestionTypeTextArray = questionController.SubquestionTypeTextArray,
                SubquestionTemplate = questionController.LoadSubquestionTemplate(testNameIdentifier, questionNumberIdentifier, subquestionIdentifier)
            });
        }

        public IActionResult ManageSolvedTestList()
        {
            return View(new ManageSolvedTestListModel
            {
                Title = "Správa vyřešených testů",
                TestResults = testController.LoadTestResults()
            });
        }

        public IActionResult ManageSolvedTest(string testNameIdentifier, string testResultIdentifier)
        {
            return View(new ManageSolvedTestModel
            {
                Title = "Správa vyřešeného testu " + testResultIdentifier,
                TestResult = testController.LoadTestResult(testNameIdentifier, testResultIdentifier),
                TestTemplate = testController.LoadTestTemplate(testNameIdentifier, testController.GetTestNumberIdentifier(testNameIdentifier))
            });
        }

        [HttpGet]
        public IActionResult ManageSolvedQuestion(string testNameIdentifier, string testNumberIdentifier, string questionNameIdentifier, string questionNumberIdentifier, string testResultIdentifier)
        {
            return View(new ManageSolvedQuestionModel
            {
                Title = "Správa vyřešeného testu " + testResultIdentifier + ", otázka " + questionNumberIdentifier,
                QuestionTemplate = questionController.LoadQuestionTemplate(testNameIdentifier, testNumberIdentifier, questionNameIdentifier, questionNumberIdentifier),
                QuestionResult = questionController.LoadQuestionResult(testNameIdentifier, 
                questionController.LoadQuestionTemplate(testNameIdentifier, testNumberIdentifier, questionNameIdentifier, questionNumberIdentifier), testResultIdentifier,
                questionController.LoadQuestionTemplate(testNameIdentifier, testNumberIdentifier, questionNameIdentifier, questionNumberIdentifier).SubquestionTemplateList[0].SubquestionIdentifier),
                TestNameIdentifier = testNameIdentifier,
                TestNumberIdentifier = testNumberIdentifier,
                TestResultIdentifier = testResultIdentifier,
                SubquestionTypeTextArray = questionController.SubquestionTypeTextArray
            });
        }

        [HttpPost]
        public IActionResult ManageSolvedQuestion(string testNameIdentifier, string testNumberIdentifier, string questionNameIdentifier, string questionNumberIdentifier, string testResultIdentifier, string subquestionIdentifier)
        {
            return View(new ManageSolvedQuestionModel
            {
                Title = "Správa vyřešeného testu " + testResultIdentifier + ", otázka " + questionNumberIdentifier,
                QuestionTemplate = questionController.LoadQuestionTemplate(testNameIdentifier, testNumberIdentifier, questionNameIdentifier, questionNumberIdentifier),
                QuestionResult = questionController.LoadQuestionResult(testNameIdentifier, 
                questionController.LoadQuestionTemplate(testNameIdentifier, testNumberIdentifier, questionNameIdentifier, questionNumberIdentifier), testResultIdentifier, subquestionIdentifier),
                TestNameIdentifier = testNameIdentifier,
                TestNumberIdentifier = testNumberIdentifier,
                TestResultIdentifier = testResultIdentifier,
                SubquestionTypeTextArray = questionController.SubquestionTypeTextArray,
                SubquestionTemplate = questionController.LoadSubquestionTemplate(testNameIdentifier, questionNumberIdentifier, subquestionIdentifier)
            });
        }

        public IActionResult StudentMenu()
        {
            return View(new StudentMenuModel
            {
                Title = "Student",
                Students = studentController.LoadStudents()
            });
        }

        public IActionResult BrowseSolvedTestList(string studentIdentifier)
        {
            return View(new BrowseSolvedTestListModel
            {
                Title = "Prohlížení vyřešených testů",
                TestResults = testController.LoadTestResults(studentIdentifier),
                Student = studentController.LoadStudent(studentIdentifier)
            });
        }

        public IActionResult BrowseSolvedTest(string testNameIdentifier, string testResultIdentifier, string studentIdentifier)
        {
            return View(new BrowseSolvedTestModel
            {
                Title = "Prohlížení vyřešeného testu " + testResultIdentifier,
                TestResult = testController.LoadTestResult(testNameIdentifier, testResultIdentifier),
                TestTemplate = testController.LoadTestTemplate(testNameIdentifier, testController.GetTestNumberIdentifier(testNameIdentifier))
            });
        }

        [HttpGet]
        public IActionResult BrowseSolvedQuestion(string testNameIdentifier, string testNumberIdentifier, string questionNameIdentifier, string questionNumberIdentifier, string testResultIdentifier)
        {
            return View(new BrowseSolvedQuestionModel
            {
                Title = "Prohlížení vyřešeného testu " + testResultIdentifier + ", otázka " + questionNumberIdentifier,
                QuestionTemplate = questionController.LoadQuestionTemplate(testNameIdentifier, testNumberIdentifier, questionNameIdentifier, questionNumberIdentifier),
                QuestionResult = questionController.LoadQuestionResult(testNameIdentifier,
                questionController.LoadQuestionTemplate(testNameIdentifier, testNumberIdentifier, questionNameIdentifier, questionNumberIdentifier), testResultIdentifier,
                questionController.LoadQuestionTemplate(testNameIdentifier, testNumberIdentifier, questionNameIdentifier, questionNumberIdentifier).SubquestionTemplateList[0].SubquestionIdentifier),
                TestNameIdentifier = testNameIdentifier,
                TestNumberIdentifier = testNumberIdentifier,
                TestResultIdentifier = testResultIdentifier,
                SubquestionTypeTextArray = questionController.SubquestionTypeTextArray
            });
        }

        [HttpPost]
        public IActionResult BrowseSolvedQuestion(string testNameIdentifier, string testNumberIdentifier, string questionNameIdentifier, string questionNumberIdentifier, string testResultIdentifier, string subquestionIdentifier)
        {
            return View(new BrowseSolvedQuestionModel
            {
                Title = "Prohlížení vyřešeného testu " + testResultIdentifier + ", otázka " + questionNumberIdentifier,
                QuestionTemplate = questionController.LoadQuestionTemplate(testNameIdentifier, testNumberIdentifier, questionNameIdentifier, questionNumberIdentifier),
                QuestionResult = questionController.LoadQuestionResult(testNameIdentifier,
                questionController.LoadQuestionTemplate(testNameIdentifier, testNumberIdentifier, questionNameIdentifier, questionNumberIdentifier), testResultIdentifier, subquestionIdentifier),
                TestNameIdentifier = testNameIdentifier,
                TestNumberIdentifier = testNumberIdentifier,
                TestResultIdentifier = testResultIdentifier,
                SubquestionTypeTextArray = questionController.SubquestionTypeTextArray,
                SubquestionTemplate = questionController.LoadSubquestionTemplate(testNameIdentifier, questionNumberIdentifier, subquestionIdentifier)
            });
        }

        [ResponseCache(Duration = 0, Location = ResponseCacheLocation.None, NoStore = true)]
        public IActionResult Error()
        {
            return View(new ErrorViewModel { RequestId = Activity.Current?.Id ?? HttpContext.TraceIdentifier });
        }
    }
}