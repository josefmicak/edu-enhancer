using Microsoft.AspNetCore.Mvc;
using System.Diagnostics;
using ViewLayer.Models;
using DomainModel;

namespace ViewLayer.Controllers
{
    public class HomeController : Controller
    {
        private QuestionController questionController = new QuestionController();
        // private StudentController studentController = new StudentController();
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

        public IActionResult StudentMenu()
        {
            return View(new PageModel
            {
                Title = "Student"
            });
        }

        public IActionResult ManageSolvedTestList()
        {
            return View();
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
                SelectedTestTemplate = testController.LoadTestTemplate(testNameIdentifier, testNumberIdentifier)
            });
        }

        [HttpGet]
        public IActionResult QuestionTemplate(string testNameIdentifier, string testNumberIdentifier, string questionNameIdentifier, string questionNumberIdentifier)
        {
            return View(new QuestionTemplateModel
            {
                Title = "Správa zadání otázky " + questionNameIdentifier,
                SelectedQuestionTemplate = questionController.LoadQuestionTemplate(testNameIdentifier, testNumberIdentifier, questionNameIdentifier, questionNumberIdentifier),
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
                SelectedQuestionTemplate = questionController.LoadQuestionTemplate(testNameIdentifier, testNumberIdentifier, questionNameIdentifier, questionNumberIdentifier),
                TestNameIdentifier = testNameIdentifier,
                TestNumberIdentifier = testNumberIdentifier,
                SubquestionTypeTextArray = questionController.SubquestionTypeTextArray,
                SelectedSubquestionTemplate = questionController.LoadSubquestionTemplate(testNameIdentifier, questionNumberIdentifier, subquestionIdentifier)
            });
        }

        [ResponseCache(Duration = 0, Location = ResponseCacheLocation.None, NoStore = true)]
        public IActionResult Error()
        {
            return View(new ErrorViewModel { RequestId = Activity.Current?.Id ?? HttpContext.TraceIdentifier });
        }
    }
}