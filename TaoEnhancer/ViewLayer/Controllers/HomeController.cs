using Microsoft.AspNetCore.Mvc;
using System.Diagnostics;
using ViewLayer.Models;
using DomainModel;
using Microsoft.EntityFrameworkCore;
using ViewLayer.Data;

namespace ViewLayer.Controllers
{
    public class HomeController : Controller
    {
        private readonly CourseContext _context;
        private QuestionController questionController = new QuestionController();
        private StudentController studentController = new StudentController();
        private TestController testController = new TestController();

        private readonly ILogger<HomeController> _logger;

        public HomeController(CourseContext context)
        {
            _context = context;
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

        public async Task<IActionResult> TestTemplateList()
        {
            if(TempData["Message"] != null)
            {
                ViewBag.Message = TempData["Message"].ToString();
            }
            return _context.TestTemplates != null ?
            View(await _context.TestTemplates.ToListAsync()) :
            Problem("Entity set 'CourseContext.TestTemplates'  is null.");
        }

        [HttpPost]
        [ValidateAntiForgeryToken]
        public async Task<IActionResult> TestTemplateList(string action)
        {
            if(action == "add")
            {
                List<TestTemplate> testTemplates = testController.LoadTestTemplates();
                int successCount = 0;
                int errorCount = 0;
                for (int i = 0; i < testTemplates.Count; i++)
                {
                    try
                    {
                        TestTemplate testTemplate = testTemplates[i];
                        _context.Add(testTemplate);
                        await _context.SaveChangesAsync();
                        List<QuestionTemplate> questionTemplates = questionController.LoadQuestionTemplates(testTemplate);
                        for (int j = 0; j < questionTemplates.Count; j++)
                        {
                            QuestionTemplate questionTemplate = questionTemplates[j];
                            _context.Add(questionTemplate);
                            await _context.SaveChangesAsync();
                            List<SubquestionTemplate> subquestionTemplates = questionController.LoadSubquestionTemplates(testTemplate.TestNameIdentifier, questionTemplate);
                            for (int k = 0; k < subquestionTemplates.Count; k++)
                            {
                                SubquestionTemplate subquestionTemplate = subquestionTemplates[k];
                                _context.Add(subquestionTemplate);
                                await _context.SaveChangesAsync();
                            }
                        }
                        successCount++;
                    }
                    catch//todo: error log
                    {
                        errorCount++;
                    }
                }
                TempData["Message"] = "Přidáno " + successCount + "testů (" + errorCount + " duplikátů nebo chyb).";
            }
            else
            {
                _context.Database.ExecuteSqlRaw("delete from TestTemplate");
                TempData["Message"] = "Byly odebrány všechny existující testy.";
            }
            return RedirectToAction(nameof(TestTemplateList));
        }

        public async Task<IActionResult> TestTemplate(string testNameIdentifier, string testNumberIdentifier)
        {
            return _context.QuestionTemplates != null ?
            View(await _context.QuestionTemplates
                .Include(q => q.TestTemplate)
                .Where(q => q.TestTemplate.TestNumberIdentifier == testNumberIdentifier).ToListAsync()) :
            //View(await _context.QuestionTemplates.ToListAsync()) :
            Problem("Entity set 'CourseContext.QuestionTemplates' is null.");
        }

        public async Task<IActionResult> QuestionTemplate(string testNameIdentifier, string testNumberIdentifier, string questionNameIdentifier, string questionNumberIdentifier)
        {
            ViewBag.SubquestionTypeTextArray = questionController.SubquestionTypeTextArray;
            return _context.SubquestionTemplates != null ?
            View(await _context.SubquestionTemplates
                .Include(q => q.QuestionTemplate)
                .Include(q => q.QuestionTemplate.TestTemplate)
                .Where(q => q.QuestionNumberIdentifier == questionNumberIdentifier).ToListAsync()) :
            Problem("Entity set 'CourseContext.QuestionTemplates' is null.");
        }

        [HttpPost]
        public async Task<IActionResult> QuestionTemplate(string testNameIdentifier, string testNumberIdentifier, string questionNameIdentifier, string questionNumberIdentifier, string subquestionIdentifier)
        {
            ViewBag.subquestionIdentifier = subquestionIdentifier;
            ViewBag.SubquestionTypeTextArray = questionController.SubquestionTypeTextArray;
            return _context.SubquestionTemplates != null ?
            View(await _context.SubquestionTemplates
                .Include(q => q.QuestionTemplate)
                .Include(q => q.QuestionTemplate.TestTemplate)
                .Where(q => q.QuestionNumberIdentifier == questionNumberIdentifier).ToListAsync()) :
            Problem("Entity set 'CourseContext.QuestionTemplates' is null.");
        }

        /*   [HttpGet]
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
           }*/

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
                /*QuestionResult = questionController.LoadQuestionResult(testNameIdentifier, 
                questionController.LoadQuestionTemplate(testNameIdentifier, testNumberIdentifier, questionNameIdentifier, questionNumberIdentifier), testResultIdentifier,
                questionController.LoadQuestionTemplate(testNameIdentifier, testNumberIdentifier, questionNameIdentifier, questionNumberIdentifier).SubquestionTemplateList[0].SubquestionIdentifier),*/
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
                /*QuestionResult = questionController.LoadQuestionResult(testNameIdentifier,
                questionController.LoadQuestionTemplate(testNameIdentifier, testNumberIdentifier, questionNameIdentifier, questionNumberIdentifier), testResultIdentifier,
                questionController.LoadQuestionTemplate(testNameIdentifier, testNumberIdentifier, questionNameIdentifier, questionNumberIdentifier).SubquestionTemplateList[0].SubquestionIdentifier),*/
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