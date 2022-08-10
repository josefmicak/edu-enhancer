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
        //private QuestionController questionController = new QuestionController();
        private QuestionController questionController;
        private StudentController studentController = new StudentController();
        private TestController testController;

        private readonly ILogger<HomeController> _logger;

        public HomeController(CourseContext context)
        {
            _context = context;
            questionController = new QuestionController(context);
            testController = new TestController(context);
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
                        _context.ChangeTracker.Clear();
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

                for (int i = 0; i < testTemplates.Count; i++)
                {
                    TestTemplate testTemplate = testTemplates[i];
                    List<QuestionTemplate> questionTemplates = questionController.LoadQuestionTemplates(testTemplate);
                    for (int j = 0; j < questionTemplates.Count; j++)
                    {
                        try
                        {
                            QuestionTemplate questionTemplate = questionTemplates[j];
                            _context.ChangeTracker.Clear();
                            _context.TestTemplates.Attach(testTemplate);
                            _context.QuestionTemplates.Add(questionTemplate);
                            await _context.SaveChangesAsync();
                            successCount++;
                        }
                        catch (Exception ex)
                        {
                            Debug.WriteLine(ex.Message);
                            errorCount++;
                        }
                    }
                }

                TempData["Message"] += "Přidáno " + successCount + " šablon otázek (" + errorCount + " duplikátů nebo chyb).";
                successCount = 0;
                errorCount = 0;

                for (int i = 0; i < testTemplates.Count; i++)
                {
                    TestTemplate testTemplate = testTemplates[i];
                    List<QuestionTemplate> questionTemplates = questionController.LoadQuestionTemplates(testTemplate);
                    for (int j = 0; j < questionTemplates.Count; j++)
                    {
                        QuestionTemplate questionTemplate = questionTemplates[j];
                        List<SubquestionTemplate> subquestionTemplates = questionController.LoadSubquestionTemplates(testTemplate.TestNameIdentifier, questionTemplate);
                        for (int k = 0; k < subquestionTemplates.Count; k++)
                        {
                            try
                            {
                                SubquestionTemplate subquestionTemplate = subquestionTemplates[k];
                                _context.ChangeTracker.Clear();
                                _context.TestTemplates.Attach(testTemplate);
                                _context.QuestionTemplates.Attach(questionTemplate);
                                _context.SubquestionTemplates.Add(subquestionTemplate);
                                await _context.SaveChangesAsync();
                                successCount++;
                            }
                            catch (Exception ex)
                            {
                                Debug.WriteLine(ex.Message);
                                errorCount++;
                            }
                        }
                    }
                }

                TempData["Message"] += "Přidáno " + successCount + " šablon podotázek (" + errorCount + " duplikátů nebo chyb).";
            }
            else
            {
                _context.Database.ExecuteSqlRaw("delete from SubquestionTemplate");
                _context.Database.ExecuteSqlRaw("delete from QuestionTemplate");
                _context.Database.ExecuteSqlRaw("delete from TestTemplate");
                TempData["Message"] = "Byly smazány všechny existující testy.";
            }
            return RedirectToAction(nameof(TestTemplateList));
        }

        public async Task<IActionResult> TestTemplate(string testNameIdentifier, string testNumberIdentifier)
        {
            return _context.QuestionTemplates != null ?
            View(await _context.QuestionTemplates
                .Include(q => q.TestTemplate)
                .Where(q => q.TestTemplate.TestNumberIdentifier == testNumberIdentifier).ToListAsync()) :
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

        public async Task<IActionResult> ManageStudentList()
        {
            if (TempData["Message"] != null)
            {
                ViewBag.Message = TempData["Message"].ToString();
            }
            return _context.Students != null ?
            View(await _context.Students.ToListAsync()) :
            Problem("Entity set 'CourseContext.Students'  is null.");
        }

        [HttpPost]
        [ValidateAntiForgeryToken]
        public async Task<IActionResult> ManageStudentList(string action)
        {
            if (action == "add")
            {
                List<Student> students = studentController.LoadStudents();

                int successCount = 0;
                int errorCount = 0;
                for (int i = 0; i < students.Count; i++)
                {
                    try
                    {
                        Student student = students[i];
                        _context.Add(student);
                        await _context.SaveChangesAsync();
                        successCount++;
                    }
                    catch(Exception ex)
                    {
                        Debug.WriteLine(ex.Message);
                        errorCount++;
                    }
                }
                TempData["Message"] = "Přidáno " + successCount + "studentů (" + errorCount + " duplikátů nebo chyb).";
            }
            else
            {
                _context.Database.ExecuteSqlRaw("delete from Student");
                TempData["Message"] = "Byly smazáni všichni existující studenti.";
            }
            return RedirectToAction(nameof(ManageStudentList));
        }

        public async Task<IActionResult> ManageSolvedTestList()
        {
            if (TempData["Message"] != null)
            {
                ViewBag.Message = TempData["Message"].ToString();
            }
            return _context.TestResults != null ?
            View(await _context.TestResults.
                Include(s => s.Student).
                Include(t => t.TestTemplate).ToListAsync()) :
            Problem("Entity set 'CourseContext.TestResults'  is null.");
        }

        [HttpPost]
        [ValidateAntiForgeryToken]
        public async Task<IActionResult> ManageSolvedTestList(string action)
        {
            if (action == "add")
            {
                List<TestResult> testResults = testController.LoadTestResults();
                int successCount = 0;
                int errorCount = 0;
                for (int i = 0; i < testResults.Count; i++)
                {
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
                    catch(Exception ex)
                    {
                        Debug.WriteLine(ex.Message);
                        errorCount++;
                    }
                }

                TempData["Message"] = "Přidáno " + successCount + " řešení testů (" + errorCount + " duplikátů nebo chyb).";
                successCount = 0;
                errorCount = 0;

                for (int i = 0; i < testResults.Count; i++)
                {
                    TestResult testResult = testResults[i];
                    List<QuestionResult> questionResults = questionController.LoadQuestionResults(testResult, testResult.TestTemplate);
                    for (int j = 0; j < questionResults.Count; j++)
                    {
                        try
                        {
                            QuestionResult questionResult = questionResults[j];
                            _context.ChangeTracker.Clear();
                            _context.Students.Attach(testResult.Student);
                            _context.QuestionTemplates.Attach(questionResult.QuestionTemplate);
                            _context.TestResults.Attach(testResult);
                            _context.QuestionResults.Add(questionResult);
                            await _context.SaveChangesAsync();
                            successCount++;
                        }
                        catch (Exception ex)
                        {
                            Debug.WriteLine(ex.Message);
                            errorCount++;
                        }
                    }
                }

                TempData["Message"] += "\nPřidáno " + successCount + " řešení otázek (" + errorCount + " duplikátů nebo chyb).";
                successCount = 0;
                errorCount = 0;

                for (int i = 0; i < testResults.Count; i++)
                {
                    try
                    {
                        TestResult testResult = testResults[i];
                        List<QuestionResult> questionResults = questionController.LoadQuestionResults(testResult, testResult.TestTemplate);
                        for (int j = 0; j < questionResults.Count; j++)
                        {
                            QuestionResult questionResult = questionResults[j];
                            List<SubquestionResult> subquestionResults = questionController.LoadSubquestionResults(questionResult);
                            for (int k = 0; k < subquestionResults.Count; k++)
                            {
                                SubquestionResult subquestionResult = subquestionResults[k];
                                _context.ChangeTracker.Clear();
                                _context.Students.Attach(testResult.Student);
                                _context.TestResults.Attach(testResult);
                                _context.QuestionResults.Attach(questionResult);
                                _context.QuestionTemplates.Attach(questionResult.QuestionTemplate);
                                _context.SubquestionTemplates.Attach(subquestionResult.SubquestionTemplate);
                                _context.SubquestionResults.Add(subquestionResult);
                                await _context.SaveChangesAsync();
                                successCount++;
                            }
                        }
                    }
                    catch (Exception ex)
                    {
                        Debug.WriteLine(ex.Message);
                        errorCount++;
                    }
                }

                TempData["Message"] += "\nPřidáno " + successCount + " řešení podotázek (" + errorCount + " duplikátů nebo chyb).";
            }
            else
            {
                _context.Database.ExecuteSqlRaw("delete from SubquestionResult");
                _context.Database.ExecuteSqlRaw("delete from QuestionResult");
                _context.Database.ExecuteSqlRaw("delete from TestResult");
                TempData["Message"] = "Byly smazány všechna existující řešení testů.";
            }
            return RedirectToAction(nameof(ManageSolvedTestList));
        }

        public async Task<IActionResult> ManageSolvedTest(string testResultIdentifier)
        {
            return _context.QuestionResults != null ?
            View(await _context.QuestionResults
                .Include(t => t.TestResult)
                .Include(q => q.QuestionTemplate)
                .Include(q => q.QuestionTemplate.TestTemplate)
                .Include(s => s.TestResult.Student)
                .Where(t => t.TestResultIdentifier == testResultIdentifier).ToListAsync()) :
            Problem("Entity set 'CourseContext.QuestionResults'  is null.");
        }

        public async Task<IActionResult> ManageSolvedQuestion(string testNameIdentifier, string testNumberIdentifier, string questionNameIdentifier, string questionNumberIdentifier, string testResultIdentifier)
        {
            ViewBag.SubquestionTypeTextArray = questionController.SubquestionTypeTextArray;
            return _context.SubquestionResults != null ?
            View(await _context.SubquestionResults
                .Include(s => s.SubquestionTemplate)
                .Include(s => s.QuestionResult)
                .Include(s => s.QuestionResult.QuestionTemplate)
                .Include(s => s.QuestionResult.QuestionTemplate.TestTemplate)
                //.Include(s => s.QuestionResult.TestResult)//todo: include mozna odebrat?
                .Where(s => s.TestResultIdentifier == testResultIdentifier && s.QuestionNumberIdentifier == questionNumberIdentifier).ToListAsync()) :
            Problem("Entity set 'CourseContext.SubquestionResults' is null.");

        }

        [HttpPost]
        public async Task<IActionResult> ManageSolvedQuestion(string testNameIdentifier, string testNumberIdentifier, string questionNameIdentifier, string questionNumberIdentifier, string testResultIdentifier, string subquestionIdentifier)
        {
            ViewBag.subquestionIdentifier = subquestionIdentifier;
            ViewBag.SubquestionTypeTextArray = questionController.SubquestionTypeTextArray;
            return _context.SubquestionResults != null ?
            View(await _context.SubquestionResults
                .Include(s => s.SubquestionTemplate)
                .Include(s => s.QuestionResult)
                .Include(s => s.QuestionResult.QuestionTemplate)
                .Include(s => s.QuestionResult.QuestionTemplate.TestTemplate)
                //.Include(s => s.QuestionResult.TestResult)//todo: include mozna odebrat?
                .Where(s => s.TestResultIdentifier == testResultIdentifier && s.QuestionNumberIdentifier == questionNumberIdentifier).ToListAsync()) :
            Problem("Entity set 'CourseContext.SubquestionTemplates' is null.");
        }

        /*[HttpGet]
        public IActionResult ManageSolvedQuestion(string testNameIdentifier, string testNumberIdentifier, string questionNameIdentifier, string questionNumberIdentifier, string testResultIdentifier)
        {
            return View(new ManageSolvedQuestionModel
            {
                Title = "Správa vyřešeného testu " + testResultIdentifier + ", otázka " + questionNumberIdentifier,
                QuestionTemplate = questionController.LoadQuestionTemplate(testNameIdentifier, testNumberIdentifier, questionNameIdentifier, questionNumberIdentifier),
                /*QuestionResult = questionController.LoadQuestionResult(testNameIdentifier, 
                questionController.LoadQuestionTemplate(testNameIdentifier, testNumberIdentifier, questionNameIdentifier, questionNumberIdentifier), testResultIdentifier,
                questionController.LoadQuestionTemplate(testNameIdentifier, testNumberIdentifier, questionNameIdentifier, questionNumberIdentifier).SubquestionTemplateList[0].SubquestionIdentifier),*/
        /*  TestNameIdentifier = testNameIdentifier,
          TestNumberIdentifier = testNumberIdentifier,
          TestResultIdentifier = testResultIdentifier,
          SubquestionTypeTextArray = questionController.SubquestionTypeTextArray
      });
  }*/
        /*
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
        }*/

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