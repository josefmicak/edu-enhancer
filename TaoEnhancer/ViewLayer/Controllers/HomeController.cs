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

        public async Task<IActionResult> TestTemplate(string testNumberIdentifier)
        {
            return _context.QuestionTemplates != null ?
            View(await _context.QuestionTemplates
                .Include(q => q.TestTemplate)
                .Include(q => q.SubquestionTemplateList)
                .Where(q => q.TestTemplate.TestNumberIdentifier == testNumberIdentifier).ToListAsync()) :
            Problem("Entity set 'CourseContext.QuestionTemplates' is null.");
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

            ViewBag.Message = message;
            return _context.QuestionTemplates != null ?
            View(await _context.QuestionTemplates
                .Include(q => q.TestTemplate)
                .Include(q => q.SubquestionTemplateList)
                .Where(q => q.TestTemplate.TestNumberIdentifier == testNumberIdentifier).ToListAsync()) :
            Problem("Entity set 'CourseContext.QuestionTemplates' is null.");
        }

        public async Task<IActionResult> QuestionTemplate(string questionNumberIdentifier)
        {
            ViewBag.SubquestionTypeTextArray = questionController.SubquestionTypeTextArray;
            return _context.SubquestionTemplates != null ?
            View(await _context.SubquestionTemplates
                .Include(q => q.QuestionTemplate)
                .Include(q => q.QuestionTemplate.TestTemplate)
                .Where(q => q.QuestionNumberIdentifier == questionNumberIdentifier).ToListAsync()) :
            Problem("Entity set 'CourseContext.SubquestionTemplates' is null.");
        }

        [HttpPost]
        public async Task<IActionResult> QuestionTemplate(string questionNumberIdentifier, string subquestionIdentifier, string subquestionPoints)
        {
            if(subquestionPoints != null)
            {
                subquestionPoints = subquestionPoints.Replace(".", ",");
            }
            string? message = null;
            //the teacher is changing points of the subquestion
            if (subquestionPoints != null)
            {
                var subquestionTemplate = _context.SubquestionTemplates.FirstOrDefault(s => s.QuestionNumberIdentifier == questionNumberIdentifier && s.SubquestionIdentifier == subquestionIdentifier);
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
            ViewBag.Message = message;
            ViewBag.subquestionIdentifier = subquestionIdentifier;
            ViewBag.SubquestionTypeTextArray = questionController.SubquestionTypeTextArray;
            return _context.SubquestionTemplates != null ?
            View(await _context.SubquestionTemplates
                .Include(q => q.QuestionTemplate)
                .Include(q => q.QuestionTemplate.TestTemplate)
                .Where(q => q.QuestionNumberIdentifier == questionNumberIdentifier).ToListAsync()) :
            Problem("Entity set 'CourseContext.SubquestionTemplates' is null.");
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
                .Include(q => q.QuestionTemplate.SubquestionTemplateList)
                .Include(s => s.TestResult.Student)
                .Include(q => q.SubquestionResultList)
                .Where(t => t.TestResultIdentifier == testResultIdentifier).ToListAsync()) :
            Problem("Entity set 'CourseContext.QuestionResults'  is null.");
        }

        public async Task<IActionResult> ManageSolvedQuestion(string questionNumberIdentifier, string testResultIdentifier)
        {
            ViewBag.SubquestionTypeTextArray = questionController.SubquestionTypeTextArray;
            return _context.SubquestionResults != null ?
            View(await _context.SubquestionResults
                .Include(s => s.SubquestionTemplate)
                .Include(s => s.QuestionResult)
                .Include(s => s.QuestionResult.QuestionTemplate)
                .Include(s => s.QuestionResult.QuestionTemplate.TestTemplate)
                .Where(s => s.TestResultIdentifier == testResultIdentifier && s.QuestionNumberIdentifier == questionNumberIdentifier).ToListAsync()) :
            Problem("Entity set 'CourseContext.SubquestionResults' is null.");
        }

        [HttpPost]
        public async Task<IActionResult> ManageSolvedQuestion(string questionNumberIdentifier, string testResultIdentifier, string subquestionIdentifier, string studentsPoints, string subquestionPoints, string negativePoints)
        {
            if(studentsPoints != null)
            {
                studentsPoints = studentsPoints.Replace(".", ",");
            }
            string? message = null;
            //the teacher is changing student's points
            if(studentsPoints != null)
            {
                var subquestionResult = _context.SubquestionResults
                    .FirstOrDefault(s => s.TestResultIdentifier == testResultIdentifier && s.QuestionNumberIdentifier == questionNumberIdentifier && s.SubquestionIdentifier == subquestionIdentifier);
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
            ViewBag.Message = message;
            ViewBag.subquestionIdentifier = subquestionIdentifier;
            ViewBag.SubquestionTypeTextArray = questionController.SubquestionTypeTextArray;
            return _context.SubquestionResults != null ?
            View(await _context.SubquestionResults
                .Include(s => s.SubquestionTemplate)
                .Include(s => s.QuestionResult)
                .Include(s => s.QuestionResult.QuestionTemplate)
                .Include(s => s.QuestionResult.QuestionTemplate.TestTemplate)
                .Where(s => s.TestResultIdentifier == testResultIdentifier && s.QuestionNumberIdentifier == questionNumberIdentifier).ToListAsync()) :
            Problem("Entity set 'CourseContext.SubquestionTemplates' is null.");
        }

        public IActionResult StudentMenu()
        {
            return View(new StudentMenuModel
            {
                Title = "Student",
                Students = studentController.LoadStudents()
            });
        }

        public async Task<IActionResult> BrowseSolvedTestList(string studentIdentifier)
        {
            return _context.TestResults != null ?
            View(await _context.TestResults
                .Include(t => t.Student)
                .Where(t => t.Student.StudentIdentifier == studentIdentifier).ToListAsync()) :
            Problem("Entity set 'CourseContext.TestResults'  is null.");
        }

        public async Task<IActionResult> BrowseSolvedTest(string testResultIdentifier)
        {
            return _context.QuestionResults != null ?
            View(await _context.QuestionResults
                .Include(t => t.TestResult)
                .Include(q => q.QuestionTemplate)
                .Include(q => q.QuestionTemplate.TestTemplate)
                .Include(q => q.QuestionTemplate.SubquestionTemplateList)
                .Include(s => s.TestResult.Student)
                .Include(q => q.SubquestionResultList)
                .Where(q => q.TestResultIdentifier == testResultIdentifier).ToListAsync()) :
            Problem("Entity set 'CourseContext.QuestionResults'  is null.");
        }

        public async Task<IActionResult> BrowseSolvedQuestion(string questionNumberIdentifier, string testResultIdentifier)
        {
            ViewBag.SubquestionTypeTextArray = questionController.SubquestionTypeTextArray;
            return _context.SubquestionResults != null ?
            View(await _context.SubquestionResults
                .Include(s => s.SubquestionTemplate)
                .Include(s => s.QuestionResult)
                .Include(s => s.QuestionResult.QuestionTemplate)
                .Include(s => s.QuestionResult.QuestionTemplate.TestTemplate)
                .Where(s => s.TestResultIdentifier == testResultIdentifier && s.QuestionNumberIdentifier == questionNumberIdentifier).ToListAsync()) :
            Problem("Entity set 'CourseContext.SubquestionResults' is null.");
        }

        [HttpPost]
        public async Task<IActionResult> BrowseSolvedQuestion(string questionNumberIdentifier, string testResultIdentifier, string subquestionIdentifier)
        {
            ViewBag.subquestionIdentifier = subquestionIdentifier;
            ViewBag.SubquestionTypeTextArray = questionController.SubquestionTypeTextArray;
            return _context.SubquestionResults != null ?
            View(await _context.SubquestionResults
                .Include(s => s.SubquestionTemplate)
                .Include(s => s.QuestionResult)
                .Include(s => s.QuestionResult.QuestionTemplate)
                .Include(s => s.QuestionResult.QuestionTemplate.TestTemplate)
                .Where(s => s.TestResultIdentifier == testResultIdentifier && s.QuestionNumberIdentifier == questionNumberIdentifier).ToListAsync()) :
            Problem("Entity set 'CourseContext.SubquestionResults' is null.");
        }

        [ResponseCache(Duration = 0, Location = ResponseCacheLocation.None, NoStore = true)]
        public IActionResult Error()
        {
            return View(new ErrorViewModel { RequestId = Activity.Current?.Id ?? HttpContext.TraceIdentifier });
        }
    }
}