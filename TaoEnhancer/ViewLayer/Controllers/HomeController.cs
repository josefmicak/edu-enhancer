using Microsoft.AspNetCore.Mvc;
using System.Diagnostics;
using ViewLayer.Models;

namespace ViewLayer.Controllers
{
    public class HomeController : Controller
    {
        private readonly ILogger<HomeController> _logger;

        public HomeController(ILogger<HomeController> logger)
        {
            _logger = logger;
        }

        public IActionResult Index()
        {
            return View();
        }

        public IActionResult TeacherMenu()
        {
            return View();
        }

        public IActionResult StudentMenu()
        {
            return View();
        }

        public IActionResult TestList()
        {
            return View();
        }

        public IActionResult SolvedTestList()
        {
            return View();
        }

        [HttpGet]
        public IActionResult TestTemplate(string testNameIdentifier, string testNumberIdentifier)
        {
            ViewBag.testNameIdentifier = testNameIdentifier;
            ViewBag.testNumberIdentifier = testNumberIdentifier;
            return View();
        }

        [HttpPost]
        public IActionResult TestTemplate(string testNameIdentifier, string testNumberIdentifier, string negativePoints)
        {
            string testPath = "C:\\xampp\\exported\\tests\\" + testNameIdentifier + "\\tests\\" + testNumberIdentifier;
            string textToWrite = "";
            ViewBag.testNameIdentifier = testNameIdentifier;
            ViewBag.testNumberIdentifier = testNumberIdentifier;
            ViewBag.negativePoints = negativePoints;

            if (negativePoints == "negativePoints_no")
            {
                textToWrite = "0";
            }
            else if (negativePoints == "negativePoints_yes")
            {
                textToWrite = "1";
            }
            System.IO.File.WriteAllText(testPath + "\\NegativePoints.txt", textToWrite);
            return View();
        }

        [HttpGet]
        public IActionResult ItemTemplate(string testNameIdentifier, string testNumberIdentifier, string itemNumberIdentifier, string itemNameIdentifier)
        {
            ViewBag.testNameIdentifier = testNameIdentifier;
            ViewBag.testNumberIdentifier = testNumberIdentifier;
            ViewBag.itemNumberIdentifier = itemNumberIdentifier;
            ViewBag.itemNameIdentifier = itemNameIdentifier;
            return View();
        }

        [HttpPost]
        public IActionResult ItemTemplate(string testNameIdentifier, string testNumberIdentifier, string itemNumberIdentifier, string itemNameIdentifier, string selectedSubitem, string subquestionPoints, string wrongChoicePoints, string recommendedWrongChoicePoints, string selectedWrongChoicePoints, int correctChoicePoints)
        {
            ViewBag.testNameIdentifier = testNameIdentifier;
            ViewBag.testNumberIdentifier = testNumberIdentifier;
            ViewBag.itemNumberIdentifier = itemNumberIdentifier;
            ViewBag.itemNameIdentifier = itemNameIdentifier;
            ViewBag.selectedSubitem = selectedSubitem;

            if(subquestionPoints != null)
            {
                ViewBag.subquestionPoints = subquestionPoints;
                bool isNumber = int.TryParse(subquestionPoints, out _);
                if(wrongChoicePoints == "wrongChoicePoints_recommended")
                {
                    wrongChoicePoints = recommendedWrongChoicePoints;
                }
                else
                {
                    wrongChoicePoints = selectedWrongChoicePoints;
                }

                if(recommendedWrongChoicePoints == "N/A")
                {
                    wrongChoicePoints = (correctChoicePoints * (-1)).ToString();
                }
                
                bool isWrongChoicePointsNumber = double.TryParse(wrongChoicePoints, out _);

                if (!isNumber || !isWrongChoicePointsNumber)
                {
                    ViewBag.ErrorText = "Chyba: je nutné zadat číslo.";
                }
                else
                {
                    bool performSave = true;
                    int subquestionPointsToSave = int.Parse(subquestionPoints);
                    double wrongChoicePointsToSave = double.Parse(wrongChoicePoints);

                    if (Math.Abs(wrongChoicePointsToSave) > subquestionPointsToSave)
                    {
                        performSave = false;
                        ViewBag.ErrorText = "Chyba: za špatný výběr bude studentovi odečteno více bodů, než kolik může dostat za otázku.";
                    }

                    if(wrongChoicePointsToSave > 0)
                    {
                        performSave = false;
                        ViewBag.ErrorText = "Chyba: za špatnou volbu nemůže být udělen kladný počet bodů.";
                    }

                    if (subquestionPointsToSave < 0)
                    {
                        performSave = false;
                        ViewBag.ErrorText = "Chyba: zza správnou odpověď nemůže být udělen záporný počet bodů.";
                    }

                    if (performSave)
                    {
                        string itemParentPath = "C:\\xampp\\exported\\tests\\" + testNameIdentifier + "\\items\\" + itemNumberIdentifier;

                        string[] importedFileLines = System.IO.File.ReadAllLines(itemParentPath + "\\Points.txt");
                        string fileLinesToExport = "";
                        for (int i = 0; i < importedFileLines.Length; i++)
                        {
                            string[] splitImportedFileLineBySemicolon = importedFileLines[i].Split(";");
                            if (splitImportedFileLineBySemicolon[0] == selectedSubitem)
                            {
                                importedFileLines[i] = selectedSubitem + ";" + subquestionPointsToSave + ";" + Math.Round(wrongChoicePointsToSave, 2);
                            }
                            fileLinesToExport += importedFileLines[i] + "\n";
                        }
                        System.IO.File.WriteAllText(itemParentPath + "\\Points.txt", fileLinesToExport);
                    }
                }
            }
            return View();
        }

        [ResponseCache(Duration = 0, Location = ResponseCacheLocation.None, NoStore = true)]
        public IActionResult Error()
        {
            return View(new ErrorViewModel { RequestId = Activity.Current?.Id ?? HttpContext.TraceIdentifier });
        }
    }
}