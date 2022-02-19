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
        public IActionResult ItemTemplate(string testNameIdentifier, string testNumberIdentifier, string itemNumberIdentifier, string itemNameIdentifier, string selectedSubitem)
        {
            ViewBag.testNameIdentifier = testNameIdentifier;
            ViewBag.testNumberIdentifier = testNumberIdentifier;
            ViewBag.itemNumberIdentifier = itemNumberIdentifier;
            ViewBag.itemNameIdentifier = itemNameIdentifier;
            ViewBag.selectedSubitem = selectedSubitem;
            return View();
        }

        [ResponseCache(Duration = 0, Location = ResponseCacheLocation.None, NoStore = true)]
        public IActionResult Error()
        {
            return View(new ErrorViewModel { RequestId = Activity.Current?.Id ?? HttpContext.TraceIdentifier });
        }
    }
}