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

        public IActionResult TestTemplate(string testNameIdentifier, string testNumberIdentifier)
        {
            ViewBag.testNameIdentifier = testNameIdentifier;
            ViewBag.testNumberIdentifier = testNumberIdentifier;
            return View();
        }

        [ResponseCache(Duration = 0, Location = ResponseCacheLocation.None, NoStore = true)]
        public IActionResult Error()
        {
            return View(new ErrorViewModel { RequestId = Activity.Current?.Id ?? HttpContext.TraceIdentifier });
        }
    }
}